"""Online same-target recombination from selected physical train windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import normalize_row_col


EVENTS_PER_WINDOW = 400
EVENTS_FROM_EACH_SOURCE = 200


@dataclass(frozen=True)
class SameTargetAnchor:
    source_index: int
    preferred_partner_indices: tuple[int, ...]
    adjacent_partner_indices: tuple[int, ...]


def sample_event_occurrences(
    event_frame: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample event occurrences without replacement from a count frame."""
    flat = np.asarray(event_frame).reshape(-1)
    if np.any(flat < 0):
        raise ValueError("event counts must be non-negative")
    total = int(flat.sum())
    if not 0 <= sample_count <= total:
        raise ValueError(f"cannot sample {sample_count} occurrences from {total}")
    occurrences = np.repeat(np.arange(flat.size, dtype=np.int32), flat.astype(np.int64))
    chosen = rng.choice(total, size=sample_count, replace=False)
    sampled = np.bincount(occurrences[chosen], minlength=flat.size)
    return sampled.reshape(event_frame.shape).astype(np.uint16, copy=False)


def select_same_target_anchors(
    *,
    target_indices: np.ndarray,
    start_event_indices: np.ndarray,
    augmented: np.ndarray,
    backfilled: np.ndarray,
    include_stride_sources: bool,
) -> tuple[np.ndarray, list[SameTargetAnchor]]:
    """Build source-wise non-overlapping same-target partner pools."""
    arrays = tuple(
        np.asarray(value)
        for value in (target_indices, start_event_indices, augmented, backfilled)
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("anchor-selection arrays must have equal length")
    if np.any(arrays[3].astype(bool)):
        raise ValueError("the no-tail cache must not contain backfilled samples")

    source_pool = (
        np.arange(len(arrays[0]), dtype=np.int64)
        if include_stride_sources
        else np.flatnonzero(~arrays[2].astype(bool)).astype(np.int64)
    )
    by_target: dict[int, list[int]] = {}
    for index in source_pool:
        by_target.setdefault(int(arrays[0][index]), []).append(int(index))

    anchors: list[SameTargetAnchor] = []
    for source_index in source_pool:
        source_start = int(arrays[1][source_index])
        preferred: list[int] = []
        adjacent: list[int] = []
        for partner_index in by_target[int(arrays[0][source_index])]:
            if partner_index == source_index:
                continue
            difference = abs(int(arrays[1][partner_index]) - source_start)
            if difference > EVENTS_PER_WINDOW:
                preferred.append(partner_index)
            elif difference == EVENTS_PER_WINDOW:
                adjacent.append(partner_index)
        if preferred or adjacent:
            key = lambda index: (int(arrays[1][index]), int(index))
            anchors.append(
                SameTargetAnchor(
                    source_index=int(source_index),
                    preferred_partner_indices=tuple(sorted(preferred, key=key)),
                    adjacent_partner_indices=tuple(sorted(adjacent, key=key)),
                )
            )
    return source_pool, anchors


def select_nonstride_same_target_anchors(
    *,
    target_indices: np.ndarray,
    start_event_indices: np.ndarray,
    augmented: np.ndarray,
    backfilled: np.ndarray,
) -> tuple[np.ndarray, list[SameTargetAnchor]]:
    """Build partner pools using complete non-stride windows only."""
    return select_same_target_anchors(
        target_indices=target_indices,
        start_event_indices=start_event_indices,
        augmented=augmented,
        backfilled=backfilled,
        include_stride_sources=False,
    )


class NonstrideSameTargetRecombinationDataset(Dataset):
    """Keep all physical train samples and append one mix per eligible base source."""

    def __init__(
        self,
        source_dataset: Dataset,
        *,
        seed: int,
        include_stride_sources: bool = False,
    ) -> None:
        required = (
            "event_frames",
            "labels",
            "target_indices",
            "start_event_indices",
            "augmented",
            "backfilled",
            "sample_weights",
            "block_names",
            "target_counts",
        )
        missing = [name for name in required if not hasattr(source_dataset, name)]
        if missing:
            raise ValueError(f"source dataset lacks recombination metadata: {missing}")
        self.source_dataset = source_dataset
        self.seed = int(seed)
        self.epoch = 0
        self.include_stride_sources = bool(include_stride_sources)
        self.source_pool_indices, self.anchors = select_same_target_anchors(
            target_indices=source_dataset.target_indices,
            start_event_indices=source_dataset.start_event_indices,
            augmented=source_dataset.augmented,
            backfilled=source_dataset.backfilled,
            include_stride_sources=self.include_stride_sources,
        )
        self.original_sample_count = len(source_dataset)
        self.synthetic_sample_count = len(self.anchors)
        self.sample_weights = source_dataset.sample_weights
        self.target_counts = source_dataset.target_counts
        self.preferred_source_count = sum(
            bool(anchor.preferred_partner_indices) for anchor in self.anchors
        )
        self.adjacent_fallback_source_count = sum(
            not anchor.preferred_partner_indices for anchor in self.anchors
        )
        self.ineligible_source_count = len(self.source_pool_indices) - len(self.anchors)
        self.synthetic_target_coverage = len(
            {
                int(source_dataset.target_indices[anchor.source_index])
                for anchor in self.anchors
            }
        )
        self.eligible_stride_source_count = sum(
            bool(source_dataset.augmented[anchor.source_index])
            for anchor in self.anchors
        )
        selectable_partners = {
            partner_index
            for anchor in self.anchors
            for partner_index in (
                anchor.preferred_partner_indices
                if anchor.preferred_partner_indices
                else anchor.adjacent_partner_indices
            )
        }
        self.selectable_partner_stride_count = sum(
            bool(source_dataset.augmented[index]) for index in selectable_partners
        )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.original_sample_count + self.synthetic_sample_count

    def _select_partner(self, anchor: SameTargetAnchor) -> int:
        partners = (
            anchor.preferred_partner_indices
            if anchor.preferred_partner_indices
            else anchor.adjacent_partner_indices
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, anchor.source_index, 0])
        )
        return int(partners[int(rng.integers(len(partners)))])

    def _sample_half(self, source_index: int, anchor_source_index: int, slot: int) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch, anchor_source_index, slot]
            )
        )
        return sample_event_occurrences(
            self.source_dataset.event_frames[source_index],
            EVENTS_FROM_EACH_SOURCE,
            rng,
        )

    def __getitem__(self, index: int) -> dict:
        if index < self.original_sample_count:
            return self.source_dataset[index]
        anchor = self.anchors[index - self.original_sample_count]
        partner_index = self._select_partner(anchor)
        source_index = anchor.source_index
        if not self.include_stride_sources and (
            bool(self.source_dataset.augmented[source_index])
            or bool(self.source_dataset.augmented[partner_index])
        ):
            raise RuntimeError("stride sample entered recombination")
        if int(self.source_dataset.target_indices[source_index]) != int(
            self.source_dataset.target_indices[partner_index]
        ):
            raise RuntimeError("recombination partner target mismatch")
        difference = abs(
            int(self.source_dataset.start_event_indices[source_index])
            - int(self.source_dataset.start_event_indices[partner_index])
        )
        if difference < EVENTS_PER_WINDOW:
            raise RuntimeError("recombination source intervals overlap")

        event_frame = self._sample_half(source_index, source_index, 1)
        event_frame += self._sample_half(partner_index, source_index, 2)
        if int(event_frame.sum()) != EVENTS_PER_WINDOW:
            raise RuntimeError("recombined frame does not contain exactly 400 events")
        label_px = torch.from_numpy(
            self.source_dataset.labels[source_index].astype(np.float32, copy=False)
        )
        return {
            "event_frame": torch.from_numpy(event_frame).float(),
            "target_norm": normalize_row_col(label_px),
            "target_row_col_px": label_px,
            "target_index": int(self.source_dataset.target_indices[source_index]),
            "sample_index": index,
            "sample_weight": float(self.source_dataset.sample_weights[source_index]),
            "block_name": str(self.source_dataset.block_names[source_index]),
            "duration_us": -1,
            "augmented": False,
            "backfilled": False,
            "synthetic": True,
            "source_indices": torch.tensor(
                [source_index, partner_index], dtype=torch.int64
            ),
        }

    def audit_summary(self) -> dict:
        return {
            "enabled": True,
            "online": True,
            "new_cache": False,
            "physical_train_samples": self.original_sample_count,
            "physical_stride200_samples": int(
                np.asarray(self.source_dataset.augmented, dtype=bool).sum()
            ),
            "source_pool_rule": (
                "all physical train samples"
                if self.include_stride_sources
                else "physical train augmented == False"
            ),
            "source_pool_samples": len(self.source_pool_indices),
            "source_pool_stride_samples": int(
                np.asarray(
                    self.source_dataset.augmented[self.source_pool_indices],
                    dtype=bool,
                ).sum()
            ),
            "eligible_source_stride_samples": self.eligible_stride_source_count,
            "eligible_source_nonstride_samples": (
                self.synthetic_sample_count - self.eligible_stride_source_count
            ),
            "partner_pool_stride_samples": self.selectable_partner_stride_count,
            "synthetic_samples_per_epoch": self.synthetic_sample_count,
            "preferred_nonadjacent_sources": self.preferred_source_count,
            "adjacent_fallback_sources": self.adjacent_fallback_source_count,
            "ineligible_sources": self.ineligible_source_count,
            "synthetic_target_coverage": self.synthetic_target_coverage,
            "partner_reuse": True,
            "events_from_each_source": EVENTS_FROM_EACH_SOURCE,
            "events_per_synthetic_sample": EVENTS_PER_WINDOW,
        }


class AllPhysicalSameTargetRecombinationDataset(
    NonstrideSameTargetRecombinationDataset
):
    """Append one mix for every eligible physical source, including stride windows."""

    def __init__(self, source_dataset: Dataset, *, seed: int) -> None:
        super().__init__(
            source_dataset,
            seed=seed,
            include_stride_sources=True,
        )


__all__ = [
    "AllPhysicalSameTargetRecombinationDataset",
    "NonstrideSameTargetRecombinationDataset",
    "SameTargetAnchor",
    "sample_event_occurrences",
    "select_same_target_anchors",
    "select_nonstride_same_target_anchors",
]
