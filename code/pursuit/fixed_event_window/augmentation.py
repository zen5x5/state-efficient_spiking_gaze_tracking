"""Local 200+200 event recombination for pursuit train windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .ann.metrics import normalize_row_col


RECOMBINATION_MODE_EPOCH_RANDOM = "epoch_random"
RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS = "fixed_inner_contiguous"
RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM = "source_wise_epoch_random"
RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM = (
    "all_physical_source_epoch_random"
)
RECOMBINATION_MODES = (
    RECOMBINATION_MODE_EPOCH_RANDOM,
    RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS,
    RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM,
    RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM,
)
SOURCE_WISE_START_DIFFERENCES_EVENTS = (800, 1200, 1600)


@dataclass(frozen=True)
class LocalRecombinationPair:
    first_index: int
    second_index: int
    label_distance_px: float


@dataclass(frozen=True)
class LocalRecombinationAnchor:
    source_index: int
    partner_indices: tuple[int, ...]


@dataclass(frozen=True)
class AllPhysicalRecombinationAnchor:
    source_index: int
    preferred_partner_indices: tuple[int, ...]
    adjacent_partner_indices: tuple[int, ...]


def select_local_recombination_pairs(
    *,
    labels: np.ndarray,
    group_indices: np.ndarray,
    start_event_indices: np.ndarray,
    augmented: np.ndarray,
    backfilled: np.ndarray,
    events_per_window: int = 400,
    max_label_distance_px: float = 32.0,
) -> list[LocalRecombinationPair]:
    """Select train pairs separated by one complete 400-event window slot.

    Only the two source windows are read.  The intervening window may belong to
    train, validation or test; its events and label are never used.  Stride and
    tail-backfilled windows are never sources.
    """
    lengths = {
        len(labels),
        len(group_indices),
        len(start_event_indices),
        len(augmented),
        len(backfilled),
    }
    if len(lengths) != 1:
        raise ValueError("pair-selection arrays must have equal length")
    if events_per_window <= 0:
        raise ValueError("events_per_window must be positive")
    if max_label_distance_px < 0:
        raise ValueError("max_label_distance_px must be non-negative")

    complete = np.flatnonzero(~augmented.astype(bool) & ~backfilled.astype(bool))
    by_group_and_start: dict[tuple[int, int], int] = {}
    for index in complete:
        key = (int(group_indices[index]), int(start_event_indices[index]))
        if key in by_group_and_start:
            raise ValueError(f"duplicate complete train window key: {key}")
        by_group_and_start[key] = int(index)

    pairs: list[LocalRecombinationPair] = []
    for first_index in complete:
        group = int(group_indices[first_index])
        start = int(start_event_indices[first_index])
        second_index = by_group_and_start.get((group, start + 2 * events_per_window))
        if second_index is None:
            continue
        distance = float(
            np.linalg.norm(
                labels[second_index].astype(np.float64)
                - labels[first_index].astype(np.float64)
            )
        )
        if distance > max_label_distance_px:
            continue
        pairs.append(
            LocalRecombinationPair(
                first_index=int(first_index),
                second_index=second_index,
                label_distance_px=distance,
            )
        )
    return pairs


def select_source_wise_recombination_anchors(
    *,
    labels: np.ndarray,
    group_indices: np.ndarray,
    start_event_indices: np.ndarray,
    augmented: np.ndarray,
    backfilled: np.ndarray,
    allowed_start_differences_events: tuple[int, ...] = (
        SOURCE_WISE_START_DIFFERENCES_EVENTS
    ),
    max_label_distance_px: float = 32.0,
) -> list[LocalRecombinationAnchor]:
    """Give every eligible complete train source its local partner pool."""
    lengths = {
        len(labels),
        len(group_indices),
        len(start_event_indices),
        len(augmented),
        len(backfilled),
    }
    if len(lengths) != 1:
        raise ValueError("anchor-selection arrays must have equal length")
    if max_label_distance_px < 0:
        raise ValueError("max_label_distance_px must be non-negative")
    differences = tuple(sorted(set(int(value) for value in allowed_start_differences_events)))
    if not differences or any(value <= 0 for value in differences):
        raise ValueError("allowed start differences must be positive")

    complete = np.flatnonzero(~augmented.astype(bool) & ~backfilled.astype(bool))
    by_group_and_start: dict[tuple[int, int], int] = {}
    for index in complete:
        key = (int(group_indices[index]), int(start_event_indices[index]))
        if key in by_group_and_start:
            raise ValueError(f"duplicate complete train window key: {key}")
        by_group_and_start[key] = int(index)

    anchors: list[LocalRecombinationAnchor] = []
    for source_index in complete:
        group = int(group_indices[source_index])
        start = int(start_event_indices[source_index])
        partner_indices: list[int] = []
        for difference in differences:
            for partner_start in (start - difference, start + difference):
                partner_index = by_group_and_start.get((group, partner_start))
                if partner_index is None:
                    continue
                distance = float(
                    np.linalg.norm(
                        labels[partner_index].astype(np.float64)
                        - labels[source_index].astype(np.float64)
                    )
                )
                if distance <= max_label_distance_px:
                    partner_indices.append(partner_index)
        if partner_indices:
            anchors.append(
                LocalRecombinationAnchor(
                    source_index=int(source_index),
                    partner_indices=tuple(
                        sorted(
                            set(partner_indices),
                            key=lambda index: (
                                int(start_event_indices[index]), int(index)
                            ),
                        )
                    ),
                )
            )
    return anchors


def select_all_physical_source_recombination_anchors(
    *,
    labels: np.ndarray,
    group_indices: np.ndarray,
    start_event_indices: np.ndarray,
    augmented: np.ndarray,
    backfilled: np.ndarray,
    events_per_window: int = 400,
    max_label_distance_px: float = 32.0,
) -> tuple[np.ndarray, list[AllPhysicalRecombinationAnchor]]:
    """Use every physical train window and mirror the old saccade rule.

    Partners must be in the same pursuit group and within the label-distance
    limit. A source uses non-adjacent partners first; exactly adjacent,
    non-overlapping partners are a fallback only when no preferred partner
    exists. Stride windows are deliberately included on both sides.
    """
    arrays = tuple(
        np.asarray(value)
        for value in (
            labels,
            group_indices,
            start_event_indices,
            augmented,
            backfilled,
        )
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("anchor-selection arrays must have equal length")
    if events_per_window <= 0:
        raise ValueError("events_per_window must be positive")
    if max_label_distance_px < 0:
        raise ValueError("max_label_distance_px must be non-negative")
    if np.any(arrays[4].astype(bool)):
        raise ValueError("all-physical mode requires a no-tail cache")

    source_pool = np.arange(len(arrays[0]), dtype=np.int64)
    by_group: dict[int, list[int]] = {}
    for index in source_pool:
        by_group.setdefault(int(arrays[1][index]), []).append(int(index))

    anchors: list[AllPhysicalRecombinationAnchor] = []
    for source_index in source_pool:
        source_start = int(arrays[2][source_index])
        source_label = arrays[0][source_index].astype(np.float64)
        preferred: list[int] = []
        adjacent: list[int] = []
        for partner_index in by_group[int(arrays[1][source_index])]:
            if partner_index == source_index:
                continue
            distance = float(
                np.linalg.norm(
                    arrays[0][partner_index].astype(np.float64) - source_label
                )
            )
            if distance > max_label_distance_px:
                continue
            start_difference = abs(
                int(arrays[2][partner_index]) - source_start
            )
            if start_difference > events_per_window:
                preferred.append(partner_index)
            elif start_difference == events_per_window:
                adjacent.append(partner_index)

        if preferred or adjacent:
            key = lambda index: (int(arrays[2][index]), int(index))
            anchors.append(
                AllPhysicalRecombinationAnchor(
                    source_index=int(source_index),
                    preferred_partner_indices=tuple(sorted(preferred, key=key)),
                    adjacent_partner_indices=tuple(sorted(adjacent, key=key)),
                )
            )
    return source_pool, anchors


def sample_event_occurrences(
    event_frame: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample event occurrences without replacement from one count frame."""
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


def load_aligned_sequential_event_bins(
    source_dataset: Dataset,
    sequential_cache_dir: Path,
    split: str,
) -> np.ndarray:
    """Load four 100-event bins and prove that they match the static cache."""
    sequential_cache_dir = Path(sequential_cache_dir)
    with np.load(sequential_cache_dir / f"{split}.npz") as cached:
        required = (
            "event_bins",
            "labels",
            "block_name",
            "start_event_index",
            "group_index",
            "augmented",
            "backfilled",
        )
        missing = [name for name in required if name not in cached]
        if missing:
            raise ValueError(f"sequential cache lacks fields: {missing}")
        event_bins = cached["event_bins"].copy()
        comparisons = {
            "labels": source_dataset.labels,
            "block_name": source_dataset.block_names,
            "start_event_index": source_dataset.start_event_indices,
            "group_index": source_dataset.group_indices,
            "augmented": source_dataset.augmented,
            "backfilled": source_dataset.backfilled,
        }
        for field, expected in comparisons.items():
            if not np.array_equal(cached[field], expected):
                raise ValueError(
                    f"sequential {field} is not aligned with the static {split} cache"
                )

    expected_shape = (len(source_dataset), 4, 2, source_dataset.height, source_dataset.width)
    if event_bins.shape != expected_shape:
        raise ValueError(
            f"expected sequential bins {expected_shape}, got {event_bins.shape}"
        )
    if not np.all(event_bins.sum(axis=(2, 3, 4), dtype=np.int64) == 100):
        raise ValueError("every ordered bin must contain exactly 100 events")
    if not np.array_equal(
        event_bins.sum(axis=1, dtype=np.uint16), source_dataset.event_frames
    ):
        raise ValueError("four ordered bins do not sum to the aligned static frame")
    return event_bins


def fixed_inner_contiguous_recombination(
    first_event_bins: np.ndarray,
    second_event_bins: np.ndarray,
) -> np.ndarray:
    """Combine the earlier window's last 200 and later window's first 200 events."""
    if first_event_bins.shape != second_event_bins.shape:
        raise ValueError("source sequential bins must have equal shapes")
    if first_event_bins.ndim != 4 or first_event_bins.shape[0] != 4:
        raise ValueError("each source must contain four ordered 100-event bins")
    combined = (
        first_event_bins[2:].sum(axis=0, dtype=np.uint16)
        + second_event_bins[:2].sum(axis=0, dtype=np.uint16)
    )
    if int(combined.sum()) != 400:
        raise ValueError("fixed contiguous recombination must contain 400 events")
    return combined.astype(np.uint16, copy=False)


class LocalRecombinationPursuitDataset(Dataset):
    """Retain original train samples and append one controlled mix per item."""

    events_from_each_source = 200

    def __init__(
        self,
        source_dataset: Dataset,
        *,
        seed: int,
        max_label_distance_px: float = 32.0,
        mode: str = RECOMBINATION_MODE_EPOCH_RANDOM,
        sequential_event_bins: np.ndarray | None = None,
    ) -> None:
        required = (
            "event_frames",
            "labels",
            "block_names",
            "group_indices",
            "start_event_indices",
            "augmented",
            "backfilled",
        )
        missing = [name for name in required if not hasattr(source_dataset, name)]
        if missing:
            raise ValueError(f"source dataset lacks recombination metadata: {missing}")
        self.source_dataset = source_dataset
        self.seed = int(seed)
        self.epoch = 0
        self.max_label_distance_px = float(max_label_distance_px)
        if mode not in RECOMBINATION_MODES:
            raise ValueError(f"unknown local recombination mode: {mode}")
        self.mode = mode
        self.pairs: list[LocalRecombinationPair] = []
        self.anchors: list[LocalRecombinationAnchor] = []
        self.all_physical_anchors: list[AllPhysicalRecombinationAnchor] = []
        self.source_pool_indices = np.empty(0, dtype=np.int64)
        if self.mode == RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM:
            (
                self.source_pool_indices,
                self.all_physical_anchors,
            ) = select_all_physical_source_recombination_anchors(
                labels=source_dataset.labels,
                group_indices=source_dataset.group_indices,
                start_event_indices=source_dataset.start_event_indices,
                augmented=source_dataset.augmented,
                backfilled=source_dataset.backfilled,
                max_label_distance_px=self.max_label_distance_px,
            )
        elif self.mode == RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM:
            self.anchors = select_source_wise_recombination_anchors(
                labels=source_dataset.labels,
                group_indices=source_dataset.group_indices,
                start_event_indices=source_dataset.start_event_indices,
                augmented=source_dataset.augmented,
                backfilled=source_dataset.backfilled,
                max_label_distance_px=self.max_label_distance_px,
            )
        else:
            self.pairs = select_local_recombination_pairs(
                labels=source_dataset.labels,
                group_indices=source_dataset.group_indices,
                start_event_indices=source_dataset.start_event_indices,
                augmented=source_dataset.augmented,
                backfilled=source_dataset.backfilled,
                events_per_window=400,
                max_label_distance_px=self.max_label_distance_px,
            )
        self.original_sample_count = len(source_dataset)
        if self.mode == RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM:
            self.synthetic_sample_count = len(self.all_physical_anchors)
        elif self.mode == RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM:
            self.synthetic_sample_count = len(self.anchors)
        else:
            self.synthetic_sample_count = len(self.pairs)
        if self.all_physical_anchors:
            selected_edges = {
                tuple(sorted((anchor.source_index, partner_index)))
                for anchor in self.all_physical_anchors
                for partner_index in (
                    anchor.preferred_partner_indices
                    if anchor.preferred_partner_indices
                    else anchor.adjacent_partner_indices
                )
            }
            self.candidate_edge_count = len(selected_edges)
        elif self.anchors:
            self.candidate_edge_count = len(
                {
                    tuple(sorted((anchor.source_index, partner_index)))
                    for anchor in self.anchors
                    for partner_index in anchor.partner_indices
                }
            )
        else:
            self.candidate_edge_count = len(self.pairs)
        self.source_sample_count = self.original_sample_count
        self.height = source_dataset.height
        self.width = source_dataset.width
        self.block_count = source_dataset.block_count
        self.physical_stride_sample_count = int(
            np.asarray(source_dataset.augmented, dtype=bool).sum()
        )
        self.preferred_source_count = sum(
            bool(anchor.preferred_partner_indices)
            for anchor in self.all_physical_anchors
        )
        self.adjacent_fallback_source_count = sum(
            not anchor.preferred_partner_indices
            for anchor in self.all_physical_anchors
        )
        self.ineligible_source_count = (
            len(self.source_pool_indices) - len(self.all_physical_anchors)
        )
        self.eligible_stride_source_count = sum(
            bool(source_dataset.augmented[anchor.source_index])
            for anchor in self.all_physical_anchors
        )
        selectable_partners = {
            partner_index
            for anchor in self.all_physical_anchors
            for partner_index in (
                anchor.preferred_partner_indices
                if anchor.preferred_partner_indices
                else anchor.adjacent_partner_indices
            )
        }
        self.selectable_partner_count = len(selectable_partners)
        self.selectable_partner_stride_count = sum(
            bool(source_dataset.augmented[index]) for index in selectable_partners
        )
        self.fixed_synthetic_frames: np.ndarray | None = None
        if self.mode == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS:
            if sequential_event_bins is None:
                raise ValueError(
                    "fixed contiguous recombination requires aligned sequential bins"
                )
            expected_shape = (
                len(source_dataset),
                4,
                2,
                self.height,
                self.width,
            )
            if sequential_event_bins.shape != expected_shape:
                raise ValueError(
                    f"expected sequential bins {expected_shape}, "
                    f"got {sequential_event_bins.shape}"
                )
            self.fixed_synthetic_frames = np.stack(
                [
                    fixed_inner_contiguous_recombination(
                        sequential_event_bins[pair.first_index],
                        sequential_event_bins[pair.second_index],
                    )
                    for pair in self.pairs
                ],
                axis=0,
            )
        elif sequential_event_bins is not None:
            raise ValueError("epoch-random recombination must not receive sequential bins")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.original_sample_count + self.synthetic_sample_count

    def _sample_half(
        self,
        source_index: int,
        pair_index: int,
        source_slot: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, pair_index, source_slot])
        )
        return sample_event_occurrences(
            self.source_dataset.event_frames[source_index],
            self.events_from_each_source,
            rng,
        )

    def _source_wise_pair(self, synthetic_index: int) -> LocalRecombinationPair:
        anchor = self.anchors[synthetic_index]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, synthetic_index, 2])
        )
        partner_index = anchor.partner_indices[
            int(rng.integers(len(anchor.partner_indices)))
        ]
        distance = float(
            np.linalg.norm(
                self.source_dataset.labels[partner_index].astype(np.float64)
                - self.source_dataset.labels[anchor.source_index].astype(np.float64)
            )
        )
        return LocalRecombinationPair(
            first_index=anchor.source_index,
            second_index=partner_index,
            label_distance_px=distance,
        )

    def _all_physical_source_pair(
        self, synthetic_index: int
    ) -> LocalRecombinationPair:
        anchor = self.all_physical_anchors[synthetic_index]
        partners = (
            anchor.preferred_partner_indices
            if anchor.preferred_partner_indices
            else anchor.adjacent_partner_indices
        )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.epoch, anchor.source_index, 0]
            )
        )
        partner_index = partners[int(rng.integers(len(partners)))]
        distance = float(
            np.linalg.norm(
                self.source_dataset.labels[partner_index].astype(np.float64)
                - self.source_dataset.labels[anchor.source_index].astype(np.float64)
            )
        )
        return LocalRecombinationPair(
            first_index=anchor.source_index,
            second_index=partner_index,
            label_distance_px=distance,
        )

    def __getitem__(self, index: int) -> dict:
        if index < self.original_sample_count:
            return self.source_dataset[index]
        pair_index = index - self.original_sample_count
        if self.mode == RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM:
            pair = self._all_physical_source_pair(pair_index)
        elif self.mode == RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM:
            pair = self._source_wise_pair(pair_index)
        else:
            pair = self.pairs[pair_index]
        if self.mode in (
            RECOMBINATION_MODE_EPOCH_RANDOM,
            RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM,
            RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM,
        ):
            sampling_key = (
                pair.first_index
                if self.mode
                == RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM
                else pair_index
            )
            first = self._sample_half(pair.first_index, sampling_key, 0)
            second = self._sample_half(pair.second_index, sampling_key, 1)
            event_frame = first + second
        else:
            if self.fixed_synthetic_frames is None:
                raise RuntimeError("fixed synthetic frames were not initialized")
            event_frame = self.fixed_synthetic_frames[pair_index]
        if int(event_frame.sum()) != 400:
            raise RuntimeError("recombined frame does not contain exactly 400 events")
        label_px = (
            self.source_dataset.labels[pair.first_index].astype(np.float32)
            + self.source_dataset.labels[pair.second_index].astype(np.float32)
        ) / 2.0
        label_px_tensor = torch.from_numpy(label_px)
        return {
            "event_frame": torch.from_numpy(event_frame).float(),
            "target_norm": normalize_row_col(label_px_tensor),
            "target_row_col_px": label_px_tensor,
            "block": self.source_dataset.block_names[pair.first_index],
            "sample_index": index,
            "duration_us": -1,
            "synthetic": True,
            "source_indices": torch.tensor(
                [pair.first_index, pair.second_index], dtype=torch.int64
            ),
        }


__all__ = [
    "AllPhysicalRecombinationAnchor",
    "LocalRecombinationAnchor",
    "LocalRecombinationPair",
    "LocalRecombinationPursuitDataset",
    "RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM",
    "RECOMBINATION_MODE_EPOCH_RANDOM",
    "RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS",
    "RECOMBINATION_MODE_SOURCE_WISE_EPOCH_RANDOM",
    "RECOMBINATION_MODES",
    "SOURCE_WISE_START_DIFFERENCES_EVENTS",
    "fixed_inner_contiguous_recombination",
    "load_aligned_sequential_event_bins",
    "sample_event_occurrences",
    "select_all_physical_source_recombination_anchors",
    "select_local_recombination_pairs",
    "select_source_wise_recombination_anchors",
]
