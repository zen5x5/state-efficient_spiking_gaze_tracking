"""Load the formal static saccade cache and attach optional target weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import normalize_row_col


LOSS_WEIGHTING_ORDINARY = "ordinary"
LOSS_WEIGHTING_TARGET_SQRT = "target_sqrt"
LOSS_WEIGHTINGS = (LOSS_WEIGHTING_ORDINARY, LOSS_WEIGHTING_TARGET_SQRT)


def target_sqrt_sample_weights(target_indices: np.ndarray) -> np.ndarray:
    """Return normalized 1/sqrt(n_target) weights with sample mean one."""
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if target_indices.ndim != 1 or len(target_indices) == 0:
        raise ValueError("target_indices must be a non-empty one-dimensional array")
    unique, inverse, counts = np.unique(
        target_indices, return_inverse=True, return_counts=True
    )
    if len(unique) == 0 or np.any(counts <= 0):
        raise ValueError("invalid target counts")
    raw = 1.0 / np.sqrt(counts[inverse].astype(np.float64))
    return (raw / raw.mean()).astype(np.float32)


class StaticSaccadeDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
        loss_weighting: str = LOSS_WEIGHTING_ORDINARY,
    ) -> None:
        if loss_weighting not in LOSS_WEIGHTINGS:
            raise ValueError(f"unknown loss weighting: {loss_weighting}")
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        with np.load(cache_dir / f"{split}.npz") as cached:
            self.event_frames = cached["event_frames"].copy()
            self.labels = cached["labels"].copy()
            self.target_indices = cached["target_index"].astype(np.int64, copy=True)
            self.block_names = cached["block_name"].copy()
            self.duration_us = cached["duration_us"].copy()
            self.start_event_indices = cached["start_event_index"].astype(
                np.int64, copy=True
            )
            self.augmented = cached["augmented"].copy()
            self.backfilled = cached["backfilled"].copy()
        if tuple(manifest["input_shape"]) != (2, 96, 128):
            raise ValueError(f"unexpected input shape: {manifest['input_shape']}")
        if not (
            len(self.event_frames)
            == len(self.labels)
            == len(self.target_indices)
            == len(self.block_names)
        ):
            raise ValueError("cache arrays have inconsistent lengths")
        if loss_weighting == LOSS_WEIGHTING_TARGET_SQRT:
            self.sample_weights = target_sqrt_sample_weights(self.target_indices)
        else:
            self.sample_weights = np.ones(len(self.target_indices), dtype=np.float32)
        self.loss_weighting = loss_weighting
        self.target_counts = {
            int(target): int(count)
            for target, count in zip(*np.unique(self.target_indices, return_counts=True))
        }

    def __len__(self) -> int:
        return len(self.event_frames)

    def __getitem__(self, index: int) -> dict:
        label_px = torch.from_numpy(self.labels[index])
        return {
            "event_frame": torch.from_numpy(self.event_frames[index]).float(),
            "target_norm": normalize_row_col(label_px),
            "target_row_col_px": label_px,
            "target_index": int(self.target_indices[index]),
            "sample_index": index,
            "sample_weight": float(self.sample_weights[index]),
            "block_name": str(self.block_names[index]),
            "duration_us": int(self.duration_us[index]),
            "augmented": bool(self.augmented[index]),
            "backfilled": bool(self.backfilled[index]),
            "synthetic": False,
            "source_indices": torch.tensor([-1, -1], dtype=torch.int64),
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    *,
    drop_last: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def make_dataloader(
    cache_dir: Path,
    split: str,
    batch_size: int,
    seed: int,
    *,
    drop_last: bool = False,
    shuffle: bool | None = None,
    loss_weighting: str = LOSS_WEIGHTING_ORDINARY,
) -> tuple[StaticSaccadeDataset, DataLoader]:
    dataset = StaticSaccadeDataset(cache_dir, split, loss_weighting)
    if shuffle is None:
        shuffle = split == "train"
    loader = make_loader(
        dataset,
        batch_size,
        seed,
        drop_last=drop_last,
        shuffle=shuffle,
    )
    return dataset, loader
