"""Load static fixed-event-window pursuit ANN caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import normalize_row_col


class StaticPursuitDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        split: str,
    ) -> None:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        with np.load(cache_dir / f"{split}.npz") as cached:
            self.event_frames = cached["event_frames"].copy()
            self.labels = cached["labels"].copy()
            self.block_names = cached["block_name"].copy()
            self.duration_us = cached["duration_us"].copy()
            self.group_indices = cached["group_index"].copy()
            self.start_event_indices = cached["start_event_index"].copy()
            self.augmented = cached["augmented"].copy()
            self.backfilled = cached["backfilled"].copy()
        self.height = int(manifest["input_shape"][1])
        self.width = int(manifest["input_shape"][2])
        self.block_count = len(np.unique(self.block_names))
        self.source_sample_count = len(self.event_frames)

    def __len__(self) -> int:
        return self.source_sample_count

    def __getitem__(self, index: int) -> dict:
        event_frame = torch.from_numpy(self.event_frames[index]).float()
        label_px = torch.from_numpy(self.labels[index])
        return {
            "event_frame": event_frame,
            "target_norm": normalize_row_col(label_px),
            "target_row_col_px": label_px,
            "block": self.block_names[index],
            "sample_index": index,
            "duration_us": int(self.duration_us[index]),
            "synthetic": False,
            "source_indices": torch.tensor([index, index], dtype=torch.int64),
        }


def make_dataloader(
    cache_dir: Path,
    split: str,
    batch_size: int,
    seed: int,
    drop_last: bool = False,
    local_recombination: bool = False,
    recombination_max_label_distance_px: float = 32.0,
    local_recombination_mode: str = "epoch_random",
    recombination_sequential_cache_dir: Path | None = None,
) -> tuple[Dataset, DataLoader]:
    source_dataset = StaticPursuitDataset(cache_dir, split)
    if local_recombination:
        if split != "train":
            raise ValueError("local recombination is train-only")
        from ..augmentation import (
            LocalRecombinationPursuitDataset,
            RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS,
            load_aligned_sequential_event_bins,
        )

        sequential_event_bins = None
        if local_recombination_mode == RECOMBINATION_MODE_FIXED_INNER_CONTIGUOUS:
            if recombination_sequential_cache_dir is None:
                raise ValueError(
                    "fixed contiguous recombination requires its sequential cache"
                )
            sequential_event_bins = load_aligned_sequential_event_bins(
                source_dataset,
                recombination_sequential_cache_dir,
                split,
            )
        elif recombination_sequential_cache_dir is not None:
            raise ValueError(
                "a sequential recombination cache is valid only for fixed mode"
            )

        dataset: Dataset = LocalRecombinationPursuitDataset(
            source_dataset,
            seed=seed,
            max_label_distance_px=recombination_max_label_distance_px,
            mode=local_recombination_mode,
            sequential_event_bins=sequential_event_bins,
        )
    else:
        dataset = source_dataset
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    return dataset, loader
