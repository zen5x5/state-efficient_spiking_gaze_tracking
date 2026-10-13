"""Data loading and explicit temporal organization for pursuit SNNs."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ..ann.dataloader import StaticPursuitDataset, make_dataloader
from ..ann.metrics import normalize_row_col

import json
from pathlib import Path

import numpy as np


def repeat_full_event_frame(
    event_frame: torch.Tensor,
    timesteps: int,
) -> torch.Tensor:
    """Repeat one complete 400-event count frame at every model time step."""
    if event_frame.ndim != 4:
        raise ValueError(
            "expected event_frame with shape (batch, channel, height, width), "
            f"got {tuple(event_frame.shape)}"
        )
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    return event_frame.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)


class SequentialPursuitDataset(Dataset):
    """Load four ordered 100-event count bins for each retained 400-event window."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if int(manifest["cached_bins_per_sample"]) != 4:
            raise ValueError("sequential cache must contain four bins per sample")
        if int(manifest["events_per_bin"]) != 100:
            raise ValueError("sequential cache must contain 100 events per bin")
        with np.load(cache_dir / f"{split}.npz") as cached:
            self.event_bins = cached["event_bins"].copy()
            self.labels = cached["labels"].copy()
            self.block_names = cached["block_name"].copy()
            self.duration_us = cached["duration_us"].copy()
        self.source_sample_count = len(self.event_bins)

    def __len__(self) -> int:
        return self.source_sample_count

    def __getitem__(self, index: int) -> dict:
        event_bins = torch.from_numpy(self.event_bins[index]).float()
        label_px = torch.from_numpy(self.labels[index])
        return {
            "event_bins": event_bins,
            "target_norm": normalize_row_col(label_px),
            "target_row_col_px": label_px,
            "block": self.block_names[index],
            "sample_index": index,
            "duration_us": int(self.duration_us[index]),
        }


def make_sequential_dataloader(
    cache_dir: Path,
    split: str,
    batch_size: int,
    seed: int,
    drop_last: bool = False,
) -> tuple[SequentialPursuitDataset, DataLoader]:
    dataset = SequentialPursuitDataset(cache_dir, split)
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


def make_snn_dataloader(
    cache_dir: Path,
    split: str,
    batch_size: int,
    seed: int,
    temporal_organization: str,
    drop_last: bool = False,
    local_recombination: bool = False,
    recombination_max_label_distance_px: float = 32.0,
    local_recombination_mode: str = "epoch_random",
    recombination_sequential_cache_dir: Path | None = None,
):
    if temporal_organization == "repeat_full_400":
        return make_dataloader(
            cache_dir,
            split,
            batch_size,
            seed,
            drop_last,
            local_recombination=local_recombination,
            recombination_max_label_distance_px=recombination_max_label_distance_px,
            local_recombination_mode=local_recombination_mode,
            recombination_sequential_cache_dir=recombination_sequential_cache_dir,
        )
    if temporal_organization == "sequential_equal_events":
        if local_recombination:
            raise ValueError(
                "local 200+200 recombination is defined only for static count frames"
            )
        return make_sequential_dataloader(
            cache_dir, split, batch_size, seed, drop_last
        )
    raise ValueError(f"unknown temporal organization: {temporal_organization}")


def prepare_event_bins(
    batch: dict,
    timesteps: int,
    temporal_organization: str,
) -> torch.Tensor:
    if temporal_organization == "repeat_full_400":
        return repeat_full_event_frame(batch["event_frame"], timesteps)
    if temporal_organization != "sequential_equal_events":
        raise ValueError(f"unknown temporal organization: {temporal_organization}")
    bins = batch["event_bins"]
    if bins.ndim != 5 or bins.shape[1] != 4:
        raise ValueError(f"expected four sequential bins, got {tuple(bins.shape)}")
    if timesteps == 4:
        return bins
    if timesteps == 2:
        batch_size, _, channels, height, width = bins.shape
        return bins.reshape(batch_size, 2, 2, channels, height, width).sum(dim=2)
    raise ValueError("sequential equal-event input supports only T=2 or T=4")


__all__ = [
    "StaticPursuitDataset",
    "SequentialPursuitDataset",
    "make_dataloader",
    "make_sequential_dataloader",
    "make_snn_dataloader",
    "prepare_event_bins",
    "repeat_full_event_frame",
]
