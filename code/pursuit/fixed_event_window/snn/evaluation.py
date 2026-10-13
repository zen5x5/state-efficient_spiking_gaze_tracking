"""Evaluation for repeated or sequential pursuit SNN experiments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..ann.metrics import angular_error_deg, pixel_error
from .dataloader import prepare_event_bins


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    timesteps: int,
    temporal_organization: str = "repeat_full_400",
    max_batches: int | None = None,
) -> dict:
    model.eval()
    by_block = defaultdict(lambda: {"pixel": [], "degree": []})
    batches = 0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target_px = batch["target_row_col_px"].to(device)
        event_bins = prepare_event_bins(
            batch, timesteps, temporal_organization
        ).to(device)
        prediction = model(event_bins)
        pixels = pixel_error(prediction, target_px).cpu().numpy()
        degrees = angular_error_deg(prediction, target_px).cpu().numpy()
        for block, pixel, degree in zip(batch["block"], pixels, degrees):
            by_block[block]["pixel"].append(float(pixel))
            by_block[block]["degree"].append(float(degree))
        batches += 1
        samples += len(target_px)

    block_metrics = {
        block: {
            "samples": len(values["pixel"]),
            "mean_pixel_error": float(np.mean(values["pixel"])),
            "mean_angular_error_deg": float(np.mean(values["degree"])),
        }
        for block, values in sorted(by_block.items())
    }
    return {
        "samples": samples,
        "batches": batches,
        "mean_pixel_error": float(
            np.mean([value for values in by_block.values() for value in values["pixel"]])
        ),
        "mean_angular_error_deg": float(
            np.mean([value for values in by_block.values() for value in values["degree"]])
        ),
        "blocks": block_metrics,
    }
