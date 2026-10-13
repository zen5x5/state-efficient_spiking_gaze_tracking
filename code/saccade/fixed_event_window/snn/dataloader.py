"""Saccade SNN loading and repeated-complete-frame temporal organization."""

from __future__ import annotations

import torch

from ..ann.dataloader import (
    LOSS_WEIGHTING_ORDINARY,
    LOSS_WEIGHTING_TARGET_SQRT,
    LOSS_WEIGHTINGS,
    StaticSaccadeDataset,
    make_dataloader,
    make_loader,
)


def repeat_full_event_frame(
    event_frame: torch.Tensor,
    timesteps: int = 2,
) -> torch.Tensor:
    if event_frame.ndim != 4:
        raise ValueError(
            "expected event_frame with shape (batch, channel, height, width), "
            f"got {tuple(event_frame.shape)}"
        )
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    return event_frame.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)


__all__ = [
    "LOSS_WEIGHTING_ORDINARY",
    "LOSS_WEIGHTING_TARGET_SQRT",
    "LOSS_WEIGHTINGS",
    "StaticSaccadeDataset",
    "make_dataloader",
    "make_loader",
    "repeat_full_event_frame",
]
