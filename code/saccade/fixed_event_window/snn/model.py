"""Saccade SNNs with continuous or grid-count spiking output heads."""

from __future__ import annotations

import torch
from torch import nn

from code.pursuit.fixed_event_window.snn.model import (
    Backend,
    INPUT_SHAPE,
    LearnableThresholdSpikingLayer,
    SNN_ARCHITECTURE,
    SpikingPursuitSNN,
    folded_batch_norm_bias,
    learnable_threshold_count,
    membrane_state_count,
    trainable_parameter_count,
)


GRID_LEVELS = 11
GRID_INDEX_MIN = 0
GRID_INDEX_MAX = GRID_LEVELS - 1
GRID_ROW_START_PX = 150.0
GRID_ROW_STEP_PX = 78.0
GRID_COL_START_PX = 150.0
GRID_COL_STEP_PX = 162.0
SCREEN_CENTER_X_PX = 960.0
SCREEN_CENTER_Y_PX = 540.0
GRID_SPIKE_COUNT_ARCHITECTURE = (
    "eight_layer_8_16_24_32_40_40_40_40_"
    "spiking_hidden60_12_grid_count_spiking_output2_96x128"
)


class SpikingSaccadeSNN(SpikingPursuitSNN):
    """The retained T-step SNN applied to formal saccade count frames."""


class GridCountSpikingReadout(nn.Module):
    """Bias-free linear readout followed by two MultiSpike output neurons."""

    def __init__(
        self,
        *,
        alpha: float,
        threshold_init: float,
        min_v_mem: float | None,
        backend: Backend,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 2, bias=False)
        self.spike = LearnableThresholdSpikingLayer(
            2,
            alpha=alpha,
            threshold_init=threshold_init,
            min_v_mem=min_v_mem,
            backend=backend,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spike(self.linear(values))


def decode_grid_spike_counts(
    spike_counts_xy: torch.Tensor,
    *,
    clamp_to_grid: bool,
) -> torch.Tensor:
    """Decode integer [column, row] spike counts to normalized [x, y]."""
    if spike_counts_xy.shape[-1] != 2:
        raise ValueError(
            f"expected final dimension 2, got {tuple(spike_counts_xy.shape)}"
        )
    grid_indices = spike_counts_xy
    if clamp_to_grid:
        grid_indices = grid_indices.clamp(GRID_INDEX_MIN, GRID_INDEX_MAX)
    col_px = GRID_COL_START_PX + GRID_COL_STEP_PX * grid_indices[..., 0]
    row_px = GRID_ROW_START_PX + GRID_ROW_STEP_PX * grid_indices[..., 1]
    x = (col_px - SCREEN_CENTER_X_PX) / SCREEN_CENTER_X_PX
    y = (row_px - SCREEN_CENTER_Y_PX) / SCREEN_CENTER_Y_PX
    return torch.stack((x, y), dim=-1)


class GridCountSpikingSaccadeSNN(SpikingPursuitSNN):
    """Encode each gaze axis as the total count of one output neuron.

    Across the two model steps, a total count ``k`` represents grid level
    ``k``. The first neuron encodes column and the second encodes row. Training
    can decode the raw count without clipping so out-of-range counts retain a
    corrective gradient; inference clips to the eleven valid levels 0..10.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        threshold_init: float = 1.0,
        min_v_mem: float | None = -1.0,
        backend: Backend = "exodus",
    ) -> None:
        super().__init__(
            alpha=alpha,
            threshold_init=threshold_init,
            min_v_mem=min_v_mem,
            backend=backend,
        )
        self.readout = GridCountSpikingReadout(
            alpha=self.alpha,
            threshold_init=self.threshold_init,
            min_v_mem=self.min_v_mem,
            backend=self.backend,
        )

    def output_spike_counts(self, event_bins: torch.Tensor) -> torch.Tensor:
        if event_bins.ndim != 5 or event_bins.shape[2:] != INPUT_SHAPE:
            raise ValueError(
                "expected (batch, time, 2, 96, 128), "
                f"got {tuple(event_bins.shape)}"
            )
        if event_bins.shape[1] <= 0:
            raise ValueError("time dimension must be positive")

        self.reset_states()
        spikes = self.blocks(event_bins)
        batch, timesteps = spikes.shape[:2]
        spikes = spikes.reshape(batch, timesteps, 480)
        spikes = self.hidden1(spikes)
        spikes = self.hidden2(spikes)
        output_spikes = self.readout(spikes)
        return output_spikes.sum(dim=1)

    def forward(
        self,
        event_bins: torch.Tensor,
        *,
        clamp_to_grid: bool = True,
    ) -> torch.Tensor:
        return decode_grid_spike_counts(
            self.output_spike_counts(event_bins),
            clamp_to_grid=clamp_to_grid,
        )


def grid_count_membrane_state_count() -> dict[str, int]:
    counts = membrane_state_count()
    counts["Output"] = 2
    counts["total"] += 2
    return counts


__all__ = [
    "Backend",
    "GRID_COL_START_PX",
    "GRID_COL_STEP_PX",
    "GRID_INDEX_MAX",
    "GRID_INDEX_MIN",
    "GRID_LEVELS",
    "GRID_ROW_START_PX",
    "GRID_ROW_STEP_PX",
    "GRID_SPIKE_COUNT_ARCHITECTURE",
    "GridCountSpikingReadout",
    "GridCountSpikingSaccadeSNN",
    "LearnableThresholdSpikingLayer",
    "SNN_ARCHITECTURE",
    "SpikingSaccadeSNN",
    "decode_grid_spike_counts",
    "folded_batch_norm_bias",
    "grid_count_membrane_state_count",
    "learnable_threshold_count",
    "membrane_state_count",
    "trainable_parameter_count",
]
