"""SNN paired with the retained 2x96x128 eight-layer pursuit ANN.

The model keeps an explicit ``(batch, time, channel, height, width)`` input.
Temporal binning or repetition is intentionally handled outside this module.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Literal, Mapping, Sequence

import sinabs
import torch
from sinabs.activation import MembraneSubtract, MultiSpike, PeriodicExponential
from torch import nn


CHANNELS = (2, 8, 16, 24, 32, 40, 40, 40, 40)
INPUT_SHAPE = (2, 96, 128)
SNN_ARCHITECTURE = (
    "eight_layer_8_16_24_32_40_40_40_40_"
    "spiking_hidden60_12_learnable_threshold_maxpool_96x128"
)
Backend = Literal["exodus", "sinabs"]


def _spiking_layer_class(alpha: float, backend: Backend) -> type[nn.Module]:
    if backend == "exodus":
        from sinabs.exodus.layers import IAF, LIF
    elif backend == "sinabs":
        from sinabs.layers import IAF, LIF
    else:
        raise ValueError(f"backend must be 'exodus' or 'sinabs', got {backend!r}")
    return IAF if alpha == 1.0 else LIF


class LearnableThresholdSpikingLayer(nn.Module):
    """Fixed-alpha spiking layer with one positive learnable threshold per channel.

    Sinabs keeps its internal threshold fixed at one. The effective threshold is
    ``theta = exp(log_threshold)`` and is implemented by dividing each channel's
    input current by ``theta`` before the spiking neuron.
    """

    def __init__(
        self,
        channels: int,
        *,
        alpha: float = 0.5,
        threshold_init: float = 1.0,
        min_v_mem: float | None = -1.0,
        backend: Backend = "exodus",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha <= 1")
        if threshold_init <= 0.0:
            raise ValueError("threshold_init must be positive")

        self.channels = channels
        self.alpha = float(alpha)
        self.backend = backend
        self.log_threshold = nn.Parameter(
            torch.full((channels,), math.log(float(threshold_init)))
        )
        # This mask is architectural metadata, not a learned checkpoint value.
        # ``True`` means that the channel still computes and emits spikes in the
        # current bin, but its membrane is not carried into the next bin.
        self.register_buffer(
            "_stateless_channels",
            torch.zeros(channels, dtype=torch.bool),
            persistent=False,
        )
        self._stateless_indices: tuple[int, ...] = ()

        layer_class = _spiking_layer_class(self.alpha, backend)
        common = dict(
            spike_threshold=1.0,
            spike_fn=MultiSpike,
            surrogate_grad_fn=PeriodicExponential(),
            reset_fn=MembraneSubtract(),
            min_v_mem=min_v_mem,
            record_states=False,
        )
        if self.alpha == 1.0:
            self.neuron = layer_class(**common)
        elif self.alpha == 0.0:
            # alpha_mem=0 gives exact per-step membrane independence while
            # retaining the same MultiSpike/reset/surrogate implementation.
            self.neuron = layer_class(
                tau_mem=0.0,
                tau_syn=None,
                train_alphas=True,
                norm_input=False,
                **common,
            )
            self.neuron.alpha_mem.requires_grad_(False)
        else:
            tau_mem = -1.0 / math.log(self.alpha)
            self.neuron = layer_class(
                tau_mem=tau_mem,
                tau_syn=None,
                train_alphas=False,
                norm_input=False,
                **common,
            )
            # Sinabs represents tau as a Parameter. Alpha is fixed here.
            self.neuron.tau_mem.requires_grad_(False)

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    @property
    def stateless_channel_indices(self) -> tuple[int, ...]:
        return self._stateless_indices

    def set_stateless_channels(self, indices: Sequence[int]) -> None:
        unique = sorted({int(index) for index in indices})
        if any(index < 0 or index >= self.channels for index in unique):
            raise ValueError(
                f"stateless channel indices must be in [0, {self.channels}), "
                f"got {unique}"
            )
        if tuple(unique) == self._stateless_indices:
            return
        mask = torch.zeros(
            self.channels,
            dtype=torch.bool,
            device=self._stateless_channels.device,
        )
        if unique:
            mask[unique] = True
        self._stateless_channels.copy_(mask)
        self._stateless_indices = tuple(unique)

    def _clear_selected_membrane_state(self) -> None:
        if not self._stateless_indices:
            return
        state = self.neuron.v_mem
        if state.numel() == 0:
            raise RuntimeError("membrane state was not initialized")
        retain = (~self._stateless_channels).to(dtype=state.dtype)
        retain = retain.view(1, self.channels, *([1] * (state.ndim - 2)))
        # Do not detach: stateful channels retain their exact temporal gradient,
        # while stateless channels receive a mathematically zero cross-bin path.
        self.neuron.v_mem = state * retain

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim < 3:
            raise ValueError(
                "expected explicit batch and time dimensions plus a channel dimension"
            )
        if current.shape[2] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels/features, got {current.shape[2]}"
            )
        broadcast_shape = (1, 1, self.channels, *([1] * (current.ndim - 3)))
        normalized_current = current / self.threshold.view(broadcast_shape)
        if not self._stateless_indices or current.shape[1] == 1:
            return self.neuron(normalized_current)

        outputs = []
        for timestep in range(current.shape[1]):
            outputs.append(self.neuron(normalized_current[:, timestep : timestep + 1]))
            if timestep + 1 < current.shape[1]:
                self._clear_selected_membrane_state()
        return torch.cat(outputs, dim=1)


class TemporalConvSpikingBlock(nn.Module):
    """Conv-BN-(optional MaxPool)-spiking block with explicit time handling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        pool: bool,
        alpha: float,
        threshold_init: float,
        min_v_mem: float | None,
        backend: Backend,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self.spike = LearnableThresholdSpikingLayer(
            out_channels,
            alpha=alpha,
            threshold_init=threshold_init,
            min_v_mem=min_v_mem,
            backend=backend,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels, height, width = values.shape
        values = values.reshape(batch * timesteps, channels, height, width)
        values = self.pool(self.batch_norm(self.conv(values)))
        _, out_channels, out_height, out_width = values.shape
        values = values.reshape(
            batch, timesteps, out_channels, out_height, out_width
        )
        return self.spike(values)


class TemporalLinearSpikingBlock(nn.Module):
    """Linear layer followed by a channel-wise learnable-threshold spiking layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        alpha: float,
        threshold_init: float,
        min_v_mem: float | None,
        backend: Backend,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.spike = LearnableThresholdSpikingLayer(
            out_features,
            alpha=alpha,
            threshold_init=threshold_init,
            min_v_mem=min_v_mem,
            backend=backend,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.spike(self.linear(values))


class SpikingPursuitSNN(nn.Module):
    """Eight-layer SNN with two spiking MLP hidden layers and linear readout."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        threshold_init: float = 1.0,
        min_v_mem: float | None = -1.0,
        backend: Backend = "exodus",
        stateless_channels: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.threshold_init = float(threshold_init)
        self.min_v_mem = min_v_mem
        self.backend = backend

        blocks = OrderedDict()
        for layer_index, (in_channels, out_channels) in enumerate(
            zip(CHANNELS[:-1], CHANNELS[1:]), start=1
        ):
            blocks[f"layer{layer_index}"] = TemporalConvSpikingBlock(
                in_channels,
                out_channels,
                pool=layer_index <= 5,
                alpha=self.alpha,
                threshold_init=self.threshold_init,
                min_v_mem=self.min_v_mem,
                backend=self.backend,
            )
        self.blocks = nn.Sequential(blocks)
        self.hidden1 = TemporalLinearSpikingBlock(
            480,
            60,
            alpha=self.alpha,
            threshold_init=self.threshold_init,
            min_v_mem=self.min_v_mem,
            backend=self.backend,
        )
        self.hidden2 = TemporalLinearSpikingBlock(
            60,
            12,
            alpha=self.alpha,
            threshold_init=self.threshold_init,
            min_v_mem=self.min_v_mem,
            backend=self.backend,
        )
        self.readout = nn.Linear(12, 2, bias=False)
        self.set_stateless_channels(stateless_channels or {})

    def set_stateless_channels(
        self,
        channels_by_layer: Mapping[str, Sequence[int]],
    ) -> None:
        valid_names = {f"L{index}" for index in range(1, len(self.blocks) + 1)}
        unknown = sorted(set(channels_by_layer) - valid_names)
        if unknown:
            raise ValueError(f"unknown stateless layer names: {unknown}")
        for layer_index, block in enumerate(self.blocks, start=1):
            block.spike.set_stateless_channels(
                channels_by_layer.get(f"L{layer_index}", ())
            )

    def stateless_channels(self) -> dict[str, list[int]]:
        return {
            f"L{layer_index}": list(block.spike.stateless_channel_indices)
            for layer_index, block in enumerate(self.blocks, start=1)
            if block.spike.stateless_channel_indices
        }

    def reset_states(self) -> None:
        sinabs.reset_states(self)

    def forward(self, event_bins: torch.Tensor) -> torch.Tensor:
        if event_bins.ndim != 5 or event_bins.shape[2:] != INPUT_SHAPE:
            raise ValueError(
                "expected (batch, time, 2, 96, 128), "
                f"got {tuple(event_bins.shape)}"
            )
        if event_bins.shape[1] <= 0:
            raise ValueError("time dimension must be positive")

        # Cached windows are independent samples; state never crosses windows.
        self.reset_states()
        spikes = self.blocks(event_bins)
        batch, timesteps = spikes.shape[:2]
        spikes = spikes.reshape(batch, timesteps, 480)
        spikes = self.hidden1(spikes)
        spikes = self.hidden2(spikes)
        return self.readout(spikes.mean(dim=1))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def learnable_threshold_count(model: nn.Module) -> int:
    return sum(
        module.log_threshold.numel()
        for module in model.modules()
        if isinstance(module, LearnableThresholdSpikingLayer)
    )


def folded_batch_norm_bias(model: SpikingPursuitSNN) -> dict[str, torch.Tensor]:
    """Return inference-time BN constants in threshold-normalized units."""
    values = {}
    for layer_index, block in enumerate(model.blocks, start=1):
        batch_norm = block.batch_norm
        scale = torch.sqrt(batch_norm.running_var + batch_norm.eps)
        folded_constant = (
            batch_norm.bias
            - batch_norm.weight * batch_norm.running_mean / scale
        )
        values[f"L{layer_index}"] = folded_constant / block.spike.threshold
    return values


@torch.no_grad()
def project_folded_batch_norm_bias_(
    model: SpikingPursuitSNN,
    through_layer: int,
) -> None:
    """Project L1..Ln so their normalized folded BN bias is <= 1-alpha."""
    if not 0 <= through_layer <= len(model.blocks):
        raise ValueError(
            f"through_layer must be in [0, {len(model.blocks)}], got {through_layer}"
        )
    if through_layer == 0:
        return

    upper_bound = 1.0 - model.alpha
    for layer_index, block in enumerate(model.blocks, start=1):
        if layer_index > through_layer:
            break
        batch_norm = block.batch_norm
        scale = torch.sqrt(batch_norm.running_var + batch_norm.eps)
        maximum_beta = (
            batch_norm.weight * batch_norm.running_mean / scale
            + upper_bound * block.spike.threshold
        )
        batch_norm.bias.copy_(torch.minimum(batch_norm.bias, maximum_beta))


def membrane_state_count() -> dict[str, int]:
    """Per-sample membrane values retained between adjacent time steps."""
    counts = OrderedDict(
        [
            ("L1", 8 * 48 * 64),
            ("L2", 16 * 24 * 32),
            ("L3", 24 * 12 * 16),
            ("L4", 32 * 6 * 8),
            ("L5", 40 * 3 * 4),
            ("L6", 40 * 3 * 4),
            ("L7", 40 * 3 * 4),
            ("L8", 40 * 3 * 4),
            ("H1", 60),
            ("H2", 12),
        ]
    )
    counts["total"] = sum(counts.values())
    return dict(counts)
