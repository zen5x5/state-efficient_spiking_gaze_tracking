"""Static eight-layer ANNs for fixed-event-window pursuit inputs."""

from __future__ import annotations

import torch
from torch import nn


CHANNELS = (2, 8, 16, 24, 32, 40, 40, 40, 40)
COMPACT_CHANNELS = (2, 8, 16, 24, 32, 36, 36, 36, 36)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class StaticPursuitANNHiddenReadout(nn.Module):
    """Eight Conv-BN-ReLU layers with a two-hidden-layer readout."""

    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for layer_index, (in_channels, out_channels) in enumerate(
            zip(CHANNELS[:-1], CHANNELS[1:]), start=1
        ):
            blocks.append(ConvBNReLU(in_channels, out_channels))
            if layer_index <= 5:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(480, 60, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(60, 12, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(12, 2, bias=False),
        )

    def forward(self, event_frame: torch.Tensor) -> torch.Tensor:
        if event_frame.shape[1:] != (2, 96, 128):
            raise ValueError(f"expected (batch, 2, 96, 128), got {tuple(event_frame.shape)}")
        return self.readout(self.blocks(event_frame))


class StaticPursuitANNCompact90K(nn.Module):
    """Compact ANN with about 90k parameters at 2×120×220.

    The five pooling stages make the flattened feature size depend on the input
    resolution.  The convolutional channels and hidden widths stay fixed, so
    the same class can also quantify a 2×120×220 version without changing the
    feature extractor design.
    """

    def __init__(self, input_height: int = 96, input_width: int = 128) -> None:
        super().__init__()
        if input_height < 32 or input_width < 32:
            raise ValueError("input height and width must both be at least 32")
        self.input_height = input_height
        self.input_width = input_width

        blocks: list[nn.Module] = []
        for layer_index, (in_channels, out_channels) in enumerate(
            zip(COMPACT_CHANNELS[:-1], COMPACT_CHANNELS[1:]), start=1
        ):
            blocks.append(ConvBNReLU(in_channels, out_channels))
            if layer_index <= 5:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.blocks = nn.Sequential(*blocks)

        pooled_height = input_height // 32
        pooled_width = input_width // 32
        self.flattened_features = COMPACT_CHANNELS[-1] * pooled_height * pooled_width
        self.readout = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.flattened_features, 48, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(48, 12, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(12, 2, bias=False),
        )

    def forward(self, event_frame: torch.Tensor) -> torch.Tensor:
        expected_shape = (2, self.input_height, self.input_width)
        if event_frame.shape[1:] != expected_shape:
            raise ValueError(
                f"expected (batch, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}), got {tuple(event_frame.shape)}"
            )
        return self.readout(self.blocks(event_frame))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
