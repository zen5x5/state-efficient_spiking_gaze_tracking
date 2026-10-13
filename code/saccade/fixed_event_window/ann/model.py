"""Task-named wrapper around the retained 96x128 eight-layer ANN."""

from __future__ import annotations

from code.pursuit.fixed_event_window.ann.model import (
    StaticPursuitANNHiddenReadout,
    trainable_parameter_count,
)


class StaticSaccadeANN(StaticPursuitANNHiddenReadout):
    """The retained 96,480-parameter ANN applied to saccade count frames."""


__all__ = ["StaticSaccadeANN", "trainable_parameter_count"]
