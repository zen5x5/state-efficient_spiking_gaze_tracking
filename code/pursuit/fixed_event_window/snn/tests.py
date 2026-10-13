"""Structural and gradient checks for the paired pursuit SNN."""

from __future__ import annotations

import copy

import torch
from sinabs.activation import MultiSpike, PeriodicExponential
from torch import nn

from .dataloader import prepare_event_bins
from .model import (
    LearnableThresholdSpikingLayer,
    SpikingPursuitSNN,
    folded_batch_norm_bias,
    learnable_threshold_count,
    membrane_state_count,
    project_folded_batch_norm_bias_,
    trainable_parameter_count,
)


def main() -> int:
    model = SpikingPursuitSNN(backend="sinabs")
    model.eval()
    assert model(torch.zeros(2, 2, 2, 96, 128)).shape == (2, 2)
    assert trainable_parameter_count(model) == 96_792
    assert learnable_threshold_count(model) == 312
    assert membrane_state_count()["total"] == 45_000
    assert sum(isinstance(module, nn.Conv2d) for module in model.modules()) == 8
    assert sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()) == 8
    assert sum(isinstance(module, nn.Linear) for module in model.modules()) == 3

    spiking_layers = [
        module
        for module in model.modules()
        if isinstance(module, LearnableThresholdSpikingLayer)
    ]
    assert len(spiking_layers) == 10
    for layer in spiking_layers:
        assert layer.alpha == 0.5
        assert torch.equal(layer.threshold, torch.ones_like(layer.threshold))
        assert layer.neuron.spike_fn is MultiSpike
        assert isinstance(layer.neuron.surrogate_grad_fn, PeriodicExponential)
        assert not layer.neuron.spike_threshold.requires_grad
        assert not layer.neuron.tau_mem.requires_grad

    threshold_layer = LearnableThresholdSpikingLayer(3, backend="sinabs")
    current = torch.full((2, 2, 3), 1.2)
    threshold_layer(current).sum().backward()
    assert threshold_layer.log_threshold.grad is not None
    assert torch.isfinite(threshold_layer.log_threshold.grad).all()

    alpha_zero = LearnableThresholdSpikingLayer(
        1, alpha=0.0, backend="sinabs"
    )
    zero_decay_input = torch.tensor([[[1.2], [0.2]]], requires_grad=True)
    zero_decay_output = alpha_zero(zero_decay_input)
    assert torch.equal(zero_decay_output, torch.tensor([[[1.0], [0.0]]]))
    assert torch.equal(
        alpha_zero.neuron.alpha_mem_calculated,
        torch.tensor(0.0),
    )
    zero_decay_output.sum().backward()
    assert torch.isfinite(zero_decay_input.grad).all()

    alpha_one = LearnableThresholdSpikingLayer(
        1, alpha=1.0, backend="sinabs"
    )
    assert alpha_one.neuron.__class__.__name__ == "IAF"

    four_bins = torch.stack(
        [torch.full((2, 96, 128), value) for value in (1.0, 2.0, 3.0, 4.0)]
    ).unsqueeze(0)
    batch = {"event_bins": four_bins}
    assert torch.equal(
        prepare_event_bins(batch, 4, "sequential_equal_events"), four_bins
    )
    two_bins = prepare_event_bins(batch, 2, "sequential_equal_events")
    assert torch.equal(two_bins[:, 0], four_bins[:, 0] + four_bins[:, 1])
    assert torch.equal(two_bins[:, 1], four_bins[:, 2] + four_bins[:, 3])

    constrained = SpikingPursuitSNN(alpha=0.75, backend="sinabs")
    with torch.no_grad():
        for block in constrained.blocks:
            block.batch_norm.bias.fill_(2.0)
    project_folded_batch_norm_bias_(constrained, through_layer=3)
    folded_biases = folded_batch_norm_bias(constrained)
    for layer_index in range(1, 4):
        assert torch.all(folded_biases[f"L{layer_index}"] <= 0.25 + 1e-6)
    assert torch.all(folded_biases["L4"] > 0.25)

    # Empty masks preserve the original vectorized implementation exactly.
    baseline = LearnableThresholdSpikingLayer(3, alpha=0.75, backend="sinabs")
    empty_mask = copy.deepcopy(baseline)
    empty_mask.set_stateless_channels([])
    baseline_current = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    empty_current = baseline_current.detach().clone().requires_grad_(True)
    baseline_output = baseline(baseline_current)
    empty_output = empty_mask(empty_current)
    torch.testing.assert_close(baseline_output, empty_output, rtol=0, atol=0)
    baseline_output.sum().backward()
    empty_output.sum().backward()
    torch.testing.assert_close(
        baseline_current.grad, empty_current.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.log_threshold.grad,
        empty_mask.log_threshold.grad,
        rtol=0,
        atol=0,
    )

    # A full stateless mask must equal the deliberately slow reference that
    # clears the entire membrane after every non-final bin.
    masked = LearnableThresholdSpikingLayer(3, alpha=0.75, backend="sinabs")
    reference = copy.deepcopy(masked)
    masked.set_stateless_channels(range(3))
    masked_current = torch.randn(2, 2, 3, 4, 5, requires_grad=True)
    reference_current = masked_current.detach().clone().requires_grad_(True)
    masked_output = masked(masked_current)
    reference_outputs = []
    for timestep in range(reference_current.shape[1]):
        reference_outputs.append(
            reference(reference_current[:, timestep : timestep + 1])
        )
        if timestep + 1 < reference_current.shape[1]:
            reference.neuron.v_mem = torch.zeros_like(reference.neuron.v_mem)
    reference_output = torch.cat(reference_outputs, dim=1)
    torch.testing.assert_close(masked_output, reference_output, rtol=0, atol=0)
    masked_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(
        masked_current.grad, reference_current.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        masked.log_threshold.grad,
        reference.log_threshold.grad,
        # Whole-sequence versus per-bin threshold broadcasting has a different
        # floating-point reduction order in its parameter gradient.
        rtol=1e-6,
        atol=2e-6,
    )

    l1_all = SpikingPursuitSNN(
        alpha=0.75,
        backend="sinabs",
        stateless_channels={"L1": range(8)},
    )
    assert l1_all.stateless_channels() == {"L1": list(range(8))}
    print("paired pursuit SNN checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
