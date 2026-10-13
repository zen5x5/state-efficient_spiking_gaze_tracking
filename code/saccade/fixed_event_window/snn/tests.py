"""Focused tests for the formal paired saccade SNN."""

from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from code.pursuit.fixed_event_window.snn.model import SpikingPursuitSNN
from code.paths import saccade_cache_root

from ..ann.augmentation import AllPhysicalSameTargetRecombinationDataset
from .analyze_activity import make_analysis_model
from .dataloader import StaticSaccadeDataset, repeat_full_event_frame
from .model import (
    GridCountSpikingSaccadeSNN,
    SpikingSaccadeSNN,
    decode_grid_spike_counts,
    grid_count_membrane_state_count,
    learnable_threshold_count,
    trainable_parameter_count,
)
from .run_all_source_recombination import BATCHES, jobs_for_batch
from .run_grid_spike_count import (
    BATCHES as GRID_BATCHES,
    jobs_for_batch as grid_jobs_for_batch,
)
from .run_l1_l2_stateless_continuation import training_command
from .training import (
    OUTPUT_ENCODING_CONTINUOUS,
    capture_training_state,
    cross_bin_state_counts,
    make_scheduler,
    parse_args,
    persistent_model_state_dict,
    restore_training_state,
)


CACHE = saccade_cache_root() / "user15/eye0"


def test_model_is_exact_pursuit_pair() -> None:
    torch.manual_seed(19)
    pursuit = SpikingPursuitSNN(alpha=0.75, backend="sinabs")
    torch.manual_seed(19)
    saccade = SpikingSaccadeSNN(alpha=0.75, backend="sinabs")
    assert pursuit.state_dict().keys() == saccade.state_dict().keys()
    for key, value in pursuit.state_dict().items():
        assert torch.equal(value, saccade.state_dict()[key]), key
    assert trainable_parameter_count(saccade) == 96_792


def test_repeat_full_frame_is_exact() -> None:
    frame = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    repeated = repeat_full_event_frame(frame, 2)
    assert repeated.shape == (2, 2, 2, 3, 4)
    assert torch.equal(repeated[:, 0], frame)
    assert torch.equal(repeated[:, 1], frame)


def test_formal_all_physical_recombination_counts() -> None:
    physical = StaticSaccadeDataset(CACHE, "train", "target_sqrt")
    dataset = AllPhysicalSameTargetRecombinationDataset(physical, seed=2021)
    audit = dataset.audit_summary()
    assert len(physical) == 4228
    assert len(dataset) == 8445
    assert dataset.synthetic_sample_count == 4217
    assert audit["eligible_source_stride_samples"] == 1828
    assert audit["eligible_source_nonstride_samples"] == 2389
    assert audit["ineligible_sources"] == 11


def test_manual_batches_cover_six_runs_once() -> None:
    jobs = [job for batch in sorted(BATCHES) for job in jobs_for_batch(batch)]
    pairs = [(job["loss_weighting"], job["seed"]) for job in jobs]
    assert len(pairs) == len(set(pairs)) == 6
    assert set(pairs) == {
        (loss, seed)
        for loss in ("ordinary", "target_sqrt")
        for seed in (2021, 2022, 2023)
    }


def test_grid_count_decoding_covers_all_declared_levels() -> None:
    counts = torch.arange(11, dtype=torch.float32)
    count_xy = torch.stack((counts, counts), dim=-1)
    prediction = decode_grid_spike_counts(count_xy, clamp_to_grid=True)
    col = prediction[:, 0] * 960.0 + 960.0
    row = prediction[:, 1] * 540.0 + 540.0
    assert torch.allclose(col, 150.0 + 162.0 * counts)
    assert torch.allclose(row, 150.0 + 78.0 * counts)
    outside = torch.tensor([[-3.0, 14.0]])
    clipped = decode_grid_spike_counts(outside, clamp_to_grid=True)
    assert torch.allclose(clipped, prediction[[0]] * torch.tensor([[1.0, 0.0]]) + prediction[[10]] * torch.tensor([[0.0, 1.0]]))


def test_grid_count_model_structure_and_surrogate_gradient() -> None:
    torch.manual_seed(23)
    model = GridCountSpikingSaccadeSNN(alpha=0.75, backend="sinabs")
    assert trainable_parameter_count(model) == 96_794
    assert learnable_threshold_count(model) == 314
    assert grid_count_membrane_state_count()["total"] == 45_002
    hidden_spikes = torch.ones(2, 2, 12)
    output_spikes = model.readout(hidden_spikes)
    counts = output_spikes.sum(dim=1)
    assert torch.equal(counts, counts.round())
    assert bool((counts >= 0).all())
    prediction = decode_grid_spike_counts(counts, clamp_to_grid=False)
    loss = prediction.square().mean()
    loss.backward()
    assert model.readout.linear.weight.grad is not None
    assert bool(model.readout.linear.weight.grad.ne(0).any())
    assert model.readout.spike.log_threshold.grad is not None


def test_grid_manual_batches_cover_three_seeds_once() -> None:
    jobs = [
        job
        for batch in sorted(GRID_BATCHES)
        for job in grid_jobs_for_batch(batch)
    ]
    seeds = [job["seed"] for job in jobs]
    assert seeds == [2021, 2022, 2023]
    assert len(seeds) == len(set(seeds)) == 3


def test_resume_and_stateless_train_cli_fields_are_exposed() -> None:
    args = parse_args(
        [
            "train",
            "--cache-dir",
            "/tmp/cache",
            "--output-dir",
            "/tmp/output",
            "--stateless-mask-json",
            "/tmp/mask.json",
            "--save-training-state-epoch",
            "100",
            "--resume-training-state",
            "/tmp/training_state_epoch100.pt",
        ]
    )
    assert args.output_encoding == OUTPUT_ENCODING_CONTINUOUS
    assert args.stateless_mask_json == Path("/tmp/mask.json")
    assert args.save_training_state_epoch == 100
    assert args.resume_training_state == Path(
        "/tmp/training_state_epoch100.pt"
    )


def test_stateless_continuation_commands_forward_state_options() -> None:
    baseline = training_command(
        cache_dir=Path("/tmp/cache"),
        output_dir=Path("/tmp/baseline"),
        seed=2021,
        epochs=100,
        save_epoch=100,
    )
    assert baseline[baseline.index("--save-training-state-epoch") + 1] == "100"
    assert "--stateless-mask-json" not in baseline
    assert "--resume-training-state" not in baseline

    continuation = training_command(
        cache_dir=Path("/tmp/cache"),
        output_dir=Path("/tmp/continuation"),
        seed=2021,
        epochs=200,
        mask_path=Path("/tmp/mask.json"),
        resume_path=Path("/tmp/training_state_epoch100.pt"),
    )
    assert continuation[continuation.index("--stateless-mask-json") + 1] == (
        "/tmp/mask.json"
    )
    assert continuation[continuation.index("--resume-training-state") + 1] == (
        "/tmp/training_state_epoch100.pt"
    )


def test_l1_l2_stateless_state_count_and_activity_model() -> None:
    stateless_channels = {
        "L1": list(range(8)),
        "L2": list(range(16)),
    }
    retained, removed = cross_bin_state_counts(stateless_channels)
    assert retained == 8_136
    assert removed == 36_864
    configuration = {
        "model": {
            "alpha": 0.75,
            "output_encoding": OUTPUT_ENCODING_CONTINUOUS,
            "stateless_channels": stateless_channels,
            "retained_cross_bin_membrane_states": retained,
            "removed_cross_bin_membrane_states": removed,
        }
    }
    model = make_analysis_model(
        configuration,
        torch.device("cpu"),
        backend="sinabs",
    )
    assert model.stateless_channels() == stateless_channels


def test_training_state_roundtrip_restores_rng_and_sampling() -> None:
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = make_scheduler(optimizer)
    generator = torch.Generator().manual_seed(31)
    loader = DataLoader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    best_model = {
        key: value.detach().clone()
        for key, value in persistent_model_state_dict(model).items()
    }
    state = copy.deepcopy(
        capture_training_state(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loader,
            history=[{"epoch": 1}],
            best_epoch=1,
            best_angle=1.25,
            best_model=best_model,
        )
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=generator)
    with torch.no_grad():
        model.weight.add_(10.0)
    random.random()
    np.random.random()
    torch.rand(3)
    torch.rand(3, generator=generator)
    restore_training_state(state, model, optimizer, scheduler, loader)
    for key, value in best_model.items():
        assert torch.equal(model.state_dict()[key], value)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=generator), expected_loader)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
