"""Checks for formal-cache ANN loading, weighting, and architecture pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from code.pursuit.fixed_event_window.ann.model import StaticPursuitANNHiddenReadout
from code.paths import saccade_cache_root

from .dataloader import (
    LOSS_WEIGHTING_ORDINARY,
    LOSS_WEIGHTING_TARGET_SQRT,
    StaticSaccadeDataset,
    target_sqrt_sample_weights,
)
from .augmentation import (
    AllPhysicalSameTargetRecombinationDataset,
)
from .model import StaticSaccadeANN, trainable_parameter_count
from .run_all_source_recombination import (
    BATCHES as ALL_SOURCE_BATCHES,
    DEFAULT_RESULT_ROOT as ALL_SOURCE_RESULT_ROOT,
    default_result_root as all_source_result_root,
)


FORMAL_CACHE = saccade_cache_root() / "user15/eye0"


def test_model_is_exact_pursuit_pair() -> None:
    torch.manual_seed(2021)
    pursuit = StaticPursuitANNHiddenReadout()
    torch.manual_seed(2021)
    saccade = StaticSaccadeANN()
    for left, right in zip(pursuit.state_dict().values(), saccade.state_dict().values()):
        assert torch.equal(left, right)
    assert trainable_parameter_count(saccade) == 96_480
    assert saccade(torch.zeros(2, 2, 96, 128)).shape == (2, 2)


def test_target_sqrt_formula() -> None:
    targets = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)
    weights = target_sqrt_sample_weights(targets)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[1] / weights[4], 0.5)
    assert np.isclose(weights[:4].sum() / weights[4], 2.0)


def test_formal_cache_loading_and_weights() -> None:
    ordinary = StaticSaccadeDataset(
        FORMAL_CACHE, "train", LOSS_WEIGHTING_ORDINARY
    )
    weighted = StaticSaccadeDataset(
        FORMAL_CACHE, "train", LOSS_WEIGHTING_TARGET_SQRT
    )
    validation = StaticSaccadeDataset(FORMAL_CACHE, "validation")
    test = StaticSaccadeDataset(FORMAL_CACHE, "test")
    assert (len(ordinary), len(validation), len(test)) == (4228, 614, 614)
    assert np.array_equal(ordinary.target_indices, weighted.target_indices)
    assert np.all(ordinary.sample_weights == 1.0)
    assert np.isclose(weighted.sample_weights.mean(), 1.0)
    assert len(weighted.target_counts) == 120
    assert int(ordinary.event_frames.sum(axis=(1, 2, 3)).min()) == 400
    assert int(ordinary.event_frames.sum(axis=(1, 2, 3)).max()) == 400
    assert int(ordinary.augmented.sum()) == 1834
    assert int(ordinary.backfilled.sum()) == 0


def test_all_physical_recombination_is_epoch_deterministic() -> None:
    physical = StaticSaccadeDataset(FORMAL_CACHE, "train")
    dataset = AllPhysicalSameTargetRecombinationDataset(physical, seed=2021)
    synthetic_index = dataset.original_sample_count
    dataset.set_epoch(1)
    first = dataset[synthetic_index]
    repeated = dataset[synthetic_index]
    assert torch.equal(first["event_frame"], repeated["event_frame"])
    assert int(first["event_frame"].sum()) == 400
    source_indices = first["source_indices"].numpy()
    assert np.all(~physical.backfilled[source_indices])
    assert len(set(physical.target_indices[source_indices].tolist())) == 1
    dataset.set_epoch(2)
    changed = dataset[synthetic_index]
    assert not torch.equal(first["event_frame"], changed["event_frame"])


def test_all_physical_recombination_includes_stride_sources_and_partners() -> None:
    physical = StaticSaccadeDataset(FORMAL_CACHE, "train")
    dataset = AllPhysicalSameTargetRecombinationDataset(physical, seed=2021)
    assert len(dataset.source_pool_indices) == 4228
    assert len(dataset.anchors) == 4217
    assert len(dataset) == 8445
    assert dataset.preferred_source_count == 4196
    assert dataset.adjacent_fallback_source_count == 21
    assert dataset.ineligible_source_count == 11
    assert dataset.synthetic_target_coverage == 115
    assert dataset.eligible_stride_source_count == 1828
    assert dataset.selectable_partner_stride_count == 1828
    assert int(physical.augmented[dataset.source_pool_indices].sum()) == 1834
    assert any(
        bool(physical.augmented[partner_index])
        for anchor in dataset.anchors
        for partner_index in (
            anchor.preferred_partner_indices
            if anchor.preferred_partner_indices
            else anchor.adjacent_partner_indices
        )
    )
    stride_anchor_offset = next(
        offset
        for offset, anchor in enumerate(dataset.anchors)
        if bool(physical.augmented[anchor.source_index])
    )
    dataset.set_epoch(1)
    synthetic = dataset[dataset.original_sample_count + stride_anchor_offset]
    source_indices = synthetic["source_indices"].numpy()
    assert bool(physical.augmented[source_indices[0]])
    assert int(synthetic["event_frame"].sum()) == 400
    assert len(set(physical.target_indices[source_indices].tolist())) == 1
    assert abs(
        int(physical.start_event_indices[source_indices[0]])
        - int(physical.start_event_indices[source_indices[1]])
    ) >= 400


def test_all_source_manual_batches_cover_three_seeds_once() -> None:
    assert tuple(ALL_SOURCE_BATCHES) == (1, 2)
    observed = [
        (int(job["seed"]), int(job["gpu"]))
        for jobs in ALL_SOURCE_BATCHES.values()
        for job in jobs
    ]
    assert observed == [(2021, 0), (2022, 1), (2023, 0)]
    assert "all_physical_source_recombination_ordinary" in str(
        ALL_SOURCE_RESULT_ROOT
    )
    assert "all_physical_source_recombination_target_sqrt" in str(
        all_source_result_root("target_sqrt")
    )


def main() -> int:
    tests = [
        test_model_is_exact_pursuit_pair,
        test_target_sqrt_formula,
        test_formal_cache_loading_and_weights,
        test_all_physical_recombination_is_epoch_deterministic,
        test_all_physical_recombination_includes_stride_sources_and_partners,
        test_all_source_manual_batches_cover_three_seeds_once,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
