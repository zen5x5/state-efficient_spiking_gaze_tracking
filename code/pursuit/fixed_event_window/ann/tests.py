"""Checks for the retained 2×96×128 pursuit ANN."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..augmentation import (
    RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM,
    fixed_inner_contiguous_recombination,
    sample_event_occurrences,
    select_all_physical_source_recombination_anchors,
    select_local_recombination_pairs,
    select_source_wise_recombination_anchors,
)
from .metrics import denormalize_xy, normalize_row_col, squared_pixel_loss
from .model import StaticPursuitANNCompact90K, trainable_parameter_count
from .training import ANN_ARCHITECTURES, make_model


def main() -> int:
    labels = torch.tensor([[0.0, 0.0], [540.0, 960.0], [1080.0, 1920.0]])
    normalized = normalize_row_col(labels)
    assert torch.equal(
        normalized,
        torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
    )
    assert torch.equal(denormalize_xy(normalized), labels)
    assert torch.equal(squared_pixel_loss(normalized, normalized), torch.zeros(3))

    expected_parameter_counts = {
        ANN_ARCHITECTURES[0]: 96_480,
        ANN_ARCHITECTURES[1]: 78_868,
    }
    for architecture, expected_parameter_count in expected_parameter_counts.items():
        model = make_model(architecture)
        assert model(torch.zeros(2, 2, 96, 128)).shape == (2, 2)
        assert trainable_parameter_count(model) == expected_parameter_count
        assert sum(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()) == 8

    compact = StaticPursuitANNCompact90K()
    assert compact.flattened_features == 432
    assert compact(torch.zeros(2, 2, 96, 128)).shape == (2, 2)
    assert trainable_parameter_count(compact) == 78_868
    assert np.isclose(trainable_parameter_count(compact) / 96_480, 0.8174543946932007)

    compact_120x220 = StaticPursuitANNCompact90K(input_height=120, input_width=220)
    assert compact_120x220.flattened_features == 648
    assert compact_120x220(torch.zeros(2, 2, 120, 220)).shape == (2, 2)
    assert trainable_parameter_count(compact_120x220) == 89_236

    labels = np.asarray(
        [[150.0, 150.0], [150.0, 170.0], [150.0, 210.0]], dtype=np.float32
    )
    pairs = select_local_recombination_pairs(
        labels=labels,
        group_indices=np.zeros(3, dtype=np.int32),
        start_event_indices=np.asarray([0, 800, 1600], dtype=np.int32),
        augmented=np.zeros(3, dtype=bool),
        backfilled=np.zeros(3, dtype=bool),
        max_label_distance_px=32.0,
    )
    assert [(pair.first_index, pair.second_index) for pair in pairs] == [(0, 1)]
    anchors = select_source_wise_recombination_anchors(
        labels=labels,
        group_indices=np.zeros(3, dtype=np.int32),
        start_event_indices=np.asarray([0, 800, 1600], dtype=np.int32),
        augmented=np.zeros(3, dtype=bool),
        backfilled=np.zeros(3, dtype=bool),
        max_label_distance_px=64.0,
    )
    assert [anchor.source_index for anchor in anchors] == [0, 1, 2]
    assert [anchor.partner_indices for anchor in anchors] == [(1, 2), (0, 2), (0, 1)]

    all_source_labels = np.repeat(
        np.asarray([[150.0, 150.0]], dtype=np.float32), 8, axis=0
    )
    source_pool, all_source_anchors = (
        select_all_physical_source_recombination_anchors(
            labels=all_source_labels,
            group_indices=np.asarray([0, 0, 0, 0, 0, 1, 1, 2]),
            start_event_indices=np.asarray(
                [0, 200, 400, 600, 800, 2000, 2400, 3000]
            ),
            augmented=np.asarray(
                [False, True, False, True, False, False, False, False]
            ),
            backfilled=np.zeros(8, dtype=bool),
        )
    )
    assert (
        RECOMBINATION_MODE_ALL_PHYSICAL_SOURCE_EPOCH_RANDOM
        == "all_physical_source_epoch_random"
    )
    assert source_pool.tolist() == list(range(8))
    assert [anchor.source_index for anchor in all_source_anchors] == list(range(7))
    assert all_source_anchors[0].preferred_partner_indices == (3, 4)
    assert all_source_anchors[0].adjacent_partner_indices == (2,)
    assert all_source_anchors[2].preferred_partner_indices == ()
    assert all_source_anchors[2].adjacent_partner_indices == (0, 4)
    assert all_source_anchors[5].preferred_partner_indices == ()
    assert all_source_anchors[5].adjacent_partner_indices == (6,)

    count_frame = np.zeros((2, 4, 4), dtype=np.uint16)
    count_frame[0, 0, 0] = 250
    count_frame[1, 3, 3] = 150
    sampled_a = sample_event_occurrences(
        count_frame, 200, np.random.default_rng(2021)
    )
    sampled_b = sample_event_occurrences(
        count_frame, 200, np.random.default_rng(2021)
    )
    assert np.array_equal(sampled_a, sampled_b)
    assert int(sampled_a.sum()) == 200
    assert np.all(sampled_a <= count_frame)

    first_bins = np.zeros((4, 2, 4, 4), dtype=np.uint16)
    second_bins = np.zeros_like(first_bins)
    for bin_index in range(4):
        first_bins[bin_index, 0, bin_index, 0] = 100
        second_bins[bin_index, 1, bin_index, 3] = 100
    fixed = fixed_inner_contiguous_recombination(first_bins, second_bins)
    assert int(fixed.sum()) == 400
    assert int(fixed[0, 0, 0]) == 0
    assert int(fixed[0, 2, 0]) == 100
    assert int(fixed[0, 3, 0]) == 100
    assert int(fixed[1, 0, 3]) == 100
    assert int(fixed[1, 1, 3]) == 100
    assert int(fixed[1, 2, 3]) == 0
    print("retained ANN checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
