"""Summarize paired ANN/SNN compute and cross-bin state costs for user15 eye0.

The command is deliberately read-only with respect to checkpoints and activity
analysis.  It consumes existing ``summary.json`` and current-schema
``analysis/activity_validation.json`` files, then writes one machine JSON and
one report rendered from the checked-in Markdown template.  Missing or legacy
activity files are reported together with commands to regenerate them; this
module never starts GPU analysis itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
from string import Template
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


SEEDS = (2021, 2022, 2023)
ACTIVITY_SCHEMA_VERSION = 2
COMPARISON_SCHEMA_VERSION = 1
ANN_DENSE_MACS = 10_390_632
TOTAL_CROSS_BIN_MEMBRANE_STATES = 45_000
LAYER_CHANNELS = {
    "L1": 8,
    "L2": 16,
    "L3": 24,
    "L4": 32,
    "L5": 40,
    "L6": 40,
    "L7": 40,
    "L8": 40,
}
HIDDEN_SPIKE_LAYERS = tuple(LAYER_CHANNELS) + ("H1", "H2")
REPEAT_TEMPORAL_ORGANIZATIONS = {
    "repeat_full_400",
    "repeat_identical_complete_400_event_count_frame",
}

from code.paths import DATA_ROOT, PROJECT_ROOT, RESULT_ROOT
DEFAULT_REPORT_TEMPLATE = (
    PROJECT_ROOT / "code/templates/user15_eye0_ann_snn_compute_comparison.md"
)


@dataclass(frozen=True)
class ModelRoots:
    ann: Path
    all_state_snn: Path
    l1_l2_stateless_snn: Path


@dataclass(frozen=True)
class TaskInputs:
    name: str
    roots: ModelRoots
    cache_dir: Path
    activity_module: str


DEFAULT_TASK_INPUTS = {
    "pursuit": TaskInputs(
        name="pursuit",
        roots=ModelRoots(
            ann=RESULT_ROOT
            / "pursuit/fixed_event_window/ann/user15/eye0/"
            "no_tail_all_physical_source_recombination_epoch200",
            all_state_snn=RESULT_ROOT
            / "pursuit/fixed_event_window/snn/user15/eye0/"
            "no_tail_all_physical_source_recombination_t2_alpha0p75_epoch200",
            l1_l2_stateless_snn=RESULT_ROOT
            / "pursuit/fixed_event_window/snn/user15/eye0/"
            "channelwise_state_allocation_l1_l2_sixteen_all_physical/"
            "l1_l2_all/continue_epoch100_to200",
        ),
        cache_dir=DATA_ROOT
        / "pursuit/fixed_event_window/"
        "ann_static_400events_full_pursuit_visible_corner_guard80ms_"
        "closed_core_no_tail_trainstride200_96x128_sum2x2_top0_left32_"
        "seed2021/user15/eye0",
        activity_module="code.pursuit.fixed_event_window.snn.analyze_activity",
    ),
    "saccade": TaskInputs(
        name="saccade",
        roots=ModelRoots(
            ann=RESULT_ROOT
            / "saccade/fixed_event_window/ann/user15/eye0/"
            "formal_visual_fixation_400event_notail_trainstride200_"
            "all_physical_source_recombination_ordinary_epoch200",
            all_state_snn=RESULT_ROOT
            / "saccade/fixed_event_window/snn/user15/eye0/"
            "formal_visual_fixation_400event_notail_trainstride200_"
            "all_physical_source_recombination_t2_alpha0p75_epoch200/ordinary",
            l1_l2_stateless_snn=RESULT_ROOT
            / "saccade/fixed_event_window/snn/user15/eye0/"
            "channelwise_state_allocation_l1_l2_all_physical/"
            "l1_l2_all/continue_epoch100_to200",
        ),
        cache_dir=DATA_ROOT
        / "saccade/fixed_event_window/"
        "ann_static_400events_visual_fixation_closedcore_notail_"
        "trainstride200_96x128_sum2x2_top0_left32_seed2021/user15/eye0",
        activity_module="code.saccade.fixed_event_window.snn.analyze_activity",
    ),
}


class ComparisonInputError(RuntimeError):
    """Raised when the result tree is incomplete or violates the contract."""


def _is_repeat_organization(value: str) -> bool:
    return value in REPEAT_TEMPORAL_ORGANIZATIONS


def reusable_mapping_transitions(
    *,
    temporal_organization: str,
    timesteps: int,
    stateless_channels: Mapping[str, Sequence[int]],
    available_transitions: Iterable[str],
) -> list[str]:
    """Return the structurally reusable feed-forward mapping prefix.

    Repeating a complete static input makes ``input_to_L1`` reusable.  A
    subsequent mapping is reusable only while every output channel of every
    preceding spiking layer is cross-bin stateless.  This deliberately does
    not infer reuse from equal aggregate counts on one validation set.
    """

    available = set(available_transitions)
    if timesteps < 2 or not _is_repeat_organization(temporal_organization):
        return []
    result = ["input_to_L1"] if "input_to_L1" in available else []
    for layer_index in range(1, 9):
        layer = f"L{layer_index}"
        actual = {int(value) for value in stateless_channels.get(layer, ())}
        expected = set(range(LAYER_CHANNELS[layer]))
        if actual != expected:
            break
        destination = (
            f"L{layer_index}_to_L{layer_index + 1}"
            if layer_index < 8
            else "L8_to_H1"
        )
        if destination in available:
            result.append(destination)
    return result


def build_repeated_input_mapping_reuse_candidate(
    *,
    temporal_organization: str,
    timesteps: int,
    stateless_channels: Mapping[str, Sequence[int]],
    synops_per_timestep_per_sample: Sequence[Mapping[str, float]],
    raw_total_synops_per_sample: float,
) -> dict:
    """Build a mapping-reuse estimate while preserving raw SynOps separately."""

    if len(synops_per_timestep_per_sample) != timesteps:
        raise ValueError(
            "per-timestep SynOps length does not match configured timesteps"
        )
    if not synops_per_timestep_per_sample:
        raise ValueError("per-timestep SynOps cannot be empty")
    transition_names = tuple(synops_per_timestep_per_sample[0])
    expected = set(transition_names)
    for step in synops_per_timestep_per_sample:
        if set(step) != expected:
            raise ValueError("per-timestep SynOps transition keys differ")
    raw_from_steps = sum(
        float(step[name])
        for step in synops_per_timestep_per_sample
        for name in transition_names
    )
    if not math.isclose(
        raw_from_steps,
        raw_total_synops_per_sample,
        rel_tol=1e-6,
        abs_tol=1e-3,
    ):
        raise ValueError(
            "per-timestep transition SynOps do not sum to raw total SynOps"
        )

    reusable = reusable_mapping_transitions(
        temporal_organization=temporal_organization,
        timesteps=timesteps,
        stateless_channels=stateless_channels,
        available_transitions=transition_names,
    )
    transition_costs: dict[str, dict[str, object]] = {}
    raw_recomputed = 0.0
    cached_once = 0.0
    for name in reusable:
        costs = [float(step[name]) for step in synops_per_timestep_per_sample]
        if not all(
            math.isclose(cost, costs[0], rel_tol=1e-7, abs_tol=1e-3)
            for cost in costs[1:]
        ):
            raise ValueError(
                f"structurally reusable transition {name} has unequal "
                "per-timestep costs"
            )
        transition_costs[name] = {
            "per_timestep": costs,
            "raw_recomputed_accumulations_per_sample": sum(costs),
            "cached_once_accumulations_per_sample": costs[0],
            "avoided_recomputation_accumulations_per_sample": (
                sum(costs) - costs[0]
            ),
        }
        raw_recomputed += sum(costs)
        cached_once += costs[0]
    avoided = raw_recomputed - cached_once
    estimated = raw_total_synops_per_sample - avoided
    return {
        "accounting_kind": "mapping_reuse_estimate_separate_from_raw_synops",
        "applicable": bool(reusable),
        "temporal_organization": temporal_organization,
        "timesteps": timesteps,
        "reusable_transitions": reusable,
        "transition_costs": transition_costs,
        "raw_total_synops_per_sample": raw_total_synops_per_sample,
        "raw_recomputed_mapping_accumulations_per_sample": raw_recomputed,
        "cached_once_mapping_accumulations_per_sample": cached_once,
        "avoided_recomputation_accumulations_per_sample": avoided,
        "estimated_total_accumulations_after_mapping_reuse_per_sample": estimated,
        "estimated_total_over_ann_dense_macs": estimated / ANN_DENSE_MACS,
        "raw_synops_is_not_rewritten": True,
        "estimate_scope": (
            "feed-forward Conv/Linear mappings whose source tensor is "
            "structurally identical across repeated bins"
        ),
        "not_removed_by_estimate": [
            "LIF membrane update",
            "threshold and reset",
            "spike routing",
            "pooling and buffering",
            "cross-bin state access",
        ],
    }


def mean_sd_by_seed(values: Mapping[int, float]) -> dict:
    ordered = {str(seed): float(values[seed]) for seed in sorted(values)}
    sequence = list(ordered.values())
    if not sequence:
        raise ValueError("cannot summarize an empty value set")
    return {
        "mean": statistics.fmean(sequence),
        "sd": statistics.stdev(sequence) if len(sequence) > 1 else 0.0,
        "n": len(sequence),
        "by_seed": ordered,
    }


def cross_bin_state_accounting(
    *, retained_states: int, removed_states: int, timesteps: int, word_bits: int
) -> dict:
    """Return logical boundary capacity and read/write traffic estimates."""

    if word_bits <= 0 or word_bits % 8:
        raise ValueError("state word width must be a positive whole number of bytes")
    if retained_states < 0 or removed_states < 0:
        raise ValueError("state counts cannot be negative")
    if retained_states + removed_states != TOTAL_CROSS_BIN_MEMBRANE_STATES:
        raise ValueError("retained and removed state counts do not sum to 45,000")
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    boundaries = timesteps - 1
    bytes_per_word = word_bits // 8
    reads = retained_states * boundaries
    writes = retained_states * boundaries
    return {
        "accounting_kind": "logical_time_major_cross_bin_boundary_estimate",
        "state_word_bits": word_bits,
        "timesteps": timesteps,
        "cross_bin_boundaries_per_sample": boundaries,
        "retained_state_elements": retained_states,
        "removed_state_elements": removed_states,
        "removed_fraction": removed_states / TOTAL_CROSS_BIN_MEMBRANE_STATES,
        "persistent_capacity_bytes": retained_states * bytes_per_word,
        "boundary_read_elements_per_sample": reads,
        "boundary_write_elements_per_sample": writes,
        "boundary_read_bytes_per_sample": reads * bytes_per_word,
        "boundary_write_bytes_per_sample": writes * bytes_per_word,
        "boundary_read_plus_write_bytes_per_sample": (
            (reads + writes) * bytes_per_word
        ),
        "scope_note": (
            "Counts one logical write and one logical read per retained state "
            "at each bin boundary; this is not an RTL or external-memory measurement."
        ),
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonInputError(f"cannot read valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonInputError(f"expected JSON object: {path}")
    return value


def _split_metrics(summary: Mapping[str, object]) -> dict[str, dict[str, float]]:
    if isinstance(summary.get("metrics"), dict):
        source = summary["metrics"]
        keys = {"train": "train", "validation": "validation", "test": "test"}
    else:
        source = summary
        keys = {
            "train": "best_train",
            "validation": "best_validation",
            "test": "test",
        }
    result = {}
    for split, key in keys.items():
        row = source.get(key) if isinstance(source, Mapping) else None
        if not isinstance(row, Mapping):
            raise ComparisonInputError(f"summary is missing {key} metrics")
        try:
            result[split] = {
                "samples": int(row["samples"]),
                "mean_pixel_error": float(row["mean_pixel_error"]),
                "mean_angular_error_deg": float(row["mean_angular_error_deg"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonInputError(f"invalid {key} metrics") from error
        for optional in (
            "target_macro_mean_pixel_error",
            "target_macro_mean_angular_error_deg",
        ):
            if optional in row:
                result[split][optional] = float(row[optional])
    return result


def _summarize_accuracy(seed_rows: Mapping[int, Mapping[str, object]]) -> dict:
    return {
        split: {
            "samples": {
                str(seed): int(seed_rows[seed][split]["samples"])
                for seed in sorted(seed_rows)
            },
            "mean_pixel_error": mean_sd_by_seed(
                {
                    seed: float(seed_rows[seed][split]["mean_pixel_error"])
                    for seed in seed_rows
                }
            ),
            "mean_angular_error_deg": mean_sd_by_seed(
                {
                    seed: float(
                        seed_rows[seed][split]["mean_angular_error_deg"]
                    )
                    for seed in seed_rows
                }
            ),
        }
        for split in ("train", "validation", "test")
    }


def _seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed{seed}"


def _required_files(task: TaskInputs) -> list[Path]:
    result = []
    for root in (task.roots.ann, task.roots.all_state_snn, task.roots.l1_l2_stateless_snn):
        for seed in SEEDS:
            result.append(_seed_dir(root, seed) / "summary.json")
    for root in (task.roots.all_state_snn, task.roots.l1_l2_stateless_snn):
        for seed in SEEDS:
            run_dir = _seed_dir(root, seed)
            result.extend(
                [
                    run_dir / "configuration.json",
                    run_dir / "analysis/activity_validation.json",
                ]
            )
    return result


def activity_refresh_commands(tasks: Sequence[TaskInputs]) -> list[str]:
    """Return commands for missing or outdated validation activity files."""

    commands = []
    for task in tasks:
        for root in (task.roots.all_state_snn, task.roots.l1_l2_stateless_snn):
            for seed in SEEDS:
                run_dir = _seed_dir(root, seed)
                activity_path = run_dir / "analysis/activity_validation.json"
                refresh = not activity_path.is_file()
                if not refresh:
                    try:
                        value = json.loads(activity_path.read_text(encoding="utf-8"))
                        refresh = (
                            int(value.get("activity_schema_version", 0))
                            < ACTIVITY_SCHEMA_VERSION
                            or "synops_per_timestep_per_sample" not in value
                            or "mapping_reuse_candidate" not in value
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        refresh = True
                if refresh:
                    commands.append(
                        f"python -m {task.activity_module} "
                        f"--result-dir {run_dir} --cache-dir {task.cache_dir} "
                        "--device cuda:0 --split validation"
                    )
    return commands


def preflight(tasks: Sequence[TaskInputs]) -> None:
    missing = [path for task in tasks for path in _required_files(task) if not path.is_file()]
    refresh = activity_refresh_commands(tasks)
    messages = []
    if missing:
        messages.append("missing required files:")
        messages.extend(f"  - {path}" for path in missing)
    if refresh:
        messages.append("activity files requiring current-schema regeneration:")
        messages.extend(f"  - {command}" for command in refresh)
    if messages:
        raise ComparisonInputError("\n".join(messages))


def _collect_ann(root: Path) -> dict:
    metrics = {}
    parameters = {}
    best_epochs = {}
    for seed in SEEDS:
        summary = _read_json(_seed_dir(root, seed) / "summary.json")
        metrics[seed] = _split_metrics(summary)
        parameters[seed] = int(summary["trainable_parameters"])
        best_epochs[seed] = int(summary["best_epoch"])
    if set(parameters.values()) != {96_480}:
        raise ComparisonInputError(f"unexpected ANN parameter counts under {root}")
    return {
        "result_root": str(root.resolve()),
        "seeds": list(SEEDS),
        "trainable_parameters": 96_480,
        "best_epoch": {str(seed): best_epochs[seed] for seed in SEEDS},
        "accuracy": _summarize_accuracy(metrics),
        "compute": {
            "accounting_kind": "dense_conv_and_linear_macs",
            "dense_macs_per_sample": ANN_DENSE_MACS,
        },
    }


def _consistent_scalar(values: Mapping[int, object], label: str) -> object:
    unique = {json.dumps(value, sort_keys=True) for value in values.values()}
    if len(unique) != 1:
        raise ComparisonInputError(f"{label} differs across seeds: {values}")
    return values[min(values)]


def _collect_snn(root: Path, *, expected_stateless: bool, word_bits: int) -> dict:
    metrics = {}
    activities = {}
    best_epochs = {}
    retained = {}
    removed = {}
    timesteps = {}
    organizations = {}
    masks = {}
    parameters = {}
    for seed in SEEDS:
        run_dir = _seed_dir(root, seed)
        summary = _read_json(run_dir / "summary.json")
        configuration = _read_json(run_dir / "configuration.json")
        activity = _read_json(run_dir / "analysis/activity_validation.json")
        if int(activity.get("activity_schema_version", 0)) < ACTIVITY_SCHEMA_VERSION:
            raise ComparisonInputError(f"legacy activity schema: {run_dir}")
        if activity.get("split") != "validation":
            raise ComparisonInputError(f"activity is not validation: {run_dir}")
        metrics[seed] = _split_metrics(summary)
        best_epochs[seed] = int(summary["best_epoch"])
        if int(activity["checkpoint_epoch"]) != best_epochs[seed]:
            raise ComparisonInputError(f"activity checkpoint mismatch: {run_dir}")
        if int(activity["samples"]) != metrics[seed]["validation"]["samples"]:
            raise ComparisonInputError(f"activity sample-count mismatch: {run_dir}")
        if int(activity["ann_dense_macs_per_sample"]) != ANN_DENSE_MACS:
            raise ComparisonInputError(f"ANN dense-MAC reference mismatch: {run_dir}")
        input_config = configuration["input"]
        model_config = configuration["model"]
        seed_config = int(configuration["training"]["seed"])
        if seed_config != seed:
            raise ComparisonInputError(f"configuration seed mismatch: {run_dir}")
        timesteps[seed] = int(input_config["temporal_steps"])
        organizations[seed] = str(input_config["temporal_organization"])
        masks[seed] = activity.get("stateless_channels", {})
        retained[seed] = int(activity["retained_cross_bin_membrane_states"])
        removed[seed] = int(activity["removed_cross_bin_membrane_states"])
        parameters[seed] = int(summary["trainable_parameters"])
        if int(model_config.get("retained_cross_bin_membrane_states", retained[seed])) != retained[seed]:
            raise ComparisonInputError(f"configured retained states mismatch: {run_dir}")
        if int(model_config.get("removed_cross_bin_membrane_states", removed[seed])) != removed[seed]:
            raise ComparisonInputError(f"configured removed states mismatch: {run_dir}")
        activities[seed] = activity

    timestep_count = int(_consistent_scalar(timesteps, "timesteps"))
    organization = str(_consistent_scalar(organizations, "temporal organization"))
    mask = _consistent_scalar(masks, "stateless mask")
    retained_count = int(_consistent_scalar(retained, "retained states"))
    removed_count = int(_consistent_scalar(removed, "removed states"))
    if set(parameters.values()) != {96_792}:
        raise ComparisonInputError(f"unexpected SNN parameter counts under {root}")
    expected_mask = (
        {"L1": list(range(8)), "L2": list(range(16))}
        if expected_stateless
        else {}
    )
    if mask != expected_mask:
        raise ComparisonInputError(f"unexpected stateless mask under {root}: {mask}")
    expected_counts = (8_136, 36_864) if expected_stateless else (45_000, 0)
    if (retained_count, removed_count) != expected_counts:
        raise ComparisonInputError(
            f"unexpected retained/removed states under {root}: "
            f"{retained_count}/{removed_count}"
        )

    layer_names = tuple(activities[SEEDS[0]]["per_layer"])
    transition_names = tuple(activities[SEEDS[0]]["synops_per_sample"])
    for seed in SEEDS[1:]:
        if tuple(activities[seed]["per_layer"]) != layer_names:
            raise ComparisonInputError(f"activity layer keys differ under {root}")
        if tuple(activities[seed]["synops_per_sample"]) != transition_names:
            raise ComparisonInputError(f"transition keys differ under {root}")

    hidden_spikes = {
        seed: sum(
            float(activities[seed]["per_layer"][name]["spikes_per_sample"])
            for name in HIDDEN_SPIKE_LAYERS
        )
        for seed in SEEDS
    }
    for seed in SEEDS:
        reported = float(activities[seed]["total_hidden_spikes_per_sample"])
        if not math.isclose(hidden_spikes[seed], reported, rel_tol=1e-9, abs_tol=1e-6):
            raise ComparisonInputError(f"hidden-spike total mismatch under {root}")

    timestep_transition: dict[str, dict[str, dict]] = {}
    for timestep in range(timestep_count):
        timestep_transition[str(timestep)] = {
            name: mean_sd_by_seed(
                {
                    seed: float(
                        activities[seed]["synops_per_timestep_per_sample"]
                        [timestep][name]
                    )
                    for seed in SEEDS
                }
            )
            for name in transition_names
        }

    reuse_lists = {
        seed: activities[seed]["mapping_reuse_candidate"]["reusable_transitions"]
        for seed in SEEDS
    }
    reusable = _consistent_scalar(reuse_lists, "reusable mapping transitions")
    mapping_fields = (
        "raw_recomputed_mapping_accumulations_per_sample",
        "cached_once_mapping_accumulations_per_sample",
        "avoided_recomputation_accumulations_per_sample",
        "estimated_total_accumulations_after_mapping_reuse_per_sample",
        "estimated_total_over_ann_dense_macs",
    )
    validation_activity = {
        "split": "validation",
        "samples": {
            str(seed): int(activities[seed]["samples"]) for seed in SEEDS
        },
        "hidden_spikes_per_sample": mean_sd_by_seed(hidden_spikes),
        "raw_synops_per_sample": mean_sd_by_seed(
            {
                seed: float(activities[seed]["total_synops_per_sample"])
                for seed in SEEDS
            }
        ),
        "raw_synops_over_ann_dense_macs": mean_sd_by_seed(
            {
                seed: float(activities[seed]["synops_over_ann_dense_macs"])
                for seed in SEEDS
            }
        ),
        "continuous_readout_macs_per_sample": _consistent_scalar(
            {
                seed: int(activities[seed]["continuous_readout_macs_per_sample"])
                for seed in SEEDS
            },
            "continuous readout MACs",
        ),
        "per_layer_spikes_per_sample": {
            name: mean_sd_by_seed(
                {
                    seed: float(
                        activities[seed]["per_layer"][name]["spikes_per_sample"]
                    )
                    for seed in SEEDS
                }
            )
            for name in layer_names
        },
        "transition_synops_per_sample": {
            name: mean_sd_by_seed(
                {
                    seed: float(activities[seed]["synops_per_sample"][name])
                    for seed in SEEDS
                }
            )
            for name in transition_names
        },
        "transition_synops_per_timestep_per_sample": timestep_transition,
        "mapping_reuse_candidate": {
            "accounting_kind": "mapping_reuse_estimate_separate_from_raw_synops",
            "reusable_transitions": reusable,
            **{
                field: mean_sd_by_seed(
                    {
                        seed: float(
                            activities[seed]["mapping_reuse_candidate"][field]
                        )
                        for seed in SEEDS
                    }
                )
                for field in mapping_fields
            },
            "scope_note": (
                "The estimate caches structurally identical feed-forward mappings; "
                "it does not rewrite raw SynOps or remove neuron/state work."
            ),
        },
    }
    return {
        "result_root": str(root.resolve()),
        "seeds": list(SEEDS),
        "trainable_parameters": 96_792,
        "best_epoch": {str(seed): best_epochs[seed] for seed in SEEDS},
        "timesteps": timestep_count,
        "temporal_organization": organization,
        "stateless_channels": mask,
        "accuracy": _summarize_accuracy(metrics),
        "validation_activity": validation_activity,
        "cross_bin_state": cross_bin_state_accounting(
            retained_states=retained_count,
            removed_states=removed_count,
            timesteps=timestep_count,
            word_bits=word_bits,
        ),
    }


def _paired_delta(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict:
    left_values = {int(seed): float(value) for seed, value in left["by_seed"].items()}
    right_values = {int(seed): float(value) for seed, value in right["by_seed"].items()}
    if set(left_values) != set(right_values):
        raise ComparisonInputError("paired comparison seed sets differ")
    return mean_sd_by_seed(
        {seed: left_values[seed] - right_values[seed] for seed in left_values}
    )


def _task_deltas(models: Mapping[str, Mapping[str, object]]) -> dict:
    ann = models["ann"]
    full = models["all_state_snn"]
    stateless = models["l1_l2_stateless_snn"]
    accuracy = {}
    for split in ("train", "validation", "test"):
        accuracy[split] = {}
        for metric in ("mean_pixel_error", "mean_angular_error_deg"):
            accuracy[split][metric] = {
                "all_state_snn_minus_ann": _paired_delta(
                    full["accuracy"][split][metric], ann["accuracy"][split][metric]
                ),
                "l1_l2_stateless_snn_minus_ann": _paired_delta(
                    stateless["accuracy"][split][metric],
                    ann["accuracy"][split][metric],
                ),
                "l1_l2_stateless_snn_minus_all_state_snn": _paired_delta(
                    stateless["accuracy"][split][metric],
                    full["accuracy"][split][metric],
                ),
            }
    return {
        "accuracy": accuracy,
        "validation_activity_l1_l2_stateless_minus_all_state": {
            metric: _paired_delta(
                stateless["validation_activity"][metric],
                full["validation_activity"][metric],
            )
            for metric in ("hidden_spikes_per_sample", "raw_synops_per_sample")
        },
        "cross_bin_state_l1_l2_stateless_minus_all_state": {
            "retained_state_elements": (
                stateless["cross_bin_state"]["retained_state_elements"]
                - full["cross_bin_state"]["retained_state_elements"]
            ),
            "persistent_capacity_bytes": (
                stateless["cross_bin_state"]["persistent_capacity_bytes"]
                - full["cross_bin_state"]["persistent_capacity_bytes"]
            ),
            "boundary_read_plus_write_bytes_per_sample": (
                stateless["cross_bin_state"][
                    "boundary_read_plus_write_bytes_per_sample"
                ]
                - full["cross_bin_state"][
                    "boundary_read_plus_write_bytes_per_sample"
                ]
            ),
        },
    }


def collect_comparison(tasks: Sequence[TaskInputs], *, state_word_bits: int) -> dict:
    preflight(tasks)
    task_results = {}
    for task in tasks:
        models = {
            "ann": _collect_ann(task.roots.ann),
            "all_state_snn": _collect_snn(
                task.roots.all_state_snn,
                expected_stateless=False,
                word_bits=state_word_bits,
            ),
            "l1_l2_stateless_snn": _collect_snn(
                task.roots.l1_l2_stateless_snn,
                expected_stateless=True,
                word_bits=state_word_bits,
            ),
        }
        task_results[task.name] = {
            "cache_dir": str(task.cache_dir.resolve()),
            "models": models,
            "paired_deltas": _task_deltas(models),
        }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "subject": "user15",
            "eye": 0,
            "tasks": [task.name for task in tasks],
            "seeds": list(SEEDS),
            "activity_split": "validation",
            "ann_dense_macs_per_sample": ANN_DENSE_MACS,
            "state_word_bits": state_word_bits,
        },
        "accounting_boundaries": {
            "raw_synops": (
                "Measured sparse Conv/Linear accumulations from validation activity; "
                "continuous readout MACs remain separate."
            ),
            "mapping_reuse": (
                "Schedule candidate for structurally identical repeated-bin mappings; "
                "never substituted for raw SynOps."
            ),
        "cross_bin_state": (
                "Logical time-major boundary capacity/read/write estimate under the "
                "selected configurable state-word width; the width is an accounting "
                "assumption, not a finalized RTL choice or a BRAM-block, "
                "external-memory, latency, or energy measurement."
            ),
        },
        "tasks": task_results,
    }


def _fmt_stats(value: Mapping[str, object], digits: int = 3, scale: float = 1.0) -> str:
    return f"{float(value['mean']) / scale:.{digits}f}±{float(value['sd']) / scale:.{digits}f}"


def _accuracy_cell(model: Mapping[str, object], metric: str, digits: int) -> str:
    return "<br>".join(
        f"{split}: {_fmt_stats(model['accuracy'][split][metric], digits)}"
        for split in ("train", "validation", "test")
    )


def _render_accuracy_table(comparison: Mapping[str, object]) -> str:
    labels = {
        "ann": "ANN",
        "all_state_snn": "全状态SNN",
        "l1_l2_stateless_snn": "L1/L2全无状态SNN",
    }
    rows = [
        "| 任务 | 模型 | 像素误差（px） | 角误差（°） |",
        "|---|---|---:|---:|",
    ]
    for task_name, task in comparison["tasks"].items():
        for model_name, label in labels.items():
            model = task["models"][model_name]
            rows.append(
                f"| {task_name} | {label} | "
                f"{_accuracy_cell(model, 'mean_pixel_error', 2)} | "
                f"{_accuracy_cell(model, 'mean_angular_error_deg', 3)} |"
            )
    return "\n".join(rows)


def _kib(value: float) -> str:
    return f"{value / 1024.0:.2f}"


def _render_compute_table(comparison: Mapping[str, object]) -> str:
    state_word_bits = int(comparison["scope"]["state_word_bits"])
    rows = [
        f"| 任务 | 模型 | ANN稠密MAC | validation隐藏脉冲 | 原始SynOps | 连续读出MAC | 原始SynOps/MAC | 跨bin状态 | {state_word_bits}-bit容量 | 边界读+写 | 映射复用候选 | 映射估算总累加 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for task_name, task in comparison["tasks"].items():
        ann = task["models"]["ann"]
        rows.append(
            f"| {task_name} | ANN | {ann['compute']['dense_macs_per_sample'] / 1e6:.3f}M | — | — | 含于稠密MAC | — | — | — | — | — | — |"
        )
        for model_name, label in (
            ("all_state_snn", "全状态SNN"),
            ("l1_l2_stateless_snn", "L1/L2全无状态SNN"),
        ):
            model = task["models"][model_name]
            activity = model["validation_activity"]
            state = model["cross_bin_state"]
            reuse = activity["mapping_reuse_candidate"]
            rows.append(
                f"| {task_name} | {label} | {ANN_DENSE_MACS / 1e6:.3f}M（参照） | "
                f"{_fmt_stats(activity['hidden_spikes_per_sample'], 2, 1e3)}k | "
                f"{_fmt_stats(activity['raw_synops_per_sample'], 3, 1e6)}M | "
                f"{activity['continuous_readout_macs_per_sample']} | "
                f"{_fmt_stats(activity['raw_synops_over_ann_dense_macs'], 3)} | "
                f"{state['retained_state_elements']:,} | "
                f"{_kib(state['persistent_capacity_bytes'])} KiB | "
                f"{_kib(state['boundary_read_plus_write_bytes_per_sample'])} KiB | "
                f"{', '.join(reuse['reusable_transitions']) or '无'} | "
                f"{_fmt_stats(reuse['estimated_total_accumulations_after_mapping_reuse_per_sample'], 3, 1e6)}M |"
            )
    return "\n".join(rows)


def _render_transition_table(comparison: Mapping[str, object]) -> str:
    rows = [
        "| 任务 | SNN结构 | 时间步 | transition | 原始SynOps/样本 |",
        "|---|---|---:|---|---:|",
    ]
    for task_name, task in comparison["tasks"].items():
        for model_name, label in (
            ("all_state_snn", "全状态"),
            ("l1_l2_stateless_snn", "L1/L2全无状态"),
        ):
            by_step = task["models"][model_name]["validation_activity"][
                "transition_synops_per_timestep_per_sample"
            ]
            for timestep, transitions in by_step.items():
                for name, value in transitions.items():
                    rows.append(
                        f"| {task_name} | {label} | {int(timestep) + 1} | {name} | "
                        f"{_fmt_stats(value, 3, 1e6)}M |"
                    )
    return "\n".join(rows)


def _render_source_list(comparison: Mapping[str, object]) -> str:
    lines = []
    for task_name, task in comparison["tasks"].items():
        lines.append(f"- {task_name} cache: `{task['cache_dir']}`")
        for model_name, model in task["models"].items():
            lines.append(f"- {task_name}/{model_name}: `{model['result_root']}`")
    return "\n".join(lines)


def render_report(comparison: Mapping[str, object], template_text: str) -> str:
    values = {
        "document_notice": (
            "本报告由统一汇总器从三 seed summary 与当前正式 validation 活动 "
            "JSON 自动生成；没有手工填入最终数字。"
        ),
        "generated_at_utc": comparison["generated_at_utc"],
        "state_word_bits": str(comparison["scope"]["state_word_bits"]),
        "accuracy_table": _render_accuracy_table(comparison),
        "compute_table": _render_compute_table(comparison),
        "transition_table": _render_transition_table(comparison),
        "source_list": _render_source_list(comparison),
    }
    try:
        return Template(template_text).substitute(values)
    except KeyError as error:
        raise ComparisonInputError(f"unknown report-template placeholder: {error}") from error


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _task_inputs_from_args(args: argparse.Namespace) -> list[TaskInputs]:
    return [
        TaskInputs(
            name="pursuit",
            roots=ModelRoots(
                ann=args.pursuit_ann_root,
                all_state_snn=args.pursuit_all_state_root,
                l1_l2_stateless_snn=args.pursuit_stateless_root,
            ),
            cache_dir=args.pursuit_cache_dir,
            activity_module=DEFAULT_TASK_INPUTS["pursuit"].activity_module,
        ),
        TaskInputs(
            name="saccade",
            roots=ModelRoots(
                ann=args.saccade_ann_root,
                all_state_snn=args.saccade_all_state_root,
                l1_l2_stateless_snn=args.saccade_stateless_root,
            ),
            cache_dir=args.saccade_cache_dir,
            activity_module=DEFAULT_TASK_INPUTS["saccade"].activity_module,
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    pursuit = DEFAULT_TASK_INPUTS["pursuit"]
    saccade = DEFAULT_TASK_INPUTS["saccade"]
    parser.add_argument("--pursuit-ann-root", type=Path, default=pursuit.roots.ann)
    parser.add_argument(
        "--pursuit-all-state-root", type=Path, default=pursuit.roots.all_state_snn
    )
    parser.add_argument(
        "--pursuit-stateless-root",
        type=Path,
        default=pursuit.roots.l1_l2_stateless_snn,
    )
    parser.add_argument("--pursuit-cache-dir", type=Path, default=pursuit.cache_dir)
    parser.add_argument("--saccade-ann-root", type=Path, default=saccade.roots.ann)
    parser.add_argument(
        "--saccade-all-state-root", type=Path, default=saccade.roots.all_state_snn
    )
    parser.add_argument(
        "--saccade-stateless-root",
        type=Path,
        default=saccade.roots.l1_l2_stateless_snn,
    )
    parser.add_argument("--saccade-cache-dir", type=Path, default=saccade.cache_dir)
    parser.add_argument(
        "--state-word-bits",
        type=int,
        default=16,
        help=(
            "configurable accounting assumption for state capacity and boundary "
            "traffic (default: 16; not a finalized RTL width)"
        ),
    )
    parser.add_argument("--report-template", type=Path, default=DEFAULT_REPORT_TEMPLATE)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and (args.output_json is None or args.output_report is None):
        parser.error("--output-json and --output-report are required unless --check-only")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = _task_inputs_from_args(args)
    try:
        comparison = collect_comparison(tasks, state_word_bits=args.state_word_bits)
    except ComparisonInputError as error:
        print(f"comparison inputs are not ready:\n{error}", file=sys.stderr)
        return 2
    if args.check_only:
        print("comparison inputs are complete and internally consistent")
        return 0
    template_text = args.report_template.read_text(encoding="utf-8")
    report = render_report(comparison, template_text)
    if not args.overwrite:
        existing = [path for path in (args.output_json, args.output_report) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing outputs: "
                + ", ".join(str(path) for path in existing)
            )
    _atomic_write(
        args.output_json,
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        overwrite=args.overwrite,
    )
    _atomic_write(args.output_report, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_report": str(args.output_report.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
