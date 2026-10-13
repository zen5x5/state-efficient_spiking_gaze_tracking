"""Measure validation-time spikes, sparse SynOps, touched sites, and bias-only firing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sinabs
import torch
import torch.nn.functional as functional

from code.summarize_user15_eye0_compute import (
    ACTIVITY_SCHEMA_VERSION,
    build_repeated_input_mapping_reuse_candidate,
)

from .dataloader import make_snn_dataloader, prepare_event_bins
from .model import (
    SpikingPursuitSNN,
    folded_batch_norm_bias,
    membrane_state_count,
)
from .training import load_persistent_model_state_dict, seed_everything, write_json


ANN_DENSE_MACS = 10_390_632
CONV_OUTPUT_CHANNELS = {
    "input_to_L1": 8,
    "L1_to_L2": 16,
    "L2_to_L3": 24,
    "L3_to_L4": 32,
    "L4_to_L5": 40,
    "L5_to_L6": 40,
    "L6_to_L7": 40,
    "L7_to_L8": 40,
}


def convolution_synops(source_spikes: torch.Tensor, output_channels: int) -> float:
    """Count dense-kernel accumulations triggered by integer source spikes."""
    _, _, _, height, width = source_spikes.shape
    row_fanout = torch.full(
        (height,), 3.0, device=source_spikes.device, dtype=source_spikes.dtype
    )
    column_fanout = torch.full(
        (width,), 3.0, device=source_spikes.device, dtype=source_spikes.dtype
    )
    row_fanout[[0, -1]] = 2.0
    column_fanout[[0, -1]] = 2.0
    spatial_fanout = row_fanout[:, None] * column_fanout[None, :]
    counts_per_location = source_spikes.sum(dim=2)
    return float((counts_per_location * spatial_fanout).sum()) * output_channels


def convolution_touched_positions(
    source_spikes: torch.Tensor,
    output_channels: int,
    *,
    pool: bool,
) -> int:
    source_support = source_spikes.ne(0).any(dim=2).float()
    batch, timesteps, height, width = source_support.shape
    touched = functional.max_pool2d(
        source_support.reshape(batch * timesteps, 1, height, width),
        kernel_size=3,
        stride=1,
        padding=1,
    )
    if pool:
        touched = functional.max_pool2d(touched, kernel_size=2, stride=2)
    touched = touched.ne(0)
    return int(touched.sum()) * output_channels


def linear_touched_positions(source_spikes: torch.Tensor, output_features: int) -> int:
    return int(source_spikes.ne(0).any(dim=2).sum()) * output_features


def bias_only_spikes(
    model: SpikingPursuitSNN,
    timesteps: int,
) -> dict[str, float]:
    currents = folded_batch_norm_bias(model)
    positions = membrane_state_count()
    layers = {
        **{
            f"L{index}": (
                block.spike,
                currents[f"L{index}"],
                positions[f"L{index}"] // block.spike.channels,
            )
            for index, block in enumerate(model.blocks, start=1)
        },
        "H1": (
            model.hidden1.spike,
            model.hidden1.linear.bias / model.hidden1.spike.threshold,
            1,
        ),
        "H2": (
            model.hidden2.spike,
            model.hidden2.linear.bias / model.hidden2.spike.threshold,
            1,
        ),
    }
    result = {}
    for name, (spike_layer, current, spatial_positions) in layers.items():
        sinabs.reset_states(spike_layer.neuron)
        sequence = current.reshape(1, 1, -1).expand(1, timesteps, -1)
        if spike_layer.stateless_channel_indices:
            emitted_steps = []
            for step in range(timesteps):
                emitted_steps.append(spike_layer.neuron(sequence[:, step:step+1]))
                if step + 1 < timesteps:
                    spike_layer._clear_selected_membrane_state()
            emitted = torch.cat(emitted_steps, dim=1)
        else:
            emitted = spike_layer.neuron(sequence)
        result[name] = float(emitted.sum()) * spatial_positions
    sinabs.reset_states(model)
    return result


@torch.no_grad()
def analyze(
    result_dir: Path,
    cache_dir: Path,
    device: torch.device,
    split: str,
) -> dict:
    configuration = json.loads(
        (result_dir / "configuration.json").read_text(encoding="utf-8")
    )
    alpha = float(configuration["model"]["alpha"])
    timesteps = int(configuration["input"]["temporal_steps"])
    configured_organization = configuration["input"]["temporal_organization"]
    temporal_organization = (
        "repeat_full_400"
        if configured_organization == "repeat_identical_complete_400_event_count_frame"
        else "sequential_equal_events"
    )
    seed = int(configuration["training"]["seed"])
    constraint = configuration["model"].get("folded_bn_bias_constraint", {})
    constrained_layers = 0
    if constraint.get("enabled", False):
        constrained_layers = int(str(constraint["layers"]).split("L")[-1])

    seed_everything(seed)
    dataset, loader = make_snn_dataloader(
        cache_dir, split, 16, seed, temporal_organization
    )
    model = SpikingPursuitSNN(
        alpha=alpha,
        threshold_init=1.0,
        min_v_mem=-1.0,
        backend="exodus",
        stateless_channels=configuration["model"].get("stateless_channels", {}),
    ).to(device)
    checkpoint = torch.load(result_dir / "best_checkpoint.pt", map_location=device)
    load_persistent_model_state_dict(model, checkpoint["model"])
    model.eval()

    layer_modules = {
        **{
            f"L{index}": block.spike
            for index, block in enumerate(model.blocks, start=1)
        },
        "H1": model.hidden1.spike,
        "H2": model.hidden2.spike,
    }
    layer_totals = {
        name: {"spikes": 0.0, "nonzero_positions": 0, "positions": 0}
        for name in layer_modules
    }
    latest_outputs: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in layer_modules.items():
        def capture(_module, _inputs, output, layer_name=name):
            detached = output.detach()
            latest_outputs[layer_name] = detached
            layer_totals[layer_name]["spikes"] += float(detached.sum())
            layer_totals[layer_name]["nonzero_positions"] += int(detached.ne(0).sum())
            layer_totals[layer_name]["positions"] += detached.numel()

        hooks.append(module.register_forward_hook(capture))

    transition_synops = {name: 0.0 for name in CONV_OUTPUT_CHANNELS}
    transition_synops.update({"L8_to_H1": 0.0, "H1_to_H2": 0.0})
    transition_synops_by_timestep = [
        {name: 0.0 for name in transition_synops}
        for _ in range(timesteps)
    ]
    touched_positions = {f"L{index}": 0 for index in range(1, 9)}
    touched_positions.update({"H1": 0, "H2": 0})
    samples = 0
    for batch in loader:
        event_bins = prepare_event_bins(
            batch, timesteps, temporal_organization
        ).to(device)
        latest_outputs.clear()
        model(event_bins)
        samples += event_bins.shape[0]

        transition_synops["input_to_L1"] += convolution_synops(event_bins, 8)
        for timestep in range(timesteps):
            transition_synops_by_timestep[timestep][
                "input_to_L1"
            ] += convolution_synops(event_bins[:, timestep : timestep + 1], 8)
        touched_positions["L1"] += convolution_touched_positions(
            event_bins, 8, pool=True
        )
        for layer_index in range(1, 8):
            source_name = f"L{layer_index}"
            destination_name = f"L{layer_index + 1}"
            transition_name = f"{source_name}_to_{destination_name}"
            output_channels = CONV_OUTPUT_CHANNELS[transition_name]
            source = latest_outputs[source_name]
            transition_synops[transition_name] += convolution_synops(
                source, output_channels
            )
            for timestep in range(timesteps):
                transition_synops_by_timestep[timestep][
                    transition_name
                ] += convolution_synops(
                    source[:, timestep : timestep + 1], output_channels
                )
            touched_positions[destination_name] += convolution_touched_positions(
                source,
                output_channels,
                pool=layer_index + 1 <= 5,
            )
        transition_synops["L8_to_H1"] += float(latest_outputs["L8"].sum()) * 60
        transition_synops["H1_to_H2"] += float(latest_outputs["H1"].sum()) * 12
        for timestep in range(timesteps):
            transition_synops_by_timestep[timestep]["L8_to_H1"] += (
                float(latest_outputs["L8"][:, timestep].sum()) * 60
            )
            transition_synops_by_timestep[timestep]["H1_to_H2"] += (
                float(latest_outputs["H1"][:, timestep].sum()) * 12
            )
        touched_positions["H1"] += linear_touched_positions(latest_outputs["L8"].flatten(2), 60)
        touched_positions["H2"] += linear_touched_positions(latest_outputs["H1"], 12)

    for hook in hooks:
        hook.remove()

    folded_biases = folded_batch_norm_bias(model)
    robust_bound = 1.0 - alpha
    per_layer = {}
    for name, totals in layer_totals.items():
        per_layer[name] = {
            "spikes_per_sample": totals["spikes"] / samples,
            "nonzero_positions_per_sample": totals["nonzero_positions"] / samples,
            "spike_position_fraction": totals["nonzero_positions"] / totals["positions"],
            "synaptically_touched_positions_per_sample": touched_positions[name] / samples,
            "all_positions_per_sample": totals["positions"] / samples,
            "untouched_position_fraction": 1.0 - touched_positions[name] / totals["positions"],
        }
        if name.startswith("L"):
            values = folded_biases[name]
            per_layer[name]["folded_bias_minimum"] = float(values.min())
            per_layer[name]["folded_bias_mean"] = float(values.mean())
            per_layer[name]["folded_bias_maximum"] = float(values.max())
            per_layer[name]["channels_above_robust_bound"] = int(
                (values > robust_bound + 1e-6).sum()
            )
            per_layer[name]["robust_constraint_enabled"] = (
                0 < int(name[1:]) <= constrained_layers
            )

    transition_synops = {
        name: value / samples for name, value in transition_synops.items()
    }
    transition_synops_by_timestep = [
        {name: value / samples for name, value in step.items()}
        for step in transition_synops_by_timestep
    ]
    total_synops = sum(transition_synops.values())
    for name, value in transition_synops.items():
        per_timestep_total = sum(
            step[name] for step in transition_synops_by_timestep
        )
        if not math.isclose(
            value, per_timestep_total, rel_tol=1e-6, abs_tol=1e-3
        ):
            raise ValueError(
                f"per-timestep SynOps do not reproduce aggregate {name}"
            )
    isolated_bias_spikes = bias_only_spikes(model, timesteps)
    total_states = membrane_state_count()["total"]
    retained_states = int(
        configuration["model"].get(
            "retained_cross_bin_membrane_states", total_states
        )
    )
    removed_states = int(
        configuration["model"].get(
            "removed_cross_bin_membrane_states", total_states - retained_states
        )
    )
    if retained_states + removed_states != total_states:
        raise ValueError(
            "configured retained and removed membrane states do not sum to total"
        )
    total_hidden_spikes = sum(
        per_layer[name]["spikes_per_sample"] for name in per_layer
    )
    mapping_reuse_candidate = build_repeated_input_mapping_reuse_candidate(
        temporal_organization=configured_organization,
        timesteps=timesteps,
        stateless_channels=model.stateless_channels(),
        synops_per_timestep_per_sample=transition_synops_by_timestep,
        raw_total_synops_per_sample=total_synops,
    )
    return {
        "activity_schema_version": ACTIVITY_SCHEMA_VERSION,
        "result_dir": str(result_dir.resolve()),
        "split": split,
        "samples": len(dataset),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "timesteps": timesteps,
        "temporal_organization": configured_organization,
        "alpha": alpha,
        "stateless_channels": model.stateless_channels(),
        "retained_cross_bin_membrane_states": retained_states,
        "removed_cross_bin_membrane_states": removed_states,
        "robust_bias_upper_bound": robust_bound,
        "constrained_through_layer": constrained_layers,
        "per_layer": per_layer,
        "total_hidden_spikes_per_sample": total_hidden_spikes,
        "synops_per_sample": transition_synops,
        "synops_per_timestep_per_sample": transition_synops_by_timestep,
        "total_synops_per_sample": total_synops,
        "ann_dense_macs_per_sample": ANN_DENSE_MACS,
        "synops_over_ann_dense_macs": total_synops / ANN_DENSE_MACS,
        "mapping_reuse_candidate": mapping_reuse_candidate,
        "bias_only_spikes_per_sample": isolated_bias_spikes,
        "total_bias_only_spikes_per_sample": sum(isolated_bias_spikes.values()),
        "continuous_readout_macs_per_sample": 24,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(args.result_dir, args.cache_dir, torch.device(args.device), args.split)
    output_dir = args.result_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"activity_{args.split}.json"
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "timesteps": report["timesteps"],
                "alpha": report["alpha"],
                "constrained_through_layer": report["constrained_through_layer"],
                "total_synops_per_sample": report["total_synops_per_sample"],
                "total_bias_only_spikes_per_sample": report[
                    "total_bias_only_spikes_per_sample"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
