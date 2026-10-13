"""Measure validation spikes and sparse SynOps for a formal saccade SNN run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from code.summarize_user15_eye0_compute import (
    ACTIVITY_SCHEMA_VERSION,
    build_repeated_input_mapping_reuse_candidate,
)

from code.pursuit.fixed_event_window.snn.analyze_activity import (
    ANN_DENSE_MACS,
    CONV_OUTPUT_CHANNELS,
    bias_only_spikes,
    convolution_synops,
    convolution_touched_positions,
    linear_touched_positions,
)

from .dataloader import make_dataloader, repeat_full_event_frame
from .model import (
    GridCountSpikingSaccadeSNN,
    SpikingSaccadeSNN,
    folded_batch_norm_bias,
    grid_count_membrane_state_count,
)
from .training import (
    OUTPUT_ENCODING_GRID_SPIKE_COUNT,
    cross_bin_state_counts,
    load_persistent_model_state_dict,
    seed_everything,
    write_json,
)


def make_analysis_model(
    configuration: dict,
    device: torch.device,
    *,
    backend: str = "exodus",
) -> GridCountSpikingSaccadeSNN | SpikingSaccadeSNN:
    """Rebuild the configured architecture before loading persistent weights."""
    alpha = float(configuration["model"]["alpha"])
    output_encoding = configuration["model"].get(
        "output_encoding", "continuous_xy"
    )
    stateless_channels = configuration["model"].get("stateless_channels", {})
    if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        if stateless_channels:
            raise ValueError(
                "grid spike-count activity analysis does not support stateless masks"
            )
        model = GridCountSpikingSaccadeSNN(
            alpha=alpha,
            threshold_init=1.0,
            min_v_mem=-1.0,
            backend=backend,
        )
    else:
        model = SpikingSaccadeSNN(
            alpha=alpha,
            threshold_init=1.0,
            min_v_mem=-1.0,
            backend=backend,
            stateless_channels=stateless_channels,
        )
    return model.to(device)


@torch.no_grad()
def analyze(
    result_dir: Path,
    cache_dir: Path,
    device: torch.device,
    split: str,
) -> dict:
    configuration = json.loads((result_dir / "configuration.json").read_text())
    seed = int(configuration["training"]["seed"])
    timesteps = int(configuration["input"]["temporal_steps"])
    alpha = float(configuration["model"]["alpha"])
    output_encoding = configuration["model"].get(
        "output_encoding", "continuous_xy"
    )
    seed_everything(seed)
    dataset, loader = make_dataloader(
        cache_dir, split, 16, seed, shuffle=False
    )
    model = make_analysis_model(configuration, device)
    checkpoint = torch.load(
        result_dir / "best_checkpoint.pt", map_location=device, weights_only=False
    )
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
    if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        layer_modules["Output"] = model.readout.spike
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
            layer_totals[layer_name]["nonzero_positions"] += int(
                detached.ne(0).sum()
            )
            layer_totals[layer_name]["positions"] += detached.numel()

        hooks.append(module.register_forward_hook(capture))

    transition_synops = {name: 0.0 for name in CONV_OUTPUT_CHANNELS}
    transition_synops.update({"L8_to_H1": 0.0, "H1_to_H2": 0.0})
    if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        transition_synops["H2_to_Output"] = 0.0
    transition_synops_by_timestep = [
        {name: 0.0 for name in transition_synops}
        for _ in range(timesteps)
    ]
    touched_positions = {f"L{index}": 0 for index in range(1, 9)}
    touched_positions.update({"H1": 0, "H2": 0})
    if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        touched_positions["Output"] = 0
    samples = 0
    for batch in loader:
        event_bins = repeat_full_event_frame(
            batch["event_frame"].to(device), timesteps
        )
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
                source, output_channels, pool=layer_index + 1 <= 5
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
        if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
            transition_synops["H2_to_Output"] += (
                float(latest_outputs["H2"].sum()) * 2
            )
            for timestep in range(timesteps):
                transition_synops_by_timestep[timestep]["H2_to_Output"] += (
                    float(latest_outputs["H2"][:, timestep].sum()) * 2
                )
        touched_positions["H1"] += linear_touched_positions(
            latest_outputs["L8"].flatten(2), 60
        )
        touched_positions["H2"] += linear_touched_positions(
            latest_outputs["H1"], 12
        )
        if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
            touched_positions["Output"] += linear_touched_positions(
                latest_outputs["H2"], 2
            )
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
            stateless_indices = list(
                layer_modules[name].stateless_channel_indices
            )
            per_layer[name].update(
                {
                    "folded_bias_minimum": float(values.min()),
                    "folded_bias_mean": float(values.mean()),
                    "folded_bias_maximum": float(values.max()),
                    "channels_above_robust_bound": int(
                        (values > robust_bound + 1e-6).sum()
                    ),
                    "robust_constraint_enabled": False,
                    "stateless_channel_indices": stateless_indices,
                    "stateless_channel_count": len(stateless_indices),
                }
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
    if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT:
        isolated_bias_spikes["Output"] = 0.0
        retained_membrane_states = grid_count_membrane_state_count()["total"]
        removed_membrane_states = 0
        stateless_channels = {}
    else:
        stateless_channels = model.stateless_channels()
        retained_membrane_states, removed_membrane_states = (
            cross_bin_state_counts(stateless_channels)
        )
    configured_retained = int(
        configuration["model"].get(
            "retained_cross_bin_membrane_states", retained_membrane_states
        )
    )
    configured_removed = int(
        configuration["model"].get(
            "removed_cross_bin_membrane_states", removed_membrane_states
        )
    )
    if (
        configured_retained != retained_membrane_states
        or configured_removed != removed_membrane_states
    ):
        raise ValueError(
            "configured cross-bin membrane-state counts do not match the "
            "configured stateless mask"
        )
    total_hidden_spikes = sum(
        per_layer[name]["spikes_per_sample"]
        for name in per_layer
        if name != "Output"
    )
    mapping_reuse_candidate = build_repeated_input_mapping_reuse_candidate(
        temporal_organization=configuration["input"]["temporal_organization"],
        timesteps=timesteps,
        stateless_channels=stateless_channels,
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
        "temporal_organization": configuration["input"]["temporal_organization"],
        "alpha": alpha,
        "loss_weighting": configuration["training"]["loss_weighting"],
        "output_encoding": output_encoding,
        "stateless_channels": stateless_channels,
        "retained_cross_bin_membrane_states": retained_membrane_states,
        "removed_cross_bin_membrane_states": removed_membrane_states,
        "robust_bias_upper_bound": robust_bound,
        "constrained_through_layer": 0,
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
        "continuous_readout_macs_per_sample": (
            0 if output_encoding == OUTPUT_ENCODING_GRID_SPIKE_COUNT else 24
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(
        args.result_dir, args.cache_dir, torch.device(args.device), args.split
    )
    output_dir = args.result_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"activity_{args.split}.json"
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
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
