"""Audit the retained seed2021 from the frozen grid-count SNN route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from code.paths import RESULT_ROOT
from .run_grid_spike_count import BATCHES, DEFAULT_RESULT_ROOT


DEFAULT_SEEDS = (2021,)
SPLIT_COUNTS = {"train": 4228, "validation": 614, "test": 614}
CONTINUOUS_SNN_ROOT = RESULT_ROOT / (
    "saccade/fixed_event_window/snn/user15/eye0/"
    "formal_visual_fixation_400event_notail_trainstride200_"
    "all_physical_source_recombination_t2_alpha0p75_epoch200"
)
ANN_ROOT = RESULT_ROOT / (
    "saccade/fixed_event_window/ann/user15/eye0/"
    "formal_visual_fixation_400event_notail_trainstride200_"
    "all_physical_source_recombination_ordinary_epoch200"
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def close(left: float, right: float) -> None:
    if not np.isclose(left, right, rtol=1e-6, atol=1e-6):
        raise AssertionError((left, right))


def mean_sd(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
    }


def audit_selected_jobs(result_root: Path, seeds: tuple[int, ...]) -> dict:
    selected = set(seeds)
    jobs_by_seed = {}
    progress_files = []
    for batch in BATCHES:
        progress_path = result_root / f"batch{batch}_progress.json"
        if not progress_path.is_file():
            continue
        progress = json.loads(progress_path.read_text())
        progress_files.append(str(progress_path))
        for job in progress["jobs"]:
            jobs_by_seed[int(job["seed"])] = job
    missing = sorted(selected.difference(jobs_by_seed))
    if missing:
        raise FileNotFoundError({"missing_selected_seed_jobs": missing})
    for seed in seeds:
        job = jobs_by_seed[seed]
        if job.get("status") != "complete" or job.get("return_code") != 0:
            raise AssertionError({"selected_seed": seed, "job": job})
    frozen_path = result_root / "frozen_status.json"
    frozen_status = json.loads(frozen_path.read_text()) if frozen_path.is_file() else None
    return {
        "progress_files": progress_files,
        "selected_jobs_complete": True,
        "unselected_jobs": {
            str(seed): {
                "status": job.get("status"),
                "return_code": job.get("return_code"),
            }
            for seed, job in sorted(jobs_by_seed.items())
            if seed not in selected
        },
        "frozen_status": frozen_status,
    }


def audit_run(run_dir: Path, seed: int) -> dict:
    required = (
        "configuration.json",
        "training_history.json",
        "best_checkpoint.pt",
        "epoch_200_training_state.pt",
        "summary.json",
        "predictions_train.npz",
        "predictions_validation.npz",
        "predictions_test.npz",
        "analysis/activity_validation.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError({"run_dir": str(run_dir), "missing": missing})

    configuration = json.loads((run_dir / "configuration.json").read_text())
    history = json.loads((run_dir / "training_history.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    model = configuration["model"]
    for key, value in {
        "parameters": 96_794,
        "learnable_thresholds": 314,
        "membrane_states_per_sample": 45_002,
        "output_encoding": "grid_spike_count_xy",
        "alpha": 0.75,
        "bias_constraint_layers": 0,
        "backend": "exodus",
    }.items():
        if model[key] != value:
            raise AssertionError((key, model[key], value))
    training = configuration["training"]
    if training["seed"] != seed or training["loss_weighting"] != "ordinary":
        raise AssertionError((training["seed"], training["loss_weighting"]))
    if len(history) != 200 or [row["epoch"] for row in history] != list(
        range(1, 201)
    ):
        raise AssertionError(f"invalid history: {run_dir}")
    if any(row["train"]["batches"] != 527 for row in history):
        raise AssertionError("unexpected batches per epoch")
    if any(row["train"]["samples"] != 8432 for row in history):
        raise AssertionError("unexpected samples per epoch")
    if summary["status"] != "complete" or summary["optimizer_updates"] != 105_400:
        raise AssertionError(summary)
    if summary["output_encoding"] != "grid_spike_count_xy":
        raise AssertionError(summary["output_encoding"])

    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    state = torch.load(
        run_dir / "epoch_200_training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise AssertionError((checkpoint["epoch"], summary["best_epoch"]))
    if state["epoch"] != 200 or len(state["history"]) != 200:
        raise AssertionError((state["epoch"], len(state["history"])))
    if any(key.endswith(".v_mem") for key in checkpoint["model"]):
        raise AssertionError("transient membrane state entered checkpoint")

    metrics = {}
    for split, count in SPLIT_COUNTS.items():
        with np.load(run_dir / f"predictions_{split}.npz") as values:
            sample_index = values["sample_index"].astype(np.int64)
            target_index = values["target_index"].astype(np.int64)
            raw_count = values["raw_output_spike_count_xy"].astype(np.float64)
            predicted_grid = values["prediction_grid_index_xy"].astype(np.int64)
            target_grid = values["target_grid_index_xy"].astype(np.int64)
            pixel = values["pixel_error"].astype(np.float64)
            angle = values["angular_error_deg"].astype(np.float64)
            prediction_px = values["prediction_row_col_px"].astype(np.float64)
        if len(sample_index) != count or not np.array_equal(
            sample_index, np.arange(count)
        ):
            raise AssertionError((split, len(sample_index), count))
        if not np.array_equal(raw_count, np.round(raw_count)):
            raise AssertionError("saved raw output count is not integer")
        if predicted_grid.min() < 0 or predicted_grid.max() > 10:
            raise AssertionError("saved grid index is outside [0, 10]")
        expected_col = 150 + 162 * predicted_grid[:, 0]
        expected_row = 150 + 78 * predicted_grid[:, 1]
        coordinate_residual = np.stack(
            (
                prediction_px[:, 0] - expected_row,
                prediction_px[:, 1] - expected_col,
            ),
            axis=-1,
        )
        maximum_coordinate_residual = float(np.abs(coordinate_residual).max())
        if maximum_coordinate_residual > 1e-3:
            raise AssertionError(
                ("decoded coordinate differs from declared grid", maximum_coordinate_residual)
            )
        target_pixel = np.asarray(
            [pixel[target_index == value].mean() for value in np.unique(target_index)]
        )
        target_angle = np.asarray(
            [angle[target_index == value].mean() for value in np.unique(target_index)]
        )
        joint = np.all(predicted_grid == target_grid, axis=1)
        target_joint = np.asarray(
            [joint[target_index == value].mean() for value in np.unique(target_index)]
        )
        recorded = summary["metrics"][split]
        close(pixel.mean(), recorded["mean_pixel_error"])
        close(angle.mean(), recorded["mean_angular_error_deg"])
        close(target_pixel.mean(), recorded["target_macro_mean_pixel_error"])
        close(target_angle.mean(), recorded["target_macro_mean_angular_error_deg"])
        close(joint.mean(), recorded["joint_grid_accuracy"])
        metrics[split] = {
            "samples": count,
            "targets": int(np.unique(target_index).size),
            "mean_pixel_error": float(pixel.mean()),
            "mean_angular_error_deg": float(angle.mean()),
            "target_macro_mean_pixel_error": float(target_pixel.mean()),
            "target_macro_mean_angular_error_deg": float(target_angle.mean()),
            "joint_grid_accuracy": float(joint.mean()),
            "target_macro_joint_grid_accuracy": float(target_joint.mean()),
            "maximum_decoded_coordinate_residual_px": maximum_coordinate_residual,
            "any_axis_out_of_range_samples": int(
                ((raw_count < 0) | (raw_count > 10)).any(axis=1).sum()
            ),
        }

    activity = json.loads(
        (run_dir / "analysis/activity_validation.json").read_text()
    )
    if activity["samples"] != 614 or activity["output_encoding"] != "grid_spike_count_xy":
        raise AssertionError((activity["samples"], activity["output_encoding"]))
    if activity["checkpoint_epoch"] != summary["best_epoch"]:
        raise AssertionError((activity["checkpoint_epoch"], summary["best_epoch"]))
    if activity["retained_cross_bin_membrane_states"] != 45_002:
        raise AssertionError(activity["retained_cross_bin_membrane_states"])
    return {
        "seed": seed,
        "best_epoch": int(summary["best_epoch"]),
        "final_learning_rate": float(summary["final_learning_rate"]),
        "total_seconds": float(summary["total_seconds"]),
        "metrics": metrics,
        "activity": activity,
    }


def aggregate(runs: list[dict]) -> dict:
    metric_keys = (
        "mean_pixel_error",
        "mean_angular_error_deg",
        "target_macro_mean_pixel_error",
        "target_macro_mean_angular_error_deg",
        "joint_grid_accuracy",
        "target_macro_joint_grid_accuracy",
        "maximum_decoded_coordinate_residual_px",
        "any_axis_out_of_range_samples",
    )
    return {
        "seeds": [run["seed"] for run in runs],
        "best_epochs": [run["best_epoch"] for run in runs],
        "total_seconds": mean_sd([run["total_seconds"] for run in runs]),
        "metrics": {
            split: {
                key: mean_sd([run["metrics"][split][key] for run in runs])
                for key in metric_keys
            }
            for split in SPLIT_COUNTS
        },
        "validation_activity": {
            "total_synops_per_sample": mean_sd(
                [run["activity"]["total_synops_per_sample"] for run in runs]
            ),
            "synops_over_ann_dense_macs": mean_sd(
                [run["activity"]["synops_over_ann_dense_macs"] for run in runs]
            ),
            "hidden_and_output_spikes_per_sample": mean_sd(
                [
                    sum(
                        layer["spikes_per_sample"]
                        for layer in run["activity"]["per_layer"].values()
                    )
                    for run in runs
                ]
            ),
            "output_spikes_per_sample": mean_sd(
                [run["activity"]["per_layer"]["Output"]["spikes_per_sample"] for run in runs]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Completed seeds to audit and aggregate (default: 2021).",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be non-empty and unique: {seeds}")
    job_audit = audit_selected_jobs(args.result_root, seeds)
    runs = [audit_run(args.result_root / f"seed{seed}", seed) for seed in seeds]
    continuous_snn = json.loads(
        (CONTINUOUS_SNN_ROOT / "study_summary.json").read_text()
    )["groups"]["ordinary"]
    ann = json.loads((ANN_ROOT / "study_summary.json").read_text())["group"]
    output = {
        "status": "complete_single_seed" if len(seeds) == 1 else "complete",
        "research_status": (
            "frozen"
            if (job_audit.get("frozen_status") or {}).get("status") == "frozen"
            else "active"
        ),
        "result_root": str(args.result_root),
        "configuration": {
            "timesteps": 2,
            "alpha": 0.75,
            "bias_constraint_layers": 0,
            "source_mode": "all_physical",
            "loss_weighting": "ordinary",
            "output_encoding": "grid_spike_count_xy",
            "seeds": list(seeds),
        },
        "audit": {
            "runs": len(seeds),
            "epochs_per_run": 200,
            "optimizer_updates_per_run": 105_400,
            "physical_prediction_counts": SPLIT_COUNTS,
            "integer_bounded_grid_indices": True,
            "decoded_coordinates_match_grid_with_float_tolerance": True,
            "decoded_coordinate_atol_px": 1e-3,
            "validation_activity_analyzed": True,
            "retained_cross_bin_membrane_states": 45_002,
            "job_status": job_audit,
        },
        "runs": runs,
        "group": aggregate(runs),
        "references": {
            "continuous_output_snn_ordinary": {
                "result_root": str(CONTINUOUS_SNN_ROOT),
                "continuous_metrics": continuous_snn["metrics"],
                "nearest_grid_projection": continuous_snn[
                    "nearest_grid_projection"
                ],
            },
            "continuous_output_ann_ordinary": {
                "result_root": str(ANN_ROOT),
                "continuous_metrics": ann["metrics"],
                "nearest_grid_projection": ann["nearest_grid_projection"],
            },
        },
    }
    atomic_json(args.result_root / "study_summary.json", output)
    print(json.dumps(output["group"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
