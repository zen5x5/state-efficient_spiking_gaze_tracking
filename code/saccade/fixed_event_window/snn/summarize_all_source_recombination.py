"""Audit and summarize all six formal alpha-0.75 saccade SNN runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from code.paths import RESULT_ROOT
from ..ann.grid_projection import evaluate_prediction_file
from .run_all_source_recombination import (
    BATCHES,
    DEFAULT_RESULT_ROOT,
    LOSS_WEIGHTINGS,
)


SEEDS = (2021, 2022, 2023)
SPLIT_COUNTS = {"train": 4228, "validation": 614, "test": 614}
ANN_ROOT = RESULT_ROOT / "saccade/fixed_event_window/ann/user15/eye0"


def ann_result_root(loss_weighting: str) -> Path:
    return ANN_ROOT / (
        "formal_visual_fixation_400event_notail_trainstride200_"
        f"all_physical_source_recombination_{loss_weighting}_epoch200"
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


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}


def audit_run(run_dir: Path, seed: int, loss_weighting: str) -> dict:
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
    expected_input = {
        "events_per_cached_window": 400,
        "shape": [2, 96, 128],
        "temporal_steps": 2,
        "temporal_organization": "repeat_identical_complete_400_event_count_frame",
        "events_represented_per_step": 400,
        "logical_replayed_event_count": 800,
        "train_samples": 8445,
        "physical_train_samples": 4228,
        "validation_samples": 614,
        "test_samples": 614,
    }
    if configuration["input"] != expected_input:
        raise AssertionError(configuration["input"])
    model = configuration["model"]
    for key, value in {
        "parameters": 96_792,
        "learnable_thresholds": 312,
        "membrane_states_per_sample": 45_000,
        "alpha": 0.75,
        "bias_constraint_layers": 0,
        "backend": "exodus",
    }.items():
        if model[key] != value:
            raise AssertionError((key, model[key], value))
    training = configuration["training"]
    if training["seed"] != seed or training["loss_weighting"] != loss_weighting:
        raise AssertionError((training["seed"], training["loss_weighting"]))
    augmentation = configuration["augmentation"]
    for key, value in {
        "physical_train_samples": 4228,
        "physical_stride200_samples": 1834,
        "source_pool_samples": 4228,
        "source_pool_stride_samples": 1834,
        "eligible_source_stride_samples": 1828,
        "eligible_source_nonstride_samples": 2389,
        "partner_pool_stride_samples": 1828,
        "synthetic_samples_per_epoch": 4217,
        "preferred_nonadjacent_sources": 4196,
        "adjacent_fallback_sources": 21,
        "ineligible_sources": 11,
        "synthetic_target_coverage": 115,
    }.items():
        if augmentation[key] != value:
            raise AssertionError((key, augmentation[key], value))
    if len(history) != 200 or [row["epoch"] for row in history] != list(range(1, 201)):
        raise AssertionError(f"invalid history: {run_dir}")
    if any(row["train"]["batches"] != 527 for row in history):
        raise AssertionError("unexpected batches per epoch")
    if any(row["train"]["samples"] != 8432 for row in history):
        raise AssertionError("unexpected samples per epoch")
    if summary["status"] != "complete" or summary["optimizer_updates"] != 105_400:
        raise AssertionError(summary)
    if summary["timesteps"] != 2 or summary["alpha"] != 0.75:
        raise AssertionError((summary["timesteps"], summary["alpha"]))

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
    if any(key.endswith(".v_mem") for key in state["model"]):
        raise AssertionError("transient membrane state entered training state")

    metrics = {}
    grid = {}
    for split, count in SPLIT_COUNTS.items():
        prediction_path = run_dir / f"predictions_{split}.npz"
        with np.load(prediction_path) as predictions:
            sample_index = predictions["sample_index"].astype(np.int64)
            target = predictions["target_index"].astype(np.int64)
            pixel = predictions["pixel_error"].astype(np.float64)
            angular = predictions["angular_error_deg"].astype(np.float64)
        if len(sample_index) != count or not np.array_equal(sample_index, np.arange(count)):
            raise AssertionError((split, len(sample_index), count))
        target_pixel = np.asarray(
            [pixel[target == value].mean() for value in np.unique(target)]
        )
        target_angular = np.asarray(
            [angular[target == value].mean() for value in np.unique(target)]
        )
        recorded = summary["metrics"][split]
        close(pixel.mean(), recorded["mean_pixel_error"])
        close(angular.mean(), recorded["mean_angular_error_deg"])
        close(target_pixel.mean(), recorded["target_macro_mean_pixel_error"])
        close(target_angular.mean(), recorded["target_macro_mean_angular_error_deg"])
        metrics[split] = {
            "samples": count,
            "targets": int(np.unique(target).size),
            "mean_pixel_error": float(pixel.mean()),
            "mean_angular_error_deg": float(angular.mean()),
            "target_macro_mean_pixel_error": float(target_pixel.mean()),
            "target_macro_mean_angular_error_deg": float(target_angular.mean()),
        }
        grid[split] = evaluate_prediction_file(prediction_path)
    activity = json.loads(
        (run_dir / "analysis/activity_validation.json").read_text()
    )
    if activity["split"] != "validation" or activity["samples"] != 614:
        raise AssertionError((activity["split"], activity["samples"]))
    if activity["checkpoint_epoch"] != summary["best_epoch"]:
        raise AssertionError((activity["checkpoint_epoch"], summary["best_epoch"]))
    return {
        "seed": seed,
        "loss_weighting": loss_weighting,
        "best_epoch": int(summary["best_epoch"]),
        "final_learning_rate": float(summary["final_learning_rate"]),
        "total_seconds": float(summary["total_seconds"]),
        "metrics": metrics,
        "grid": grid,
        "activity": activity,
    }


def aggregate(runs: list[dict]) -> dict:
    metric_keys = (
        "mean_pixel_error",
        "mean_angular_error_deg",
        "target_macro_mean_pixel_error",
        "target_macro_mean_angular_error_deg",
    )
    grid_keys = (*metric_keys, "joint_accuracy")
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
        "nearest_grid_projection": {
            split: {
                key: mean_sd(
                    [
                        run["grid"][split]["nearest_grid_projection"][key]
                        for run in runs
                    ]
                )
                for key in grid_keys
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
            "hidden_spikes_per_sample": mean_sd(
                [
                    sum(
                        layer["spikes_per_sample"]
                        for layer in run["activity"]["per_layer"].values()
                    )
                    for run in runs
                ]
            ),
            "total_bias_only_spikes_per_sample": mean_sd(
                [
                    run["activity"]["total_bias_only_spikes_per_sample"]
                    for run in runs
                ]
            ),
        },
    }


def paired_difference(
    left: list[dict],
    right: list[dict],
    definition: str,
) -> dict:
    left_by_seed = {run["seed"]: run for run in left}
    right_by_seed = {run["seed"]: run for run in right}
    if set(left_by_seed) != set(SEEDS) or set(right_by_seed) != set(SEEDS):
        raise AssertionError((sorted(left_by_seed), sorted(right_by_seed)))
    metric_keys = (
        "mean_pixel_error",
        "mean_angular_error_deg",
        "target_macro_mean_pixel_error",
        "target_macro_mean_angular_error_deg",
    )
    output = {"definition": definition, "metrics": {}}
    for split in SPLIT_COUNTS:
        output["metrics"][split] = {}
        for key in metric_keys:
            values = [
                left_by_seed[seed]["metrics"][split][key]
                - right_by_seed[seed]["metrics"][split][key]
                for seed in SEEDS
            ]
            output["metrics"][split][key] = {
                "per_seed": values,
                **mean_sd(values),
            }
    return output


def ann_runs(loss_weighting: str) -> list[dict]:
    summary_path = ann_result_root(loss_weighting) / "study_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete":
        raise AssertionError(summary["status"])
    return summary["runs"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    for batch in BATCHES:
        progress = json.loads(
            (args.result_root / f"batch{batch}_progress.json").read_text()
        )
        if progress["status"] != "complete":
            raise AssertionError((batch, progress["status"]))
        if any(job["return_code"] != 0 for job in progress["jobs"]):
            raise AssertionError(progress["jobs"])
    runs_by_loss = {
        loss: [
            audit_run(args.result_root / loss / f"seed{seed}", seed, loss)
            for seed in SEEDS
        ]
        for loss in LOSS_WEIGHTINGS
    }
    output = {
        "status": "complete",
        "result_root": str(args.result_root),
        "configuration": {
            "timesteps": 2,
            "alpha": 0.75,
            "bias_constraint_layers": 0,
            "source_mode": "all_physical",
            "loss_weightings": list(LOSS_WEIGHTINGS),
            "seeds": list(SEEDS),
        },
        "audit": {
            "runs": 6,
            "epochs_per_run": 200,
            "optimizer_updates_per_run": 105_400,
            "physical_prediction_counts": SPLIT_COUNTS,
            "checkpoint_and_training_state_deserialized": True,
            "prediction_metrics_match_summary": True,
            "validation_activity_analyzed": True,
        },
        "runs": runs_by_loss,
        "groups": {loss: aggregate(runs) for loss, runs in runs_by_loss.items()},
        "paired_target_sqrt_minus_ordinary": paired_difference(
            runs_by_loss["target_sqrt"],
            runs_by_loss["ordinary"],
            "target_sqrt SNN minus ordinary SNN, paired by seed",
        ),
        "paired_snn_minus_ann": {
            loss: paired_difference(
                runs_by_loss[loss],
                ann_runs(loss),
                f"alpha0.75 T2 SNN minus ANN for {loss}, paired by seed",
            )
            for loss in LOSS_WEIGHTINGS
        },
    }
    atomic_json(args.result_root / "study_summary.json", output)
    print(json.dumps(output["groups"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
