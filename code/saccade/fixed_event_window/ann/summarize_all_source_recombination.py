"""Audit and summarize all-physical-source recombination ANN runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .grid_projection import evaluate_prediction_file
from .run_all_source_recombination import (
    LOSS_WEIGHTINGS,
    default_result_root,
)


SEEDS = (2021, 2022, 2023)
SPLIT_COUNTS = {"train": 4228, "validation": 614, "test": 614}


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
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError({"run_dir": str(run_dir), "missing": missing})

    configuration = json.loads((run_dir / "configuration.json").read_text())
    history = json.loads((run_dir / "training_history.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    expected_input = {
        "events_per_sample": 400,
        "shape": [2, 96, 128],
        "train_samples": 8445,
        "physical_train_samples": 4228,
        "validation_samples": 614,
        "test_samples": 614,
    }
    if configuration["input"] != expected_input:
        raise AssertionError(configuration["input"])
    augmentation = configuration["augmentation"]
    expected_augmentation = {
        "physical_train_samples": 4228,
        "physical_stride200_samples": 1834,
        "source_pool_rule": "all physical train samples",
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
    }
    for key, value in expected_augmentation.items():
        if augmentation[key] != value:
            raise AssertionError((key, augmentation[key], value))
    training = configuration["training"]
    if training["seed"] != seed or training["loss_weighting"] != loss_weighting:
        raise AssertionError((training["seed"], training["loss_weighting"]))
    if len(history) != 200 or [row["epoch"] for row in history] != list(range(1, 201)):
        raise AssertionError(f"invalid history: {run_dir}")
    if any(row["train"]["batches"] != 527 for row in history):
        raise AssertionError("unexpected batches per epoch")
    if any(row["train"]["samples"] != 8432 for row in history):
        raise AssertionError("unexpected samples seen per epoch")
    if summary["status"] != "complete" or summary["optimizer_updates"] != 105_400:
        raise AssertionError(summary)
    if summary["same_target_recombination_nonstride_sources"]:
        raise AssertionError("nonstride-only recombination flag is true")
    if not summary["same_target_recombination_all_sources"]:
        raise AssertionError("all-source recombination flag is false")
    if summary["physical_train_samples"] != 4228:
        raise AssertionError(summary["physical_train_samples"])
    if summary["synthetic_train_samples_per_epoch"] != 4217:
        raise AssertionError(summary["synthetic_train_samples_per_epoch"])

    best = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    state = torch.load(
        run_dir / "epoch_200_training_state.pt", map_location="cpu", weights_only=False
    )
    if best["epoch"] != summary["best_epoch"]:
        raise AssertionError((best["epoch"], summary["best_epoch"]))
    if state["epoch"] != 200 or len(state["history"]) != 200:
        raise AssertionError((state["epoch"], len(state["history"])))

    compact_metrics = {}
    grid = {}
    for split, count in SPLIT_COUNTS.items():
        prediction_path = run_dir / f"predictions_{split}.npz"
        with np.load(prediction_path) as predictions:
            if len(predictions["sample_index"]) != count:
                raise AssertionError((split, len(predictions["sample_index"]), count))
            if not np.array_equal(predictions["sample_index"], np.arange(count)):
                raise AssertionError(f"noncanonical sample order: {run_dir} {split}")
            pixel = predictions["pixel_error"].astype(np.float64)
            angular = predictions["angular_error_deg"].astype(np.float64)
            target = predictions["target_index"].astype(np.int64)
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
        compact_metrics[split] = {
            "samples": count,
            "targets": int(np.unique(target).size),
            "mean_pixel_error": float(pixel.mean()),
            "mean_angular_error_deg": float(angular.mean()),
            "target_macro_mean_pixel_error": float(target_pixel.mean()),
            "target_macro_mean_angular_error_deg": float(target_angular.mean()),
        }
        grid[split] = evaluate_prediction_file(prediction_path)

    return {
        "seed": seed,
        "loss_weighting": loss_weighting,
        "best_epoch": int(summary["best_epoch"]),
        "final_learning_rate": float(summary["final_learning_rate"]),
        "total_seconds": float(summary["total_seconds"]),
        "metrics": compact_metrics,
        "grid": grid,
    }


def aggregate(runs: list[dict]) -> dict:
    metric_keys = (
        "mean_pixel_error",
        "mean_angular_error_deg",
        "target_macro_mean_pixel_error",
        "target_macro_mean_angular_error_deg",
    )
    grid_keys = (
        "mean_pixel_error",
        "mean_angular_error_deg",
        "target_macro_mean_pixel_error",
        "target_macro_mean_angular_error_deg",
        "joint_accuracy",
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loss-weighting", choices=LOSS_WEIGHTINGS, default="ordinary"
    )
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root or default_result_root(args.loss_weighting)
    batches = []
    for batch in (1, 2):
        progress = json.loads(
            (result_root / f"batch{batch}_progress.json").read_text()
        )
        if progress["status"] != "complete":
            raise AssertionError((batch, progress["status"]))
        if progress["loss_weighting"] != args.loss_weighting:
            raise AssertionError((batch, progress["loss_weighting"]))
        if progress["source_mode"] != "all_physical":
            raise AssertionError((batch, progress["source_mode"]))
        batches.append(progress)
    runs = [
        audit_run(result_root / f"seed{seed}", seed, args.loss_weighting)
        for seed in SEEDS
    ]
    output = {
        "status": "complete",
        "result_root": str(result_root),
        "loss_weighting": args.loss_weighting,
        "source_mode": "all_physical",
        "audit": {
            "runs": 3,
            "epochs_per_run": 200,
            "optimizer_updates_per_run": 105_400,
            "physical_prediction_counts": SPLIT_COUNTS,
            "checkpoint_and_training_state_deserialized": True,
            "prediction_metrics_match_summary": True,
            "eligible_stride_sources": 1828,
            "eligible_stride_partners": 1828,
        },
        "manual_batches": [
            {
                "batch": progress["batch"],
                "status": progress["status"],
                "jobs": [
                    {
                        "seed": job["seed"],
                        "gpu": job["gpu"],
                        "return_code": job["return_code"],
                    }
                    for job in progress["jobs"]
                ],
            }
            for progress in batches
        ],
        "runs": runs,
        "group": aggregate(runs),
    }
    atomic_json(result_root / "study_summary.json", output)
    print(json.dumps(output["group"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
