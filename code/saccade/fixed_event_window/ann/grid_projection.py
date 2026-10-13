"""Nearest known-target grid diagnostics for saved ANN predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import angular_error_deg, normalize_row_col, pixel_error


GRID_ROWS = np.arange(150, 931, 78, dtype=np.float32)
GRID_COLS = np.arange(150, 1771, 162, dtype=np.float32)


def nearest_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("grid projection expects a one-dimensional coordinate array")
    indices = np.abs(values[:, None] - grid[None, :]).argmin(axis=1)
    return grid[indices]


def evaluate_prediction_file(path: Path) -> dict:
    with np.load(path) as predictions:
        required = (
            "sample_index",
            "target_index",
            "target_row_col_px",
            "prediction_row_col_px",
            "pixel_error",
            "angular_error_deg",
        )
        missing = [name for name in required if name not in predictions]
        if missing:
            raise ValueError(f"{path} lacks prediction fields: {missing}")
        sample_index = predictions["sample_index"].astype(np.int64, copy=True)
        target_index = predictions["target_index"].astype(np.int64, copy=True)
        target_px = predictions["target_row_col_px"].astype(np.float32, copy=True)
        prediction_px = predictions["prediction_row_col_px"].astype(
            np.float32, copy=True
        )
        continuous_pixel = predictions["pixel_error"].astype(np.float64, copy=True)
        continuous_angular = predictions["angular_error_deg"].astype(
            np.float64, copy=True
        )

    if not np.array_equal(sample_index, np.arange(len(sample_index))):
        raise AssertionError(f"noncanonical sample order: {path}")
    if target_px.shape != prediction_px.shape or target_px.shape[1:] != (2,):
        raise ValueError(f"unexpected coordinate shapes: {path}")

    target_rows = nearest_grid(target_px[:, 0], GRID_ROWS)
    target_cols = nearest_grid(target_px[:, 1], GRID_COLS)
    if not np.array_equal(target_px[:, 0], target_rows) or not np.array_equal(
        target_px[:, 1], target_cols
    ):
        raise AssertionError(f"targets are not on the declared 11x11 grid: {path}")

    projected_px = np.stack(
        (
            nearest_grid(prediction_px[:, 0], GRID_ROWS),
            nearest_grid(prediction_px[:, 1], GRID_COLS),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    projected_px_tensor = torch.from_numpy(projected_px)
    target_px_tensor = torch.from_numpy(target_px)
    projected_norm = normalize_row_col(projected_px_tensor)
    projected_pixel = pixel_error(projected_norm, target_px_tensor).numpy().astype(
        np.float64
    )
    projected_angular = angular_error_deg(
        projected_norm, target_px_tensor
    ).numpy().astype(np.float64)

    row_correct = projected_px[:, 0] == target_px[:, 0]
    col_correct = projected_px[:, 1] == target_px[:, 1]
    joint_correct = row_correct & col_correct
    unique_targets = np.unique(target_index)
    macro_pixel = np.asarray(
        [projected_pixel[target_index == value].mean() for value in unique_targets]
    )
    macro_angular = np.asarray(
        [projected_angular[target_index == value].mean() for value in unique_targets]
    )
    return {
        "samples": int(len(sample_index)),
        "targets": int(len(unique_targets)),
        "continuous": {
            "mean_pixel_error": float(continuous_pixel.mean()),
            "mean_angular_error_deg": float(continuous_angular.mean()),
        },
        "nearest_grid_projection": {
            "mean_pixel_error": float(projected_pixel.mean()),
            "mean_angular_error_deg": float(projected_angular.mean()),
            "target_macro_mean_pixel_error": float(macro_pixel.mean()),
            "target_macro_mean_angular_error_deg": float(macro_angular.mean()),
            "row_correct": int(row_correct.sum()),
            "col_correct": int(col_correct.sum()),
            "joint_correct": int(joint_correct.sum()),
            "row_accuracy": float(row_correct.mean()),
            "col_accuracy": float(col_correct.mean()),
            "joint_accuracy": float(joint_correct.mean()),
        },
        "grid_minus_continuous": {
            "mean_pixel_error": float(projected_pixel.mean() - continuous_pixel.mean()),
            "mean_angular_error_deg": float(
                projected_angular.mean() - continuous_angular.mean()
            ),
        },
    }


__all__ = ["GRID_COLS", "GRID_ROWS", "evaluate_prediction_file", "nearest_grid"]
