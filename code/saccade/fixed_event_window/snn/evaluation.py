"""Evaluation and physical-sample prediction export for the saccade SNN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..ann.metrics import angular_error_deg, denormalize_xy, pixel_error
from .dataloader import repeat_full_event_frame
from .model import (
    GRID_COL_START_PX,
    GRID_COL_STEP_PX,
    GRID_INDEX_MAX,
    GRID_INDEX_MIN,
    GRID_ROW_START_PX,
    GRID_ROW_STEP_PX,
    GridCountSpikingSaccadeSNN,
    decode_grid_spike_counts,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    timesteps: int = 2,
    max_batches: int | None = None,
    prediction_path: Path | None = None,
) -> dict:
    model.eval()
    sample_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions_norm: list[np.ndarray] = []
    predictions_px: list[np.ndarray] = []
    pixel_errors: list[np.ndarray] = []
    angular_errors: list[np.ndarray] = []
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        event_bins = repeat_full_event_frame(
            batch["event_frame"].to(device), timesteps
        )
        target_px = batch["target_row_col_px"].to(device)
        prediction = model(event_bins)
        sample_indices.append(batch["sample_index"].numpy())
        target_indices.append(batch["target_index"].numpy())
        labels.append(target_px.cpu().numpy())
        predictions_norm.append(prediction.cpu().numpy())
        predictions_px.append(denormalize_xy(prediction).cpu().numpy())
        pixel_errors.append(pixel_error(prediction, target_px).cpu().numpy())
        angular_errors.append(angular_error_deg(prediction, target_px).cpu().numpy())
        batches += 1
    if not pixel_errors:
        raise ValueError("evaluation loader produced no samples")
    arrays = {
        "sample_index": np.concatenate(sample_indices).astype(np.int64),
        "target_index": np.concatenate(target_indices).astype(np.int64),
        "target_row_col_px": np.concatenate(labels).astype(np.float32),
        "prediction_xy_norm": np.concatenate(predictions_norm).astype(np.float32),
        "prediction_row_col_px": np.concatenate(predictions_px).astype(np.float32),
        "pixel_error": np.concatenate(pixel_errors).astype(np.float32),
        "angular_error_deg": np.concatenate(angular_errors).astype(np.float32),
    }
    per_target = {}
    for target in np.unique(arrays["target_index"]):
        mask = arrays["target_index"] == target
        per_target[str(int(target))] = {
            "samples": int(mask.sum()),
            "mean_pixel_error": float(arrays["pixel_error"][mask].mean()),
            "mean_angular_error_deg": float(
                arrays["angular_error_deg"][mask].mean()
            ),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, **arrays)
    return {
        "samples": int(len(arrays["pixel_error"])),
        "batches": batches,
        "targets": len(per_target),
        "mean_pixel_error": float(arrays["pixel_error"].mean()),
        "mean_angular_error_deg": float(arrays["angular_error_deg"].mean()),
        "target_macro_mean_pixel_error": float(
            np.mean([value["mean_pixel_error"] for value in per_target.values()])
        ),
        "target_macro_mean_angular_error_deg": float(
            np.mean(
                [
                    value["mean_angular_error_deg"]
                    for value in per_target.values()
                ]
            )
        ),
        "per_target": per_target,
    }


@torch.no_grad()
def evaluate_grid_count(
    model: GridCountSpikingSaccadeSNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    timesteps: int = 2,
    max_batches: int | None = None,
    prediction_path: Path | None = None,
) -> dict:
    """Evaluate the two-neuron grid-count head after valid-level clipping."""
    model.eval()
    sample_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    raw_counts: list[np.ndarray] = []
    grid_indices: list[np.ndarray] = []
    target_grid_indices: list[np.ndarray] = []
    predictions_norm: list[np.ndarray] = []
    predictions_px: list[np.ndarray] = []
    pixel_errors: list[np.ndarray] = []
    angular_errors: list[np.ndarray] = []
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        event_bins = repeat_full_event_frame(
            batch["event_frame"].to(device), timesteps
        )
        target_px = batch["target_row_col_px"].to(device)
        count_xy = model.output_spike_counts(event_bins)
        if not torch.equal(count_xy, count_xy.round()):
            raise AssertionError("MultiSpike output count is not integer-valued")
        clipped_xy = count_xy.clamp(GRID_INDEX_MIN, GRID_INDEX_MAX)
        prediction = decode_grid_spike_counts(
            clipped_xy, clamp_to_grid=False
        )
        target_col_index = torch.round(
            (target_px[:, 1] - GRID_COL_START_PX) / GRID_COL_STEP_PX
        )
        target_row_index = torch.round(
            (target_px[:, 0] - GRID_ROW_START_PX) / GRID_ROW_STEP_PX
        )
        target_grid_xy = torch.stack(
            (target_col_index, target_row_index), dim=-1
        )
        reconstructed_col = (
            GRID_COL_START_PX + GRID_COL_STEP_PX * target_grid_xy[:, 0]
        )
        reconstructed_row = (
            GRID_ROW_START_PX + GRID_ROW_STEP_PX * target_grid_xy[:, 1]
        )
        if not torch.equal(reconstructed_col, target_px[:, 1]) or not torch.equal(
            reconstructed_row, target_px[:, 0]
        ):
            raise AssertionError("target is not on the declared 11x11 grid")

        sample_indices.append(batch["sample_index"].numpy())
        target_indices.append(batch["target_index"].numpy())
        labels.append(target_px.cpu().numpy())
        raw_counts.append(count_xy.cpu().numpy())
        grid_indices.append(clipped_xy.cpu().numpy())
        target_grid_indices.append(target_grid_xy.cpu().numpy())
        predictions_norm.append(prediction.cpu().numpy())
        predictions_px.append(denormalize_xy(prediction).cpu().numpy())
        pixel_errors.append(pixel_error(prediction, target_px).cpu().numpy())
        angular_errors.append(angular_error_deg(prediction, target_px).cpu().numpy())
        batches += 1
    if not pixel_errors:
        raise ValueError("evaluation loader produced no samples")

    arrays = {
        "sample_index": np.concatenate(sample_indices).astype(np.int64),
        "target_index": np.concatenate(target_indices).astype(np.int64),
        "target_row_col_px": np.concatenate(labels).astype(np.float32),
        "raw_output_spike_count_xy": np.concatenate(raw_counts).astype(np.float32),
        "prediction_grid_index_xy": np.concatenate(grid_indices).astype(np.int64),
        "target_grid_index_xy": np.concatenate(target_grid_indices).astype(np.int64),
        "prediction_xy_norm": np.concatenate(predictions_norm).astype(np.float32),
        "prediction_row_col_px": np.concatenate(predictions_px).astype(np.float32),
        "pixel_error": np.concatenate(pixel_errors).astype(np.float32),
        "angular_error_deg": np.concatenate(angular_errors).astype(np.float32),
    }
    raw = arrays["raw_output_spike_count_xy"]
    predicted_grid = arrays["prediction_grid_index_xy"]
    target_grid = arrays["target_grid_index_xy"]
    out_of_range = (raw < GRID_INDEX_MIN) | (raw > GRID_INDEX_MAX)
    joint_correct = np.all(predicted_grid == target_grid, axis=1)
    per_target = {}
    for target in np.unique(arrays["target_index"]):
        mask = arrays["target_index"] == target
        per_target[str(int(target))] = {
            "samples": int(mask.sum()),
            "mean_pixel_error": float(arrays["pixel_error"][mask].mean()),
            "mean_angular_error_deg": float(
                arrays["angular_error_deg"][mask].mean()
            ),
            "joint_grid_accuracy": float(joint_correct[mask].mean()),
        }
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, **arrays)
    return {
        "samples": int(len(arrays["pixel_error"])),
        "batches": batches,
        "targets": len(per_target),
        "mean_pixel_error": float(arrays["pixel_error"].mean()),
        "mean_angular_error_deg": float(arrays["angular_error_deg"].mean()),
        "target_macro_mean_pixel_error": float(
            np.mean([value["mean_pixel_error"] for value in per_target.values()])
        ),
        "target_macro_mean_angular_error_deg": float(
            np.mean(
                [value["mean_angular_error_deg"] for value in per_target.values()]
            )
        ),
        "joint_grid_accuracy": float(joint_correct.mean()),
        "target_macro_joint_grid_accuracy": float(
            np.mean([value["joint_grid_accuracy"] for value in per_target.values()])
        ),
        "raw_output_spike_count": {
            "x_col": {
                "minimum": float(raw[:, 0].min()),
                "mean": float(raw[:, 0].mean()),
                "maximum": float(raw[:, 0].max()),
                "out_of_range_samples": int(out_of_range[:, 0].sum()),
            },
            "y_row": {
                "minimum": float(raw[:, 1].min()),
                "mean": float(raw[:, 1].mean()),
                "maximum": float(raw[:, 1].max()),
                "out_of_range_samples": int(out_of_range[:, 1].sum()),
            },
            "any_axis_out_of_range_samples": int(out_of_range.any(axis=1).sum()),
        },
        "per_target": per_target,
    }
