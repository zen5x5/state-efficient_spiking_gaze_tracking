"""Evaluation and per-sample prediction export for the saccade ANN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import angular_error_deg, denormalize_xy, pixel_error


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
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
        event_frame = batch["event_frame"].to(device)
        target_px = batch["target_row_col_px"].to(device)
        prediction = model(event_frame)
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
                [value["mean_angular_error_deg"] for value in per_target.values()]
            )
        ),
        "per_target": per_target,
    }
