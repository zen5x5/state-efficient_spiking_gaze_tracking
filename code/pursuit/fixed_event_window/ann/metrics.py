"""Coordinate transforms, training loss, and metrics for static ANNs."""

from __future__ import annotations

import torch


SCREEN_WIDTH_PX = 1920.0
SCREEN_HEIGHT_PX = 1080.0
SCREEN_CENTER_X_PX = SCREEN_WIDTH_PX / 2
SCREEN_CENTER_Y_PX = SCREEN_HEIGHT_PX / 2
SCREEN_HALF_DIAGONAL_PX = (
    SCREEN_CENTER_X_PX**2 + SCREEN_CENTER_Y_PX**2
) ** 0.5
SCREEN_DIAGONAL_INCH = 40.0
VIEW_DISTANCE_MM = 400.0
MM_PER_PX = SCREEN_DIAGONAL_INCH * 25.4 / (
    SCREEN_WIDTH_PX**2 + SCREEN_HEIGHT_PX**2
) ** 0.5


def normalize_row_col(labels_px: torch.Tensor) -> torch.Tensor:
    """Convert cached [row, col] pixels to full-screen normalized [x, y]."""
    x = (labels_px[..., 1] - SCREEN_CENTER_X_PX) / SCREEN_CENTER_X_PX
    y = (labels_px[..., 0] - SCREEN_CENTER_Y_PX) / SCREEN_CENTER_Y_PX
    return torch.stack((x, y), dim=-1)


def denormalize_xy(prediction_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized model [x, y] to screen [row, col] pixels."""
    col = prediction_norm[..., 0] * SCREEN_CENTER_X_PX + SCREEN_CENTER_X_PX
    row = prediction_norm[..., 1] * SCREEN_CENTER_Y_PX + SCREEN_CENTER_Y_PX
    return torch.stack((row, col), dim=-1)


def squared_pixel_loss(
    prediction_norm: torch.Tensor, target_norm: torch.Tensor
) -> torch.Tensor:
    """Per-sample squared pixel distance divided by half-diagonal squared."""
    delta = prediction_norm - target_norm
    dx_px = delta[..., 0] * SCREEN_CENTER_X_PX
    dy_px = delta[..., 1] * SCREEN_CENTER_Y_PX
    return (dx_px.square() + dy_px.square()) / SCREEN_HALF_DIAGONAL_PX**2


def pixel_error(
    prediction_norm: torch.Tensor, target_row_col_px: torch.Tensor
) -> torch.Tensor:
    prediction_px = denormalize_xy(prediction_norm)
    return torch.linalg.vector_norm(prediction_px - target_row_col_px, dim=-1)


def angular_error_deg(
    prediction_norm: torch.Tensor, target_row_col_px: torch.Tensor
) -> torch.Tensor:
    prediction_px = denormalize_xy(prediction_norm)

    def gaze_vectors(row_col_px: torch.Tensor) -> torch.Tensor:
        x_mm = (row_col_px[..., 1] - SCREEN_CENTER_X_PX) * MM_PER_PX
        y_mm = (row_col_px[..., 0] - SCREEN_CENTER_Y_PX) * MM_PER_PX
        z_mm = torch.full_like(x_mm, VIEW_DISTANCE_MM)
        return torch.stack((x_mm, y_mm, z_mm), dim=-1)

    prediction_vector = gaze_vectors(prediction_px)
    target_vector = gaze_vectors(target_row_col_px)
    cosine = (prediction_vector * target_vector).sum(dim=-1) / (
        prediction_vector.norm(dim=-1) * target_vector.norm(dim=-1)
    )
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
