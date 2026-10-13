"""Portable filesystem defaults for EBV-Eye commands.

Every command still accepts explicit paths.  These environment-backed defaults
only remove assumptions about the login node or scheduler host.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("EBV_EYE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
DATA_ROOT = Path(
    os.environ.get("EBV_EYE_DATA_ROOT", "/data/xuzx/data/ebv-eye")
).resolve()
RAW_ROOT = Path(
    os.environ.get("EBV_EYE_RAW_ROOT", DATA_ROOT / "eye_data")
).resolve()
CACHE_ROOT = Path(
    os.environ.get("EBV_EYE_CACHE_ROOT", DATA_ROOT / "cache" / "formal_eye0")
).resolve()
RESULT_ROOT = Path(
    os.environ.get("EBV_EYE_RESULT_ROOT", "/data/xuzx/result/ebv-eye")
).resolve()


def raw_eye_dir(user: int, eye: int = 0) -> Path:
    return RAW_ROOT / f"user{user}" / str(eye)


def pursuit_cache_root() -> Path:
    return CACHE_ROOT / "pursuit" / "fixed_event_window_400_stride200_96x128"


def saccade_cache_root() -> Path:
    return CACHE_ROOT / "saccade" / "fixed_event_window_400_stride200_96x128"


def saccade_last500ms_cache_root() -> Path:
    return CACHE_ROOT / "saccade" / "fixed_last500ms_event_window_400_stride200_96x128"


def project_relative_path(path: Path) -> str:
    """Return a repository-relative provenance path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
