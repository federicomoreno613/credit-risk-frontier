"""Shared helpers for the thesis Kedro pipelines."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repository-relative path without hiding absolute paths."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_script_module(filename: str, module_name: str) -> ModuleType:
    """Load one of the historical research scripts whose filename starts with digits."""
    script_path = PROJECT_ROOT / "scripts" / filename
    if not script_path.exists():
        raise FileNotFoundError(f"No existe el script esperado: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: str | Path) -> dict[str, Any]:
    with open(resolve_repo_path(path), encoding="utf-8") as f:
        return json.load(f)


def json_safe(value: Any) -> Any:
    """Convert common numpy/pandas scalars to plain JSON-compatible values."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # pragma: no cover - defensive fallback
        np = None
        pd = None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
    if pd is not None:
        if value is pd.NA:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    return value
