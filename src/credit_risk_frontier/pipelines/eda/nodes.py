"""Lightweight EDA summary nodes for thesis reproducibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .._utils import PROJECT_ROOT


def build_eda_report(dataset: pd.DataFrame, params: dict) -> dict:
    """Create a compact EDA report and verify thesis figure availability."""
    df = dataset.copy()
    target_col = params.get("target_col", "target")
    set_col = params.get("set_col", "set")
    date_col = params.get("date_col", "fecha_desembolso")
    figure_files = params.get("figure_files", [])

    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_distribution": {},
        "set_target_rate": {},
        "top_missing_columns": {},
        "figure_checks": {},
    }
    if target_col in df.columns:
        report["target_distribution"] = {
            str(k): int(v) for k, v in df[target_col].value_counts(dropna=False).sort_index().to_dict().items()
        }
    if target_col in df.columns and set_col in df.columns:
        report["set_target_rate"] = {
            str(k): float(v) for k, v in df.groupby(set_col)[target_col].mean().to_dict().items()
        }
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        report["monthly_rows"] = {
            str(k): int(v) for k, v in dates.dt.to_period("M").value_counts().sort_index().to_dict().items()
        }
    missing = df.isna().mean().sort_values(ascending=False).head(15)
    report["top_missing_columns"] = {str(k): float(v) for k, v in missing.to_dict().items()}

    for rel in figure_files:
        path = PROJECT_ROOT / rel
        report["figure_checks"][rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    missing_figures = [rel for rel, check in report["figure_checks"].items() if not check["exists"]]
    if missing_figures and params.get("fail_on_missing_figures", True):
        raise FileNotFoundError(f"Faltan figuras EDA esperadas: {missing_figures}")
    return report
