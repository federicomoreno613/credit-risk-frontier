"""Report artifact validation nodes."""

from __future__ import annotations

import pandas as pd

from .._utils import PROJECT_ROOT


def validate_thesis_artifacts(
    comparison_table: pd.DataFrame,
    dataset_report: dict,
    eda_report: dict,
    params: dict,
) -> dict:
    """Validate that thesis-facing docs, figures and comparison outputs are present."""
    required_files = params.get("required_files", [])
    checks = {}
    for rel in required_files:
        path = PROJECT_ROOT / rel
        checks[rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    missing = [rel for rel, check in checks.items() if not check["exists"]]
    if missing and params.get("fail_on_missing_files", True):
        raise FileNotFoundError(f"Faltan artefactos de tesis esperados: {missing}")

    return {
        "status": "ok" if not missing else "missing_files",
        "missing_files": missing,
        "required_file_checks": checks,
        "comparison_rows": int(len(comparison_table)),
        "comparison_models": sorted(comparison_table["model"].dropna().unique().tolist()) if "model" in comparison_table else [],
        "dataset_rows": dataset_report.get("n_rows"),
        "dataset_segments": dataset_report.get("segment_counts", {}),
        "eda_figures_checked": len(eda_report.get("figure_checks", {})),
    }
