"""Model comparison nodes."""

from __future__ import annotations

import pandas as pd

from .._utils import load_script_module, resolve_repo_path


def build_model_comparison(
    xgboost_metrics: dict,
    logreg_metrics: dict,
    segment_metrics: dict,
    paths: dict,
    params: dict,
) -> pd.DataFrame:
    """Build or load the final model-comparison table after upstream metrics exist."""
    results_dir = resolve_repo_path(paths["results_dir"])
    table_path = results_dir / "tabla_comparacion_final.csv"
    if table_path.exists() and not params.get("rebuild", False):
        return pd.read_csv(table_path)

    module = load_script_module("07_comparacion_final.py", "credit_risk_frontier_script_07_compare")
    table = module.compare_models(
        metrics_dir=str(resolve_repo_path(paths["models_dir"])),
        output_dir=str(results_dir),
    )
    return table
