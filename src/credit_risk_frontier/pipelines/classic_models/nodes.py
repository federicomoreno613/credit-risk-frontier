"""Classic model orchestration nodes."""

from __future__ import annotations

from .._utils import json_safe, load_script_module, read_json, resolve_repo_path


def run_xgboost_baseline(paths: dict, params: dict) -> dict:
    """Return XGBoost metrics, retraining only when requested or needed."""
    models_dir = resolve_repo_path(paths["models_dir"])
    metrics_path = models_dir / "xgboost_metrics.json"
    if metrics_path.exists() and not params.get("retrain", False):
        return {"source": "existing_artifact", **read_json(metrics_path)}

    module = load_script_module("03_baseline_xgboost.py", "credit_risk_frontier_script_03_xgb")
    if "n_trials" in params:
        module.N_TRIALS = int(params["n_trials"])
    if "seed" in params:
        module.SEED = int(params["seed"])
    result = module.train_xgboost(
        dataset_path=str(resolve_repo_path(paths["dataset"])),
        output_dir=str(models_dir),
    )
    return {"source": "retrained", **json_safe(result)}


def run_logreg_baseline(paths: dict, params: dict) -> dict:
    """Return Logistic Regression metrics, retraining only when requested or needed."""
    models_dir = resolve_repo_path(paths["models_dir"])
    metrics_path = models_dir / "logreg_metrics.json"
    if metrics_path.exists() and not params.get("retrain", False):
        return {"source": "existing_artifact", **read_json(metrics_path)}

    module = load_script_module("04_baseline_logreg.py", "credit_risk_frontier_script_04_logreg")
    if "seed" in params:
        module.SEED = int(params["seed"])
    result = module.train_logreg(
        dataset_path=str(resolve_repo_path(paths["dataset"])),
        output_dir=str(models_dir),
    )
    return {"source": "retrained", **json_safe(result)}
