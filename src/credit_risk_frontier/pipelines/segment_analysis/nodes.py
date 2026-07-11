"""Segment analysis nodes for clientes con buró esparso."""

from __future__ import annotations

from .._utils import json_safe, load_script_module, read_json, resolve_repo_path


def run_segment_analysis(paths: dict, params: dict, _xgboost_metrics: dict | None = None) -> dict:
    """Return temporal CV metrics by bureau-density segment.

    `_xgboost_metrics` is intentionally unused: it is a Kedro dependency edge so
    segment CV recomputes after a retrained XGBoost writes fresh parameters.
    """
    models_dir = resolve_repo_path(paths["models_dir"])
    metrics_path = models_dir / "cv_seg_metrics.json"
    if metrics_path.exists() and not params.get("recompute", False):
        return {"source": "existing_artifact", **read_json(metrics_path)}

    module = load_script_module("05_segmentacion_thin_file.py", "credit_risk_frontier_script_05_segment")
    if "seed" in params:
        module.SEED = int(params["seed"])
    if "n_splits" in params:
        module.N_SPLITS = int(params["n_splits"])
    if "segment_cutoff" in params:
        module.CORTE_ESPARSO = int(params["segment_cutoff"])
    result = module.segment_cv(
        dataset_path=str(resolve_repo_path(paths["dataset"])),
        output_dir=str(models_dir),
    )
    return {"source": "recomputed", **json_safe(result)}
