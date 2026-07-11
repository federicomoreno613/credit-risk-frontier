"""Local LLM few-shot orchestration nodes."""

from __future__ import annotations

from .._utils import json_safe, load_script_module, read_json, resolve_repo_path


def run_llm_few_shot_suite(paths: dict, params: dict) -> dict:
    """Run or reuse the few-shot LLM metrics suite."""
    models_dir = resolve_repo_path(paths["models_dir"])
    combined_path = models_dir / "llm_metrics_fewshot.json"
    enabled = bool(params.get("run_inference", False))
    if combined_path.exists() and not enabled:
        return {"source": "existing_artifact", **read_json(combined_path)}
    if not enabled:
        return {
            "source": "skipped",
            "status": "skipped",
            "reason": "Few-shot local deshabilitado por parámetros; activar llm_few_shot.run_inference para recalcular.",
            "expected_artifact": str(combined_path),
        }

    module = load_script_module("08_llm_fewshot.py", "credit_risk_frontier_script_08_fewshot")
    results = {}
    for n_shots in params.get("shots", [8, 16, 32]):
        result = module.run_fewshot(
            dataset_path=str(resolve_repo_path(paths["dataset"])),
            n_shots=int(n_shots),
            output_dir=str(models_dir),
        )
        results[f"{n_shots}_shot"] = json_safe(result)
    return {"source": "recomputed", **results}
