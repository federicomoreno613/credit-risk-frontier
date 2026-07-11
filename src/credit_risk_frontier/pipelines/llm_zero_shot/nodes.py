"""Local LLM zero-shot orchestration nodes."""

from __future__ import annotations

from .._utils import json_safe, load_script_module, read_json, resolve_repo_path


def _load_or_skip(path, enabled: bool, label: str) -> dict | None:
    if path.exists() and not enabled:
        return {"source": "existing_artifact", **read_json(path)}
    if not enabled:
        return {
            "source": "skipped",
            "status": "skipped",
            "reason": "LLM local deshabilitado por parámetros; activar llm_zero_shot.run_inference para recalcular.",
            "expected_artifact": str(path),
            "label": label,
        }
    return None


def run_llm_zero_shot_mode(paths: dict, params: dict, mode: str) -> dict:
    """Run or reuse a single TabLLM zero-shot mode (`nothink` or `think`)."""
    models_dir = resolve_repo_path(paths["models_dir"])
    metrics_path = models_dir / f"llm_metrics_tabllm_{mode}.json"
    enabled = bool(params.get("run_inference", False))
    existing = _load_or_skip(metrics_path, enabled, f"zero_shot_{mode}")
    if existing is not None:
        return existing

    module = load_script_module("06b_llm_prompting.py", f"credit_risk_frontier_script_06b_{mode}")
    if params.get("ollama_model"):
        module.OLLAMA_MODEL = params["ollama_model"]
    cache_dir = resolve_repo_path(params.get("cache_dir", paths.get("llm_cache_dir", "data/07_model_output")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = module.run_llm_inference(
        dataset_path=str(resolve_repo_path(paths["dataset"])),
        cache_path=str(cache_dir / f"llm_probs_tabllm_{mode}.parquet"),
        mode=mode,
        output_dir=str(models_dir),
    )
    return {"source": "recomputed", **json_safe(result)}


def run_llm_zero_shot_nothink(paths: dict, params: dict) -> dict:
    return run_llm_zero_shot_mode(paths, params, "nothink")


def run_llm_zero_shot_think(paths: dict, params: dict) -> dict:
    return run_llm_zero_shot_mode(paths, params, "think")
