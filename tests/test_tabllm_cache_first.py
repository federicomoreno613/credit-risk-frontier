"""Semántica cache-first del pipeline tabllm + paridad recompute-from-cache.

Verifica que:
* ``score_llm`` hace pass-through puro sin tocar la red cuando el cache está completo,
  y levanta error (no infiere) cuando faltan ids y ``allow_infer`` es falso.
* ``canonicalize_predictions`` + ``evaluate_predictions`` reproducen los JSON de
  métricas LLM committeados @1e-12 (que la evaluación es fiel a la evidencia).
* la estructura namespaced es coherente (cada variante tiene cache + variant params).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_frontier import utils
from credit_risk_frontier.pipelines.tabllm import nodes

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def mit():
    df = pd.read_csv(ROOT / "data" / "dataset_tesis.csv", parse_dates=["fecha_desembolso"])
    return utils.annotate_segments(df)


# ---------------------------------------------------------------------------
# Cache-first: pass-through sin red / error sin allow_infer
# ---------------------------------------------------------------------------

def test_score_llm_passthrough_no_network(monkeypatch, mit):
    """Cache completo → pass-through: nunca importa requests ni contacta la red."""
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no debe tocar la red"))

    cache = pd.read_parquet(MODELS / "llm_probs_tabllm_nothink.parquet")
    target = mit[mit["set"] == "test"][["credito_id_anon", "target", "set"]]
    variant = {"name": "qwen_nothink", "scope": "test", "allow_infer": False}
    out = nodes.score_llm(target, cache, mit, variant)
    assert set(target["credito_id_anon"]).issubset(set(out["credito_id_anon"]))


def test_score_llm_raises_when_incomplete_and_frozen(mit):
    """Cache incompleto + allow_infer=false → RuntimeError (evidencia frozen)."""
    tiny_cache = pd.DataFrame({"credito_id_anon": [], "prob_llm": []})
    target = mit[mit["set"] == "test"][["credito_id_anon", "target", "set"]].head(10)
    variant = {"name": "x", "scope": "test", "allow_infer": False}
    with pytest.raises(RuntimeError, match="frozen"):
        nodes.score_llm(target, tiny_cache, mit, variant)


# ---------------------------------------------------------------------------
# Paridad recompute-from-cache vs JSON committeados
# ---------------------------------------------------------------------------

CASES = [
    ("qwen_nothink", "llm_probs_tabllm_nothink.parquet", "llm_metrics_tabllm_nothink.json", "test"),
    ("qwen_few16", "llm_probs_fewshot_16.parquet", "llm_metrics_fewshot_16.json", "test"),
    ("qwen_think_zero", "llm_probs_thinking_zero_test.parquet", "llm_metrics_thinking_zero_test.json", "test"),
    ("gpt_zero", "gpt/gpt_probs_zero_test.parquet", "gpt/gpt_metrics_zero_test.json", "test"),
]


@pytest.mark.parametrize("name,cache_rel,metrics_rel,scope", CASES)
def test_recompute_matches_committed(mit, name, cache_rel, metrics_rel, scope):
    cache_path = MODELS / cache_rel
    metrics_path = MODELS / metrics_rel
    if not cache_path.exists() or not metrics_path.exists():
        pytest.skip(f"cache/metrics ausente para {name}")

    cache = pd.read_parquet(cache_path)
    variant = {"name": name, "scope": scope, "allow_infer": False}
    preds = nodes.canonicalize_predictions(cache, mit)
    got = nodes.evaluate_predictions(preds, variant)

    expected = json.loads(metrics_path.read_text())["test"]["total"]
    got_total = got["test"]["total"]
    for k in ("AUC", "Gini", "KS", "Brier"):
        assert got_total[k] == pytest.approx(expected[k], abs=1e-12), f"{name}:{k}"
    got_pr = got_total["PR_AUC"]
    exp_pr = expected.get("PR_AUC", expected.get("PR-AUC"))
    assert got_pr == pytest.approx(exp_pr, abs=1e-12), f"{name}:PR_AUC"
    assert got_total["n"] == expected["n"], f"{name}:n"


# ---------------------------------------------------------------------------
# Estructura namespaced
# ---------------------------------------------------------------------------

def test_every_variant_has_cache_and_params():
    import yaml
    params = yaml.safe_load((ROOT / "conf" / "base" / "parameters_tabllm.yml").read_text())
    catalog = yaml.safe_load((ROOT / "conf" / "base" / "catalog.yml").read_text())
    from credit_risk_frontier.pipelines.tabllm import VARIANTS

    for v in VARIANTS:
        assert v in params and "variant" in params[v], f"falta variant params para {v}"
        assert f"{v}.cache" in catalog, f"falta {v}.cache en el catálogo"
