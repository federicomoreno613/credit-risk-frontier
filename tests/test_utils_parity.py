"""Paridad de ``utils.py`` contra los scripts autoritativos y la evidencia E4.

Gate de la Fase 2: antes de reemplazar los scripts por nodos Kedro, se prueba que
el módulo compartido produce EXACTAMENTE lo mismo que los scripts congelados:

* ``serialize_tabllm`` byte-idéntico vs ``06b`` en filas reales × 4 perfiles de ablación.
* ``credit_metrics``/``evaluate_on_test`` recomputan los JSON de métricas committeados
  desde los caches parquet reales (prueba que la evaluación es fiel).

Los tests que dependen de artefactos no versionados (caches parquet git-ignored)
se saltan con ``skipif`` para que un clone público sin caches igual pase su subset.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_frontier import utils

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
MODELS = REPO / "models"
DATASET = REPO / "data" / "dataset_tesis.csv"


def _load_script(fname: str, alias: str):
    """Carga un script hermano (nombre con dígitos) como módulo aislado."""
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m06b():
    return _load_script("06b_llm_prompting.py", "m06b_parity")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATASET, parse_dates=["fecha_desembolso"])


@pytest.fixture(scope="module")
def sample_rows(df):
    """~12 filas variadas: mezcla de segmentos y clases, incluye centinelas y NaN de texto."""
    return df.sample(n=12, random_state=7).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Serialización TabLLM byte-idéntica vs script 06b (perfil full == defaults)
# ---------------------------------------------------------------------------

def test_serialize_full_byte_identical(m06b, sample_rows):
    """Con defaults (categóricas+texto), utils reproduce 06b.serialize_tabllm exacto."""
    feats = utils.TOP_FEATURES_LLM
    assert feats == m06b.TOP_FEATURES_LLM, "TOP_FEATURES_LLM debe coincidir con el script"
    for _, row in sample_rows.iterrows():
        esperado = m06b.serialize_tabllm(row, feats)
        obtenido = utils.serialize_tabllm(row, feats)
        assert obtenido == esperado


@pytest.mark.parametrize("profile", ["full", "tu", "tu-sem", "sin-tu"])
def test_serialize_profiles_match_ablation(m06b, sample_rows, profile):
    """utils.features_for_profile + serialize reproduce la ablación --features de 09.

    El script 09 muta ``m06b.TOP_FEATURES_LLM`` y los globals INCLUDE_*; acá se
    replica ese efecto sobre el módulo cargado y se compara byte a byte.
    """
    feats, inc_cat, inc_text = utils.features_for_profile(profile)

    # Reproducir el efecto de 09 sobre el módulo-script para tener el oráculo.
    tu_feats = [f for f in m06b.TOP_FEATURES_LLM if f in m06b.TU_DESC]
    neg_feats = [f for f in m06b.TOP_FEATURES_LLM if f not in m06b.TU_DESC]
    m06b.INCLUDE_CATEGORICALS = True
    m06b.INCLUDE_TEXT = True
    if profile == "tu":
        script_feats = tu_feats
        m06b.INCLUDE_CATEGORICALS = False
        m06b.INCLUDE_TEXT = False
    elif profile == "tu-sem":
        script_feats = tu_feats
    elif profile == "sin-tu":
        script_feats = neg_feats
    else:
        script_feats = list(m06b.TOP_FEATURES_LLM)

    assert feats == script_feats

    for _, row in sample_rows.iterrows():
        esperado = m06b.serialize_tabllm(row, script_feats)
        obtenido = utils.serialize_tabllm(
            row, feats, include_categoricals=inc_cat, include_text=inc_text)
        assert obtenido == esperado

    # Limpiar los globals que dejó el oráculo para no contaminar otros tests.
    m06b.INCLUDE_CATEGORICALS = True
    m06b.INCLUDE_TEXT = True


# ---------------------------------------------------------------------------
# 2. credit_metrics idéntico a la versión del script (misma fórmula)
# ---------------------------------------------------------------------------

def test_credit_metrics_matches_script(m06b):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    p = rng.random(400)
    a = utils.credit_metrics(y, p)
    b = m06b.credit_metrics(y, p)
    for k in ("AUC", "Gini", "KS", "Brier"):
        assert a[k] == pytest.approx(b[k], abs=1e-12)
    # 06b usa clave PR_AUC; utils también.
    assert a["PR_AUC"] == pytest.approx(b["PR_AUC"], abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Recompute-from-cache: reproducir JSON committeados desde caches reales
# ---------------------------------------------------------------------------

def _recompute_from_cache(df, cache_path: Path) -> dict:
    """Segmenta, une el cache por id y evalúa en test — como hará el nodo evaluate."""
    seg = utils.annotate_segments(df)
    cache = pd.read_parquet(cache_path).drop_duplicates("credito_id_anon", keep="last")
    merged = seg[["credito_id_anon"]].merge(cache, on="credito_id_anon", how="left")
    return utils.evaluate_on_test(seg, merged["prob_llm"].values)


def _assert_metrics_close(got: dict, expected: dict, abs_tol=1e-12):
    """Compara el bloque test.total (AUC/Gini/KS/Brier/PR_AUC/n) tolerando la grafía de PR."""
    exp = expected["test"]["total"]
    for k in ("AUC", "Gini", "KS", "Brier"):
        assert got["total"][k] == pytest.approx(exp[k], abs=abs_tol), k
    got_pr = got["total"]["PR_AUC"]
    exp_pr = exp.get("PR_AUC", exp.get("PR-AUC"))
    assert got_pr == pytest.approx(exp_pr, abs=abs_tol), "PR_AUC"
    assert got["total"]["n"] == exp["n"]


CACHE_CASES = [
    ("llm_probs_tabllm_nothink.parquet", "llm_metrics_tabllm_nothink.json"),
    ("llm_probs_fewshot_16.parquet", "llm_metrics_fewshot_16.json"),
    ("llm_probs_thinking_zero_test.parquet", "llm_metrics_thinking_zero_test.json"),
]


@pytest.mark.parametrize("cache_name,metrics_name", CACHE_CASES)
def test_recompute_from_cache_matches_committed_json(df, cache_name, metrics_name):
    cache = MODELS / cache_name
    metrics = MODELS / metrics_name
    if not cache.exists() or not metrics.exists():
        pytest.skip(f"cache/metrics ausente ({cache_name}) — clone sin evidencia frozen")
    got = _recompute_from_cache(df, cache)
    expected = json.loads(metrics.read_text())
    _assert_metrics_close(got, expected)
