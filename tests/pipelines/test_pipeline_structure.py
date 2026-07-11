"""Estructura del proyecto Kedro canónico y paridad de los modelos clásicos.

Verifica que ``find_pipelines()`` descubre los pipelines esperados, que el
``__default__`` es offline (sin ``tabllm`` ni nodos que requieran torch/servidor),
que cada variante namespaced de ``data_science`` tiene su ``model_options``, y que
las métricas clásicas ya regeneradas reproducen los valores de paridad E4.

Los pipelines ``tabllm`` y ``reporting`` se agregan en fases posteriores; los tests
que dependen de la ``comparison_table`` y de las métricas LLM se incorporan con
``reporting``. Aquí no se re-corre nada (rápido): se leen los artefactos ya escritos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pipelines():
    from kedro.framework.project import configure_project
    from credit_risk_frontier.pipeline_registry import register_pipelines

    configure_project("credit_risk_frontier")
    return register_pipelines()


# ---------------------------------------------------------------------------
# Estructura / registry canónico
# ---------------------------------------------------------------------------

def test_find_pipelines_discovers_expected_folders(pipelines):
    for name in ("data_processing", "data_science"):
        assert name in pipelines, f"find_pipelines debe descubrir {name}"
    assert "__default__" in pipelines


def test_default_is_offline(pipelines):
    """El __default__ no debe contener nodos que requieran torch/servidor."""
    default = pipelines["__default__"]
    namespaces = {n.namespace for n in default.nodes if n.namespace}
    assert "tabfm" not in namespaces, "TabFM (torch) no debe estar en __default__"
    assert not any(ns.startswith("arm_") for ns in namespaces), "los brazos de ablación no van en __default__"
    # Solo los clásicos citables + procesamiento.
    assert {"xgb", "logreg"}.issubset(namespaces)
    tags = set().union(*(n.tags for n in default.nodes))
    assert "requires_tabfm" not in tags


def test_data_science_variants_have_params(pipelines):
    """Cada variante namespaced de data_science tiene su model_options en params."""
    ds = pipelines["data_science"]
    namespaces = {n.namespace for n in ds.nodes if n.namespace}
    params = _load_params("parameters_data_science.yml")
    for ns in namespaces:
        assert ns in params and "model_options" in params[ns], f"falta model_options para {ns}"


def _load_params(fname: str) -> dict:
    import yaml
    return yaml.safe_load((ROOT / "conf" / "base" / fname).read_text())


# ---------------------------------------------------------------------------
# Paridad de los modelos clásicos regenerados (E4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metrics_file,target", [
    ("models/xgboost_metrics.json", 0.7516166960611403),   # XGB no-leak citable
    ("models/logreg_metrics.json", 0.6604611666340061),    # LogReg no-leak citable
])
def test_classic_noleak_metrics_match_thesis(metrics_file, target):
    path = ROOT / metrics_file
    if not path.exists():
        pytest.skip(f"{metrics_file} ausente — correr kedro run primero")
    auc = json.loads(path.read_text())["test"]["AUC"]
    assert auc == pytest.approx(target, abs=1e-9)


@pytest.mark.parametrize("fixture_file,target", [
    ("tests/fixtures/legacy_metrics/xgboost_metrics_leak.json", 0.7775981448820954),
    ("tests/fixtures/legacy_metrics/logreg_metrics_leak.json", 0.6910150891632373),
])
def test_leak_fixtures_preserved(fixture_file, target):
    """Los fixtures con-leakage (evidencia previa) quedan preservados para las variantes *_leak."""
    path = ROOT / fixture_file
    assert path.exists(), f"fixture con-leak ausente: {fixture_file}"
    auc = json.loads(path.read_text())["test"]["AUC"]
    assert auc == pytest.approx(target, abs=1e-9)
