"""Estructura Kedro mínima y vigente de la entrega intermedia."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pipelines():
    from kedro.framework.project import configure_project
    from credit_risk_frontier.pipeline_registry import register_pipelines

    configure_project("credit_risk_frontier")
    return register_pipelines()


def test_registry_exposes_current_delivery_pipelines(pipelines):
    assert set(pipelines) == {
        "__default__",
        "data_processing",
        "intermediate_classics",
        "intermediate_evaluate",
        "intermediate_reporting",
        "intermediate_delivery",
    }
    composed = (
        pipelines["intermediate_classics"]
        + pipelines["intermediate_evaluate"]
        + pipelines["intermediate_reporting"]
    )
    assert {node.name for node in pipelines["intermediate_delivery"].nodes} == {
        node.name for node in composed.nodes
    }


def test_default_runs_only_the_two_classic_arms(pipelines):
    default = pipelines["__default__"]
    assert [node.name for node in default.nodes] == [
        "run_intermediate_classic_experiments_node"
    ]
    assert set(default.outputs()) == {
        "intermediate_classic_predictions",
        "intermediate_classic_metrics",
        "intermediate_logreg_coefficients",
        "intermediate_xgb_shap_summary",
    }


def test_full_delivery_validates_contract_and_consumes_four_qwen_caches(pipelines):
    delivery = pipelines["intermediate_delivery"]
    names = {node.name for node in delivery.nodes}
    assert "build_intermediate_feature_contract_node" in names
    assert "combine_intermediate_predictions_node" in names
    assert "build_intermediate_metrics_table_node" in names
    assert "plot_intermediate_xgb_shap_node" in names
    qwen_inputs = {
        name for name in delivery.inputs()
        if name.startswith("qwen_intermediate_")
    }
    assert qwen_inputs == {
        "qwen_intermediate_tu_form_zero_cache",
        "qwen_intermediate_tu_form_few8_cache",
        "qwen_intermediate_tu_form_description_zero_cache",
        "qwen_intermediate_tu_form_description_few8_cache",
    }
