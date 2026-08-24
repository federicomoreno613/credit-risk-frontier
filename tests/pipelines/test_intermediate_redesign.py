"""Contrato del rediseño solicitado para la entrega intermedia."""

import pandas as pd
import pytest
import json
import yaml

from credit_risk_frontier import utils
from credit_risk_frontier.pipelines.intermediate_delivery.nodes import (
    build_intermediate_feature_contract,
)


def _real_row():
    frame = pd.read_parquet("data/05_model_input/model_input_table.parquet")
    return frame.iloc[0], frame


def _configured_contract():
    params = yaml.safe_load(open("conf/base/parameters_intermediate_delivery.yml"))
    return params["intermediate_delivery"]


def test_intermediate_feature_profiles_are_exact_and_predecision():
    _, frame = _real_row()
    configured = _configured_contract()["feature_contract"]
    tu = utils.intermediate_feature_columns(frame, "tu", configured)
    tu_form = utils.intermediate_feature_columns(frame, "tu_form", configured)

    assert tu == utils.TU_VARS
    assert tu_form == utils.TU_VARS + utils.FORM_DIRECT_VARS
    assert len(tu) == 20
    assert len(tu_form) == 29
    assert not set(tu_form) & set(
        utils.TEXT
        + utils.LEAK_COLS
        + utils.SCORES_INTERNOS
        + utils.TEMPORAL_PROXY_COLS
        + ["estimated_income", "free_cash_flow", "cost_ingress_ratio"]
    )


def test_intermediate_serialization_adds_only_the_free_description():
    row, frame = _real_row()
    features = utils.intermediate_feature_columns(frame, "tu_form")
    structured = utils.serialize_intermediate_profile(row, features, "tu_form")
    with_description = utils.serialize_intermediate_profile(
        row, features, "tu_form_description"
    )

    assert with_description.startswith(structured)
    assert "descripción del negocio:" in with_description
    assert "subcategoría:" not in with_description
    assert "rubro declarado:" not in with_description
    assert "tipo de crédito:" not in with_description
    assert "categoría negocio:" not in with_description
    assert "objetivo crédito:" not in with_description
    assert "alianza:" not in with_description
    assert "género:" not in with_description


def test_intermediate_prompt_states_target_and_horizon():
    row, frame = _real_row()
    features = utils.intermediate_feature_columns(frame, "tu_form")
    messages = utils.build_messages_intermediate(
        row, features, "tu_form_description", prompt_variant="minimum"
    )
    prompt = messages[-1]["content"]

    assert "mora mayor de 60 días" in prompt
    assert "150 días" in prompt
    assert "PROBABILIDAD_DE_MORA" in prompt
    assert "target" not in prompt
    assert "credito_id_anon" not in prompt


def test_intermediate_serialization_rejects_a_different_feature_universe():
    row, frame = _real_row()
    features = utils.intermediate_feature_columns(frame, "tu_form")
    with pytest.raises(ValueError, match="exactamente"):
        utils.serialize_intermediate_profile(row, features[:-1], "tu_form")


def test_few_shot_distance_uses_the_same_29_variables():
    _, frame = _real_row()
    train = frame[frame["set"].eq("train")].reset_index(drop=True)
    features = utils.intermediate_feature_columns(frame, "tu_form")
    _, _, _, used = utils.build_knn_space(train, features)
    assert used == features


def test_kedro_materializes_the_exact_20_plus_9_contract():
    _, frame = _real_row()
    contract = build_intermediate_feature_contract(frame, _configured_contract())

    assert contract["transunion"] == utils.TU_VARS
    assert contract["form_direct"] == utils.FORM_DIRECT_VARS
    assert contract["structured"] == utils.TU_VARS + utils.FORM_DIRECT_VARS
    assert contract["transunion_count"] == 20
    assert contract["form_direct_count"] == 9
    assert contract["structured_count"] == 29
    assert contract["free_text"] == "descripcion_negocio"
    assert contract["profiles"]["logreg_tu_form"] == (
        utils.TU_VARS + utils.FORM_DIRECT_VARS
    )
    assert contract["profiles"]["xgb_tu_form"] == (
        utils.TU_VARS + utils.FORM_DIRECT_VARS
    )
    assert contract["profiles"]["qwen_tu_form"] == utils.TU_VARS + utils.FORM_DIRECT_VARS


def test_persisted_classic_results_follow_the_new_contract():
    metrics_path = "data/08_reporting/intermedia_20260714_redesign/classic_metrics.json"
    predictions_path = "data/07_model_output/intermediate_20260714/classic_predictions.parquet"
    shap_path = "data/08_reporting/intermedia_20260714_redesign/xgboost_shap_resumen.csv"
    metrics = json.loads(open(metrics_path).read())
    predictions = pd.read_parquet(predictions_path)
    shap_summary = pd.read_csv(shap_path)

    expected = utils.TU_VARS + utils.FORM_DIRECT_VARS
    assert set(metrics) == {"logreg_tu_form", "xgb_tu_form"}
    assert metrics["logreg_tu_form"]["features"] == expected
    assert metrics["xgb_tu_form"]["features"] == expected
    assert metrics["logreg_tu_form"]["feature_count"] == 29
    assert metrics["xgb_tu_form"]["feature_count"] == 29
    assert metrics["logreg_tu_form"]["selected_parameters"]["C"] in {0.01, 0.1, 1.0, 10.0}
    assert metrics["xgb_tu_form"]["selected_parameters"]["best_iteration"] >= 0
    assert shap_summary["codigo"].tolist() != []
    assert set(shap_summary["codigo"]) == set(expected)
    assert len(shap_summary) == 29
    assert (shap_summary["shap_medio_absoluto"] >= 0).all()
    assert len(predictions) == 2 * 421
    assert not predictions.duplicated([
        "evaluation_id", "model", "feature_profile", "mode", "split"
    ]).any()
