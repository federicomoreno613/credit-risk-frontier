"""El alcance intermedio no puede incorporar otros modelos ni resultados antiguos."""

import pandas as pd
import pytest
import yaml

from credit_risk_frontier import utils
from credit_risk_frontier.pipelines.intermediate_delivery.nodes import (
    ALLOWED,
    TU_CODES,
    combine_intermediate_predictions,
    build_intermediate_results,
    build_real_serialization_example,
    build_transunion_dictionary,
)


def _prediction_rows(model, profile, mode):
    return pd.DataFrame({
        "evaluation_id": ["a", "b", "c", "d"],
        "model": model,
        "feature_profile": profile,
        "mode": mode,
        "split": "test",
        "segment": ["denso", "denso", "esparso", "esparso"],
        "y_true": [0, 1, 0, 1],
        "probability": [.1, .8, .2, .7],
    })


def test_combination_requires_the_same_test_credits_in_every_configuration():
    classic = pd.concat([
        _prediction_rows("logreg_tu_form", "tu_form", "trained"),
        _prediction_rows("xgb_tu_form", "tu_form", "trained"),
    ], ignore_index=True)
    qwen = [
        _prediction_rows("qwen3:8b", "tu_form", "zero"),
        _prediction_rows("qwen3:8b", "tu_form", "few"),
        _prediction_rows("qwen3:8b", "tu_form_description", "zero"),
        _prediction_rows("qwen3:8b", "tu_form_description", "few"),
    ]
    assert len(combine_intermediate_predictions(classic, *qwen)) == 24
    with pytest.raises(ValueError, match="mismos créditos"):
        combine_intermediate_predictions(classic, qwen[0].iloc[:-1], *qwen[1:])


def test_intermediate_results_contains_exactly_six_authorized_configurations():
    rows = []
    for index, (model, profile, mode) in enumerate(sorted(ALLOWED)):
        rows.append({
            "model": model, "feature_profile": profile, "mode": mode,
            "segment": "total", "n": 421, "n_total": 421,
            "n_valid": 421, "n_invalid": 0, "AUC": .6 + index / 100,
            "AUC_ci_low": .5, "AUC_ci_high": .7, "Gini": .2, "KS": .3,
            "Brier": .2, "PR_AUC": .4, "ECE": .1,
        })
    rows.append({
        "model": "gpt", "feature_profile": "tu_form", "mode": "zero",
        "segment": "total", "n": 421, "n_total": 421,
        "n_valid": 421, "n_invalid": 0, "AUC": .99, "AUC_ci_low": .9,
        "AUC_ci_high": 1, "Gini": .98, "KS": .9, "Brier": .1,
        "PR_AUC": .9, "ECE": .1,
    })
    source = pd.DataFrame(rows)
    result = build_intermediate_results(source)
    assert len(result) == 6
    assert set(result["modelo_presentado"]) == {"XGBoost", "Regresión logística", "Qwen3-8B"}
    assert set(result["informacion_presentada"]) == {
        "TransUnion más 9 variables directas del formulario",
        "TransUnion más 9 variables directas y descripción del negocio",
    }
    assert not result["model"].str.contains("gpt|gemma", case=False, regex=True).any()


def test_transunion_dictionary_contains_full_official_descriptions():
    source = pd.read_csv("data/diccionario_tu_oficial.csv")
    result = build_transunion_dictionary(source)
    assert result["codigo"].tolist() == TU_CODES
    assert result["definicion_oficial_CreditVision"].notna().all()
    utlmag04 = result.set_index("codigo").loc["utlmag04", "definicion_oficial_CreditVision"]
    assert utlmag04 == (
        "Magnitud de utilización de obligaciones retail en los últimos 24 meses "
        "(índice que mide la tendencia de 0 a 600)"
    )


def test_real_serialization_example_comes_from_the_actual_training_row():
    source = pd.read_parquet("data/05_model_input/model_input_table.parquet")
    delivery = yaml.safe_load(open(
        "conf/base/parameters_intermediate_delivery.yml"
    ))["intermediate_delivery"]
    result = build_real_serialization_example(
        source,
        delivery,
    )

    assert result["record_is_real_and_anonymized"] is True
    assert result["source_split"] == "train"
    assert result["target_excluded_from_report_and_prompt"] is True
    assert result["structured_feature_count"] == 29
    assert result["structured_features_in_order"] == utils.TU_VARS + utils.FORM_DIRECT_VARS
    assert result["free_text_field"] == "descripcion_negocio"
    assert result["free_text_value"] == "Comidas preparadas"
    assert result["qwen_structured_description_serialization"].endswith(
        "descripción del negocio: Comidas preparadas"
    )
    assert "PROBABILIDAD_DE_MORA" in result["qwen_user_message"]
    assert "150 días" in result["qwen_user_message"]
    assert "subcategoría:" not in result["qwen_user_message"]
    assert "rubro declarado:" not in result["qwen_user_message"]
    assert "target" not in result["qwen_user_message"]
    assert "credito_id_anon" not in result["qwen_user_message"]
