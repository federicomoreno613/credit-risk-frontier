import json
from pathlib import Path

import pandas as pd

from credit_risk_frontier import utils
from credit_risk_frontier.cohorte import validate_credit_dataset

ROOT = Path(__file__).resolve().parents[2]


def test_buro_esparso_counts_match_thesis_dataset():
    # Universo contractual anonimizado (n=4897). Los centinelas TU negativos
    # representan ausencia de información, no solamente el valor -1.
    df = pd.read_csv(ROOT / "data" / "crudo" / "credit_applications_anonymized.csv")
    n_tu_missing = (df[utils.TU_VARS] < 0).sum(axis=1)
    segmento = n_tu_missing.ge(6).map({True: "esparso", False: "denso"})

    assert len(df) == 4897
    assert int(segmento.eq("esparso").sum()) == 3538
    assert int(segmento.eq("denso").sum()) == 1359
    assert int(segmento[df["set"].eq("test")].eq("esparso").sum()) == 204
    assert int(segmento[df["set"].eq("test")].eq("denso").sum()) == 291


def test_validate_credit_dataset_report_has_expected_contract():
    df = pd.read_csv(ROOT / "data" / "crudo" / "credit_applications_anonymized.csv")
    params = {
        "required_columns": ["credito_id_anon", "fecha_desembolso", "target", "set", "wd81"],
        "meta_cols": ["credito_id_anon", "fecha_desembolso", "target", "set"],
        "text_cols": ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"],
        "tu_vars": utils.TU_VARS,
        "segment": {"missing_code": -1, "cutoff": 6},
        "expected_counts": {
            "n_rows": 4897,
            "segment_esparso_total": 3538,
            "segment_denso_total": 1359,
            "test_esparso": 204,
            "test_denso": 291,
        },
        "fail_on_missing_required": True,
        "fail_on_expected_mismatch": True,
    }

    report = validate_credit_dataset(df, params)

    assert report["status"] == "ok"
    assert report["n_rows"] == 4897
    assert report["segment_counts"]["esparso"] == 3538
    assert report["test_segment_counts"]["esparso"] == 204
    assert report["expected_count_mismatches"] == []


def test_frozen_split_matches_plan_70_15_15():
    # Cohorte v2 (2026-09): legacy deduplicado -> puente completo 4897 y corte
    # temporal ajustado a bordes de fecha (ningún día repartido entre sets).
    manifiesto = json.loads(
        (ROOT / "data" / "pipeline" / "01_manifiesto_particion.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifiesto["set_counts"]["train"] == 3232
    assert manifiesto["set_counts"]["val"] == 699
    assert manifiesto["set_counts"]["test"] == 686
