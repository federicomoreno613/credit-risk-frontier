from pathlib import Path

import pandas as pd

from credit_risk_frontier.pipelines.data_preparation.nodes import validate_credit_dataset

ROOT = Path(__file__).resolve().parents[2]
TU_VARS = [
    "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
    "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
    "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
]


def test_buro_esparso_counts_match_thesis_dataset():
    df = pd.read_csv(ROOT / "data" / "dataset_tesis.csv")
    n_tu_missing = (df[TU_VARS] == -1).sum(axis=1)
    segmento = n_tu_missing.ge(6).map({True: "esparso", False: "denso"})

    assert int(segmento.eq("esparso").sum()) == 1911
    assert int(segmento.eq("denso").sum()) == 3440
    assert int(segmento[df["set"].eq("test")].eq("esparso").sum()) == 28
    assert int(segmento[df["set"].eq("test")].eq("denso").sum()) == 467


def test_validate_credit_dataset_report_has_expected_contract():
    df = pd.read_csv(ROOT / "data" / "dataset_tesis.csv")
    params = {
        "required_columns": ["credito_id_anon", "fecha_desembolso", "target", "set", "wd81"],
        "meta_cols": ["credito_id_anon", "fecha_desembolso", "target", "set"],
        "text_cols": ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"],
        "tu_vars": TU_VARS,
        "segment": {"missing_code": -1, "cutoff": 6},
        "expected_counts": {
            "n_rows": 5351,
            "segment_esparso_total": 1911,
            "segment_denso_total": 3440,
            "test_esparso": 28,
            "test_denso": 467,
        },
        "fail_on_missing_required": True,
        "fail_on_expected_mismatch": True,
    }

    report = validate_credit_dataset(df, params)

    assert report["status"] == "ok"
    assert report["n_rows"] == 5351
    assert report["segment_counts"]["esparso"] == 1911
    assert report["test_segment_counts"]["esparso"] == 28
    assert report["expected_count_mismatches"] == []
