"""Nodos de procesamiento de datos.

Análogo canónico del ``data_processing`` de spaceflights: valida el dataset curado,
construye la ``model_input_table`` (única segmentación autoritativa) y arma un
reporte EDA liviano. Los helpers de un solo dueño viven acá; los cross-pipeline
(segmentación, constantes) vienen de ``credit_risk_frontier.utils``.
"""

from __future__ import annotations

import pandas as pd

from credit_risk_frontier import utils


def validate_credit_dataset(dataset: pd.DataFrame, params: dict) -> dict:
    """Valida el dataset curado y devuelve un reporte compacto de reproducibilidad.

    Preservado del scaffold WIP (nodo puro: df + params → dict, sin I/O). E4: el
    segmento ``esparso`` cuenta cualquier código TU ``< 0`` (no solo ``-1``).
    """
    df = dataset.copy()
    required = params.get("required_columns", [])
    meta_cols = params.get("meta_cols", [])
    text_cols = params.get("text_cols", [])
    tu_vars = params.get("tu_vars", [])
    target_col = params.get("target_col", "target")
    set_col = params.get("set_col", "set")
    date_col = params.get("date_col", "fecha_desembolso")
    segment = params.get("segment", {})
    expected = params.get("expected_counts", {})

    missing_required = [c for c in required if c not in df.columns]
    if missing_required and params.get("fail_on_missing_required", True):
        raise ValueError(f"Faltan columnas obligatorias: {missing_required}")

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        min_date = None if dates.isna().all() else str(dates.min().date())
        max_date = None if dates.isna().all() else str(dates.max().date())
    else:
        min_date = max_date = None

    feature_cols = [c for c in df.columns if c not in set(meta_cols + text_cols)]
    report = {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features_modelables": int(len(feature_cols)),
        "required_columns_missing": missing_required,
        "set_counts": {str(k): int(v) for k, v in df.get(set_col, pd.Series(dtype=str)).value_counts().to_dict().items()},
        "target_rate": None if target_col not in df.columns else float(df[target_col].mean()),
        "date_min": min_date,
        "date_max": max_date,
    }

    if tu_vars and all(c in df.columns for c in tu_vars):
        cutoff = segment.get("cutoff", 6)
        n_tu_missing = (df[tu_vars] < 0).sum(axis=1)
        segmento = n_tu_missing.ge(cutoff).map({True: "esparso", False: "denso"})
        report["segment_rule"] = f"({len(tu_vars)} TU vars < 0).sum(axis=1) >= {cutoff}"
        report["segment_counts"] = {str(k): int(v) for k, v in segmento.value_counts().to_dict().items()}
        if set_col in df.columns:
            test_mask = df[set_col].eq("test")
            report["test_segment_counts"] = {
                str(k): int(v) for k, v in segmento[test_mask].value_counts().to_dict().items()
            }
    else:
        report["segment_rule"] = "not_evaluated_missing_tu_columns"

    mismatches = []
    checks = {
        "n_rows": report.get("n_rows"),
        "segment_esparso_total": report.get("segment_counts", {}).get("esparso"),
        "segment_denso_total": report.get("segment_counts", {}).get("denso"),
        "test_esparso": report.get("test_segment_counts", {}).get("esparso"),
        "test_denso": report.get("test_segment_counts", {}).get("denso"),
    }
    for key, expected_value in expected.items():
        observed = checks.get(key)
        if observed != expected_value:
            mismatches.append({"metric": key, "expected": expected_value, "observed": observed})
    report["expected_count_mismatches"] = mismatches
    if mismatches and params.get("fail_on_expected_mismatch", True):
        raise ValueError(f"No coinciden conteos esperados del dataset: {mismatches}")
    return report


def create_model_input_table(dataset: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Construye la tabla de entrada del modelo — ÚNICA segmentación autoritativa.

    Parsea ``fecha_desembolso``, ordena temporalmente y agrega ``n_tu_missing`` y
    ``segmento`` (via ``utils.annotate_segments``). Todos los pipelines consumen
    esta tabla; nadie vuelve a segmentar. Consolida los dos ``load_segmented``
    incompatibles de los scripts 05 y 06b en una sola fuente.
    """
    df = dataset.copy()
    date_col = params.get("date_col", "fecha_desembolso")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    return utils.annotate_segments(df)


def build_eda_report(dataset: pd.DataFrame, params: dict) -> dict:
    """Reporte EDA compacto + verificación de disponibilidad de figuras de tesis.

    Preservado del scaffold WIP; usa ``utils.PROJECT_ROOT`` en vez de ``_utils``.
    """
    df = dataset.copy()
    target_col = params.get("target_col", "target")
    set_col = params.get("set_col", "set")
    date_col = params.get("date_col", "fecha_desembolso")
    figure_files = params.get("figure_files", [])

    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_distribution": {},
        "set_target_rate": {},
        "top_missing_columns": {},
        "figure_checks": {},
    }
    if target_col in df.columns:
        report["target_distribution"] = {
            str(k): int(v) for k, v in df[target_col].value_counts(dropna=False).sort_index().to_dict().items()
        }
    if target_col in df.columns and set_col in df.columns:
        report["set_target_rate"] = {
            str(k): float(v) for k, v in df.groupby(set_col)[target_col].mean().to_dict().items()
        }
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        report["monthly_rows"] = {
            str(k): int(v) for k, v in dates.dt.to_period("M").value_counts().sort_index().to_dict().items()
        }
    missing = df.isna().mean().sort_values(ascending=False).head(15)
    report["top_missing_columns"] = {str(k): float(v) for k, v in missing.to_dict().items()}

    for rel in figure_files:
        path = utils.PROJECT_ROOT / rel
        report["figure_checks"][rel] = {
            "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    missing_figures = [rel for rel, check in report["figure_checks"].items() if not check["exists"]]
    if missing_figures and params.get("fail_on_missing_figures", True):
        raise FileNotFoundError(f"Faltan figuras EDA esperadas: {missing_figures}")
    return report
