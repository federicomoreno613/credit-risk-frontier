"""Data validation nodes for the credit-risk thesis dataset."""

from __future__ import annotations

import pandas as pd


def validate_credit_dataset(dataset: pd.DataFrame, params: dict) -> dict:
    """Validate the curated thesis dataset and return a compact reproducibility report."""
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
        # E4: se considera "esparso" cualquier variable TU con código negativo (< 0),
        # no solo el -1 exacto. Otros códigos negativos también indican ausencia de
        # historial en el buró. Esto lleva el segmento esparso de ~38% a ~72%.
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
