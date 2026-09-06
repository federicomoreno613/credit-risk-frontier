"""Funciones de cohorte: puente, desenlace, partición y validación."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from credit_risk_frontier import utils


_BRIDGE_EXCLUDED = {"credito_id_anon", "target", "set"}


def _canonical_signature_value(value) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".12g")
    text = str(value).strip()
    try:
        return format(float(text), ".12g")
    except ValueError:
        return text


def _row_signatures(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    def signature(row) -> str:
        payload = [_canonical_signature_value(row[c]) for c in columns]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()

    return df[columns].apply(signature, axis=1)


def build_exact_credit_bridge(legacy_dataset: pd.DataFrame,
                              current_dataset: pd.DataFrame,
                              params: dict | None = None) -> pd.DataFrame:
    """Une IDs legacy/current solo por firmas de features únicas en ambos lados."""
    params = params or {}
    requested = params.get("signature_columns")
    common = [c for c in legacy_dataset.columns if c in current_dataset.columns]
    columns = requested or [c for c in common if c not in _BRIDGE_EXCLUDED]
    columns = [c for c in columns if c not in _BRIDGE_EXCLUDED]
    if not columns:
        raise ValueError("No hay columnas válidas para construir la firma del puente")

    legacy = legacy_dataset.copy()
    current = current_dataset.copy()
    legacy["_signature"] = _row_signatures(legacy, columns)
    current["_signature"] = _row_signatures(current, columns)
    legacy_counts = legacy["_signature"].value_counts()
    current_counts = current["_signature"].value_counts()
    unique = set(legacy_counts[legacy_counts.eq(1)].index) & set(
        current_counts[current_counts.eq(1)].index
    )
    bridge = (
        legacy.loc[legacy["_signature"].isin(unique), ["credito_id_anon", "_signature"]]
        .rename(columns={"credito_id_anon": "legacy_credito_id_anon"})
        .merge(
            current.loc[current["_signature"].isin(unique), ["credito_id_anon", "_signature"]],
            on="_signature",
            validate="one_to_one",
        )
        .rename(columns={"_signature": "signature"})
        .sort_values("credito_id_anon")
        .reset_index(drop=True)
    )
    bridge.attrs["signature_columns"] = columns
    return bridge


def _assign_temporal_splits(outcomes: pd.DataFrame, train_fraction: float,
                            validation_fraction: float) -> pd.Series:
    ordered = outcomes.sort_values(["fecha_desembolso", "credito_id_anon"], kind="stable")
    n = len(ordered)
    n_train = int(n * train_fraction)
    n_val = int(n * validation_fraction)
    labels = pd.Series("test", index=ordered.index, dtype=object)
    labels.iloc[:n_train] = "train"
    labels.iloc[n_train:n_train + n_val] = "val"
    return labels.reindex(outcomes.index)


def build_credit_outcomes(current_dataset: pd.DataFrame, payments: pd.DataFrame,
                          credit_bridge: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Construye el desenlace a horizonte fijo, con ``1 = mora``."""
    horizon = int(params.get("horizon_days", 150))
    default_dpd = int(params.get("default_dpd", 60))
    good_dpd = int(params.get("good_dpd", 30))
    cutoff_value = params.get("observation_cutoff")

    pay = payments.rename(columns={"credito_id_anon": "legacy_credito_id_anon"}).merge(
        credit_bridge[["legacy_credito_id_anon", "credito_id_anon"]],
        on="legacy_credito_id_anon",
        validate="many_to_one",
    )
    pay["fecha_t_pago"] = pd.to_datetime(pay["fecha_t_pago"], errors="coerce")
    pay["dias_retraso"] = pd.to_numeric(pay["dias_retraso"], errors="coerce").fillna(0).clip(lower=0)
    # La fecha canónica viene del dataset actual; el archivo de pagos conserva otra
    # copia histórica que generaría sufijos y no debe gobernar la elegibilidad.
    pay = pay.drop(columns=["fecha_desembolso"], errors="ignore")

    base = current_dataset[["credito_id_anon", "fecha_desembolso"]].drop_duplicates().copy()
    base["fecha_desembolso"] = pd.to_datetime(base["fecha_desembolso"], errors="coerce")
    if cutoff_value:
        cutoff = pd.Timestamp(cutoff_value)
    elif "fecha_pago" in pay.columns:
        cutoff = pd.to_datetime(pay["fecha_pago"], errors="coerce").max()
    else:
        raise ValueError("Falta observation_cutoff y no existe fecha_pago para inferirlo")
    if pd.isna(cutoff):
        raise ValueError("No se pudo determinar el corte de observación")

    base["horizon_end"] = base["fecha_desembolso"] + pd.to_timedelta(horizon, unit="D")
    base = base[base["horizon_end"].le(cutoff)]
    pay = pay.merge(base[["credito_id_anon", "fecha_desembolso", "horizon_end"]],
                    on="credito_id_anon", how="inner")
    pay = pay[
        pay["fecha_t_pago"].ge(pay["fecha_desembolso"]) &
        pay["fecha_t_pago"].le(pay["horizon_end"])
    ]
    available = (pay["horizon_end"] - pay["fecha_t_pago"]).dt.days.clip(
        lower=0, upper=horizon)
    pay["dpd_at_horizon"] = np.minimum(pay["dias_retraso"], available)
    max_dpd = pay.groupby("credito_id_anon")["dpd_at_horizon"].max()

    outcomes = base.merge(max_dpd.rename("max_dpd_horizon"), on="credito_id_anon", how="inner")
    outcomes = outcomes[
        outcomes["max_dpd_horizon"].le(good_dpd) |
        outcomes["max_dpd_horizon"].gt(default_dpd)
    ].copy()
    outcomes["target"] = outcomes["max_dpd_horizon"].gt(default_dpd).astype(int)
    outcomes["target_definition"] = f"default_dpd_gt_{default_dpd}_within_{horizon}d"
    outcomes["horizon_days"] = horizon
    outcomes["observation_cutoff"] = cutoff.strftime("%Y-%m-%d")
    outcomes["set"] = _assign_temporal_splits(
        outcomes,
        float(params.get("train_fraction", 0.8)),
        float(params.get("validation_fraction", 0.1)),
    )
    return outcomes.sort_values(["fecha_desembolso", "credito_id_anon"]).reset_index(drop=True)


def create_model_input(current_dataset: pd.DataFrame, outcomes: pd.DataFrame,
                       params: dict | None = None) -> pd.DataFrame:
    """Reemplaza target/set heredados y elimina toda variable prohibida."""
    del params
    forbidden = set(utils.LEAK_COLS + utils.SCORES_INTERNOS + utils.TEMPORAL_PROXY_COLS)
    base = current_dataset.drop(columns=["target", "set", *forbidden], errors="ignore")
    outcome_cols = ["credito_id_anon", "target", "set"]
    merged = base.merge(outcomes[outcome_cols], on="credito_id_anon", how="inner", validate="one_to_one")
    merged["fecha_desembolso"] = pd.to_datetime(merged["fecha_desembolso"], errors="coerce")
    if all(column in merged.columns for column in utils.TU_VARS):
        merged = utils.annotate_segments(merged)
    return merged.sort_values(
        ["fecha_desembolso", "credito_id_anon"], kind="stable").reset_index(drop=True)


def build_split_manifest(current_dataset: pd.DataFrame, credit_bridge: pd.DataFrame,
                         outcomes: pd.DataFrame, model_input: pd.DataFrame,
                         params: dict | None = None) -> dict:
    """Manifiesto congelable de la cohorte y su partición temporal."""
    params = params or {}
    stable = model_input.sort_values("credito_id_anon").copy()
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(stable, index=False).values.tobytes()
    ).hexdigest()
    forbidden = utils.LEAK_COLS + utils.SCORES_INTERNOS + utils.TEMPORAL_PROXY_COLS
    set_counts = model_input["set"].value_counts().to_dict()
    target_by_set = model_input.groupby("set")["target"].agg(["count", "mean"]).to_dict("index")
    segment_counts = model_input.get("segmento", pd.Series(dtype=str)).value_counts().to_dict()
    ids_by_set = {split: sorted(group["credito_id_anon"].astype(str).tolist())
                  for split, group in model_input.groupby("set")}
    split_sha256 = {split: hashlib.sha256("\n".join(ids).encode()).hexdigest()
                    for split, ids in ids_by_set.items()}
    return utils.json_safe({
        "contract": "target_60dpd_150d_temporal_70_15_15",
        "dataset_sha256": digest,
        "source_rows": len(current_dataset),
        "bridge_rows": len(credit_bridge),
        "bridge_coverage": len(credit_bridge) / len(current_dataset) if len(current_dataset) else 0,
        "outcome_rows": len(outcomes),
        "model_input_rows": len(model_input),
        "target_definition": outcomes["target_definition"].iloc[0] if len(outcomes) else None,
        "observation_cutoff": outcomes["observation_cutoff"].iloc[0] if len(outcomes) else None,
        "set_counts": set_counts,
        "target_by_set": target_by_set,
        "segment_counts": segment_counts,
        "n_columns": model_input.shape[1],
        "columns": model_input.columns.tolist(),
        "ids_by_set": ids_by_set,
        "ids_sha256_by_set": split_sha256,
        "forbidden_columns_present": [c for c in forbidden if c in model_input.columns],
        "signature_columns": credit_bridge.attrs.get("signature_columns", []),
        "parameters": params,
    })


def validate_credit_dataset(dataset: pd.DataFrame, params: dict) -> dict:
    """Valida el dataset curado y devuelve un reporte compacto de reproducibilidad.

    El segmento ``esparso`` cuenta cualquier código TU ``< 0`` y no solamente
    el centinela ``-1``.
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
