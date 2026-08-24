"""Ejecución recuperable de Qwen para el experimento intermedio.

La lógica recibe tablas, parámetros y funciones de persistencia explícitas. El
ejecutable de ``scripts/`` se limita a cargar y guardar datasets por sus nombres
del Data Catalog.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk_frontier import utils


CACHE_DATASETS = {
    ("tu_form", 0): "qwen_intermediate_tu_form_zero_cache",
    ("tu_form", 8): "qwen_intermediate_tu_form_few8_cache",
    ("tu_form_description", 0): "qwen_intermediate_tu_form_description_zero_cache",
    ("tu_form_description", 8): "qwen_intermediate_tu_form_description_few8_cache",
}
SUMMARY_DATASET = "qwen_intermediate_run_summary"


def experiment_configurations(params: dict) -> list[tuple[str, int]]:
    """Devuelve y valida el orden contractual de las cuatro configuraciones."""
    configurations = [
        (str(profile), int(shots))
        for profile in params["profiles"]
        for shots in params["shots"]
    ]
    if configurations != list(CACHE_DATASETS):
        raise ValueError(
            "Qwen debe ejecutar exactamente tu_form y tu_form_description, cada "
            "uno primero sin ejemplos y luego con ocho ejemplos"
        )
    return configurations


def fingerprint(
    dataset_sha: str,
    profile: str,
    shots: int,
    features: list[str],
    params: dict,
) -> str:
    """Identifica de forma estable el conjunto, prompt y parámetros de inferencia."""
    contract = {
        "dataset_sha256": dataset_sha,
        "model": params["model"],
        "feature_profile": profile,
        "features": features,
        "split": params["split"],
        "prompt": params["prompt_variant"],
        "shots": shots,
        "example_selection": params["example_selection"],
        "example_serialization": params["example_serialization"],
        "target": params["target"],
        "instruction": params["instruction_version"],
        "think_native": bool(params["think_native"]),
        "temperature": float(params["temperature"]),
        "num_ctx": int(params["num_ctx"]),
        "num_predict": int(params["num_predict"]),
        "retry_num_predict": int(params["retry_num_predict"]),
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()


def summarize_cache_status(
    frame: pd.DataFrame,
    caches: dict[tuple[str, int], pd.DataFrame],
    features: dict[str, list[str]] | list[str],
    dataset_sha: str,
    params: dict,
) -> dict:
    """Resume avance y compatibilidad sin ejecutar una sola inferencia."""
    target = frame[frame["set"].eq(params["split"])]
    target_ids = set(target["credito_id_anon"].astype(str))
    configurations = []
    total_rows = 0
    total_expected = len(target) * len(CACHE_DATASETS)

    for profile, shots in experiment_configurations(params):
        profile_features = (
            features[profile] if isinstance(features, dict) else features
        )
        expected_fingerprint = fingerprint(
            dataset_sha, profile, shots, profile_features, params
        )
        cache = caches.get((profile, shots), pd.DataFrame()).copy()
        if cache.empty:
            rows = valid = invalid = 0
            fingerprint_ok = True
        else:
            cache = cache[
                cache["evaluation_id"].astype(str).isin(target_ids)
            ].drop_duplicates("evaluation_id", keep="last")
            rows = len(cache)
            valid = int(cache["valid"].fillna(False).sum())
            invalid = rows - valid
            observed = set(cache["fingerprint"].dropna().astype(str))
            fingerprint_ok = observed in (set(), {expected_fingerprint})
        total_rows += rows
        configurations.append({
            "profile": profile,
            "shots": shots,
            "dataset": CACHE_DATASETS[(profile, shots)],
            "rows": rows,
            "expected_rows": len(target),
            "valid": valid,
            "invalid_pending_retry": invalid,
            "first_pass_remaining": max(len(target) - rows, 0),
            "complete_first_pass": rows == len(target),
            "fingerprint_ok": fingerprint_ok,
            "fingerprint": expected_fingerprint,
        })

    return {
        "test_rows_per_configuration": len(target),
        "configuration_count": len(configurations),
        "rows_present": total_rows,
        "rows_expected": total_expected,
        "first_pass_completion": (
            total_rows / total_expected if total_expected else 0.0
        ),
        "configurations": configurations,
    }


def run_configuration(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    dataset_sha: str,
    profile: str,
    shots: int,
    params: dict,
    existing: pd.DataFrame | None = None,
    save_cache: Callable[[pd.DataFrame], None] | None = None,
    limit: int | None = None,
) -> dict:
    """Ejecuta una configuración y persiste avances mediante ``save_cache``."""
    expected_fingerprint = fingerprint(
        dataset_sha, profile, shots, features, params
    )
    target = frame[frame["set"].eq(params["split"])].copy()
    if limit is not None:
        target = target.head(limit)

    previous = existing.copy() if existing is not None else pd.DataFrame()
    if not previous.empty:
        observed = set(previous["fingerprint"].dropna().astype(str))
        if observed and observed != {expected_fingerprint}:
            raise RuntimeError(
                "El cache pertenece a otro contrato; no se puede mezclar"
            )
        allowed_ids = set(target["credito_id_anon"].astype(str))
        previous = previous[
            previous["evaluation_id"].astype(str).isin(allowed_ids)
        ].drop_duplicates("evaluation_id", keep="last")

    rows = previous.to_dict("records")
    done = set(previous.get("evaluation_id", pd.Series(dtype=str)).astype(str))
    pending = list(
        target[~target["credito_id_anon"].astype(str).isin(done)].iterrows()
    )
    cache_was_complete = len(done) == len(target)
    knn = utils.build_knn_space(train, features) if shots else None

    def infer(item, budget: int) -> dict:
        _, row = item
        examples = (
            utils.knn_examples_for_case(
                row, str(row["credito_id_anon"]), train, knn, shots
            )
            if shots
            else None
        )
        messages = utils.build_messages_intermediate(
            row,
            features,
            profile,
            examples=examples,
            prompt_variant=params["prompt_variant"],
        )
        result = utils.call_ollama_think(
            messages,
            model=params["model"],
            think_native=bool(params["think_native"]),
            timeout=int(params["timeout_seconds"]),
            retries=int(params["request_retries"]),
            num_predict=budget,
            num_ctx=int(params["num_ctx"]),
            temperature=float(params["temperature"]),
        )
        probability = result.get("prob")
        return {
            "evaluation_id": str(row["credito_id_anon"]),
            "model": params["model"],
            "feature_profile": profile,
            "mode": "few" if shots else "zero",
            "split": params["split"],
            "segment": str(row.get("segmento", "")),
            "y_true": int(row["target"]),
            "probability": probability,
            "valid": bool(pd.notna(probability)),
            "n_shots": shots,
            "prompt": params["prompt_variant"],
            "eval_count": result.get("eval_count"),
            "fingerprint": expected_fingerprint,
        }

    parallel = int(os.environ.get(
        "INTERMEDIATE_QWEN_PARALLEL", params["parallel_workers"]
    ))
    initial_budget = int(params["num_predict"])
    for start in range(0, len(pending), parallel):
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [
                pool.submit(infer, item, initial_budget)
                for item in pending[start:start + parallel]
            ]
            for future in as_completed(futures):
                rows.append(future.result())
        current = pd.DataFrame(rows)
        if save_cache is not None:
            save_cache(current)
        print(f"{profile}/few{shots}: {len(rows)}/{len(target)}", flush=True)

    result = pd.DataFrame(rows)
    invalid_ids = (
        set()
        if cache_was_complete or result.empty
        else set(result.loc[~result["valid"], "evaluation_id"].astype(str))
    )
    if invalid_ids:
        retry_items = list(target[
            target["credito_id_anon"].astype(str).isin(invalid_ids)
        ].iterrows())
        replacements = {}
        retry_budget = int(params["retry_num_predict"])
        for start in range(0, len(retry_items), parallel):
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [
                    pool.submit(infer, item, retry_budget)
                    for item in retry_items[start:start + parallel]
                ]
                for future in as_completed(futures):
                    retry = future.result()
                    replacements[retry["evaluation_id"]] = retry
        rows = [
            replacements.get(str(row["evaluation_id"]), row) for row in rows
        ]
        result = pd.DataFrame(rows)
        if save_cache is not None:
            save_cache(result)

    valid = result[result["probability"].notna()]
    auc = (
        float(roc_auc_score(valid["y_true"], valid["probability"]))
        if not valid.empty and valid["y_true"].nunique() == 2
        else None
    )
    return {
        "model": params["model"],
        "feature_profile": profile,
        "shots": shots,
        "n": len(result),
        "valid": len(valid),
        "invalid": len(result) - len(valid),
        "auc": auc,
        "dataset": CACHE_DATASETS[(profile, shots)],
        "fingerprint": expected_fingerprint,
    }
