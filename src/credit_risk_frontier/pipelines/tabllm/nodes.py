"""Nodos del pipeline tabllm (LLM TabLLM: zero/few-shot, gemma, gpt, ablaciones).

Template de 4 nodos instanciado por namespace (una variante por instancia). La
restricción dura del proyecto — **los experimentos LLM no se re-corren jamás** —
se codifica en ``score_llm``: si el cache cubre todos los ids objetivo, el nodo
hace *pass-through* puro (no importa ni contacta ollama/openai). Solo con
``allow_infer: true`` y ids pendientes se puntúa (y solo los pendientes).

Los caches parquet son evidencia frozen (chmod -w, SHA256-manifested); se adoptan
como inputs del catálogo en sus rutas legacy. La lógica de merge vive en el nodo
(canónico: los joins van en nodos, no en clases de dataset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_frontier import utils


def select_target_rows(model_input_table: pd.DataFrame, cache: pd.DataFrame,
                       variant: dict) -> pd.DataFrame:
    """Filas objetivo a puntuar según ``scope`` (fuente: 09._subset_index / 08 scope).

    * ``scope: test`` → solo ``set == "test"`` (comparación honesta).
    * ``scope: all``  → todo el dataset (métrica general).
    * ``scope: cached`` → solo los ids ya presentes en el cache (para variantes cuyo
      experimento cubrió un subconjunto, p. ej. la ablación de fuentes con n=200).
      Nunca dispara inferencia: el objetivo es exactamente lo puntuado.

    ``limit_per_class`` recorta a los primeros N de CADA clase (el test está ordenado
    por target: un ``limit`` crudo daría una sola clase, sin AUC).
    """
    scope = variant.get("scope", "test")
    if scope == "all":
        target = model_input_table
    elif scope == "cached":
        cached_ids = set(cache["credito_id_anon"])
        target = model_input_table[model_input_table["credito_id_anon"].isin(cached_ids)]
    else:
        target = model_input_table[model_input_table["set"] == "test"]

    lpc = variant.get("limit_per_class", 0)
    if lpc and lpc > 0:
        tgt = target["target"]
        idx0 = target.index[(tgt == 0).values][:lpc]
        idx1 = target.index[(tgt == 1).values][:lpc]
        target = target.loc[idx0.append(idx1)]
    return target[["credito_id_anon", "target", "set"]].copy()


def score_llm(target_rows: pd.DataFrame, cache: pd.DataFrame, model_input_table: pd.DataFrame,
              variant: dict) -> pd.DataFrame:
    """Cache-first: puntúa SOLO ids pendientes; pass-through puro si el cache está completo.

    * Si no hay pendientes → devuelve el cache dedup (offline, sin red, determinista).
    * Si hay pendientes y ``allow_infer`` es falso (default) → error claro (evidencia
      frozen: no se re-corre sin autorización explícita).
    * Si hay pendientes y ``allow_infer`` es true → puntúa los pendientes vía el
      transporte del ``variant`` (import lazy dentro de la rama).
    """
    cache = cache.drop_duplicates("credito_id_anon", keep="last")
    target_ids = set(target_rows["credito_id_anon"])
    pending = target_ids - set(cache["credito_id_anon"])

    if not pending:
        return cache

    if not variant.get("allow_infer", False):
        raise RuntimeError(
            f"Cache incompleto para la variante '{variant.get('name', '?')}': "
            f"{len(pending)} ids objetivo sin puntuar. Los experimentos LLM son evidencia "
            f"frozen: correr con allow_infer=true (y el backend disponible) para completarlos."
        )

    scored = _infer_pending(pending, model_input_table, variant)
    return pd.concat([cache, scored], ignore_index=True).drop_duplicates(
        "credito_id_anon", keep="last")


def _infer_pending(pending_ids: set, model_input_table: pd.DataFrame, variant: dict) -> pd.DataFrame:
    """Puntúa los ids pendientes con el transporte del variant (solo si allow_infer)."""
    seg = utils.annotate_segments(model_input_table)
    df = seg[seg["credito_id_anon"].isin(pending_ids)]
    transport = variant.get("transport", "ollama")
    model = variant.get("model", utils.DEFAULT_OLLAMA_MODEL)
    profile = variant.get("features", "full")
    feats, inc_cat, inc_text = utils.features_for_profile(profile)
    think = variant.get("mode") == "think"

    rows = []
    for _, row in df.iterrows():
        if think:
            messages = utils.build_messages_thinking_zero(
                row, feats, include_categoricals=inc_cat, include_text=inc_text)
            out = utils.call_ollama_think(
                messages, model=model, think_native=variant.get("think_native", True))
            prob = out["prob"]
        else:
            messages = utils.build_messages_direct(
                row, feats, include_categoricals=inc_cat, include_text=inc_text)
            prob = utils.call_ollama(messages, model=model, think=False)
        rows.append({"credito_id_anon": row["credito_id_anon"], "prob_llm": prob})
    return pd.DataFrame(rows)


def canonicalize_predictions(scored: pd.DataFrame, model_input_table: pd.DataFrame) -> pd.DataFrame:
    """Une el cache crudo con la tabla segmentada → contrato de predicciones.

    Contrato: ``credito_id_anon, set, segmento, y_true, prob``. Desacopla toda figura
    y métrica del entrenamiento/inferencia.
    """
    seg = utils.annotate_segments(model_input_table)
    base = seg[["credito_id_anon", "set", "segmento", "target"]].rename(columns={"target": "y_true"})
    cache = scored.drop_duplicates("credito_id_anon", keep="last")[["credito_id_anon", "prob_llm"]]
    merged = base.merge(cache, on="credito_id_anon", how="left")
    merged = merged.rename(columns={"prob_llm": "prob"})
    merged["prob"] = pd.to_numeric(merged["prob"], errors="coerce")
    return merged[["credito_id_anon", "set", "segmento", "y_true", "prob"]]


def evaluate_predictions(predictions: pd.DataFrame, variant: dict) -> dict:
    """Métricas en test por segmento (+ general si scope=all). Fuente: utils.evaluate_on_test.

    Reconstruye el df mínimo que espera ``evaluate_on_test`` desde el contrato de
    predicciones y devuelve la misma estructura que los JSON committeados.
    """
    df = predictions.rename(columns={"y_true": "target"})[
        ["credito_id_anon", "set", "segmento", "target"]].copy()
    probs = predictions["prob"].values

    results = {
        "model": variant.get("model"),
        "mode": variant.get("mode"),
        "scope": variant.get("scope", "test"),
        "test": utils.evaluate_on_test(df, probs),
    }
    if variant.get("n_shots"):
        results["n_shots"] = variant["n_shots"]
    if variant.get("selection"):
        results["selection"] = variant["selection"]
    if variant.get("scope") == "all":
        results["general"] = utils.evaluate_general(df, probs)
        results["n_general"] = int((~np.isnan(np.asarray(probs, dtype=float))).sum())
    return utils.json_safe(results)
