#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1 post-hoc por segmento desde un parquet de probabilidades LLM (esquema compartido).

No toca credit_metrics de 06b (lo usan otros scripts): calcula F1 aparte leyendo el parquet
(`credito_id_anon`, `prob_llm`) de cualquier corrida (GPT, Qwen, Gemma) más el dataset
segmentado. Reporta por segmento (total/esparso/denso), SOLO sobre test:
  - f1_05   : F1 con umbral 0.5 — el valor honesto por defecto.
  - f1_opt  : F1 máximo con umbral elegido EN test (optimista, solo referencia) + thr_opt.
  - auc     : recomputado del parquet, para cross-check contra el JSON de la corrida.
  - n, n_nan.

Uso:
    python3 scripts/09d_f1_posthoc.py --probs models/gpt/gpt_probs_zero_test.parquet \
        --out models/gpt/gpt_f1_zero_test.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

BASE = Path(__file__).resolve().parent.parent


def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, BASE / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m06b = _load("06b_llm_prompting.py", "m06b")

MIN_CASOS = 20  # mismo criterio que MIN_SEG_CASOS de 06b


def _f1_block(y: np.ndarray, p: np.ndarray) -> dict:
    f1_05 = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.nanargmax(f1s))
    # el último punto de la curva (recall=0) no tiene umbral asociado
    thr_opt = float(thr[best]) if best < len(thr) else 1.0
    return {"f1_05": f1_05, "f1_opt": float(f1s[best]), "thr_opt": thr_opt}


def main():
    ap = argparse.ArgumentParser(description="F1 post-hoc por segmento (test) desde parquet LLM")
    ap.add_argument("--probs", required=True, help="parquet con credito_id_anon y prob_llm")
    ap.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"))
    ap.add_argument("--out", required=True, help="path del JSON de salida")
    args = ap.parse_args()

    df = m06b.load_segmented(args.dataset)
    cache = pd.read_parquet(args.probs).drop_duplicates("credito_id_anon", keep="last")
    merged = df.merge(cache[["credito_id_anon", "prob_llm"]], on="credito_id_anon", how="left")
    test = merged[merged["set"] == "test"].copy()
    test["prob_llm"] = pd.to_numeric(test["prob_llm"], errors="coerce")

    res = {
        "probs": str(args.probs),
        "nota": ("f1_opt elige el umbral EN test (optimista, solo referencia); "
                 "f1_05 es el valor honesto por defecto."),
        "segmentos": {},
    }
    for seg in ("total", "esparso", "denso"):
        sub_all = test if seg == "total" else test[test["segmento"] == seg]
        sub = sub_all[~sub_all["prob_llm"].isna()]
        block = {"n": int(len(sub)), "n_nan": int(sub_all["prob_llm"].isna().sum())}
        if len(sub) >= MIN_CASOS and sub["target"].nunique() == 2:
            y = sub["target"].values.astype(int)
            p = sub["prob_llm"].values.astype(float)
            block.update(_f1_block(y, p))
            block["auc"] = float(roc_auc_score(y, p))
        else:
            block["note"] = "insuficiente para métricas confiables"
        res["segmentos"][seg] = block

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
