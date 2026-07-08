#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnóstico de UN SOLO caso en modo razonamiento.

Muestra TODO lo que entra y sale para un único crédito, sin escribir nada:
  1) qué fila del dataset se usa (id, target real, algunas variables)
  2) el PROMPT EXACTO que se envía a Ollama (system + user, serialización incluida)
  3) la RESPUESTA CRUDA completa de Ollama (content, thinking, logprobs, eval_count)
  4) qué número se parsea como score

Uso (en el Mac, con Ollama corriendo):
    python scripts/09b_diagnostico_un_caso.py                 # zero-shot, primer caso de test
    python scripts/09b_diagnostico_un_caso.py --id 6e605c516921
    python scripts/09b_diagnostico_un_caso.py --mode few --shots 16
"""
import argparse, json
from pathlib import Path

import importlib.util
BASE = Path(__file__).resolve().parent.parent

def _load(name, alias):
    spec = importlib.util.spec_from_file_location(alias, BASE / "scripts" / name)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

m09 = _load("09_llm_thinking.py", "m09")
m06b = m09.m06b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero", "few"], default="zero")
    ap.add_argument("--shots", type=int, default=16)
    ap.add_argument("--id", default=None, help="credito_id_anon; por defecto el 1º de test")
    ap.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"))
    args = ap.parse_args()

    print("=" * 70)
    print("DIAGNÓSTICO DE UN CASO — modo:", args.mode, "| shots:", args.shots if args.mode=="few" else "-")
    print("=" * 70)

    # 0) chequear Ollama
    m06b.check_ollama()

    df = m06b.load_segmented(args.dataset)
    test = df[df["set"] == "test"]
    if args.id:
        row = df[df["credito_id_anon"] == args.id].iloc[0]
    else:
        row = test.iloc[0]

    # 1) LA FILA
    print("\n### 1) FILA DEL DATASET ###")
    print("  credito_id_anon :", row["credito_id_anon"])
    print("  target real     :", int(row["target"]), "(1 = entró en mora, 0 = pagó)")
    print("  set             :", row["set"])
    for v in ["g051s", "wd81", "wd03", "antiguedad_cliente", "descripcion_negocio"]:
        if v in row.index:
            print(f"  {v:16}:", row[v])

    # 2) EL PROMPT EXACTO
    print("\n### 2) PROMPT ENVIADO A OLLAMA ###")
    if args.mode == "zero":
        messages = m09.build_zero_messages(row)
    else:
        train_df = df[df["set"] == "train"].copy()
        knn_space = m09._build_knn_space(train_df)
        ex = m09.knn_examples_for_case(row, row["credito_id_anon"], train_df, knn_space, args.shots)
        messages = m09.build_few_messages(ex, row)
    for msg in messages:
        print(f"\n--- [{msg['role'].upper()}] ---")
        print(msg["content"])

    # 3) RESPUESTA CRUDA (una llamada)
    print("\n### 3) RESPUESTA DE OLLAMA (cruda) ###")
    out = m09.call_ollama_think(messages)
    print("  content (texto visible):")
    print("   ", repr(out["content"])[:2000])
    print("\n  thinking (razonamiento interno):")
    print("   ", repr(out["thinking"])[:2000])
    print("\n  logprob_final :", out["logprob"], "  (NaN = tu versión de Ollama no expone logprobs)")
    print("  n_tokens      :", out["eval_count"])

    # 4) SCORE PARSEADO
    print("\n### 4) SCORE EXTRAÍDO ###")
    print("  prob_llm (score para AUC):", out["prob"])
    print("  → etiqueta (>=0.5):", 1 if (out["prob"] or 0) >= 0.5 else 0)
    print("  → target real     :", int(row["target"]))
    print("\nListo. Nada se guardó (solo diagnóstico).")

if __name__ == "__main__":
    main()
