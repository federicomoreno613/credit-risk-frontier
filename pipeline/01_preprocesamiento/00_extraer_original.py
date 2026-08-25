"""Paso 00 (opcional) — Extraer del dataset original SOLO nuestra cohorte.

El dataset original-original (~114 MB, 1.146 columnas, con PII cruda) no se
puede leer entero ni versionar. Este script lo recorre por chunks, calcula el
``credito_id_anon`` de cada fila con la MISMA función de anonimización de la
tesis (sha256 con salt, primeros 12 hex) y conserva únicamente las filas cuyo
id pertenece a nuestra cohorte (data/pipeline/02_variables.parquet).

Salida (SOLO local, ignorada por git — contiene PII):
  data/00_original/original_cohorte.parquet

Uso:
  poetry run python pipeline/01_preprocesamiento/00_extraer_original.py \
      [--origen "/ruta/al/csv"] [--columnas col1,col2]  (default: todas)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

ORIGEN_DEFAULT = (
    "/Users/federicomoreno/Downloads/results_with_clip_clusters_transunion (1).csv"
)
SALIDA = C.ROOT / "data" / "00_original" / "original_cohorte.parquet"
# Misma sal y función que new_tesis/scripts_to_notebook/01_preparar_dataset.py
ANON_SALT = b"tesis_uba_credit_risk_anon_2025"


def anon_hash(valor) -> str:
    return hashlib.sha256(ANON_SALT + str(valor).strip().encode("utf-8")).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", default=ORIGEN_DEFAULT)
    parser.add_argument("--columnas", default=None,
                        help="lista separada por comas; default: todas las 1.146")
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args()

    ids_cohorte = set(C.cargar_variables()["credito_id_anon"].astype(str))
    print(f"cohorte objetivo: {len(ids_cohorte)} créditos")

    usecols = None
    if args.columnas:
        usecols = ["credits_credit_id"] + [
            c.strip() for c in args.columnas.split(",") if c.strip()
        ]

    partes = []
    leidas = 0
    for chunk in pd.read_csv(args.origen, chunksize=args.chunk,
                             usecols=usecols, low_memory=False):
        leidas += len(chunk)
        chunk["credito_id_anon"] = chunk["credits_credit_id"].map(anon_hash)
        match = chunk[chunk["credito_id_anon"].isin(ids_cohorte)]
        if not match.empty:
            partes.append(match)
        print(f"  leídas {leidas:,} filas, matcheadas "
              f"{sum(len(p) for p in partes):,}", flush=True)

    extracto = pd.concat(partes, ignore_index=True).drop_duplicates("credito_id_anon")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    extracto.to_parquet(SALIDA, index=False)

    cubiertos = len(set(extracto["credito_id_anon"]) & ids_cohorte)
    print(f"extracto: {extracto.shape} -> {SALIDA}")
    print(f"cobertura de la cohorte: {cubiertos}/{len(ids_cohorte)} "
          f"({cubiertos / len(ids_cohorte):.1%})")
    print("AVISO: contiene PII cruda; data/00_original/ está en .gitignore.")


if __name__ == "__main__":
    main()
