"""Paso 02 — Crear la tabla de variables.

Parte de la base del paso 01 y deja SOLO lo que el contrato autoriza:
meta (id, fecha, target, set), las 29 variables estructuradas en el orden
contractual, el texto libre para los LLM y el segmento esparso/denso.
Valida que no haya columnas prohibidas ni variables faltantes.

Salidas:
  data/pipeline/02_variables.parquet
  data/pipeline/02_contrato_variables.json   inventario con faltantes por variable
"""

from __future__ import annotations

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

from credit_risk_frontier import utils


def main() -> None:
    if not C.BASE.exists():
        raise FileNotFoundError(f"Falta {C.BASE}. Correr primero 01_cargar_datos.py")
    base = pd.read_parquet(C.BASE)

    faltantes = [v for v in C.META + C.FEATURES_29 + [C.TEXTO_LIBRE] if v not in base.columns]
    if faltantes:
        raise RuntimeError(f"La base no cumple el contrato; faltan: {faltantes}")
    prohibidas = [
        c for c in utils.LEAK_COLS + utils.SCORES_INTERNOS + utils.TEMPORAL_PROXY_COLS
        if c in base.columns
    ]
    if prohibidas:
        raise RuntimeError(f"Columnas con fuga presentes: {prohibidas}")

    if "segmento" not in base.columns:
        base = utils.annotate_segments(base)
    columnas = C.META + C.FEATURES_29 + [C.TEXTO_LIBRE, "segmento", "n_tu_missing"]
    variables = base.loc[:, columnas].copy()
    variables.to_parquet(C.VARIABLES, index=False)

    numerico = C.preparar_numerico(variables)
    inventario = {
        "n_filas": len(variables),
        "particion": variables["set"].value_counts().to_dict(),
        "variables_transunion": C.TU_VARS,
        "variables_formulario": C.FORM_DIRECT_VARS,
        "texto_libre": C.TEXTO_LIBRE,
        "segmentos": variables["segmento"].value_counts().to_dict(),
        "faltantes_por_variable": numerico.isna().mean().round(4).to_dict(),
    }
    C.guardar_json(C.CONTRATO_VARIABLES, inventario)

    print(f"variables: {variables.shape} -> {C.VARIABLES}")
    print(f"segmentos: {inventario['segmentos']}")
    print(f"contrato: {C.CONTRATO_VARIABLES}")


if __name__ == "__main__":
    main()
