"""Paso 01 — Cargar la data original y declarar la partición.

Lee los tres archivos crudos, reconstruye la cohorte con desenlace a 150 días
y asigna la partición temporal train/val/test UNA sola vez. Todos los pasos
posteriores heredan la columna ``set`` de acá; nadie vuelve a particionar.

Salidas:
  data/pipeline/01_base.parquet               cohorte modelable con target y set
  data/pipeline/01_manifiesto_particion.json  huellas e IDs por partición (congelable)
"""

from __future__ import annotations

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

from credit_risk_frontier.pipelines.data_processing.nodes import (
    build_credit_outcomes,
    build_exact_credit_bridge,
    build_split_manifest,
    create_model_input,
)


def main() -> None:
    creditos = pd.read_csv(C.RAW_CREDITOS, parse_dates=["fecha_desembolso"])
    pagos = pd.read_csv(
        C.RAW_PAGOS, parse_dates=["fecha_desembolso", "fecha_t_pago", "fecha_pago"]
    )
    legacy = pd.read_csv(C.RAW_LEGACY)

    puente = build_exact_credit_bridge(legacy, creditos)
    desenlaces = build_credit_outcomes(creditos, pagos, puente, C.DESENLACE)
    base = create_model_input(creditos, desenlaces)
    manifiesto = build_split_manifest(creditos, puente, desenlaces, base, C.DESENLACE)

    C.BASE.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(C.BASE, index=False)
    C.guardar_json(C.MANIFIESTO, manifiesto)

    conteos = base["set"].value_counts().to_dict()
    tasas = base.groupby("set")["target"].mean().round(4).to_dict()
    print(f"cohorte: {len(base)} créditos -> {C.BASE}")
    print(f"partición: {conteos}")
    print(f"tasa de mora por set: {tasas}")
    print(f"manifiesto: {C.MANIFIESTO}")


if __name__ == "__main__":
    main()
