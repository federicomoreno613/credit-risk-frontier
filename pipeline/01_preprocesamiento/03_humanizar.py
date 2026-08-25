"""Paso 03 — Humanizar: convertir cada fila en texto en lenguaje natural.

Este es el puente entre la tabla y los LLM: cada crédito se serializa con las
descripciones oficiales de las variables (estilo TabLLM). Se generan los dos
perfiles del diseño: 29 variables, y 29 variables más la descripción libre del
negocio. Los pasos de predicción (Qwen/GPT) consumen ESTE archivo, de modo que
todos los modelos de lenguaje ven exactamente el mismo texto.

Salida:
  data/pipeline/03_humanizado.parquet
    credito_id_anon, set, target, segmento,
    texto_tu_form (29 vars), texto_tu_form_description (29 vars + descripción)
"""

from __future__ import annotations

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402


def texto_cualitativas(fila: pd.Series) -> str:
    """Serializa las cualitativas restantes (rubro, tipo, otra categoría)."""
    partes = []
    for columna, descripcion in C.CUALITATIVAS.items():
        valor = fila.get(columna)
        if valor is not None and not pd.isna(valor) and str(valor).strip():
            partes.append(f"{descripcion}: {str(valor).strip()}")
    return ", ".join(partes)


def main() -> None:
    variables = C.cargar_variables()

    filas = []
    for _, fila in variables.iterrows():
        base_desc = C.serializar_perfil(fila, C.FEATURES_29, "tu_form_description")
        extra = texto_cualitativas(fila)
        filas.append(
            {
                "credito_id_anon": str(fila["credito_id_anon"]),
                "set": fila["set"],
                "target": int(fila["target"]),
                "segmento": fila["segmento"],
                "texto_tu_form": C.serializar_perfil(fila, C.FEATURES_29, "tu_form"),
                "texto_tu_form_description": base_desc,
                # Perfil full: 29 vars + descripción + rubro/tipo/otra categoría
                "texto_tu_form_description_full": (
                    f"{base_desc}, {extra}" if extra else base_desc
                ),
            }
        )
    humanizado = pd.DataFrame(filas)
    humanizado.to_parquet(C.HUMANIZADO, index=False)

    ejemplo = humanizado.iloc[0]
    print(f"humanizado: {len(humanizado)} créditos -> {C.HUMANIZADO}")
    print(f"ejemplo ({ejemplo['credito_id_anon']}, set={ejemplo['set']}):")
    print(f"  {ejemplo['texto_tu_form'][:300]}...")


if __name__ == "__main__":
    main()
