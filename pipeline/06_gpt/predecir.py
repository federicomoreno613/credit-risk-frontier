"""Modelo #4 del PLAN §2.2 — GPT vía API: techo de calidad (PLAN §1.5.1).

Un LLM propietario de frontera, sin fine-tuning, marca el techo de lo que se
puede esperar de un modelo de lenguaje sin límites de recursos; es además la
comparación directa con Feng et al. (2023). Recibe EXACTAMENTE el mismo texto
humanizado y el mismo prompt que Qwen.

Se usa la Responses API de OpenAI con `reasoning={"effort": "medium",
"summary": "auto"}`: el resumen del razonamiento se guarda por caso (campo
`reasoning`) en data/pipeline/razonamientos/gpt_{perfil}_few{shots}.jsonl,
junto con la respuesta, la probabilidad y el uso de tokens (campo `usage`,
para reportar el costo en dólares que pide el PLAN §2.4).

Requisitos: `poetry add openai` y la variable OPENAI_API_KEY. El modelo se
elige con TESIS_GPT_MODEL (default en contrato.py). El cache es reanudable y
los errores por caso se persisten sin abortar la corrida.

Uso:
  poetry run python pipeline/06_gpt/predecir.py --perfil tu_form --shots 0 --limite 50
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

from credit_risk_frontier import utils  # noqa: E402

SISTEMA = "Estima el riesgo de mora de un microcrédito."


def leer_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    registros = {}
    for linea in path.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            registro = json.loads(linea)
            registros[str(registro["evaluation_id"])] = registro
    return registros


def agregar_cache(path: Path, registro: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        k: (None if isinstance(v, float) and v != v else v)
        for k, v in utils.json_safe(registro).items()
    }
    with path.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def construir_mensajes(texto_caso: str, ejemplos: list[tuple[str, int]]) -> list[dict]:
    """Mismo prompt que Qwen; el system va aparte (instructions)."""
    mensajes = []
    for texto_ejemplo, target in ejemplos:
        mensajes.append({
            "role": "user",
            "content": f"DATOS DEL SOLICITANTE: {texto_ejemplo}\n\n{utils.INSTR_INTERMEDIATE}",
        })
        mensajes.append({
            "role": "assistant",
            "content": f"PROBABILIDAD_DE_MORA: {100 if target == 1 else 0}",
        })
    mensajes.append({
        "role": "user",
        "content": f"DATOS DEL SOLICITANTE: {texto_caso}\n\n{utils.INSTR_INTERMEDIATE}",
    })
    return mensajes


def ejemplos_para(fila, train, knn, perfil: str, shots: int) -> list[tuple[str, int]]:
    if not shots:
        return []
    perfil_ejemplos = "tu_form_description" if perfil.endswith("_full") else perfil
    vecinos = utils.knn_examples_for_case(
        fila, str(fila["credito_id_anon"]), train, knn, shots
    )
    return [
        (C.serializar_perfil(v, C.FEATURES_29, perfil_ejemplos, compact=True),
         int(v["target"]))
        for _, v in vecinos.iterrows()
    ]


def correr(variables, humanizado, perfil: str, shots: int, limite,
           effort: str = "medium") -> None:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "Falta la librería openai. Instalar con: poetry add openai "
            "(y definir OPENAI_API_KEY)"
        ) from error

    cliente = OpenAI()
    train, _, test = C.dividir(variables)
    # La variante "thinking" (effort=high) es una configuración aparte:
    # cache y etiqueta propios, para comparar contra el zero-shot estándar.
    etiqueta = f"{perfil}_think" if effort == "high" else perfil
    cache_path = C.RAZONAMIENTOS / f"gpt_{etiqueta}_few{shots}.jsonl"
    cache = leer_cache(cache_path)
    textos = humanizado.set_index("credito_id_anon")
    pendientes = []
    for _, fila in test.iterrows():
        if limite is not None and len(cache) + len(pendientes) >= limite:
            break
        if str(fila["credito_id_anon"]) not in cache:
            pendientes.append(fila)
    print(f"gpt {etiqueta}/few{shots}: {len(cache)} en cache, {len(pendientes)} pendientes")

    knn = utils.build_knn_space(train, C.FEATURES_29) if shots else None
    for i, fila in enumerate(pendientes, 1):
        texto = textos.loc[str(fila["credito_id_anon"]), f"texto_{perfil}"]
        mensajes = construir_mensajes(texto, ejemplos_para(fila, train, knn, perfil, shots))
        registro = {
            "evaluation_id": str(fila["credito_id_anon"]),
            "modelo": C.GPT_MODEL,
            "perfil": etiqueta,
            "shots": shots,
            "set": "test",
            "segmento": str(fila.get("segmento", "")),
            "y_true": int(fila["target"]),
            "prompt_variant": C.PROMPT_VARIANT,
            "reasoning_effort": effort,
            "ts": time.time(),
        }
        try:
            respuesta = cliente.responses.create(
                model=C.GPT_MODEL,
                instructions=SISTEMA,
                input=mensajes,
                reasoning={"effort": effort, "summary": "auto"},
                max_output_tokens=4096,
            )
            salida = respuesta.output_text or ""
            razonamiento = "\n\n".join(
                parte.text
                for item in respuesta.output
                if getattr(item, "type", "") == "reasoning"
                for parte in (item.summary or [])
            )
            probabilidad = C.parse_prob(salida)
            registro |= {
                "probabilidad": probabilidad,
                "valida": bool(pd.notna(probabilidad)),
                "reasoning": razonamiento,
                "respuesta": salida,
                "usage": respuesta.usage.model_dump() if respuesta.usage else None,
            }
        except Exception as error:  # noqa: BLE001 — se persiste el error y se sigue
            registro |= {
                "probabilidad": float("nan"),
                "valida": False,
                "reasoning": "",
                "respuesta": "",
                "error": str(error),
            }
        agregar_cache(cache_path, registro)
        print(f"  {i}/{len(pendientes)} id={registro['evaluation_id']} "
              f"prob={registro.get('probabilidad')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfil", choices=C.PERFILES, default=None)
    parser.add_argument("--shots", type=int, choices=C.SHOTS, default=None)
    parser.add_argument("--limite", type=int, default=None, help="máx. de casos por config")
    parser.add_argument("--effort", choices=("medium", "high"), default="medium",
                        help="esfuerzo de reasoning; high = variante 'thinking' con cache aparte")
    args = parser.parse_args()

    variables = C.cargar_variables()
    if not C.HUMANIZADO.exists():
        raise FileNotFoundError(
            f"Falta {C.HUMANIZADO}. Correr pipeline/01_preprocesamiento/03_humanizar.py"
        )
    humanizado = pd.read_parquet(C.HUMANIZADO)

    perfiles = [args.perfil] if args.perfil else list(C.PERFILES)
    shots = [args.shots] if args.shots is not None else list(C.SHOTS)
    for perfil in perfiles:
        for n in shots:
            correr(variables, humanizado, perfil, n, args.limite, args.effort)
    print("consolidar y medir: poetry run python pipeline/monitoreo.py")


if __name__ == "__main__":
    main()
