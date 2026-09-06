"""Qwen3 local: zero-shot, thinking y few-shot (modelos #5-7).

Lee el texto humanizado, guarda thinking en JSONL reanudable y escribe
predicciones de test. Detalle en ``pipeline/05_qwen/README.md``.
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


# ---------------------------------------------------------------------------
# Cache JSONL reanudable (un registro por caso, NaN se guarda como null)
# ---------------------------------------------------------------------------
def leer_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    registros = {}
    for linea in path.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            registro = json.loads(linea)
        except json.JSONDecodeError:
            continue  # línea truncada por una corrida interrumpida: se re-predice
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


# ---------------------------------------------------------------------------
# Prompt: texto humanizado + instrucción con formato de salida fijo
# ---------------------------------------------------------------------------
def construir_mensajes(texto_caso: str, ejemplos: list[tuple[str, int]]) -> list[dict]:
    mensajes = [{"role": "system", "content": SISTEMA}]
    for texto_ejemplo, target in ejemplos:
        # Cada ejemplo few-shot es un turno usuario->asistente. La respuesta del
        # ejemplo es el desenlace observado (CLASE), nunca una pseudo-probabilidad
        # 0/100: el ground truth es binario y verbalizar certeza absoluta ancla
        # el score del caso a evaluar en los extremos.
        mensajes.append({
            "role": "user",
            "content": f"DATOS DEL SOLICITANTE: {texto_ejemplo}\n\n{utils.INSTR_EJEMPLO}",
        })
        mensajes.append({
            "role": "assistant",
            "content": f"CLASE: {'moroso' if target == 1 else 'no moroso'}",
        })
    mensajes.append({
        "role": "user",
        "content": f"DATOS DEL SOLICITANTE: {texto_caso}\n\n{utils.INSTR_INTERMEDIATE}",
    })
    return mensajes


def ejemplos_para(fila, train, knn, perfil: str, shots: int) -> list[tuple[str, int]]:
    """Vecinos de TRAIN balanceados por clase, en el MISMO formato humanizado
    que el caso a evaluar (Min et al. 2022: las demostraciones enseñan formato).

    El perfil "full" usa la serialización del perfil con descripción: las
    cualitativas extra solo van en el caso a evaluar, no en los ejemplos.
    """
    if not shots:
        return []
    perfil_ejemplos = "tu_form_description" if perfil.endswith("_full") else perfil
    vecinos = utils.knn_examples_for_case(
        fila, str(fila["credito_id_anon"]), train, knn, shots
    )
    return [
        (C.serializar_perfil(v, C.FEATURES_29, perfil_ejemplos),
         int(v["target"]))
        for _, v in vecinos.iterrows()
    ]


def casos_pendientes(test, humanizado, cache: dict, limite):
    """Pendientes en orden de test; --limite acota el TOTAL cubierto (cache
    incluido) tomando siempre un prefijo contiguo del test, nunca un subconjunto
    salteado (un piloto sobre casos no contiguos no es interpretable)."""
    textos = humanizado.set_index("credito_id_anon")
    filas = [fila for _, fila in test.iterrows()]
    if limite is not None:
        filas = filas[:limite]
    pendientes = [f for f in filas if str(f["credito_id_anon"]) not in cache]
    return pendientes, textos


# ---------------------------------------------------------------------------
# Inferencia local vía Ollama, con reintento y thinking guardado
# ---------------------------------------------------------------------------
def inferir(mensajes: list[dict], presupuesto: int, think: bool = True) -> dict:
    return utils.call_ollama_think(
        mensajes,
        model=C.QWEN_MODEL,
        think_native=think and C.QWEN_OPCIONES["think_native"],
        timeout=C.QWEN_OPCIONES["timeout_seconds"],
        retries=C.QWEN_OPCIONES["request_retries"],
        num_predict=presupuesto,
        num_ctx=C.QWEN_OPCIONES["num_ctx"],
        temperature=C.QWEN_OPCIONES["temperature"],
    )


def correr(variables, humanizado, perfil: str, shots: int, limite,
           think: bool = True) -> None:
    utils.check_ollama(model=C.QWEN_MODEL)
    train, _, test = C.dividir(variables)
    # La variante sin thinking (modelo #5 vs #6 del PLAN) usa cache y etiqueta
    # propios: mismo prompt, mismo modelo, única diferencia el razonamiento.
    etiqueta = perfil if think else f"{perfil}_nothink"
    cache_path = C.cache_razonamientos("qwen", etiqueta, shots)
    cache = leer_cache(cache_path)
    pendientes, textos = casos_pendientes(test, humanizado, cache, limite)
    print(f"qwen {etiqueta}/few{shots}: {len(cache)} en cache, {len(pendientes)} pendientes")

    knn = utils.build_knn_space(train, C.FEATURES_29) if shots else None
    for i, fila in enumerate(pendientes, 1):
        texto = textos.loc[str(fila["credito_id_anon"]), f"texto_{perfil}"]
        mensajes = construir_mensajes(texto, ejemplos_para(fila, train, knn, perfil, shots))
        inicio = time.time()
        resultado = inferir(mensajes, C.QWEN_OPCIONES["num_predict"], think)
        if pd.isna(resultado["prob"]):  # reintento con más presupuesto de tokens
            resultado = inferir(mensajes, C.QWEN_OPCIONES["retry_num_predict"], think)
        registro = {
            "evaluation_id": str(fila["credito_id_anon"]),
            "modelo": C.QWEN_MODEL,
            "perfil": etiqueta,
            "shots": shots,
            "set": "test",
            "segmento": str(fila.get("segmento", "")),
            "y_true": int(fila["target"]),
            "prompt_variant": C.PROMPT_VARIANT,
            "probabilidad": resultado["prob"],
            "clase": resultado["clase"],
            "valida": bool(pd.notna(resultado["prob"])),
            "prob_desde_thinking": resultado["prob_desde_thinking"],
            "thinking": resultado["thinking"],
            "respuesta": resultado["content"],
            "eval_count": resultado["eval_count"],
            "duracion_s": round(time.time() - inicio, 2),
            "ts": time.time(),
        }
        agregar_cache(cache_path, registro)
        print(f"  {i}/{len(pendientes)} id={registro['evaluation_id']} "
              f"prob={registro['probabilidad']}", flush=True)


def demo(variables, humanizado, perfil: str, shots: int) -> None:
    """Un caso real completo: imprime el prompt, el thinking y la respuesta."""
    utils.check_ollama(model=C.QWEN_MODEL)
    train, _, test = C.dividir(variables)
    fila = test.iloc[0]
    texto = humanizado.set_index("credito_id_anon").loc[
        str(fila["credito_id_anon"]), f"texto_{perfil}"
    ]
    knn = utils.build_knn_space(train, C.FEATURES_29) if shots else None
    mensajes = construir_mensajes(texto, ejemplos_para(fila, train, knn, perfil, shots))
    print("=== PROMPT ===")
    for mensaje in mensajes:
        print(f"[{mensaje['role']}]\n{mensaje['content']}\n")
    resultado = inferir(mensajes, C.QWEN_OPCIONES["num_predict"])
    print("=== THINKING ===")
    print(resultado["thinking"] or "(vacío)")
    print("=== RESPUESTA ===")
    print(resultado["content"])
    print(f"=== probabilidad parseada: {resultado['prob']} | y_true: {int(fila['target'])} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfil", choices=C.PERFILES, default=None)
    parser.add_argument("--shots", type=int, choices=C.SHOTS, default=None)
    parser.add_argument("--limite", type=int, default=None, help="máx. de casos por config")
    parser.add_argument("--demo", action="store_true",
                        help="muestra un caso completo: prompt + thinking + respuesta")
    parser.add_argument("--sin-thinking", action="store_true",
                        help="desactiva el thinking nativo (contraste #5 vs #6 del PLAN); "
                             "cache y etiqueta propios con sufijo _nothink")
    args = parser.parse_args()

    variables = C.cargar_variables()
    if not C.HUMANIZADO.exists():
        raise FileNotFoundError(
            f"Falta {C.HUMANIZADO}. Correr pipeline/01_preprocesamiento/03_humanizar.py"
        )
    humanizado = pd.read_parquet(C.HUMANIZADO)

    if args.demo:
        demo(variables, humanizado, args.perfil or "tu_form", args.shots or 0)
        return

    perfiles = [args.perfil] if args.perfil else list(C.PERFILES)
    shots = [args.shots] if args.shots is not None else list(C.SHOTS)
    for perfil in perfiles:
        for n in shots:
            correr(variables, humanizado, perfil, n, args.limite,
                   think=not args.sin_thinking)
    print("consolidar y medir: poetry run python pipeline/monitoreo.py")


if __name__ == "__main__":
    main()
