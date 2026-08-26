"""Tests de contaminación de preentrenamiento sobre Qwen3-8B (LendingClub).

Adaptación de los tests de memorización de datos tabulares de Bordt, Nori y
Caruana (2024, "Elephants Never Forget", arXiv:2404.06209 / paquete
tabmemcheck) al artefacto que efectivamente pudo entrar al corpus de Qwen3:
el benchmark de Feng et al. 2023 publicado en Hugging Face (2024-03), no el
CSV de Kaggle. Tres tests, cada uno con su control:

1. Conocimiento del esquema (≈ header test): ¿Qwen lista los atributos del
   dataset sin verlos? Se mide solapamiento con los 21 atributos reales.
2. Completación de registro (≈ row completion): se corta el `text` de un caso
   real en el atributo 12 y se pide continuar VERBATIM. Control: los mismos
   casos con valores permutados entre filas (imposibles de haber memorizado).
   Señal de memorización = similitud(original) >> similitud(control).
3. Completación de atributo (≈ feature completion): se oculta `Installment`
   (casi único por caso, no derivable sin el plazo) y se pide el valor exacto.
   Control idéntico al test 2.

Un resultado negativo NO prueba ausencia de contaminación (corpus cerrado);
solo la acota. Interpretación en `data/lendingclub/salidas/contaminacion.json`.

Uso:
    poetry run python pipeline/08_lendingclub/02_contaminacion.py [--n 25]
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc  # noqa: E402
from lc import C  # noqa: E402

CORTE = 12  # se muestra hasta el atributo 12 y se pide continuar con el resto


# ---------------------------------------------------------------------------
# Controles: permutar cada atributo entre filas rompe los registros reales
# manteniendo las distribuciones marginales (mismo formato, filas inexistentes).
# ---------------------------------------------------------------------------
def permutar_columnas(campos_lista: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    resultado = [dict(c) for c in campos_lista]
    for atributo in lc.ATRIBUTOS:
        valores = [c[atributo] for c in campos_lista]
        for caso, valor in zip(resultado, rng.permutation(valores)):
            caso[atributo] = str(valor)
    return resultado


# ---------------------------------------------------------------------------
# Test 1: conocimiento del esquema
# ---------------------------------------------------------------------------
def test_esquema() -> dict:
    prompt = (
        "List the exact feature/column names of the Lending Club loan dataset "
        "as used in credit scoring benchmarks (the version with 21 client "
        "attributes). Output only a comma-separated list of names."
    )
    salida = lc.chat([{"role": "user", "content": prompt}])
    nombrados = {n.strip().lower() for n in salida.replace("\n", ",").split(",") if n.strip()}
    aciertos = sorted(a for a in lc.ATRIBUTOS if a.lower() in nombrados)
    return {
        "respuesta": salida,
        "aciertos": aciertos,
        "n_aciertos": len(aciertos),
        "n_atributos": len(lc.ATRIBUTOS),
    }


# ---------------------------------------------------------------------------
# Tests 2 y 3, corridos sobre casos originales y de control
# ---------------------------------------------------------------------------
def prefijo_y_resto(campos: dict) -> tuple[str, str]:
    texto = lc.serializar_feng(campos)
    marca = f"The state of {lc.ATRIBUTOS[CORTE]} is"
    idx = texto.index(marca)
    return texto[:idx].rstrip(), texto[idx:]


def test_completacion_registro(campos_lista: list[dict], contexto: list[dict]) -> list[float]:
    ejemplos = "\n\n".join(lc.serializar_feng(c) for c in contexto)
    similitudes = []
    for campos in campos_lista:
        prefijo, resto = prefijo_y_resto(campos)
        prompt = (
            "These are records from a dataset:\n\n"
            f"{ejemplos}\n\n"
            "Complete the next record VERBATIM, continuing exactly from where "
            "it stops. Output only the continuation, nothing else.\n\n"
            f"{prefijo}"
        )
        salida = lc.chat([{"role": "user", "content": prompt}], num_predict=400)
        similitudes.append(difflib.SequenceMatcher(
            None, salida.strip()[: len(resto)], resto).ratio())
    return similitudes


def test_completacion_atributo(campos_lista: list[dict]) -> list[bool]:
    aciertos = []
    for campos in campos_lista:
        parcial = {a: v for a, v in campos.items() if a != "Installment"}
        cuerpo = " ".join(f"The state of {a} is {v}." for a, v in parcial.items())
        prompt = (
            "The following record comes from the Lending Club dataset with one "
            "attribute removed. State the exact value of 'Installment' for this "
            "record as it appears in the dataset. Output only the number.\n\n"
            f"The client has attributes as follows: {cuerpo}"
        )
        salida = lc.chat([{"role": "user", "content": prompt}], num_predict=30)
        try:
            acierto = abs(float(salida.strip().split()[-1].rstrip(".")) -
                          float(campos["Installment"])) < 0.005
        except (ValueError, IndexError):
            acierto = False
        aciertos.append(acierto)
    return aciertos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=25, help="casos por condición")
    args = parser.parse_args()

    C.utils.check_ollama(model=C.QWEN_MODEL)
    test = pd.read_parquet(lc.parquet_path("test"))
    muestra = test.sample(n=args.n + 3, random_state=C.SEED)
    campos_lista = [lc.parsear_texto(t) for t in muestra["text"]]
    contexto, originales = campos_lista[:3], campos_lista[3:]
    control = permutar_columnas(originales, seed=C.SEED)

    print(f"== test 1: conocimiento del esquema (Qwen {C.QWEN_MODEL}) ==")
    esquema = test_esquema()
    print(f"   atributos acertados: {esquema['n_aciertos']}/{esquema['n_atributos']}")

    print(f"== test 2: completación de registro (n={args.n} + {args.n} control) ==")
    sim_orig = test_completacion_registro(originales, contexto)
    sim_ctrl = test_completacion_registro(control, contexto)
    print(f"   similitud media original={np.mean(sim_orig):.3f} control={np.mean(sim_ctrl):.3f}")

    print("== test 3: completación de Installment ==")
    acc_orig = test_completacion_atributo(originales)
    acc_ctrl = test_completacion_atributo(control)
    print(f"   exactos original={np.mean(acc_orig):.3f} control={np.mean(acc_ctrl):.3f}")

    resultado = {
        "modelo": C.QWEN_MODEL,
        "n": args.n,
        "metodologia": "Bordt et al. 2024 (arXiv:2404.06209), adaptado al benchmark HF de Feng et al. 2023",
        "esquema": esquema,
        "completacion_registro": {
            "similitud_original": sim_orig, "similitud_control": sim_ctrl,
            "media_original": float(np.mean(sim_orig)),
            "media_control": float(np.mean(sim_ctrl)),
            "brecha": float(np.mean(sim_orig) - np.mean(sim_ctrl)),
        },
        "completacion_installment": {
            "acierto_original": float(np.mean(acc_orig)),
            "acierto_control": float(np.mean(acc_ctrl)),
        },
        "interpretacion": (
            "Brecha original-control ≈ 0 en tests 2 y 3 => sin evidencia de "
            "memorización verbatim (no prueba ausencia de contaminación). "
            "Brecha grande y positiva => el benchmark estuvo en el corpus."
        ),
    }
    path = lc.guardar_json(lc.SALIDAS / "contaminacion.json", resultado)
    print(f"[ok] resultados -> {path}")


if __name__ == "__main__":
    main()
