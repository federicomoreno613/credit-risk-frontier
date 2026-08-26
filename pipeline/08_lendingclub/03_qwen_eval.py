"""Qwen3-8B zero-shot y few-shot sobre el benchmark LendingClub de Feng et al.

Cuatro condiciones (2 shots x 2 variantes), reanudables vía cache JSONL:
- shots: 0 (zero-shot) y 8 (few-shot balanceado, ejemplos fijos de train).
- variante `original`: el `query` EXACTO del benchmark (comparable con las
  métricas reportadas por Feng et al. 2023 / FinBen).
- variante `perturbado`: mismos casos y valores, pero template reescrito,
  atributos reordenados por caso y sin mencionar "Lending Club". Si el
  desempeño cae respecto de `original`, la señal dependía del formato
  memorizado y no de razonamiento sobre los atributos (complementa el 02).

Métricas: accuracy, F1 (bad=positivo) y MCC — las del paper — en
`data/lendingclub/salidas/metricas_qwen.csv`.

Uso:
    poetry run python pipeline/08_lendingclub/03_qwen_eval.py --demo
    poetry run python pipeline/08_lendingclub/03_qwen_eval.py --limite 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc  # noqa: E402
from lc import C  # noqa: E402

SHOTS = (0, 8)
VARIANTES = ("original", "perturbado")
_RE_ETIQUETA = re.compile(r"\b(good|bad)\b", re.IGNORECASE)

INSTRUCCION_PERTURBADA = (
    "You are assessing a loan applicant. Based on the record below, decide "
    "whether the loan outcome will be 'good' (repaid) or 'bad' (default). "
    "Respond with only 'good' or 'bad'.\nRecord: {registro}\nAnswer:"
)


def consulta(fila: pd.Series, variante: str) -> str:
    if variante == "original":
        return fila["query"]
    campos = lc.parsear_texto(fila["text"])
    rng = np.random.default_rng(C.SEED + int(fila["id"]))
    orden = rng.permutation(lc.ATRIBUTOS)
    registro = "; ".join(f"{a}: {campos[a]}" for a in orden)
    return INSTRUCCION_PERTURBADA.format(registro=registro)


def ejemplos_few_shot(train: pd.DataFrame, variante: str, shots: int) -> list[dict]:
    """Mensajes usuario/asistente con `shots` ejemplos balanceados y fijos."""
    por_clase = shots // 2
    ejemplos = pd.concat([
        train[train["gold"] == 0].sample(n=por_clase, random_state=C.SEED),
        train[train["gold"] == 1].sample(n=por_clase, random_state=C.SEED),
    ]).sample(frac=1, random_state=C.SEED)
    mensajes = []
    for _, fila in ejemplos.iterrows():
        mensajes.append({"role": "user", "content": consulta(fila, variante)})
        mensajes.append({"role": "assistant", "content": fila["answer"]})
    return mensajes


def parsear_etiqueta(salida: str) -> int | None:
    encontrados = _RE_ETIQUETA.findall(salida)
    if not encontrados:
        return None
    return 1 if encontrados[-1].lower() == "bad" else 0


def leer_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def correr(test: pd.DataFrame, train: pd.DataFrame, shots: int,
           variante: str, limite: int) -> pd.DataFrame:
    muestra = test.sample(n=min(limite, len(test)), random_state=C.SEED)
    cache_path = lc.SALIDAS / f"qwen_lc_{shots}shot_{variante}.jsonl"
    cache = leer_cache(cache_path)
    contexto = ejemplos_few_shot(train, variante, shots) if shots else []

    pendientes = muestra[~muestra["id"].isin(cache)]
    print(f"[{shots}-shot/{variante}] {len(cache)} en cache, {len(pendientes)} pendientes")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for i, (_, fila) in enumerate(pendientes.iterrows(), 1):
        mensajes = contexto + [{"role": "user", "content": consulta(fila, variante)}]
        salida = lc.chat(mensajes, num_predict=20)
        registro = {
            "id": int(fila["id"]), "gold": int(fila["gold"]),
            "respuesta": salida, "prediccion": parsear_etiqueta(salida),
        }
        cache[registro["id"]] = registro
        with cache_path.open("a") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        if i % 25 == 0:
            print(f"  {i}/{len(pendientes)}", flush=True)
    return pd.DataFrame([cache[i] for i in muestra["id"] if i in cache])


def medir(resultados: pd.DataFrame, shots: int, variante: str) -> dict:
    validos = resultados.dropna(subset=["prediccion"])
    y, p = validos["gold"], validos["prediccion"].astype(int)
    return {
        "modelo": C.QWEN_MODEL, "shots": shots, "variante": variante,
        "n": len(resultados), "sin_parsear": int(len(resultados) - len(validos)),
        "accuracy": round(accuracy_score(y, p), 4),
        "f1_bad": round(f1_score(y, p, pos_label=1), 4),
        "mcc": round(matthews_corrcoef(y, p), 4),
        "tasa_bad_real": round(float(y.mean()), 4),
        "tasa_bad_pred": round(float(p.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=500, help="casos de test por condición")
    parser.add_argument("--demo", action="store_true", help="imprime un caso por variante y sale")
    args = parser.parse_args()

    C.utils.check_ollama(model=C.QWEN_MODEL)
    test = pd.read_parquet(lc.parquet_path("test"))
    train = pd.read_parquet(lc.parquet_path("train"))

    if args.demo:
        fila = test.sample(n=1, random_state=C.SEED).iloc[0]
        for variante in VARIANTES:
            prompt = consulta(fila, variante)
            print(f"\n=== {variante} ===\n{prompt}")
            print(f"--> respuesta: {lc.chat([{'role': 'user', 'content': prompt}], num_predict=20)!r}"
                  f" | gold: {fila['answer']}")
        return

    metricas = []
    for shots in SHOTS:
        for variante in VARIANTES:
            resultados = correr(test, train, shots, variante, args.limite)
            metricas.append(medir(resultados, shots, variante))
            print(f"  -> {metricas[-1]}")

    tabla = pd.DataFrame(metricas)
    path = lc.SALIDAS / "metricas_qwen.csv"
    tabla.to_csv(path, index=False)
    print(f"\n{tabla.to_string(index=False)}")
    print(f"[ok] métricas -> {path}")
    print("comparar contra Acc/F1/MCC reportados por Feng et al. 2023 (Tabla LendingClub).")


if __name__ == "__main__":
    main()
