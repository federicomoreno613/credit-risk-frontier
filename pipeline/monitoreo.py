"""Módulo COMÚN de monitoreo — lo reutilizan todas las carpetas de modelos.

Como módulo:  ``import monitoreo`` -> ``metricas_grupo``, ``psi``,
``consolidar``, ``reporte_modelo``.
Como script:  consolida todos los parquet de ``PRED_DIR`` + los razonamientos
y produce el reporte global (métricas por segmento, PSI train/test, tasa de
mora por partición, auditoría de razonamientos).

Salidas del script:
  data/pipeline/predicciones_consolidadas.parquet
  data/pipeline/monitoreo/metricas.csv
  data/pipeline/monitoreo/psi_variables.csv
  data/pipeline/monitoreo/resumen.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrato as C  # noqa: E402


def psi(esperado: pd.Series, observado: pd.Series, bins: int = 10) -> float:
    """Population Stability Index con cortes por cuantiles del esperado."""
    esperado = pd.to_numeric(esperado, errors="coerce")
    observado = pd.to_numeric(observado, errors="coerce")
    cortes = esperado.quantile(np.linspace(0, 1, bins + 1)).unique()
    if len(cortes) < 3:
        return float("nan")
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = pd.cut(esperado, cortes).value_counts(normalize=True, dropna=False)
    o = pd.cut(observado, cortes).value_counts(normalize=True, dropna=False)
    e, o = e.align(o, fill_value=0.0)
    e, o = e.clip(lower=1e-4), o.clip(lower=1e-4)
    return float(((o - e) * np.log(o / e)).sum())


def metricas_grupo(grupo: pd.DataFrame) -> dict:
    """AUC/Gini/KS/Brier de un grupo de predicciones (si hay señal suficiente)."""
    validas = grupo[grupo["valida"] & grupo["probabilidad"].notna()]
    fila = {"n": len(grupo), "validas": len(validas),
            "cobertura": round(len(validas) / len(grupo), 4) if len(grupo) else 0.0}
    if len(validas) >= 10 and validas["y_true"].nunique() == 2:
        m = C.credit_metrics(validas["y_true"], validas["probabilidad"])
        fila |= {k: round(float(m[k]), 4) for k in ("AUC", "Gini", "KS", "Brier")}
    return fila


def metricas_umbral(y_true: np.ndarray, prob: np.ndarray, umbral: float) -> dict:
    """Matriz de confusión y métricas de decisión a un umbral dado.

    Decisión: prob >= umbral -> se predice mora (y se rechazaría el crédito).
    Costo esperado según C.COSTOS: FN aprueba un moroso, FP rechaza un pagador.
    """
    pred = (prob >= umbral).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    total = tp + tn + fp + fn
    return {
        "umbral": round(float(umbral), 3),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round((tp + tn) / total, 4) if total else float("nan"),
        "precision": round(tp / (tp + fp), 4) if tp + fp else float("nan"),
        "recall": round(tp / (tp + fn), 4) if tp + fn else float("nan"),
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else float("nan"),
        "costo": round(
            fn * C.COSTOS["costo_fn"] + fp * C.COSTOS["costo_fp"], 2
        ),
    }


def umbral_optimo_costo(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Umbral que minimiza el costo esperado (barrido fino de 0.01 a 0.99)."""
    candidatos = np.arange(0.01, 1.0, 0.01)
    costos = [metricas_umbral(y_true, prob, u)["costo"] for u in candidatos]
    return float(candidatos[int(np.argmin(costos))])


def tabla_confusion(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Por modelo/config: matriz y métricas a umbral 0.5 y al óptimo por costo."""
    filas = []
    for (modelo, config), grupo in predicciones.groupby(["modelo", "configuracion"]):
        validas = grupo[grupo["valida"] & grupo["probabilidad"].notna()]
        if len(validas) < 10 or validas["y_true"].nunique() != 2:
            continue
        y = validas["y_true"].to_numpy()
        p = validas["probabilidad"].to_numpy()
        base = {"modelo": modelo, "configuracion": config, "n": len(validas)}
        filas.append(base | {"criterio": "umbral_0.5"} | metricas_umbral(y, p, 0.5))
        optimo = umbral_optimo_costo(y, p)
        filas.append(base | {"criterio": "costo_minimo"} | metricas_umbral(y, p, optimo))
    return pd.DataFrame(filas)


def tabla_metricas(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Métricas por modelo/configuración, total y por segmento esparso/denso."""
    filas = []
    for (modelo, config), grupo in predicciones.groupby(["modelo", "configuracion"]):
        filas.append({"modelo": modelo, "configuracion": config, "segmento": "total"}
                     | metricas_grupo(grupo))
        for segmento, sub in grupo.groupby("segmento"):
            filas.append({"modelo": modelo, "configuracion": config,
                          "segmento": segmento} | metricas_grupo(sub))
    return pd.DataFrame(filas)


def reporte_modelo(predicciones: pd.DataFrame, titulo: str = "") -> pd.DataFrame:
    """Reporte rápido para usar al final del script de CADA modelo."""
    tabla = tabla_metricas(predicciones)
    if titulo:
        print(f"== {titulo} ==")
    print(tabla.to_string(index=False))
    return tabla


def leer_razonamientos(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def consolidar() -> pd.DataFrame:
    """Une los parquet por modelo + los caches JSONL de razonamientos."""
    partes = [pd.read_parquet(p) for p in sorted(C.PRED_DIR.glob("*.parquet"))]
    for cache in sorted(C.RAZONAMIENTOS.glob("*.jsonl")):
        registros = leer_razonamientos(cache)
        if not registros:
            continue
        marco = pd.DataFrame(registros).drop_duplicates("evaluation_id", keep="last")
        partes.append(pd.DataFrame({
            "modelo": marco["modelo"],
            "configuracion": marco["perfil"] + "_few" + marco["shots"].astype(str),
            "credito_id_anon": marco["evaluation_id"].astype(str),
            "segmento": marco["segmento"],
            "y_true": marco["y_true"].astype(int),
            "probabilidad": pd.to_numeric(marco["probabilidad"], errors="coerce"),
            "valida": marco["valida"].astype(bool),
        }))
    if not partes:
        return pd.DataFrame(columns=C.COLUMNAS_PREDICCION)
    todas = pd.concat(partes, ignore_index=True)
    todas.to_parquet(C.PREDICCIONES, index=False)
    return todas


def main() -> None:
    predicciones = consolidar()
    if predicciones.empty:
        print("sin predicciones para monitorear")
        return
    variables = C.cargar_variables()
    train, _, test = C.dividir(variables)
    C.MONITOREO.mkdir(parents=True, exist_ok=True)

    metricas = tabla_metricas(predicciones)
    metricas.to_csv(C.MONITOREO / "metricas.csv", index=False)

    X_train, X_test = C.preparar_numerico(train), C.preparar_numerico(test)
    tabla_psi = pd.DataFrame({
        "variable": C.FEATURES_29,
        "psi_train_test": [psi(X_train[v], X_test[v]) for v in C.FEATURES_29],
    }).sort_values("psi_train_test", ascending=False)
    tabla_psi.to_csv(C.MONITOREO / "psi_variables.csv", index=False)

    matrices = tabla_confusion(predicciones)
    matrices.to_csv(C.MONITOREO / "matrices_confusion.csv", index=False)

    razonamientos = {}
    for cache in sorted(C.RAZONAMIENTOS.glob("*.jsonl")):
        registros = leer_razonamientos(cache)
        textos = [r.get("thinking") or r.get("reasoning") or "" for r in registros]
        razonamientos[cache.name] = {
            "casos": len(registros),
            "con_razonamiento": sum(1 for t in textos if t.strip()),
            "largo_medio_caracteres": int(np.mean([len(t) for t in textos])) if textos else 0,
        }

    resumen = {
        "tasa_mora_por_set": variables.groupby("set")["target"].mean().round(4).to_dict(),
        "conteos_por_set": variables["set"].value_counts().to_dict(),
        "variables_psi_alto": tabla_psi[tabla_psi["psi_train_test"] > 0.25]["variable"].tolist(),
        "razonamientos": razonamientos,
        "costos": C.COSTOS,
    }
    C.guardar_json(C.MONITOREO / "resumen.json", resumen)

    print("== desempeño (total) ==")
    print(metricas[metricas["segmento"] == "total"].to_string(index=False))
    if not matrices.empty:
        print("\n== matrices de confusión (0.5 vs. umbral de costo mínimo) ==")
        print(matrices.to_string(index=False))
    print("\n== PSI train vs test (top 5) ==")
    print(tabla_psi.head(5).to_string(index=False))
    print(f"\n== tasa de mora por set == {resumen['tasa_mora_por_set']}")
    if razonamientos:
        print(f"== razonamientos guardados == {json.dumps(razonamientos, indent=2)}")
    print(f"\nsalidas -> {C.MONITOREO}")


if __name__ == "__main__":
    main()
