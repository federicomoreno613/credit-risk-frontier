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


def bootstrap_auc_ic(y_true: np.ndarray, prob: np.ndarray,
                     n_iter: int = 1000, seed: int = 42) -> tuple[float, float]:
    """IC 95% del AUC por bootstrap de casos (Efron & Tibshirani, 1993)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _midranks(x: np.ndarray) -> np.ndarray:
    """Midranks para el estimador de DeLong (Sun & Xu, 2014)."""
    orden = np.argsort(x)
    ordenado = x[orden]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and ordenado[j] == ordenado[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    resultado = np.empty(n)
    resultado[orden] = t
    return resultado


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Test de DeLong et al. (1988) para dos AUC correlacionadas (mismos casos).

    Devuelve ambas AUC, la diferencia y el p-valor bilateral.
    """
    from scipy import stats

    orden = np.argsort(-y_true)
    y = y_true[orden]
    m = int(y.sum())            # positivos primero
    n = len(y) - m
    if m == 0 or n == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "delta": float("nan"), "p_valor": float("nan")}
    scores = np.vstack([prob_a[orden], prob_b[orden]])
    k = scores.shape[0]
    tx = np.array([_midranks(fila[:m]) for fila in scores])
    ty = np.array([_midranks(fila[m:]) for fila in scores])
    tz = np.array([_midranks(fila) for fila in scores])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    contraste = np.array([1.0, -1.0])
    var = float(contraste @ cov @ contraste) if k == 2 else float("nan")
    delta = float(aucs[0] - aucs[1])
    if var <= 0:
        p = float("nan")
    else:
        z = delta / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": delta, "p_valor": p}


def metricas_grupo(grupo: pd.DataFrame, con_ic: bool = False) -> dict:
    """AUC/Gini/KS/Brier de un grupo de predicciones (si hay señal suficiente)."""
    validas = grupo[grupo["valida"] & grupo["probabilidad"].notna()]
    fila = {"n": len(grupo), "validas": len(validas),
            "cobertura": round(len(validas) / len(grupo), 4) if len(grupo) else 0.0}
    if len(validas) >= 10 and validas["y_true"].nunique() == 2:
        m = C.credit_metrics(validas["y_true"], validas["probabilidad"])
        fila |= {k: round(float(m[k]), 4) for k in ("AUC", "Gini", "KS", "Brier")}
        if con_ic:
            lo, hi = bootstrap_auc_ic(validas["y_true"].to_numpy(),
                                      validas["probabilidad"].to_numpy())
            fila |= {"AUC_ic95_lo": round(lo, 4), "AUC_ic95_hi": round(hi, 4)}
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
    """Por modelo/config y segmento: matrices a umbral 0.5, al óptimo por costo
    y (si el LLM la verbalizó) según la clase declarada por el modelo.

    ``costo_minimo_test`` optimiza el umbral SOBRE el propio test: es una cota
    optimista, no una estimación fuera de muestra — el umbral operativo honesto
    se elige en validación (los scripts de modelos tabulares lo hacen).
    """
    filas = []
    for (modelo, config), grupo in predicciones.groupby(["modelo", "configuracion"]):
        segmentos = [("total", grupo)] + list(grupo.groupby("segmento"))
        for segmento, sub in segmentos:
            validas = sub[sub["valida"] & sub["probabilidad"].notna()]
            if len(validas) < 10 or validas["y_true"].nunique() != 2:
                continue
            y = validas["y_true"].to_numpy()
            p = validas["probabilidad"].to_numpy()
            base = {"modelo": modelo, "configuracion": config,
                    "segmento": segmento, "n": len(validas)}
            filas.append(base | {"criterio": "umbral_0.5"} | metricas_umbral(y, p, 0.5))
            optimo = umbral_optimo_costo(y, p)
            filas.append(base | {"criterio": "costo_minimo_test"}
                         | metricas_umbral(y, p, optimo))
            if "clase" in validas.columns and validas["clase"].notna().sum() >= 10:
                con_clase = validas[validas["clase"].notna()]
                filas.append({**base, "n": len(con_clase),
                              "criterio": "clase_verbalizada"}
                             | metricas_umbral(con_clase["y_true"].to_numpy(),
                                               con_clase["clase"].to_numpy(), 0.5))
    return pd.DataFrame(filas)


def tabla_delong(predicciones: pd.DataFrame) -> pd.DataFrame:
    """DeLong entre las mejores configuraciones de cada modelo (pares clave).

    Se comparan sobre la intersección de casos válidos en ambas configs.
    """
    metricas = tabla_metricas(predicciones)
    total = metricas[metricas["segmento"].eq("total") & metricas["AUC"].notna()]
    mejores = (total.sort_values("AUC", ascending=False)
               .groupby("modelo", sort=False).head(1))
    claves = list(mejores[["modelo", "configuracion"]].itertuples(index=False))
    filas = []
    for i, (modelo_a, config_a) in enumerate(claves):
        for modelo_b, config_b in claves[i + 1:]:
            a = predicciones[(predicciones["modelo"] == modelo_a)
                             & (predicciones["configuracion"] == config_a)
                             & predicciones["valida"]]
            b = predicciones[(predicciones["modelo"] == modelo_b)
                             & (predicciones["configuracion"] == config_b)
                             & predicciones["valida"]]
            par = a.merge(b, on="credito_id_anon", suffixes=("_a", "_b"))
            par = par[par["probabilidad_a"].notna() & par["probabilidad_b"].notna()]
            if len(par) < 30 or par["y_true_a"].nunique() != 2:
                continue
            resultado = delong_test(par["y_true_a"].to_numpy(),
                                    par["probabilidad_a"].to_numpy(),
                                    par["probabilidad_b"].to_numpy())
            filas.append({
                "modelo_a": modelo_a, "config_a": config_a,
                "modelo_b": modelo_b, "config_b": config_b,
                "n_comunes": len(par),
                "auc_a": round(resultado["auc_a"], 4),
                "auc_b": round(resultado["auc_b"], 4),
                "delta_auc": round(resultado["delta"], 4),
                "p_valor": round(resultado["p_valor"], 5),
            })
    return pd.DataFrame(filas)


def tabla_metricas(predicciones: pd.DataFrame, con_ic: bool = False) -> pd.DataFrame:
    """Métricas por modelo/configuración, total y por segmento esparso/denso."""
    filas = []
    for (modelo, config), grupo in predicciones.groupby(["modelo", "configuracion"]):
        filas.append({"modelo": modelo, "configuracion": config, "segmento": "total"}
                     | metricas_grupo(grupo, con_ic))
        for segmento, sub in grupo.groupby("segmento"):
            filas.append({"modelo": modelo, "configuracion": config,
                          "segmento": segmento} | metricas_grupo(sub, con_ic))
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
    """Une los parquet por modelo + los caches JSONL de razonamientos.

    Solo consolida razonamientos de la variante de prompt VIGENTE
    (``C.PROMPT_VARIANT``): nunca se mezclan respuestas de prompts distintos
    en una misma tabla de métricas.
    """
    partes = [pd.read_parquet(p) for p in sorted(C.PRED_DIR.glob("*.parquet"))]
    for cache in sorted(C.RAZONAMIENTOS.glob("*.jsonl")):
        registros = [r for r in leer_razonamientos(cache)
                     if r.get("prompt_variant") == C.PROMPT_VARIANT]
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
            "clase": pd.to_numeric(marco.get("clase"), errors="coerce"),
        }))
    if not partes:
        return pd.DataFrame(columns=C.COLUMNAS_PREDICCION)
    todas = pd.concat(partes, ignore_index=True)
    todas.to_parquet(C.PREDICCIONES, index=False)
    return todas


def costos_y_tiempos() -> dict:
    """Tiempo y consumo por corrida LLM, desde los JSONL de la variante vigente.

    El costo GPT en USD se calcula si están definidos TESIS_GPT_PRECIO_IN /
    TESIS_GPT_PRECIO_OUT (dólares por millón de tokens); si no, se reportan
    solo los tokens para calcularlo aparte.
    """
    import os

    precio_in = float(os.environ.get("TESIS_GPT_PRECIO_IN", 0) or 0)
    precio_out = float(os.environ.get("TESIS_GPT_PRECIO_OUT", 0) or 0)
    reporte = {}
    for cache in sorted(C.RAZONAMIENTOS.glob("*.jsonl")):
        registros = [r for r in leer_razonamientos(cache)
                     if r.get("prompt_variant") == C.PROMPT_VARIANT]
        if not registros:
            continue
        duraciones = [r["duracion_s"] for r in registros if r.get("duracion_s")]
        entrada = {
            "casos": len(registros),
            "duracion_total_min": round(sum(duraciones) / 60, 1),
            "duracion_media_s": round(np.mean(duraciones), 2) if duraciones else None,
        }
        usos = [r["usage"] for r in registros if r.get("usage")]
        if usos:
            tokens_in = sum(u.get("input_tokens", 0) for u in usos)
            tokens_out = sum(u.get("output_tokens", 0) for u in usos)
            entrada |= {"tokens_input": tokens_in, "tokens_output": tokens_out}
            if precio_in and precio_out:
                entrada["costo_usd"] = round(
                    tokens_in / 1e6 * precio_in + tokens_out / 1e6 * precio_out, 2)
        reporte[cache.name] = entrada
    return reporte


def main() -> None:
    predicciones = consolidar()
    if predicciones.empty:
        print("sin predicciones para monitorear")
        return
    variables = C.cargar_variables()
    train, _, test = C.dividir(variables)
    C.MONITOREO.mkdir(parents=True, exist_ok=True)

    metricas = tabla_metricas(predicciones, con_ic=True)
    metricas.to_csv(C.MONITOREO / "metricas.csv", index=False)

    comparaciones = tabla_delong(predicciones)
    comparaciones.to_csv(C.MONITOREO / "comparaciones_delong.csv", index=False)

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
        "prompt_variant": C.PROMPT_VARIANT,
        "tasa_mora_por_set": variables.groupby("set")["target"].mean().round(4).to_dict(),
        "conteos_por_set": variables["set"].value_counts().to_dict(),
        "variables_psi_alto": tabla_psi[tabla_psi["psi_train_test"] > 0.25]["variable"].tolist(),
        "razonamientos": razonamientos,
        "costos_y_tiempos_llm": costos_y_tiempos(),
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
