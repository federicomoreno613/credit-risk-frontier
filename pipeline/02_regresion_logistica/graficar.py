"""Visualizaciones del modelo #1 (regresión logística, baseline).

Lee los artefactos ya generados por entrenar_y_predecir.py y produce cuatro
figuras para la tesis en figures/pipeline/logreg/:

  01_coeficientes.png   coeficientes estandarizados ordenados (positivo = más mora)
  02_roc.png            curva ROC del test con el AUC
  03_calibracion.png    diagrama de calibración por deciles de score
  04_distribucion.png   distribución de scores por clase real

Insumos:
  data/pipeline/modelos/logreg_info.json
  data/pipeline/predicciones/regresion_logistica.parquet

Uso:
  poetry run python pipeline/02_regresion_logistica/graficar.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402

FIGURAS = C.ROOT / "figures" / "pipeline" / "logreg"
DPI = 200


def figura_coeficientes(info: dict) -> Path:
    """Coeficientes estandarizados ordenados; positivo empuja hacia la mora."""
    coefs = pd.Series(info["coeficientes_estandarizados"]).sort_values()
    colores = ["#b2182b" if v > 0 else "#2166ac" for v in coefs]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(coefs.index, coefs.values, color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente estandarizado (positivo = más probabilidad de mora)")
    ax.set_title(
        "Regresión logística: coeficientes estandarizados\n"
        f"(C={info['C']}, AUC validación={info['auc_val']:.3f})"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURAS / "01_coeficientes.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figura_roc(y_true: np.ndarray, prob: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#b2182b", label=f"Regresión logística (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0,5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — regresión logística (test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURAS / "02_roc.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figura_calibracion(y_true: np.ndarray, prob: np.ndarray) -> Path:
    """Tasa de mora observada vs. score medio por decil de score."""
    deciles = pd.qcut(prob, 10, labels=False, duplicates="drop")
    tabla = pd.DataFrame({"decil": deciles, "prob": prob, "y": y_true})
    agrupado = tabla.groupby("decil").agg(
        score_medio=("prob", "mean"), mora_observada=("y", "mean"), n=("y", "size")
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    ax.plot(
        agrupado["score_medio"], agrupado["mora_observada"],
        marker="o", color="#b2182b", label="Observado por decil",
    )
    for _, fila in agrupado.iterrows():
        ax.annotate(
            f"n={fila['n']:.0f}", (fila["score_medio"], fila["mora_observada"]),
            textcoords="offset points", xytext=(6, -10), fontsize=7, color="gray",
        )
    ax.set_xlabel("Probabilidad predicha (media del decil)")
    ax.set_ylabel("Tasa de mora observada")
    ax.set_title("Calibración por deciles — regresión logística (test)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURAS / "03_calibracion.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def figura_distribucion(y_true: np.ndarray, prob: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    ax.hist(
        prob[y_true == 0], bins=bins, alpha=0.6, color="#2166ac",
        label=f"Pagó (n={(y_true == 0).sum()})", density=True,
    )
    ax.hist(
        prob[y_true == 1], bins=bins, alpha=0.6, color="#b2182b",
        label=f"Mora (n={(y_true == 1).sum()})", density=True,
    )
    ax.set_xlabel("Probabilidad de mora predicha")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de scores por clase real — regresión logística (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURAS / "04_distribucion.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    info = json.loads((C.MODELOS / "logreg_info.json").read_text(encoding="utf-8"))
    predicciones = pd.read_parquet(C.PRED_DIR / "regresion_logistica.parquet")
    y_true = predicciones["y_true"].to_numpy(dtype=int)
    prob = predicciones["probabilidad"].to_numpy(dtype=float)

    FIGURAS.mkdir(parents=True, exist_ok=True)
    rutas = [
        figura_coeficientes(info),
        figura_roc(y_true, prob),
        figura_calibracion(y_true, prob),
        figura_distribucion(y_true, prob),
    ]
    print(f"AUC test: {roc_auc_score(y_true, prob):.4f} (n={len(predicciones)})")
    for ruta in rutas:
        print(f"figura -> {ruta.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
