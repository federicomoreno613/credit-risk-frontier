# %% [markdown]
# # Baseline Regresión Logística — Credit Scoring
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Regresión Logística con imputación de medianas (train-only) y escalado estándar.
# Sirve como baseline lineal interpretable para contrastar con XGBoost y LLM.
# Los coeficientes son directamente interpretables como log-odds.
#
# **Validación temporal (no StratifiedKFold).** El split train/val/test viene
# dado por la columna `set`, que respeta el orden de `fecha_desembolso`. La
# imputación de medianas y el escalado se ajustan **solo sobre train** (vía
# `Pipeline`), evitando *data leakage*. Se reporta el AUC de train, val y test
# por separado para cuantificar el sobreajuste de forma honesta.

# %% [markdown]
# ## Setup

# %%
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]

SEED = 42

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = {"xgb": "#4878CF", "logreg": "#6ACC65", "llm": "#D65F5F"}


# %% [markdown]
# ## Métricas estándar de credit scoring

# %%
def credit_metrics(y_true, y_prob) -> dict:
    """Calcula AUC, KS, Gini, Brier y PR-AUC para un conjunto de predicciones."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return {"AUC": auc, "Gini": gini, "KS": ks, "Brier": brier, "PR-AUC": prauc}


# %% [markdown]
# ## Carga de datos y splits temporales

# %%
def load_splits(dataset_path: str) -> dict:
    """Carga el dataset y separa train/val/test por la columna temporal `set`.

    Retorna un dict con X/y por split y la lista de features usadas.
    """
    df = pd.read_csv(dataset_path, parse_dates=["fecha_desembolso"])
    features = [c for c in df.columns if c not in META + TEXT]
    splits = {}
    for name in ("train", "val", "test"):
        subset = df[df["set"] == name]
        splits[name] = {"X": subset[features], "y": subset["target"]}
    splits["features"] = features
    return splits


# %% [markdown]
# ## Pipeline: imputación + escalado + LogReg
#
# Imputación con mediana y escalado se ajustan dentro del `Pipeline`, por lo que
# se calibran solo con train al llamar `.fit(X_train, ...)`. Las variables TU ya
# usan `-1` como código de faltante (no NaN), así que no se imputan.

# %%
def build_pipeline(seed: int = SEED) -> Pipeline:
    """Construye el pipeline imputación → escalado → regresión logística L2."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            C=0.1,           # regularización L2 moderada (equivale a λ=10)
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


# %% [markdown]
# ## Figuras de diagnóstico

# %%
def plot_roc(splits, probs, metrics, figs_dir: Path) -> None:
    """Curva ROC para validación y test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr_v, tpr_v, _ = roc_curve(splits["val"]["y"], probs["val"])
    fpr_t, tpr_t, _ = roc_curve(splits["test"]["y"], probs["test"])
    ax.plot(fpr_v, tpr_v, color=PALETTE["logreg"], lw=2,
            label=f"Validación (AUC={metrics['val']['AUC']:.3f})")
    ax.plot(fpr_t, tpr_t, color=PALETTE["logreg"], lw=2, linestyle="--",
            label=f"Test (AUC={metrics['test']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — Regresión Logística", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "15_roc_logreg.png", bbox_inches="tight")
    plt.close(fig)


# %%
def plot_coefs(model, features, figs_dir: Path) -> None:
    """Top 20 coeficientes (log-odds estandarizados) ordenados por magnitud."""
    coefs = (pd.Series(model.coef_[0], index=features)
             .sort_values(key=abs, ascending=False).head(20).sort_values())
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#D65F5F" if c > 0 else "#4878CF" for c in coefs]
    ax.barh(coefs.index, coefs.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coeficiente (log-odds) — estandarizado")
    ax.set_title("Top 20 coeficientes — Regresión Logística", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs_dir / "16_coefs_logreg.png", bbox_inches="tight")
    plt.close(fig)


# %%
def plot_calibration(splits, probs, figs_dir: Path) -> None:
    """Curva de calibración sobre el conjunto de test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(splits["test"]["y"], probs["test"], n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["logreg"], lw=2, label="LogReg")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media del bin)")
    ax.set_ylabel("Fracción de positivos observados")
    ax.set_title("Calibración — Regresión Logística", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "17_calibracion_logreg.png", bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Entrenamiento end-to-end (nodo Kedro)
#
# `train_logreg` recibe rutas como argumentos y retorna un dict de métricas.
# Es importable como módulo: la lógica no depende de estado global.

# %%
def train_logreg(dataset_path: str, output_dir: str) -> dict:
    """Entrena Regresión Logística con validación temporal y persiste artefactos.

    Args:
        dataset_path: ruta al CSV con columnas `set`, `target`, `fecha_desembolso`.
        output_dir: directorio para guardar pipeline y `logreg_metrics.json`.
                    Las figuras se guardan en `<output_dir>/../figures`.

    Returns:
        dict con métricas de train/val/test.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figs_dir = output_path.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(dataset_path)
    pipeline = build_pipeline()
    pipeline.fit(splits["train"]["X"], splits["train"]["y"])
    joblib.dump(pipeline, output_path / "logreg_baseline.pkl")

    probs = {name: pipeline.predict_proba(splits[name]["X"])[:, 1]
             for name in ("train", "val", "test")}
    metrics = {name: credit_metrics(splits[name]["y"], probs[name])
               for name in ("train", "val", "test")}

    results = {"model": "logreg", **metrics}
    with open(output_path / "logreg_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_roc(splits, probs, metrics, figs_dir)
    plot_coefs(pipeline.named_steps["model"], splits["features"], figs_dir)
    plot_calibration(splits, probs, figs_dir)

    return results


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    train_logreg(
        dataset_path=str(BASE / "data" / "dataset_tesis.csv"),
        output_dir=str(BASE / "models"),
    )
