# %% [markdown]
# # Segmentación por Densidad de Bureau — Validación Temporal (TimeSeriesSplit)
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Compara XGBoost vs. Regresión Logística en dos segmentos según la riqueza de
# información disponible en el bureau TransUnion:
#
# - **Bureau esparso** (≥6 vars TU con -1, ~36%): clientes con actividad crediticia
#   reducida. Variables TU con -1 como `lmd34s`, `rle904`, `agg2503` indican ausencia
#   de mora histórica, cartera castigada y deuda vencida — son señales protectoras.
#
# - **Bureau denso** (<6 vars TU con -1, ~64%): clientes con historial TU completo
#   donde los modelos clásicos tienen mayor ventaja informativa.
#
# ## Cambio metodológico: de StratifiedKFold a TimeSeriesSplit
#
# La versión previa de este script usaba `StratifiedKFold(shuffle=True)`. En un
# dataset con estructura temporal eso es **incorrecto**: el barajado mezcla
# créditos de 2022 y 2024 en el mismo fold, de modo que el modelo entrena con
# información del *futuro* para predecir el *pasado* (data leakage temporal). El
# resultado son métricas infladas (XGBoost denso llegaba a AUC=0.945) que no se
# sostienen en el test honesto fuera de muestra (AUC=0.820, ver script 03).
#
# Aquí se reemplaza por **`TimeSeriesSplit` con 5 splits**, ordenando los créditos
# por `fecha_desembolso`. Cada fold entrena sobre los meses anteriores y valida
# sobre el bloque siguiente — nunca usa el futuro para predecir el pasado. Es la
# validación honesta para series temporales (Bergmeir & Benítez, 2012; Hyndman &
# Athanasopoulos, 2018).
#
# **Limitación heredada del confound temporal:** el segmento esparso está
# concentrado en los primeros meses de operación (2022). Con validación temporal,
# los primeros folds de test concentran casos esparsos y los últimos casos densos,
# por lo que el N por segmento varía entre folds (se omiten los folds con <20 casos
# de un segmento). Esto es un costo metodológico honesto frente al leakage del
# k-fold barajado.

# %% [markdown]
# ## Setup

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]
TU_ALL = [
    "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
    "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
    "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
]

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = {"xgb": "#4878CF", "logreg": "#6ACC65", "llm": "#D65F5F",
           "esparso": "#E8A838", "denso": "#4878CF"}

N_SPLITS = 5
CORTE_ESPARSO = 6
MIN_SEG_CASOS = 20
SEED = 42


# %% [markdown]
# ## Métricas estándar de credit scoring

# %%
def credit_metrics(y_true, y_prob) -> dict:
    """Calcula AUC, KS, Gini, Brier y PR-AUC para un conjunto de predicciones."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks, "Brier": brier, "PR-AUC": prauc}


# %% [markdown]
# ## Carga de datos y definición de segmentos
#
# El dataset se ordena por `fecha_desembolso` para que `TimeSeriesSplit` respete
# la cronología. El segmento se define por el conteo de variables TU con código -1.

# %%
def load_segmented(dataset_path: str) -> tuple:
    """Carga el dataset ordenado por fecha y agrega columnas de segmentación.

    Retorna (df ordenado, lista de features).
    """
    df = pd.read_csv(dataset_path, parse_dates=["fecha_desembolso"])
    df = df.sort_values("fecha_desembolso").reset_index(drop=True)
    features = [c for c in df.columns if c not in META + TEXT]
    df["n_tu_missing"] = (df[TU_ALL] < 0).sum(axis=1)
    df["segmento"] = (df["n_tu_missing"] >= CORTE_ESPARSO).map({True: "esparso", False: "denso"})
    return df, features


# %% [markdown]
# ## Modelos a comparar

# %%
def make_xgb(models_dir: Path):
    """XGBoost con los mejores hiperparámetros de Optuna (script 03) si existen."""
    metrics_path = models_dir / "xgboost_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved = json.load(f)
        params = dict(saved.get("best_params", {}))
        params.pop("early_stopping_rounds", None)
        params.pop("eval_metric", None)
    else:
        params = {"n_estimators": 374, "max_depth": 4, "learning_rate": 0.106,
                  "subsample": 0.999, "colsample_bytree": 0.819, "min_child_weight": 4}
    params.update({"objective": "binary:logistic", "random_state": SEED,
                   "n_jobs": -1, "verbosity": 0})
    return xgb.XGBClassifier(**params)


def make_logreg():
    """Pipeline imputación → escalado → regresión logística L2."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(C=0.1, max_iter=1000,
                                        solver="lbfgs", random_state=SEED, n_jobs=-1)),
    ])


# %% [markdown]
# ## Validación temporal por segmento (TimeSeriesSplit)
#
# Cada fold entrena con el pasado y valida con el bloque temporal siguiente. Las
# métricas se calculan globales y por segmento (omitiendo segmentos con <20 casos
# en el fold de test, donde la estimación de AUC no sería confiable).

# %%
def run_timeseries_cv(df: pd.DataFrame, features: list, models_dir: Path) -> dict:
    """Ejecuta TimeSeriesSplit y acumula métricas por modelo y segmento."""
    X = df[features].values
    y = df["target"].values
    segmentos = df["segmento"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results_cv = {
        "XGBoost": {"esparso": [], "denso": [], "total": []},
        "LogReg":  {"esparso": [], "denso": [], "total": []},
    }
    fold_composition = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        seg_te = segmentos[test_idx]
        fold_composition.append({"esparso": int((seg_te == "esparso").sum()),
                                 "denso": int((seg_te == "denso").sum())})

        for name, model in [("XGBoost", make_xgb(models_dir)), ("LogReg", make_logreg())]:
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            results_cv[name]["total"].append(credit_metrics(y_te, probs))
            for seg in ("esparso", "denso"):
                mask = seg_te == seg
                if mask.sum() < MIN_SEG_CASOS:
                    continue
                results_cv[name][seg].append(credit_metrics(y_te[mask], probs[mask]))

    results_cv["_fold_composition"] = fold_composition
    return results_cv


# %% [markdown]
# ## Resumen media ± std por segmento

# %%
def summarize(fold_list: list) -> dict:
    """Media y std de cada métrica a través de los folds."""
    keys = fold_list[0].keys()
    return {k: {"mean": float(np.mean([f[k] for f in fold_list])),
                "std":  float(np.std([f[k] for f in fold_list]))}
            for k in keys}


def summarize_cv(results_cv: dict) -> dict:
    """Construye el resumen media±std por modelo y segmento (n folds incluido)."""
    cv_summary = {}
    for model_name in ("XGBoost", "LogReg"):
        cv_summary[model_name] = {}
        for seg in ("esparso", "denso", "total"):
            folds = results_cv[model_name][seg]
            if not folds:
                continue
            cv_summary[model_name][seg] = {**summarize(folds), "n_folds": len(folds)}
    return cv_summary


# %% [markdown]
# ## Figura 1 — Perfil temporal del segmento esparso

# %%
def plot_temporal(df: pd.DataFrame, figs_dir: Path) -> None:
    """Distribución mensual de segmentos y % esparso por mes."""
    tmp = df.copy()
    tmp["mes"] = tmp["fecha_desembolso"].dt.to_period("M")
    temporal = tmp.groupby(["mes", "segmento"]).size().unstack(fill_value=0)
    for seg in ("esparso", "denso"):
        if seg not in temporal.columns:
            temporal[seg] = 0
    temporal["pct_esparso"] = temporal["esparso"] / (temporal["esparso"] + temporal["denso"]) * 100
    temporal["mes_str"] = temporal.index.astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(temporal["mes_str"], temporal["esparso"], color=PALETTE["esparso"], alpha=0.8, label="Esparso")
    ax1.bar(temporal["mes_str"], temporal["denso"], bottom=temporal["esparso"],
            color=PALETTE["denso"], alpha=0.8, label="Denso")
    ax1.set_ylabel("N créditos")
    ax1.set_title("Distribución mensual de segmentos", fontweight="bold")
    ax1.legend()

    ax2.plot(temporal["mes_str"], temporal["pct_esparso"],
             color=PALETTE["esparso"], marker="o", markersize=4, lw=2)
    ax2.axhline(50, color="gray", linestyle="--", lw=1)
    ax2.set_ylabel("% bureau esparso")
    ax2.set_title("% de casos con bureau esparso por mes\n"
                  "(el segmento esparso se concentra en los primeros meses)",
                  fontweight="bold")

    tick_step = max(1, len(temporal) // 10)
    ticks = list(range(0, len(temporal), tick_step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([temporal["mes_str"].iloc[i] for i in ticks], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(figs_dir / "18_segmentos_temporal.png", bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Figura 2 — AUC por segmento con intervalos de confianza

# %%
def plot_auc_segmento(cv_summary: dict, figs_dir: Path) -> None:
    """Barras de AUC media±std por segmento para XGBoost y LogReg."""
    seg_order = ["esparso", "denso", "total"]
    seg_labels = ["Bureau esparso\n(≥6 vars -1)", "Bureau denso\n(<6 vars -1)", "Total"]
    x = np.arange(len(seg_order))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model_name, color) in enumerate([("XGBoost", PALETTE["xgb"]),
                                             ("LogReg", PALETTE["logreg"])]):
        aucs = [cv_summary[model_name].get(s, {}).get("AUC", {}).get("mean", 0) for s in seg_order]
        stds = [cv_summary[model_name].get(s, {}).get("AUC", {}).get("std", 0) for s in seg_order]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, aucs, width, color=color, alpha=0.85,
                      label=model_name, edgecolor="white",
                      yerr=stds, capsize=5, error_kw={"linewidth": 1.5})
        for bar, val, std in zip(bars, aucs, stds):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.008,
                        f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels)
    ax.set_ylabel(f"AUC-ROC (media ± std, {N_SPLITS}-split TimeSeriesSplit)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"AUC-ROC por segmento de bureau — validación temporal ({N_SPLITS} splits)",
                 fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figs_dir / "19_auc_por_segmento_cv.png", bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Pipeline end-to-end (nodo Kedro)
#
# `segment_cv` recibe rutas como argumentos y retorna el resumen de métricas.
# Es importable como módulo: la lógica no depende de estado global.

# %%
def segment_cv(dataset_path: str, output_dir: str) -> dict:
    """Compara XGBoost vs LogReg por segmento con validación temporal.

    Args:
        dataset_path: ruta al CSV con `fecha_desembolso`, `target` y variables TU.
        output_dir: directorio para guardar `cv_seg_metrics.json`.
                    Las figuras se guardan en `<output_dir>/../figures`.

    Returns:
        dict con AUC/Gini/KS media±std por modelo y segmento (más n_folds).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figs_dir = output_path.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df, features = load_segmented(dataset_path)
    results_cv = run_timeseries_cv(df, features, output_path)
    cv_summary = summarize_cv(results_cv)
    cv_summary["_fold_composition"] = results_cv["_fold_composition"]

    with open(output_path / "cv_seg_metrics.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    plot_temporal(df, figs_dir)
    plot_auc_segmento(cv_summary, figs_dir)

    return cv_summary


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    segment_cv(
        dataset_path=str(BASE / "data" / "dataset_tesis.csv"),
        output_dir=str(BASE / "models"),
    )
