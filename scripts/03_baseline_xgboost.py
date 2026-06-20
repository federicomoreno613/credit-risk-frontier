# %% [markdown]
# # Baseline XGBoost — Credit Scoring
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Entrena XGBoost sobre las 94 variables numéricas/one-hot del dataset.
# Reporta métricas estándar de la industria del crédito:
# AUC-ROC, KS, Gini, Brier Score, PR-AUC.
# Genera SHAP values y curva de calibración para análisis interpretativo.
#
# **Validación temporal (no StratifiedKFold).** El split train/val/test viene
# dado por la columna `set`, que respeta el orden de `fecha_desembolso`. Por la
# naturaleza temporal de los datos, mezclar fechas (como haría un k-fold
# estratificado) genera *data leakage* del futuro hacia el pasado e infla las
# métricas. Por eso se reporta el AUC de **train, val y test por separado**: el
# gap train→test es el indicador honesto de sobreajuste.
#
# **Referencias:**
# - Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System
# - Siddiqi (2006) — Credit Risk Scorecards (KS y Gini como métricas estándar)

# %% [markdown]
# ## Setup

# %%
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]

N_TRIALS = 50
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
# ## Optimización de hiperparámetros con Optuna
#
# El objective optimiza AUC sobre el conjunto de **validación**, usando *early
# stopping* contra el mismo. No se mezclan fechas: train siempre precede a val.

# %%
def tune_hyperparams(splits: dict, n_trials: int = N_TRIALS, seed: int = SEED) -> dict:
    """Busca hiperparámetros maximizando AUC en validación (Optuna TPE)."""
    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "random_state":     seed,
            "n_jobs":           -1,
        }
        model = xgb.XGBClassifier(**params, early_stopping_rounds=30, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({"objective": "binary:logistic", "eval_metric": "auc",
                        "random_state": seed, "n_jobs": -1})
    return best_params


# %% [markdown]
# ## Figuras de diagnóstico

# %%
def plot_roc(splits, probs, metrics, figs_dir: Path) -> None:
    """Curva ROC para validación y test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr_v, tpr_v, _ = roc_curve(splits["val"]["y"], probs["val"])
    fpr_t, tpr_t, _ = roc_curve(splits["test"]["y"], probs["test"])
    ax.plot(fpr_v, tpr_v, color=PALETTE["xgb"], lw=2,
            label=f"Validación (AUC={metrics['val']['AUC']:.3f})")
    ax.plot(fpr_t, tpr_t, color=PALETTE["xgb"], lw=2, linestyle="--",
            label=f"Test (AUC={metrics['test']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — XGBoost Baseline", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "11_roc_xgboost.png", bbox_inches="tight")
    plt.close(fig)


# %%
def plot_ks_panel(splits, probs, metrics, figs_dir: Path) -> None:
    """Curvas KS (distribución acumulada de buenos y malos) para val y test."""
    def _plot_ks(y_true, y_prob, ax, title):
        df_ks = pd.DataFrame({"prob": y_prob, "target": y_true.values}).sort_values("prob")
        n_bad = y_true.sum()
        n_good = len(y_true) - n_bad
        df_ks["cum_bad"] = (df_ks["target"] == 1).cumsum() / n_bad
        df_ks["cum_good"] = (df_ks["target"] == 0).cumsum() / n_good
        df_ks["ks"] = df_ks["cum_bad"] - df_ks["cum_good"]
        ks_idx = df_ks["ks"].abs().idxmax()
        ax.plot(df_ks["prob"], df_ks["cum_bad"], color="#D65F5F", label="Malos (default=1)")
        ax.plot(df_ks["prob"], df_ks["cum_good"], color="#4878CF", label="Buenos (default=0)")
        ax.axvline(df_ks.loc[ks_idx, "prob"], color="gray", linestyle="--", lw=1.5,
                   label=f"KS = {df_ks['ks'].abs().max():.3f}")
        ax.set_xlabel("Score de probabilidad")
        ax.set_ylabel("Distribución acumulada")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_ks(splits["val"]["y"], probs["val"], ax1, f"KS — Validación (KS={metrics['val']['KS']:.3f})")
    _plot_ks(splits["test"]["y"], probs["test"], ax2, f"KS — Test (KS={metrics['test']['KS']:.3f})")
    fig.suptitle("Curva KS — XGBoost Baseline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs_dir / "12_ks_xgboost.png", bbox_inches="tight")
    plt.close(fig)


# %%
def plot_calibration(splits, probs, figs_dir: Path) -> None:
    """Curva de calibración sobre el conjunto de test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(splits["test"]["y"], probs["test"], n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["xgb"], lw=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media del bin)")
    ax.set_ylabel("Fracción de positivos observados")
    ax.set_title("Calibración — XGBoost Baseline", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "13_calibracion_xgboost.png", bbox_inches="tight")
    plt.close(fig)


# %%
def plot_shap(model, splits, figs_dir: Path) -> bool:
    """SHAP summary plot de las top 20 variables sobre test.

    `shap` es una dependencia opcional (solo interpretabilidad). Si no está
    instalada, se omite la figura sin interrumpir el entrenamiento.
    Retorna True si la figura se generó.
    """
    try:
        import shap
    except ImportError:
        return False
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(splits["test"]["X"])
    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, splits["test"]["X"], max_display=20,
                      show=False, plot_size=None)
    plt.gca().set_title("SHAP Values — Top 20 variables (XGBoost)", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(figs_dir / "14_shap_xgboost.png", bbox_inches="tight")
    plt.close(fig)
    return True


# %% [markdown]
# ## Entrenamiento end-to-end (nodo Kedro)
#
# `train_xgboost` recibe rutas como argumentos y retorna un dict de métricas.
# Es importable como módulo: la lógica no depende de estado global.

# %%
def train_xgboost(dataset_path: str, output_dir: str) -> dict:
    """Entrena XGBoost con validación temporal y persiste modelo, métricas y figuras.

    Args:
        dataset_path: ruta al CSV con columnas `set`, `target`, `fecha_desembolso`.
        output_dir: directorio para guardar modelo y `xgboost_metrics.json`.
                    Las figuras se guardan en `<output_dir>/../figures`.

    Returns:
        dict con métricas de train/val/test y mejores hiperparámetros.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figs_dir = output_path.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(dataset_path)
    best_params = tune_hyperparams(splits)

    model = xgb.XGBClassifier(**best_params, early_stopping_rounds=30, verbosity=0)
    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        eval_set=[(splits["val"]["X"], splits["val"]["y"])],
        verbose=False,
    )
    model.save_model(str(output_path / "xgboost_baseline.json"))
    joblib.dump(model, output_path / "xgboost_baseline.pkl")

    probs = {name: model.predict_proba(splits[name]["X"])[:, 1]
             for name in ("train", "val", "test")}
    metrics = {name: credit_metrics(splits[name]["y"], probs[name])
               for name in ("train", "val", "test")}

    results = {"model": "xgboost", **metrics, "best_params": best_params}
    with open(output_path / "xgboost_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_roc(splits, probs, metrics, figs_dir)
    plot_ks_panel(splits, probs, metrics, figs_dir)
    plot_calibration(splits, probs, figs_dir)
    plot_shap(model, splits, figs_dir)

    return results


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    train_xgboost(
        dataset_path=str(BASE / "data" / "dataset_tesis.csv"),
        output_dir=str(BASE / "models"),
    )
