# %% [markdown]
# # Ablación de fuentes de variables — XGBoost y Regresión Logística
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Espejo ML de la ablación de fuentes corrida con el LLM (Gemma): entrena XGBoost
# y Regresión Logística sobre subconjuntos de variables ("brazos") para medir qué
# fuente de información (buró TransUnion vs. formulario/negocio) aporta señal.
#
# **Dos familias de brazos:**
# - *Espejo LLM* (`llm_*`): exactamente las variables que vio el LLM en cada brazo
#   de su ablación (TOP_FEATURES_LLM de `06b_llm_prompting.py` + one-hots decodificados).
# - *Máxima información por fuente* (`all_*`): todas las columnas de cada fuente.
#   `all_full` (todo sin leakage) es el control: debe aproximar el baseline oficial
#   sin leakage (XGB AUC test ~0,7515 / RegLog ~0,6605).
#
# **Metodología idéntica al baseline:** split temporal por columna `set`, Optuna
# (TPE, seed 42) maximizando AUC en validación con early stopping para XGBoost;
# pipeline imputación (mediana) → escalado → LogReg para la regresión logística.
# SIEMPRE se excluyen las 5 variables de leakage (datos del crédito otorgado,
# post-aprobación) además de META y TEXT. Además, el brazo `all_sin_tu` excluye
# las TU_DERIVADAS (debts_savings, score_debets, relacion_edad_deuda): variables
# derivadas del buró TransUnion que no están en TU_VARS y contaminarían el brazo
# "sin buró" con señal de buró.
#
# **Nota de reproducibilidad:** la Optuna de este script usa el espacio de búsqueda
# de `03_baseline_xgboost.py`; la corrida ad-hoc que produjo el 0,7515 usó un espacio
# no documentado (best_params fuera del espacio de 03), por lo que el control
# `all_full` reproduce el baseline de forma aproximada, no exacta. Para RegLog se
# replica la config real del baseline sin leakage (`models/baseline_metrics_noleak.json`
# y el pkl: C=1.0, lbfgs, max_iter=3000), que difiere de `04_baseline_logreg.py` (C=0.1).
#
# **Salidas** (en `models/ablacion_ml/`, NO se toca `results/`):
# - `metrics_ablacion.csv` — filas brazo×modelo, AUC/Gini/KS/Brier/PR_AUC/n por
#   segmento (total / esparso / denso).
# - `importancias_<brazo>_xgb.csv` (gain) e `importancias_<brazo>_logreg.csv`
#   (|coef| sobre features estandarizadas).
# - `univariate_auc_test.csv` — AUC univariado en test de cada variable numérica
#   no-leakage (centinelas TU `<0` excluidos del cómputo, con `pct_centinela`).
# - `best_params_xgb_ablacion.json` y `resumen_ablacion.json` — trazabilidad.
#
# Uso:
#     poetry run python scripts/12_ablacion_fuentes_ml.py                       # todo
#     poetry run python scripts/12_ablacion_fuentes_ml.py --arms all_tu all_sin_tu --n-trials 30

# %% [markdown]
# ## Setup — imports dinámicos de 03 / 04 / 06b (mismo patrón que 09c)

# %%
import argparse
import importlib.util
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent


def _load(name: str, alias: str):
    """Carga un script hermano como módulo (sin modificarlo ni ejecutar su main)."""
    spec = importlib.util.spec_from_file_location(alias, BASE / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m03 = _load("03_baseline_xgboost.py", "m03")    # credit_metrics, tune_hyperparams
m04 = _load("04_baseline_logreg.py", "m04")     # build_pipeline
m06b = _load("06b_llm_prompting.py", "m06b")    # TU_VARS, TOP_FEATURES_LLM, load_segmented, evaluate_on_test

# %% [markdown]
# ## Constantes — leakage, grupos de columnas y brazos

# %%
SEED = 42
N_TRIALS_DEFAULT = 50

META = m03.META
TEXT = m03.TEXT

# Variables del crédito OTORGADO (post-aprobación) — leakage confirmado en el
# rebaseline del 2026-07-05 (`models/baseline_metrics_noleak.json`). Se excluyen
# SIEMPRE, en todos los brazos.
LEAK_COLS = [
    "credits_amount_granted",
    "credits_fee_month_amount_granted",
    "credits_interest_amount",
    "credits_surety_bond_amount",
    "credits_digital_instrumentation_amount",
]

# Variables DERIVADAS del buró TransUnion (ver scripts/04_clean_dataset.py, grupo
# DERIVED: "deudas/ahorros (TransUnion)", "score de deudas (TransUnion)", "edad de
# la deuda más antigua"). NO están en m06b.TU_VARS, pero su información proviene
# del buró: su patrón de missing sigue la firma esparso/denso (NaN 20-26% vs 3-9%)
# y tienen señal univariada directa fuerte (debts_savings ~0,62 / score_debets ~0,63).
# Se EXCLUYEN de `all_sin_tu` para no atribuir señal de buró a la fuente
# formulario/negocio; solo participan en `all_full`.
TU_DERIVADAS = ["debts_savings", "score_debets", "relacion_edad_deuda"]

# Grupos one-hot que el LLM vio decodificados en su serialización TabLLM
# (categoría, objetivo del crédito, educación, canal). ALIANZA_* y género NO se
# incluyen en los brazos espejo porque 06b no los serializa para el LLM.
ONEHOT_PREFIXES = ("category_", "credits_credit_goal_", "credits_contact_studies_", "CANAL_")

# Columnas derivadas que agrega `m06b.load_segmented` — nunca son features.
DERIVED = ["n_tu_missing", "segmento"]


def build_arms(columns: list[str]) -> dict[str, list[str]]:
    """Construye los brazos de la ablación a partir de las columnas del dataset.

    Familia espejo LLM (mismas variables que vio Gemma en su ablación):
      - llm_tu:      solo las TU de TOP_FEATURES_LLM (sin categóricas ni texto).
      - llm_tu_sem:  TU de TOP_FEATURES_LLM + one-hots (categoría/objetivo/educación/canal).
      - llm_sin_tu:  numéricas no-TU de TOP_FEATURES_LLM + one-hots.
      - llm_full:    TOP_FEATURES_LLM completas + one-hots.

    Familia máxima información por fuente:
      - all_tu:      las 20 columnas TU del dataset (buró crudo).
      - all_sin_tu:  fuente formulario/negocio pura, sin leakage: numéricas no-TU
                     + todos los one-hots (incluidos ALIANZA_* y género), EXCLUYENDO
                     también TU_DERIVADAS (debts_savings, score_debets,
                     relacion_edad_deuda), que son derivadas del buró aunque no
                     estén en TU_VARS. Por eso all_tu ∪ all_sin_tu ⊂ all_full:
                     las TU_DERIVADAS solo participan en all_full.
      - all_full:    todas las features sin leakage (control ≈ baseline 0,7515/0,6605).
    """
    feats_noleak = [c for c in columns if c not in META + TEXT + LEAK_COLS + DERIVED]
    onehots = [c for c in feats_noleak if c.startswith(ONEHOT_PREFIXES)]
    llm_tu = [f for f in m06b.TOP_FEATURES_LLM if f in m06b.TU_VARS]
    llm_no_tu = [f for f in m06b.TOP_FEATURES_LLM if f not in m06b.TU_VARS]

    arms = {
        "llm_tu":     list(llm_tu),
        "llm_tu_sem": llm_tu + onehots,
        "llm_sin_tu": llm_no_tu + onehots,
        "llm_full":   list(m06b.TOP_FEATURES_LLM) + onehots,
        "all_tu":     [c for c in feats_noleak if c in m06b.TU_VARS],
        "all_sin_tu": [c for c in feats_noleak if c not in m06b.TU_VARS + TU_DERIVADAS],
        "all_full":   feats_noleak,
    }

    # Sanidad: las TU_DERIVADAS deben existir en el dataset (si se renombran en el
    # pipeline, la exclusión sería un no-op silencioso y volvería la contaminación).
    tu_der_faltantes = [c for c in TU_DERIVADAS if c not in columns]
    if tu_der_faltantes:
        raise ValueError(f"TU_DERIVADAS no encontradas en el dataset: {tu_der_faltantes}")

    # Sanidad: ninguna columna de leakage/meta/texto en ningún brazo, y todas existen.
    for name, cols in arms.items():
        faltantes = [c for c in cols if c not in columns]
        if faltantes:
            raise ValueError(f"Brazo {name}: columnas inexistentes en el dataset: {faltantes}")
        prohibidas = [c for c in cols if c in META + TEXT + LEAK_COLS + DERIVED]
        if prohibidas:
            raise ValueError(f"Brazo {name}: contiene columnas prohibidas: {prohibidas}")
    return arms


# %% [markdown]
# ## Splits por brazo y entrenamiento (misma metodología que el baseline)

# %%
def make_splits(df: pd.DataFrame, cols: list[str]) -> dict:
    """Arma el dict de splits (train/val/test) restringido a las columnas del brazo.

    Mismo formato que `m03.load_splits`, para poder reutilizar `m03.tune_hyperparams`.
    """
    splits = {}
    for name in ("train", "val", "test"):
        subset = df[df["set"] == name]
        splits[name] = {"X": subset[cols], "y": subset["target"]}
    splits["features"] = list(cols)
    return splits


def train_xgb_arm(splits: dict, n_trials: int, seed: int = SEED):
    """Optuna (TPE seed fijo) + fit final con early stopping en validación.

    Reutiliza `m03.tune_hyperparams` → mismo espacio de búsqueda y determinismo
    que `03_baseline_xgboost.py`.
    """
    best_params = m03.tune_hyperparams(splits, n_trials=n_trials, seed=seed)
    model = xgb.XGBClassifier(**best_params, early_stopping_rounds=30, verbosity=0)
    model.fit(
        splits["train"]["X"], splits["train"]["y"],
        eval_set=[(splits["val"]["X"], splits["val"]["y"])],
        verbose=False,
    )
    return model, best_params


def train_logreg_arm(splits: dict, seed: int = SEED):
    """Pipeline de `04_baseline_logreg.py` con la config del baseline sin leakage.

    Se reutiliza `m04.build_pipeline` (imputer mediana → scaler → logreg) y se
    ajustan C=1.0 y max_iter=3000 para replicar la config real del baseline
    oficial sin leakage (AUC test 0,6605; ver `models/baseline_metrics_noleak.json`
    y `models/logreg_baseline.pkl`), que difiere del C=0.1 de scripts/04.
    """
    pipeline = m04.build_pipeline(seed)
    pipeline.set_params(model__C=1.0, model__max_iter=3000)
    pipeline.fit(splits["train"]["X"], splits["train"]["y"])
    return pipeline


# %% [markdown]
# ## Evaluación en test por segmento (esparso / denso)
#
# Reutiliza `m06b.load_segmented` (esparso = ≥6 vars TU `<0`) y
# `m06b.evaluate_on_test` — mismo criterio y métricas que la ablación LLM.

# %%
def evaluate_arm(df: pd.DataFrame, cols: list[str], predict_proba_fn) -> dict:
    """Evalúa un modelo del brazo sobre el test, total y por segmento.

    `predict_proba_fn` recibe el X de test (DataFrame con `cols`) y devuelve la
    probabilidad de la clase 1. Se arma un vector alineado a `df` (NaN fuera de
    test) para reutilizar `m06b.evaluate_on_test` sin cambios.
    """
    probs = np.full(len(df), np.nan)
    test_mask = (df["set"] == "test").values
    probs[test_mask] = predict_proba_fn(df.loc[test_mask, cols])
    return m06b.evaluate_on_test(df, probs)


def flatten_summary(brazo: str, modelo: str, n_features: int, summary: dict) -> dict:
    """Aplana el dict de `evaluate_on_test` a una fila de la tabla de métricas."""
    row = {"brazo": brazo, "modelo": modelo, "n_features": n_features}
    for seg in ("total", "esparso", "denso"):
        seg_metrics = summary.get(seg, {})
        for met in ("AUC", "Gini", "KS", "Brier", "PR_AUC"):
            row[f"{met}_{seg}"] = seg_metrics.get(met, np.nan)
        row[f"n_{seg}"] = seg_metrics.get("n", 0)
    return row


# %% [markdown]
# ## Importancias por brazo

# %%
def save_xgb_importances(model, cols: list[str], path: Path) -> None:
    """Importancias por gain del booster (0 para features no usadas por ningún árbol)."""
    scores = model.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({
        "feature": cols,
        "gain": [scores.get(c, 0.0) for c in cols],
    }).sort_values("gain", ascending=False)
    imp.to_csv(path, index=False)


def save_logreg_importances(pipeline, cols: list[str], path: Path) -> None:
    """Coeficientes (log-odds sobre features estandarizadas) ordenados por |coef|."""
    coefs = pipeline.named_steps["model"].coef_[0]
    imp = pd.DataFrame({"feature": cols, "coef": coefs, "abs_coef": np.abs(coefs)})
    imp.sort_values("abs_coef", ascending=False).to_csv(path, index=False)


# %% [markdown]
# ## AUC univariado en test
#
# Para cada variable numérica no-leakage: AUC de la variable cruda como score.
# En las TU se excluyen del cómputo los valores centinela (`<0` = sin información),
# y se reporta `pct_centinela` (proporción de test con centinela). Un AUC < 0,5
# marca señal invertida (el mecanismo del sesgo de selección, p. ej. g051s ~0,19).

# %%
def univariate_auc_test(df: pd.DataFrame, feats: list[str], min_casos: int = 20) -> pd.DataFrame:
    """AUC univariado en test por variable, con manejo de centinelas TU."""
    test = df[df["set"] == "test"]
    y = test["target"].values.astype(int)
    rows = []
    for col in feats:
        x = test[col].astype(float).values
        es_tu = col in m06b.TU_VARS
        valid = ~np.isnan(x)
        pct_centinela = np.nan
        if es_tu:
            pct_centinela = float((x < 0).mean())
            valid &= x >= 0
        n_validos = int(valid.sum())
        y_v = y[valid]
        if n_validos >= min_casos and len(np.unique(y_v)) == 2:
            auc = float(roc_auc_score(y_v, x[valid]))
        else:
            auc = np.nan
        rows.append({
            "variable": col,
            "es_tu": es_tu,
            "es_tu_derivada": col in TU_DERIVADAS,
            "auc_test": auc,
            "n_validos": n_validos,
            "n_test": int(len(test)),
            "pct_centinela": pct_centinela,
        })
    tabla = pd.DataFrame(rows)
    # Orden por fuerza de señal en cualquier dirección (|AUC - 0,5| descendente)
    return tabla.reindex(
        (tabla["auc_test"] - 0.5).abs().sort_values(ascending=False, na_position="last").index
    )


# %% [markdown]
# ## Orquestación end-to-end

# %%
def _merge_metrics(out_dir: Path, new_rows: list[dict]) -> pd.DataFrame:
    """Fusiona filas nuevas con `metrics_ablacion.csv` existente (clave brazo×modelo).

    Permite correr subconjuntos de brazos (`--arms`) sin pisar resultados previos.
    """
    nuevos = pd.DataFrame(new_rows)
    csv_path = out_dir / "metrics_ablacion.csv"
    if csv_path.exists():
        previos = pd.read_csv(csv_path)
        claves = set(zip(nuevos["brazo"], nuevos["modelo"]))
        previos = previos[~previos.apply(lambda r: (r["brazo"], r["modelo"]) in claves, axis=1)]
        nuevos = pd.concat([previos, nuevos], ignore_index=True)
    return nuevos.sort_values(["brazo", "modelo"]).reset_index(drop=True)


def run_ablacion(dataset_path: str, out_dir: str, n_trials: int = N_TRIALS_DEFAULT,
                 arms_sel: list[str] | None = None, seed: int = SEED) -> pd.DataFrame:
    """Corre la ablación completa (o un subconjunto de brazos) y persiste artefactos.

    Args:
        dataset_path: CSV con columnas `set`, `target`, `fecha_desembolso`.
        out_dir: directorio de salida (models/ablacion_ml — NO results/).
        n_trials: trials de Optuna por brazo para XGBoost.
        arms_sel: brazos a correr (None = todos).
        seed: semilla global (Optuna TPE + modelos).

    Returns:
        DataFrame con la tabla de métricas (una fila por brazo×modelo).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = m06b.load_segmented(dataset_path)   # agrega n_tu_missing y segmento
    arms = build_arms(list(df.columns))
    arms_sel = arms_sel or list(arms)
    desconocidos = [a for a in arms_sel if a not in arms]
    if desconocidos:
        raise ValueError(f"Brazos desconocidos: {desconocidos}. Válidos: {list(arms)}")

    n_test = int((df["set"] == "test").sum())
    print(f"Dataset: {len(df)} filas | test={n_test} | brazos: {arms_sel} | n_trials={n_trials}",
          flush=True)

    # AUC univariado (independiente de los brazos, siempre se regenera)
    feats_noleak = arms["all_full"]
    univariate_auc_test(df, feats_noleak).to_csv(out_path / "univariate_auc_test.csv", index=False)
    print(f"univariate_auc_test.csv → {len(feats_noleak)} variables", flush=True)

    rows = []
    best_params_all = {}
    for name in arms_sel:
        cols = arms[name]
        splits = make_splits(df, cols)
        print(f"[{name}] {len(cols)} features — XGBoost (Optuna {n_trials} trials)...", flush=True)

        model_xgb, best_params = train_xgb_arm(splits, n_trials=n_trials, seed=seed)
        best_params_all[name] = best_params
        summary_xgb = evaluate_arm(df, cols, lambda X, m=model_xgb: m.predict_proba(X)[:, 1])
        rows.append(flatten_summary(name, "xgboost", len(cols), summary_xgb))
        save_xgb_importances(model_xgb, cols, out_path / f"importancias_{name}_xgb.csv")
        print(f"[{name}] XGB AUC test total={summary_xgb['total']['AUC']:.4f}", flush=True)

        print(f"[{name}] Regresión Logística...", flush=True)
        pipe = train_logreg_arm(splits, seed=seed)
        summary_lr = evaluate_arm(df, cols, lambda X, p=pipe: p.predict_proba(X)[:, 1])
        rows.append(flatten_summary(name, "logreg", len(cols), summary_lr))
        save_logreg_importances(pipe, cols, out_path / f"importancias_{name}_logreg.csv")
        print(f"[{name}] LogReg AUC test total={summary_lr['total']['AUC']:.4f}", flush=True)

    tabla = _merge_metrics(out_path, rows)
    tabla.to_csv(out_path / "metrics_ablacion.csv", index=False)

    with open(out_path / "best_params_xgb_ablacion.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "n_filas": int(len(df)),
        "n_test": n_test,
        "seed": seed,
        "n_trials": n_trials,
        "leakage_excluido": LEAK_COLS,
        "tu_derivadas_excluidas_de_all_sin_tu": TU_DERIVADAS,
        "criterio_esparso": f">= {m06b.CORTE_ESPARSO} vars TU < 0",
        "brazos": {name: {"n_features": len(arms[name])} for name in arms_sel},
        "logreg_config": "C=1.0, lbfgs, max_iter=3000 (config del baseline noleak, no la de scripts/04)",
    }
    with open(out_path / "resumen_ablacion.json", "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Listo. Artefactos en {out_path}", flush=True)
    return tabla


# %% [markdown]
# ## CLI / entry point

# %%
def main(argv: list[str] | None = None) -> None:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description="Ablación de fuentes de variables (XGBoost + RegLog) — espejo de la ablación LLM."
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT,
                        help=f"Trials de Optuna por brazo para XGBoost (default {N_TRIALS_DEFAULT}).")
    parser.add_argument("--arms", nargs="+", default=None,
                        choices=["llm_tu", "llm_tu_sem", "llm_sin_tu", "llm_full",
                                 "all_tu", "all_sin_tu", "all_full"],
                        help="Brazos a correr (default: todos).")
    parser.add_argument("--dataset", default=str(BASE / "data" / "dataset_tesis.csv"),
                        help="Ruta al CSV del dataset.")
    parser.add_argument("--out", default=str(BASE / "models" / "ablacion_ml"),
                        help="Directorio de salida (default models/ablacion_ml).")
    args = parser.parse_args(argv)

    run_ablacion(
        dataset_path=args.dataset,
        out_dir=args.out,
        n_trials=args.n_trials,
        arms_sel=args.arms,
    )


if __name__ == "__main__":
    main()
