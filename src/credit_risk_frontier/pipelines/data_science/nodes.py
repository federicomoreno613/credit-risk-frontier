"""Nodos de modelado clásico + TabFM (huella spaceflights: data_science).

Un único template ``split → train → evaluate`` instanciado por namespace (patrón
active/candidate). El estimador se elige por ``params:model_options['estimator']``
(``xgb`` | ``logreg`` | ``tabfm``) y las features por ``feature_arm``. Todos los
estimadores emiten el mismo dict de métricas, así una instancia = 1 stanza YAML.

Los helpers de modelado (Optuna, pipeline sklearn, carga lazy de TabFM) son de un
solo dueño → viven acá. La segmentación, splits y métricas vienen de ``utils``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from credit_risk_frontier import utils

warnings.filterwarnings("ignore")

# Espacio de búsqueda Optuna del baseline XGBoost (fuente: 03_baseline_xgboost.tune_hyperparams).
_XGB_SEARCH = {
    "n_estimators":     (200, 800),
    "max_depth":        (3, 7),
    "learning_rate":    (0.01, 0.2),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "reg_alpha":        (1e-4, 10.0),
    "reg_lambda":       (1e-4, 10.0),
}


# ---------------------------------------------------------------------------
# Arms de features (fuente: 12_ablacion_fuentes_ml.build_arms)
# ---------------------------------------------------------------------------

def _feature_arm(df: pd.DataFrame, arm: str) -> list[str]:
    """Columnas de features para el ``feature_arm`` pedido.

    * ``noleak``  → baseline citable (excluye META+TEXT+LEAK_COLS+TU_DERIVADAS+DERIVED).
    * ``leak``    → baseline con filtración de los scripts 03/04 (solo excluye META+TEXT).
    * ``all_tu`` / ``all_sin_tu`` / ``all_full`` / ``llm_*`` → brazos de la ablación ML.
    """
    if arm == "leak":
        return utils.feature_columns(df, exclude_leak=False)
    if arm == "noleak":
        return utils.feature_columns(df, exclude_leak=True)

    # Brazos de ablación de fuentes (espejo ML de la ablación LLM).
    feats_noleak = utils.feature_columns(df, exclude_leak=True)
    onehots = [c for c in feats_noleak if c.startswith(utils.ONEHOT_PREFIXES)]
    llm_tu = [f for f in utils.TOP_FEATURES_LLM if f in utils.TU_VARS]
    llm_no_tu = [f for f in utils.TOP_FEATURES_LLM if f not in utils.TU_VARS]
    arms = {
        "llm_tu":     list(llm_tu),
        "llm_tu_sem": llm_tu + onehots,
        "llm_sin_tu": llm_no_tu + onehots,
        "llm_full":   list(utils.TOP_FEATURES_LLM) + onehots,
        "all_tu":     [c for c in feats_noleak if c in utils.TU_VARS],
        "all_sin_tu": [c for c in feats_noleak if c not in utils.TU_VARS + utils.TU_DERIVADAS],
        "all_full":   feats_noleak,
    }
    if arm not in arms:
        raise ValueError(f"feature_arm desconocido: {arm!r}. Válidos: noleak, leak, {list(arms)}")
    return arms[arm]


# ---------------------------------------------------------------------------
# Nodo 1: split temporal por feature_arm
# ---------------------------------------------------------------------------

def split_by_feature_arm(model_input_table: pd.DataFrame, model_options: dict):
    """Split temporal train/val/test restringido a las features del ``feature_arm``.

    Devuelve X/y por split (tuplas) + la lista de features, para que el template
    sea genérico. El orden de retorno alimenta ``train_estimator``/``evaluate_estimator``.
    """
    arm = model_options.get("feature_arm", "noleak")
    features = _feature_arm(model_input_table, arm)
    splits = utils.load_splits(model_input_table, features)
    return (
        splits["train"]["X"], splits["train"]["y"],
        splits["val"]["X"], splits["val"]["y"],
        splits["test"]["X"], splits["test"]["y"],
        features,
    )


# ---------------------------------------------------------------------------
# Nodo 2: entrenamiento (dispatch por estimator)
# ---------------------------------------------------------------------------

def _tune_xgb(X_train, y_train, X_val, y_val, n_trials: int, seed: int) -> dict:
    """Optuna TPE maximizando AUC en validación (fuente: 03.tune_hyperparams)."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", *_XGB_SEARCH["n_estimators"]),
            "max_depth":        trial.suggest_int("max_depth", *_XGB_SEARCH["max_depth"]),
            "learning_rate":    trial.suggest_float("learning_rate", *_XGB_SEARCH["learning_rate"], log=True),
            "subsample":        trial.suggest_float("subsample", *_XGB_SEARCH["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *_XGB_SEARCH["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *_XGB_SEARCH["min_child_weight"]),
            "reg_alpha":        trial.suggest_float("reg_alpha", *_XGB_SEARCH["reg_alpha"], log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", *_XGB_SEARCH["reg_lambda"], log=True),
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
    best = dict(study.best_params)
    best.update({"objective": "binary:logistic", "eval_metric": "auc",
                 "random_state": seed, "n_jobs": -1})
    return best


def _train_xgb(X_train, y_train, X_val, y_val, model_options: dict):
    import xgboost as xgb
    seed = model_options.get("seed", utils.SEED)
    n_trials = model_options.get("n_trials", 50)
    best = _tune_xgb(X_train, y_train, X_val, y_val, n_trials, seed)
    model = xgb.XGBClassifier(**best, early_stopping_rounds=30, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    model._best_params = best  # se persiste en las métricas
    return model


def _train_logreg(X_train, y_train, model_options: dict):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seed = model_options.get("seed", utils.SEED)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=model_options.get("C", 1.0),
            max_iter=model_options.get("max_iter", 3000),
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _train_tabfm(X_train, y_train, model_options: dict):
    """TabFM (Google foundation model tabular). Import lazy: solo con el extra `tabfoundation`.

    API sklearn: ``fit`` registra el contexto (in-context learning), ``predict_proba``
    puntúa. Re-corrible como XGBoost, no requiere cache. Licencia de pesos non-commercial.
    """
    try:
        from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    except ImportError as exc:  # pragma: no cover - depende de un extra opcional
        raise ImportError(
            "TabFM no está instalado. Instalar el extra opcional: "
            "poetry install --extras tabfoundation (o pip install 'tabfm[pytorch]')."
        ) from exc
    base = tabfm_v1_0_0.load(model_type="classification")
    clf = TabFMClassifier(model=base)
    clf.fit(X_train, y_train)
    return clf


def train_estimator(X_train, y_train, X_val, y_val, model_options: dict):
    """Entrena el estimador elegido por ``model_options['estimator']`` y lo devuelve."""
    estimator = model_options.get("estimator", "xgb")
    if estimator == "xgb":
        return _train_xgb(X_train, y_train, X_val, y_val, model_options)
    if estimator == "logreg":
        return _train_logreg(X_train, y_train, model_options)
    if estimator == "tabfm":
        return _train_tabfm(X_train, y_train, model_options)
    raise ValueError(f"estimator desconocido: {estimator!r}. Válidos: xgb, logreg, tabfm")


# ---------------------------------------------------------------------------
# Nodo 3: evaluación → dict de métricas por split (misma forma que los JSON E4)
# ---------------------------------------------------------------------------

def evaluate_estimator(classifier, X_train, y_train, X_val, y_val, X_test, y_test,
                       model_options: dict) -> dict:
    """Métricas AUC/Gini/KS/Brier/PR_AUC en train/val/test (fuente: 03/04.train_*).

    Retorna ``{model, train, val, test, best_params?}``, la misma estructura que los
    ``models/*_metrics.json`` committeados (con la clave canónica ``PR_AUC``).
    """
    estimator = model_options.get("estimator", "xgb")
    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    metrics = {}
    for name, (X, y) in splits.items():
        prob = classifier.predict_proba(X)[:, 1]
        metrics[name] = utils.credit_metrics(np.asarray(y), prob)

    results = {"model": estimator, **metrics}
    best_params = getattr(classifier, "_best_params", None)
    if best_params is not None:
        results["best_params"] = best_params
    return utils.json_safe(results)


# ---------------------------------------------------------------------------
# Nodo aparte: CV temporal por segmento (fuente: 05_segmentacion_thin_file)
# ---------------------------------------------------------------------------

def _summarize_folds(fold_list: list) -> dict:
    """Media y std de cada métrica a través de los folds (fuente: 05.summarize)."""
    keys = fold_list[0].keys()
    return {k: {"mean": float(np.mean([f[k] for f in fold_list])),
                "std": float(np.std([f[k] for f in fold_list]))}
            for k in keys}


def run_segment_cv(model_input_table: pd.DataFrame, xgb_metrics: dict, params: dict) -> dict:
    """TimeSeriesSplit sobre XGBoost/LogReg por segmento (fuente: 05_segmentacion_thin_file).

    Reproduce fielmente ``cv_seg_metrics.json``: usa TODAS las features (META+TEXT
    excluidas, CON crédito otorgado, igual que el script 05), los ``best_params`` de
    XGBoost provenientes de las métricas ya calculadas (input del catálogo, no lectura
    de disco) y LogReg con C=0.1/max_iter=1000. Estructura de salida:
    ``{XGBoost, LogReg: {esparso/denso/total: {métrica: {mean,std}, n_folds}}, _fold_composition}``.
    """
    import xgboost as xgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seed = params.get("seed", utils.SEED)
    n_splits = params.get("n_splits", 5)
    min_seg = params.get("min_seg_casos", utils.MIN_SEG_CASOS)

    # El script 05 usa el feature set CON leakage (solo excluye META+TEXT) y los
    # best_params de xgboost_metrics — históricamente el baseline con leakage.
    df = model_input_table.sort_values("fecha_desembolso", kind="stable").reset_index(drop=True)
    features = [c for c in df.columns if c not in set(utils.META + utils.TEXT + utils.DERIVED)]
    X = df[features].values
    y = df["target"].values
    segmentos = df["segmento"].values

    xgb_params = dict(xgb_metrics.get("best_params", {}))
    xgb_params.pop("early_stopping_rounds", None)
    xgb_params.pop("eval_metric", None)
    xgb_params.update({"objective": "binary:logistic", "random_state": seed,
                       "n_jobs": -1, "verbosity": 0})

    def _make_xgb():
        return xgb.XGBClassifier(**xgb_params)

    def _make_logreg():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                         random_state=seed, n_jobs=-1)),
        ])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"XGBoost": {"esparso": [], "denso": [], "total": []},
               "LogReg": {"esparso": [], "denso": [], "total": []}}
    fold_composition = []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        seg_te = segmentos[te_idx]
        fold_composition.append({"esparso": int((seg_te == "esparso").sum()),
                                 "denso": int((seg_te == "denso").sum())})
        for name, model in [("XGBoost", _make_xgb()), ("LogReg", _make_logreg())]:
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            results[name]["total"].append(utils.credit_metrics(y_te, probs))
            for seg in ("esparso", "denso"):
                mask = seg_te == seg
                if mask.sum() >= min_seg:
                    results[name][seg].append(utils.credit_metrics(y_te[mask], probs[mask]))

    summary = {}
    for model_name in ("XGBoost", "LogReg"):
        summary[model_name] = {}
        for seg in ("esparso", "denso", "total"):
            folds = results[model_name][seg]
            if folds:
                summary[model_name][seg] = {**_summarize_folds(folds), "n_folds": len(folds)}
    summary["_fold_composition"] = fold_composition
    return utils.json_safe(summary)
