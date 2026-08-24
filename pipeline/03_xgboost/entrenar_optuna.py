"""Modelo #2 del PLAN §2.2 — XGBoost con búsqueda de hiperparámetros Optuna.

XGBoost es el estado del arte práctico en datos tabulares (Lessmann et al.,
2015; Chen & Guestrin, 2016) y el rival a vencer del comparativo. La búsqueda
usa Optuna (Akiba et al., 2019) con sampler TPE y objetivo = AUC de
VALIDACIÓN; el test no participa en ninguna decisión. El PLAN prevé 500
trials; el default acá es 50 para iterar (subir con --trials 500).

Uso:
  poetry run python pipeline/03_xgboost/entrenar_optuna.py --trials 50
  poetry run python pipeline/03_xgboost/entrenar_optuna.py --trials 500 --timeout 3600

Salidas:
  data/pipeline/modelos/xgboost.joblib
  data/pipeline/modelos/xgboost_trials.csv    todas las pruebas del estudio
  data/pipeline/modelos/xgboost_info.json     mejores params + AUC val
  data/pipeline/predicciones/xgboost.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402
import monitoreo  # noqa: E402


def construir_modelo(params: dict, balance: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=30,
        eval_metric="auc",
        scale_pos_weight=balance,
        random_state=C.SEED,
        verbosity=0,
        **params,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="tope en segundos")
    args = parser.parse_args()

    variables = C.cargar_variables()
    train, val, test = C.dividir(variables)
    X_train, y_train = C.preparar_numerico(train), train["target"].astype(int)
    X_val, y_val = C.preparar_numerico(val), val["target"].astype(int)
    balance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        modelo = construir_modelo(params, balance)
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
    )
    estudio.optimize(objetivo, n_trials=args.trials, timeout=args.timeout)

    mejor = construir_modelo(estudio.best_params, balance)
    mejor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    C.MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor, C.MODELOS / "xgboost.joblib")
    estudio.trials_dataframe().to_csv(C.MODELOS / "xgboost_trials.csv", index=False)
    C.guardar_json(C.MODELOS / "xgboost_info.json", {
        "auc_val": estudio.best_value,
        "trials": len(estudio.trials),
        "mejores_params": estudio.best_params,
        "mejor_iteracion": int(mejor.best_iteration),
        "scale_pos_weight": balance,
    })
    print(f"xgboost AUC val: {estudio.best_value:.4f} "
          f"({len(estudio.trials)} trials, mejores: {estudio.best_params})")

    probabilidades = mejor.predict_proba(C.preparar_numerico(test))[:, 1]
    predicciones = pd.DataFrame({
        "modelo": "xgboost",
        "configuracion": "29vars_optuna",
        "credito_id_anon": test["credito_id_anon"].astype(str),
        "segmento": test["segmento"].astype(str),
        "y_true": test["target"].astype(int),
        "probabilidad": probabilidades,
        "valida": True,
    })
    path = C.guardar_predicciones("xgboost", predicciones)
    monitoreo.reporte_modelo(predicciones, "XGBoost + Optuna (test)")
    print(f"predicciones -> {path}")


if __name__ == "__main__":
    main()
