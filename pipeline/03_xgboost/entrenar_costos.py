"""XGBoost optimizado por COSTO monetario (comparación con la variante por AUC).

entrenar_optuna.py busca hiperparámetros maximizando el AUC de validación.
Acá se repite la búsqueda con el MISMO espacio, pero el objetivo de cada trial
es el costo monetario mínimo en validación (evaluado en el umbral óptimo del
propio trial). El test no participa en ninguna decisión.

Función de costos monetaria
---------------------------
El dataset NO tiene monto prestado ni tasa del crédito (las condiciones
finales del crédito se excluyeron por fuga de información), así que se
parametrizan con defaults razonables de microcréditos:

  costo_FN = monto_medio * LGD          aprobar un moroso: pérdida esperada
                                        del capital (LGD = severidad, default 1.0
                                        = se pierde todo el capital, conservador)
  costo_FP = monto_medio * tasa_interes aprobar de menos: margen perdido por
                                        rechazar a un buen pagador

Con los defaults (monto USD 500, tasa 20% del ciclo) la razón FN:FP es 5:1,
consistente con C.COSTOS del contrato.

Uso:
  poetry run python pipeline/03_xgboost/entrenar_costos.py --trials 30
  poetry run python pipeline/03_xgboost/entrenar_costos.py --trials 200 \
      --monto-medio 800 --tasa-interes 0.25

Salidas:
  data/pipeline/modelos/xgboost_costos.joblib
  data/pipeline/modelos/xgboost_costos_info.json
  data/pipeline/predicciones/xgboost_costos.parquet  (configuracion=29vars_costos)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402
import monitoreo  # noqa: E402

from entrenar_optuna import construir_modelo  # noqa: E402  (mismo constructor)


def costo_monetario(
    y_true: np.ndarray, prob: np.ndarray, umbral: float,
    costo_fn: float, costo_fp: float,
) -> float:
    """Costo total en dinero: FN pierden capital, FP pierden margen."""
    pred = (prob >= umbral).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return fn * costo_fn + fp * costo_fp


def umbral_y_costo_minimo(
    y_true: np.ndarray, prob: np.ndarray, costo_fn: float, costo_fp: float,
) -> tuple[float, float]:
    """Barrido fino de umbrales (mismo grid que monitoreo.umbral_optimo_costo)."""
    candidatos = np.arange(0.01, 1.0, 0.01)
    costos = [costo_monetario(y_true, prob, u, costo_fn, costo_fp) for u in candidatos]
    idx = int(np.argmin(costos))
    return float(candidatos[idx]), float(costos[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="tope en segundos")
    parser.add_argument(
        "--monto-medio", type=float, default=500.0,
        help="monto medio del microcrédito en USD (default 500)",
    )
    parser.add_argument(
        "--tasa-interes", type=float, default=0.20,
        help="margen/tasa del ciclo del crédito (default 0.20)",
    )
    parser.add_argument(
        "--lgd", type=float, default=1.0,
        help="severidad de la pérdida ante mora (default 1.0 = capital completo)",
    )
    args = parser.parse_args()

    costo_fn = args.monto_medio * args.lgd            # capital perdido por moroso aprobado
    costo_fp = args.monto_medio * args.tasa_interes   # margen perdido por buen pagador rechazado
    formula = (
        f"costo_FN = monto_medio({args.monto_medio}) * LGD({args.lgd}) = {costo_fn}; "
        f"costo_FP = monto_medio({args.monto_medio}) * tasa({args.tasa_interes}) = {costo_fp}"
    )
    print(f"función de costos: {formula} (razón FN:FP = {costo_fn / costo_fp:.1f}:1)")

    variables = C.cargar_variables()
    train, val, test = C.dividir(variables)
    X_train, y_train = C.preparar_numerico(train), train["target"].astype(int)
    X_val, y_val = C.preparar_numerico(val), val["target"].astype(int)
    X_test, y_test = C.preparar_numerico(test), test["target"].astype(int)
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()
    balance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    def objetivo(trial: optuna.Trial) -> float:
        # MISMO espacio de búsqueda que entrenar_optuna.py; cambia el objetivo.
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
        prob_val = modelo.predict_proba(X_val)[:, 1]
        umbral, costo = umbral_y_costo_minimo(y_val_np, prob_val, costo_fn, costo_fp)
        trial.set_user_attr("umbral_optimo_val", umbral)
        return costo

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
    )
    estudio.optimize(objetivo, n_trials=args.trials, timeout=args.timeout)

    mejor = construir_modelo(estudio.best_params, balance)
    mejor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    prob_val = mejor.predict_proba(X_val)[:, 1]
    umbral_val, costo_val = umbral_y_costo_minimo(y_val_np, prob_val, costo_fn, costo_fp)

    C.MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor, C.MODELOS / "xgboost_costos.joblib")
    C.guardar_json(C.MODELOS / "xgboost_costos_info.json", {
        "objetivo": "costo monetario mínimo en validación",
        "formula_costos": formula,
        "monto_medio": args.monto_medio,
        "tasa_interes": args.tasa_interes,
        "lgd": args.lgd,
        "costo_val": costo_val,
        "umbral_optimo_val": umbral_val,
        "auc_val": float(roc_auc_score(y_val_np, prob_val)),
        "trials": len(estudio.trials),
        "mejores_params": estudio.best_params,
        "mejor_iteracion": int(mejor.best_iteration),
        "scale_pos_weight": balance,
    })
    print(
        f"xgboost_costos: costo val {costo_val:,.0f} (umbral {umbral_val:.2f}, "
        f"{len(estudio.trials)} trials, mejores: {estudio.best_params})"
    )

    prob_test = mejor.predict_proba(X_test)[:, 1]
    predicciones = pd.DataFrame({
        "modelo": "xgboost",
        "configuracion": "29vars_costos",
        "credito_id_anon": test["credito_id_anon"].astype(str),
        "segmento": test["segmento"].astype(str),
        "y_true": y_test,
        "probabilidad": prob_test,
        "valida": True,
    })
    path = C.guardar_predicciones("xgboost_costos", predicciones)
    monitoreo.reporte_modelo(predicciones, "XGBoost cost-sensitive (test)")
    print(f"predicciones -> {path}")

    # ---- Comparativa en TEST: AUC-optimizado (existente) vs. costo-optimizado
    ruta_auc = C.PRED_DIR / "xgboost.parquet"
    if ruta_auc.exists():
        pred_auc = pd.read_parquet(ruta_auc)
        prob_auc = pred_auc["probabilidad"].to_numpy(dtype=float)
        y_auc = pred_auc["y_true"].to_numpy(dtype=int)
        filas = []
        for nombre, y, p in (
            ("AUC-optimizado (29vars_optuna)", y_auc, prob_auc),
            ("costo-optimizado (29vars_costos)", y_test_np, prob_test),
        ):
            umbral, costo = umbral_y_costo_minimo(y, p, costo_fn, costo_fp)
            m = monitoreo.metricas_umbral(y, p, umbral)
            filas.append({
                "variante": nombre,
                "AUC_test": round(float(roc_auc_score(y, p)), 4),
                "umbral_costo": umbral,
                "costo_test_USD": round(costo, 0),
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
                "accuracy": m["accuracy"], "recall": m["recall"],
            })
        print("\n== comparativa test: optimizar por AUC vs. por costo ==")
        print(pd.DataFrame(filas).to_string(index=False))
    else:
        print(f"aviso: no existe {ruta_auc}; correr entrenar_optuna.py para comparar")


if __name__ == "__main__":
    main()
