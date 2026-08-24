"""Modelo #1 del PLAN §2.2 — Regresión Logística (baseline clásico).

Es el estándar de la industria crediticia (Hand & Henley, 1997): lineal,
interpretable por coeficientes y barato. Todo comparativo se lee contra este
piso. Entrena con train, elige la regularización C con el AUC de validación
y predice sobre el test intocado del contrato.

Salidas:
  data/pipeline/modelos/logreg.joblib
  data/pipeline/modelos/logreg_info.json      AUC val, C elegido, coeficientes
  data/pipeline/predicciones/regresion_logistica.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contrato as C  # noqa: E402
import monitoreo  # noqa: E402

GRILLA_C = (0.01, 0.1, 1.0, 10.0)


def entrenar(X_train, y_train, X_val, y_val):
    """Ajusta con train y elige C por AUC de validación (test no participa)."""
    mejor, mejor_auc = None, -1.0
    for c in GRILLA_C:
        modelo = Pipeline([
            ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("escalar", StandardScaler()),
            ("logreg", LogisticRegression(
                C=c, max_iter=3000, solver="liblinear", random_state=C.SEED
            )),
        ]).fit(X_train, y_train)
        auc = roc_auc_score(y_val, modelo.predict_proba(X_val)[:, 1])
        if auc > mejor_auc:
            mejor, mejor_auc = modelo, auc
    return mejor, mejor_auc


def main() -> None:
    variables = C.cargar_variables()
    train, val, test = C.dividir(variables)
    X_train, y_train = C.preparar_numerico(train), train["target"].astype(int)
    X_val, y_val = C.preparar_numerico(val), val["target"].astype(int)

    modelo, auc_val = entrenar(X_train, y_train, X_val, y_val)
    c_elegido = float(modelo.named_steps["logreg"].C)

    C.MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, C.MODELOS / "logreg.joblib")
    coeficientes = dict(zip(
        C.FEATURES_29, modelo.named_steps["logreg"].coef_[0].round(4).tolist()
    ))
    C.guardar_json(C.MODELOS / "logreg_info.json", {
        "auc_val": auc_val,
        "C": c_elegido,
        "n_train": len(train),
        "n_val": len(val),
        "coeficientes_estandarizados": coeficientes,
    })
    print(f"logreg AUC val: {auc_val:.4f} (C={c_elegido})")

    probabilidades = modelo.predict_proba(C.preparar_numerico(test))[:, 1]
    predicciones = pd.DataFrame({
        "modelo": "logreg",
        "configuracion": "29vars",
        "credito_id_anon": test["credito_id_anon"].astype(str),
        "segmento": test["segmento"].astype(str),
        "y_true": test["target"].astype(int),
        "probabilidad": probabilidades,
        "valida": True,
    })
    path = C.guardar_predicciones("regresion_logistica", predicciones)
    monitoreo.reporte_modelo(predicciones, "Regresión Logística (test)")
    print(f"predicciones -> {path}")


if __name__ == "__main__":
    main()
