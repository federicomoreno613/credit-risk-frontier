"""Tests de las funciones que producen los números de la tesis.

Fixtures sintéticos: no dependen de datos locales.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

import contrato as C  # noqa: E402
import monitoreo  # noqa: E402
from credit_risk_frontier import utils  # noqa: E402


# ---------------------------------------------------------------------------
# Parseo de respuestas LLM
# ---------------------------------------------------------------------------
def test_parse_prob_toma_la_ultima_ocurrencia_valida():
    texto = "quizás PROBABILIDAD_DE_MORA: 30... no, PROBABILIDAD_DE_MORA: 65"
    assert utils.parse_prob_labeled(texto) == 0.65


def test_parse_prob_sin_formato_devuelve_nan():
    assert np.isnan(utils.parse_prob_labeled("el riesgo es alto, 80%"))


def test_parse_clase_no_moroso_antes_que_moroso():
    # "no moroso" contiene "moroso": el orden de la alternancia es el caso borde.
    assert utils.parse_clase_labeled("CLASE: no moroso") == 0.0
    assert utils.parse_clase_labeled("CLASE: moroso") == 1.0
    assert utils.parse_clase_labeled("**CLASE:** no moroso") == 0.0
    assert np.isnan(utils.parse_clase_labeled("sin etiqueta"))


def test_parse_clase_ultima_ocurrencia():
    texto = "CLASE: moroso... corrigiendo:\nCLASE: no moroso\nPROBABILIDAD_DE_MORA: 20"
    assert utils.parse_clase_labeled(texto) == 0.0


# ---------------------------------------------------------------------------
# Contrato: centinelas TU
# ---------------------------------------------------------------------------
def test_preparar_numerico_convierte_centinelas_tu_en_nan():
    fila = {c: 1.0 for c in C.FEATURES_29}
    fila[utils.TU_VARS[0]] = -1.0     # centinela clásico
    fila[utils.TU_VARS[1]] = -999.0   # cualquier negativo es ausencia
    fila[utils.FORM_DIRECT_VARS[0]] = -5.0  # formulario NO se toca
    frame = pd.DataFrame([fila])
    resultado = C.preparar_numerico(frame)
    assert np.isnan(resultado.iloc[0][utils.TU_VARS[0]])
    assert np.isnan(resultado.iloc[0][utils.TU_VARS[1]])
    assert resultado.iloc[0][utils.FORM_DIRECT_VARS[0]] == -5.0


# ---------------------------------------------------------------------------
# Métricas de umbral y costo
# ---------------------------------------------------------------------------
def _grupo(y, p, **extra):
    base = {"modelo": "m", "configuracion": "c", "segmento": "total",
            "credito_id_anon": [str(i) for i in range(len(y))],
            "y_true": y, "probabilidad": p, "valida": True}
    return pd.DataFrame(base | extra)


def test_metricas_umbral_matriz_conocida():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    m = monitoreo.metricas_umbral(y, p, 0.5)
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5
    assert m["costo"] == 1 * C.COSTOS["costo_fn"] + 1 * C.COSTOS["costo_fp"]


def test_umbral_optimo_minimiza_el_costo():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0, 0.2, 400), 0, 1)
    optimo = monitoreo.umbral_optimo_costo(y, p)
    costo_optimo = monitoreo.metricas_umbral(y, p, optimo)["costo"]
    for u in (0.1, 0.5, 0.9):
        assert costo_optimo <= monitoreo.metricas_umbral(y, p, u)["costo"]


def test_tabla_confusion_incluye_segmentos_y_clase():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 100), 0, 1)
    df = _grupo(y, p, clase=y.astype(float))
    df.loc[:49, "segmento"] = "esparso"
    df.loc[50:, "segmento"] = "denso"
    tabla = monitoreo.tabla_confusion(df)
    assert set(tabla["segmento"]) == {"total", "esparso", "denso"}
    assert "clase_verbalizada" in set(tabla["criterio"])
    clase_total = tabla[(tabla["criterio"] == "clase_verbalizada")
                        & (tabla["segmento"] == "total")].iloc[0]
    assert clase_total["FP"] == 0 and clase_total["FN"] == 0  # clase == y_true


# ---------------------------------------------------------------------------
# Comparación estadística
# ---------------------------------------------------------------------------
def test_delong_identico_da_delta_cero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    r = monitoreo.delong_test(y, p, p.copy())
    assert r["delta"] == pytest.approx(0.0)


def test_delong_detecta_diferencia_grande():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    bueno = np.clip(y * 0.7 + rng.normal(0.15, 0.1, 500), 0, 1)
    azar = rng.random(500)
    r = monitoreo.delong_test(y, bueno, azar)
    assert r["auc_a"] > 0.9 > r["auc_b"]
    assert r["p_valor"] < 0.001


def test_delong_coincide_con_auc_de_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    a, b = rng.random(300), rng.random(300)
    r = monitoreo.delong_test(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a), abs=1e-9)
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, b), abs=1e-9)


def test_bootstrap_ic_contiene_el_auc_puntual():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.normal(0.3, 0.25, 400), 0, 1)
    lo, hi = monitoreo.bootstrap_auc_ic(y, p, n_iter=200)
    assert lo < roc_auc_score(y, p) < hi
