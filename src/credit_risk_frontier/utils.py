"""Módulo compartido del proyecto — única fuente de verdad para constantes,
métricas, splits/segmentación, serialización TabLLM y transportes LLM.

Vive a nivel de paquete (hermano de ``pipelines/``), no bajo ``pipelines/``, para
que ``find_pipelines()`` no lo descubra como pipeline. Consolida los helpers que
los scripts ``03``–``12`` duplicaban (``credit_metrics`` estaba definido 4 veces
con deriva de clave ``PR_AUC``/``PR-AUC``; ``load_splits``/``load_segmented`` dos
veces cada uno; la serialización y los transportes vivían en ``06b``/``09`` y se
importaban vía ``importlib``).

Diferencias respecto de los scripts (deliberadas, para nodos Kedro puros):

* ``load_splits``/``load_segmented`` reciben un ``DataFrame`` ya cargado, no una
  ruta — el I/O lo hace el catálogo de Kedro.
* ``serialize_tabllm`` recibe ``include_categoricals``/``include_text`` como
  argumentos, no como ``globals()`` mutados (así lo hacía la ablación de fuentes).
* ``credit_metrics`` usa una sola clave ``PR_AUC`` (la mayoritaria; ``03``/``04``
  usaban ``PR-AUC``). Los valores son idénticos; la normalización de clave se
  documenta en el plan y los tests toleran ambas grafías al comparar evidencia.
* Los transportes (``call_ollama``/``call_openai``) reciben ``model``/``think``
  como argumentos, sin mutar estado de módulo ni monkeypatch.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

# Raíz del repo (src/credit_risk_frontier/utils.py → ../../.. = repo root). Solo la
# usan los nodos de validación de artefactos, que legítimamente chequean existencia
# de figuras/docs publicados en disco (desviación menor documentada en el plan).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constantes de columnas (fuente: scripts 03/04/06b/12)
# ---------------------------------------------------------------------------

META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = ["subcategoria_texto", "descripcion_negocio", "otra_categoria_negocio", "tipo_credito"]

SEED = 42

# Variables TransUnion — un código < 0 significa "sin información en el buró".
TU_VARS = [
    "agg308", "wd81", "agg2503", "utlmag04", "duemag01", "aepmag01",
    "bi21s", "lmd34s", "ri27s", "rle904", "tel32s", "tranbal09",
    "at104s", "sa21s", "at103s", "tel03s", "at34af", "g051s", "agg9316", "wd03",
]

# Variables del crédito OTORGADO (post-aprobación) → leakage. Se excluyen SIEMPRE
# del baseline sin filtración (fuente: 12_ablacion_fuentes_ml.py).
LEAK_COLS = [
    "credits_amount_granted",
    "credits_fee_month_amount_granted",
    "credits_interest_amount",
    "credits_surety_bond_amount",
    "credits_digital_instrumentation_amount",
]

# Variables derivadas del buró TransUnion que no están en TU_VARS pero cuya señal
# proviene del buró. Se excluyen del brazo "sin buró"; solo participan en all_full.
TU_DERIVADAS = ["debts_savings", "score_debets", "relacion_edad_deuda"]

# Columnas que agrega la segmentación — nunca son features.
DERIVED = ["n_tu_missing", "segmento"]

# Grupos one-hot que el LLM ve decodificados en su serialización TabLLM.
ONEHOT_PREFIXES = ("category_", "credits_credit_goal_", "credits_contact_studies_", "CANAL_")

CORTE_ESPARSO = 6
MIN_SEG_CASOS = 20

# Top features para la serialización TabLLM (20 más predictivas, sin leakage).
TOP_FEATURES_LLM = [
    "g051s", "wd81", "wd03", "aepmag01", "bi21s", "ri27s",
    "rle904", "at104s", "agg308", "utlmag04",
    "antiguedad_cliente", "appusers_age",
    "shops_monthly_incomes", "shops_monthly_outcomes",
    "estimated_income", "free_cash_flow",
    "cost_ingress_ratio", "appusers_score",
]

# Features para la selección kNN de ejemplos few-shot (sin leakage).
KNN_FEATURES = [
    "g051s", "wd81", "wd03", "antiguedad_cliente", "appusers_age",
    "shops_monthly_incomes", "shops_monthly_outcomes", "estimated_income",
    "free_cash_flow", "cost_ingress_ratio", "appusers_score",
]

PALETTE = {"xgb": "#4878CF", "logreg": "#6ACC65", "llm": "#D65F5F"}


# ---------------------------------------------------------------------------
# Diccionarios de descripción y mapeos one-hot (fuente: 06b_llm_prompting.py)
# ---------------------------------------------------------------------------

NUMERIC_DESC = {
    "appusers_age":                          "la edad del solicitante en años",
    "credits_amount_granted":                "el monto del crédito otorgado en pesos colombianos",
    "credits_fee_month_amount_granted":      "la cuota mensual del crédito en pesos colombianos",
    "credits_interest_amount":               "el monto de intereses del crédito en pesos",
    "credits_surety_bond_amount":            "el monto del seguro/fianza en pesos",
    "credits_digital_instrumentation_amount":"el costo de instrumentación digital en pesos",
    "credits_dependants_amount":             "el número de dependientes del solicitante",
    "credits_family_expenses":               "los gastos familiares mensuales en pesos",
    "shops_monthly_incomes":                 "los ingresos mensuales del negocio en pesos",
    "shops_monthly_outcomes":                "los egresos mensuales del negocio en pesos",
    "shops_daily_incomes":                   "los ingresos diarios del negocio en pesos",
    "shops_initial_capital":                 "el capital inicial del negocio en pesos",
    "shops_rent_amount":                     "el arriendo mensual del negocio en pesos",
    "shops_shop_age":                        "la antigüedad del negocio en años",
    "antiguedad_cliente":                    "la antigüedad del solicitante como cliente en días",
    "estimated_income":                      "el ingreso mensual estimado del solicitante en pesos",
    "free_cash_flow":                        "el flujo de caja libre estimado en pesos",
    "cost_ingress_ratio":                    "la relación costo/ingreso del negocio (0=eficiente, 1=sin margen)",
    "debts_savings":                         "la relación deudas/ahorros del solicitante",
    "score_debets":                          "el puntaje de deudas del solicitante",
    "relacion_edad_deuda":                   "la relación entre edad del solicitante y sus deudas actuales",
    "appusers_score":                        "el puntaje interno del solicitante en el sistema",
}

# Definiciones oficiales del diccionario TransUnion CreditVision V3.8.
TU_DESC = {
    "g051s":     "el porcentaje de obligaciones que alguna vez estuvo en mora",
    "wd81":      "la mora ponderada en créditos financieros en el mes M=01 (índice)",
    "wd03":      "la mora ponderada en las obligaciones en el mes M=06 (índice)",
    "at103s":    "el porcentaje de obligaciones vigentes y al día del total de obligaciones",
    "duemag01":  "la magnitud total de todas las obligaciones en los últimos 24 meses (índice 0–600)",
    "agg308":    "el monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M=08",
    "tel03s":    "el número de obligaciones de telecomunicaciones vigentes y al día",
    "agg9316":   "el monto agregado en mora en el mes M=16",
    "utlmag04":  "la magnitud de utilización de obligaciones retail en los últimos 24 meses (índice 0–600)",
    "aepmag01":  "la magnitud del exceso de pago inferido agregado no hipotecario en 24 meses (índice 0–600)",
    "bi21s":     "los meses desde la más reciente apertura bancaria en cuotas",
    "tranbal09": "el saldo asignado a obligaciones identificadas como transactor al mes 9",
    "agg2503":   "el plazo inferido agregado (relación de saldo sobre cuota mínima) en el mes M=03",
    "at104s":    "el porcentaje de obligaciones aperturadas en los últimos 24 meses sobre el total",
    "tel32s":    "el saldo máximo en obligaciones de telecomunicaciones (últimos 12 meses) en pesos",
    "sa21s":     "los meses desde la más reciente cuenta de ahorros aperturada",
    "ri27s":     "el número de obligaciones retail vigentes y al día con 24 meses o más de antigüedad",
    "rle904":    "el exceso de pago inferido en cuentas hipotecarias en los últimos 6 meses",
    "at34af":    "la utilización de obligaciones vigentes en créditos financieros (últimos 12 meses)",
    "lmd34s":    "la utilización de obligaciones bancarias sin garantía de mediano plazo (últimos 12 meses)",
}

CATEGORY_MAP = {
    "category_agricultura_y_granja":      "Agricultura y Granja",
    "category_artesanias_y_arte":         "Artesanías y Arte",
    "category_bar_y_licores":             "Bar y Licores",
    "category_belleza_e_higiene":         "Belleza e Higiene",
    "category_carniceria_y_avicola":      "Carnicería y Avícola",
    "category_comercializacion":          "Comercialización",
    "category_confeccion":                "Confección",
    "category_drogueria_y_medicina":      "Droguería y Medicina",
    "category_electronica_y_tecnologia":  "Electrónica y Tecnología",
    "category_mercado_y_perecederos":     "Mercado y Perecederos",
    "category_miscelanea_y_papeleria":    "Miscelánea y Papelería",
    "category_restaurantes_y_comidas":    "Restaurantes y Comidas",
    "category_servicios":                 "Servicios",
    "category_servicios_y_alquiler":      "Servicios y Alquiler",
    "category_transporte_y_logistica":    "Transporte y Logística",
}

GOAL_MAP = {
    "credits_credit_goal_labor_payment":            "Pago de mano de obra",
    "credits_credit_goal_machinery_and_equipment":  "Maquinaria y equipos",
    "credits_credit_goal_marketing_and_advertising":"Marketing y publicidad",
    "credits_credit_goal_nan":                      "No especificado",
    "credits_credit_goal_payment_suppliers":        "Pago a proveedores",
    "credits_credit_goal_stocking":                 "Inventario/mercancía",
    "credits_credit_goal_supplies":                 "Insumos y materiales",
}

EDUCATION_MAP = {
    "credits_contact_studies_None":                  "No especificado",
    "credits_contact_studies_college":               "Universitario (en curso)",
    "credits_contact_studies_high_school":           "Bachillerato",
    "credits_contact_studies_nan":                   "No especificado",
    "credits_contact_studies_postgraduate_studies":  "Posgrado",
    "credits_contact_studies_primary":               "Primaria",
    "credits_contact_studies_professional":          "Profesional universitario",
    "credits_contact_studies_technical":             "Técnico o tecnólogo",
}

GENDER_MAP = {
    "appusers_gender_None":   "No especificado",
    "appusers_gender_female": "Femenino",
    "appusers_gender_male":   "Masculino",
    "appusers_gender_other":  "Otro",
}

CANAL_COLS = [f"CANAL_{n:02d}" for n in range(1, 13)]
ALIANZA_COLS = [f"ALIANZA_{n:02d}" for n in range(1, 7)]


# ---------------------------------------------------------------------------
# Métricas de credit scoring
# ---------------------------------------------------------------------------

def credit_metrics(y_true, y_prob) -> dict:
    """AUC, Gini, KS, Brier y PR-AUC para un conjunto de predicciones.

    Clave canónica ``PR_AUC`` (guión bajo). Los scripts 03/04 usaban ``PR-AUC``;
    el valor es idéntico. Fuente numérica: 06b_llm_prompting.credit_metrics.
    """
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks, "Brier": brier, "PR_AUC": prauc}


# ---------------------------------------------------------------------------
# Splits temporales y segmentación (reciben DataFrame, no ruta)
# ---------------------------------------------------------------------------

def feature_columns(df: pd.DataFrame, *, exclude_leak: bool = True) -> list[str]:
    """Columnas de features del dataset.

    ``exclude_leak=True`` (baseline citable, sin filtración = brazo ``all_full`` de
    la ablación ML, test AUC 0.7516) descarta las 5 ``LEAK_COLS`` (crédito otorgado,
    post-aprobación), META, TEXT y las columnas derivadas por la segmentación. Las
    ``TU_DERIVADAS`` NO se excluyen: son señal de buró legítima disponible al decidir
    (solo se sacan del brazo ``all_sin_tu`` para no contaminar la fuente "sin buró").
    ``exclude_leak=False`` reproduce el baseline con leakage de 03/04 (solo META+TEXT).
    """
    banned = set(META + TEXT + DERIVED)
    if exclude_leak:
        banned |= set(LEAK_COLS)
    return [c for c in df.columns if c not in banned]


def load_splits(df: pd.DataFrame, features: list[str]) -> dict:
    """Separa train/val/test por la columna temporal ``set`` restringido a ``features``.

    Fuente: 03/04 ``load_splits`` y 12 ``make_splits`` (unificados). Retorna un dict
    con ``{split: {"X", "y"}}`` más ``"features"``.
    """
    splits = {}
    for name in ("train", "val", "test"):
        subset = df[df["set"] == name]
        splits[name] = {"X": subset[features], "y": subset["target"]}
    splits["features"] = list(features)
    return splits


def annotate_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``n_tu_missing`` y ``segmento`` (esparso/denso) por densidad de buró.

    Fuente: 06b ``load_segmented`` (regla ``TU < 0``, corte ``>= CORTE_ESPARSO``).
    Devuelve una copia; no muta el DataFrame de entrada.
    """
    out = df.copy()
    out["n_tu_missing"] = (out[TU_VARS] < 0).sum(axis=1)
    out["segmento"] = (out["n_tu_missing"] >= CORTE_ESPARSO).map({True: "esparso", False: "denso"})
    return out


# ---------------------------------------------------------------------------
# Evaluación sobre test / general por segmento (fuente: 06b y 08)
# ---------------------------------------------------------------------------

def evaluate_on_test(df: pd.DataFrame, probs) -> dict:
    """Evalúa ``probs`` sobre el test, total y por segmento (esparso/denso).

    Requiere que ``df`` tenga la columna ``segmento`` (via ``annotate_segments``).
    Omite segmentos con menos de ``MIN_SEG_CASOS`` casos. Fuente: 06b.evaluate_on_test.
    """
    probs = np.asarray(probs, dtype=float)
    test_mask = (df["set"] == "test").values & ~np.isnan(probs)
    y = df["target"].values
    segmentos = df["segmento"].values

    summary = {}
    summary["total"] = {**credit_metrics(y[test_mask], probs[test_mask]),
                        "n": int(test_mask.sum())}
    for seg in ("esparso", "denso"):
        mask = test_mask & (segmentos == seg)
        if mask.sum() >= MIN_SEG_CASOS:
            summary[seg] = {**credit_metrics(y[mask], probs[mask]), "n": int(mask.sum())}
        else:
            summary[seg] = {"n": int(mask.sum()), "note": "insuficiente para AUC confiable"}
    return summary


def evaluate_general(df: pd.DataFrame, probs) -> dict:
    """AUC/Gini/KS por segmento sobre todo el dataset (filas con prob no-NaN).

    Fuente: 08.evaluate_general. Existe porque el LLM no distingue train/test:
    puntúa cualquier fila igual, y esta métrica describe todo el pool puntuado.
    """
    probs = np.asarray(probs, dtype=float)
    valid = ~np.isnan(probs)
    y = df["target"].values
    seg = df["segmento"].values

    summary = {"total": {**credit_metrics(y[valid], probs[valid]), "n": int(valid.sum())}}
    for s in ("esparso", "denso"):
        mask = valid & (seg == s)
        if mask.sum() >= MIN_SEG_CASOS:
            summary[s] = {**credit_metrics(y[mask], probs[mask]), "n": int(mask.sum())}
        else:
            summary[s] = {"n": int(mask.sum()), "note": "insuficiente para AUC confiable"}
    return summary


# ---------------------------------------------------------------------------
# Serialización TabLLM (fuente: 06b; switches como argumentos, no globals)
# ---------------------------------------------------------------------------

def _decode_onehot(row: pd.Series, col_map: dict) -> str:
    """Decodifica un grupo de one-hot a su valor categórico."""
    for col, val_label in col_map.items():
        if col in row and row[col] == 1:
            return val_label
    return "No especificado"


def _decode_canal(row: pd.Series) -> str:
    """Decodifica CANAL_XX (anónimo, mantiene el código)."""
    for col in CANAL_COLS:
        if col in row and row[col] == 1:
            return col.replace("_", " ")
    return "No especificado"


def _decode_alianza(row: pd.Series) -> str:
    """Decodifica ALIANZA_XX (anónimo, mantiene el código)."""
    for col in ALIANZA_COLS:
        if col in row and row[col] == 1:
            return col.replace("_", " ")
    return "Sin alianza"


def _fmt_val(val, is_tu: bool = False) -> str | None:
    """Formatea un valor numérico para el prompt; None si es desconocido/NaN/centinela TU."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if is_tu and val < 0:
        # Códigos centinela de TransUnion (sin información) → no se serializan como valor real.
        return None
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def serialize_tabllm(row: pd.Series, feature_names: list[str], *,
                     include_categoricals: bool = True,
                     include_text: bool = True) -> str:
    """Serialización compacta TabLLM: ``descripción: valor`` separadas por coma.

    ``include_categoricals``/``include_text`` reemplazan a los ``globals()``
    ``INCLUDE_CATEGORICALS``/``INCLUDE_TEXT`` que mutaba la ablación de fuentes
    (09 ``--features``). Con ambos en ``True`` reproduce byte a byte la salida del
    script 06b. Fuente: 06b.serialize_tabllm.
    """
    all_desc = {**NUMERIC_DESC, **TU_DESC}
    parts = []
    for col in feature_names:
        v = row.get(col)
        is_tu = col in TU_DESC
        fv = _fmt_val(v, is_tu=is_tu)
        desc = all_desc.get(col, col)
        parts.append(f"{desc}: {fv}" if fv is not None else f"{desc}: sin historial")

    if include_categoricals:
        parts.append(f"categoría negocio: {_decode_onehot(row, CATEGORY_MAP)}")
        parts.append(f"objetivo crédito: {_decode_onehot(row, GOAL_MAP)}")
        parts.append(f"educación: {_decode_onehot(row, EDUCATION_MAP)}")
        parts.append(f"canal: {_decode_canal(row)}")

    if include_text:
        for text_col, label in [
            ("subcategoria_texto", "subcategoría"),
            ("descripcion_negocio", "negocio"),
            ("tipo_credito", "tipo crédito"),
        ]:
            v = row.get(text_col, "")
            if v and not (isinstance(v, float) and pd.isna(v)):
                parts.append(f"{label}: {v}")
    return ", ".join(parts)


def features_for_profile(profile: str) -> tuple[list[str], bool, bool]:
    """Resuelve un perfil de ablación de fuentes a ``(features, include_cat, include_text)``.

    Fuente: rama ``--features`` de 09_llm_thinking.py.
      * ``full``    → TOP_FEATURES_LLM completas + categóricas + texto.
      * ``tu``      → solo variables de buró (TU), sin categóricas ni texto.
      * ``tu-sem``  → solo TU + categóricas + texto (semántica).
      * ``sin-tu``  → solo variables no-TU + categóricas + texto.
    """
    tu_feats = [f for f in TOP_FEATURES_LLM if f in TU_DESC]
    neg_feats = [f for f in TOP_FEATURES_LLM if f not in TU_DESC]
    if profile == "full":
        return list(TOP_FEATURES_LLM), True, True
    if profile == "tu":
        return tu_feats, False, False
    if profile == "tu-sem":
        return tu_feats, True, True
    if profile == "sin-tu":
        return neg_feats, True, True
    raise ValueError(f"perfil de features desconocido: {profile!r}")


# ---------------------------------------------------------------------------
# Prompts (fuente: 06b directo, 09 razonamiento, 09e variantes de prompt)
# ---------------------------------------------------------------------------

SYSTEM_BASE = (
    "Eres evaluador de riesgo crediticio de microcréditos en Colombia. "
    "Responde SOLO con un número entero."
)

SYSTEM_FEWSHOT = (
    "Eres evaluador de riesgo crediticio de microcréditos en Colombia. "
    "Aprende de los ejemplos resueltos y responde SOLO con un número entero."
)

SYSTEM_THINKING = (
    "Eres un evaluador experto de riesgo crediticio de microcréditos en Colombia. "
    "Para cada solicitud analizas la capacidad de pago, la estabilidad del negocio y el "
    "historial disponible, y estimas la probabilidad de mora. Razonas paso a paso antes de "
    "dar tu estimación final."
)

SYSTEM_THINKING_FEWSHOT = (
    "Eres un evaluador experto de riesgo crediticio de microcréditos en Colombia. "
    "Se te muestran casos resueltos con su desenlace real; aprende de ellos. Para el nuevo "
    "solicitante analizas la capacidad de pago, la estabilidad del negocio y el historial "
    "disponible, y razonas paso a paso antes de dar tu estimación final de probabilidad de mora."
)

INSTR_THINKING = (
    "Analiza el caso y estima la probabilidad de que este crédito entre en mora "
    "(atraso mayor a 60 días). Considera la capacidad de pago, la coherencia entre ingresos "
    "y monto solicitado, y el historial de buró. Razona brevemente y termina tu respuesta con "
    "una única línea con este formato EXACTO:\n"
    "PROBABILIDAD_DE_MORA: <entero de 0 a 100>"
)

USER_TEMPLATE_DIRECT = "DATOS: {serialized}\n\nProbabilidad de mora (0-100, entero):"


def build_messages_direct(row: pd.Series, feature_names: list[str] | None = None,
                          **serialize_kwargs) -> list[dict]:
    """Mensajes zero-shot directo (fuente: 06b._build_messages)."""
    feats = feature_names if feature_names is not None else TOP_FEATURES_LLM
    serialized = serialize_tabllm(row, feats, **serialize_kwargs)
    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": USER_TEMPLATE_DIRECT.format(serialized=serialized)},
    ]


def build_messages_thinking_zero(row: pd.Series, feature_names: list[str] | None = None,
                                 **serialize_kwargs) -> list[dict]:
    """Mensajes zero-shot razonado (fuente: 09.build_zero_messages)."""
    feats = feature_names if feature_names is not None else TOP_FEATURES_LLM
    serialized = serialize_tabllm(row, feats, **serialize_kwargs)
    user = f"DATOS DEL SOLICITANTE: {serialized}\n\n{INSTR_THINKING}"
    return [
        {"role": "system", "content": SYSTEM_THINKING},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Parseo de probabilidad (fuente: 06b directo, 09 razonamiento estricto)
# ---------------------------------------------------------------------------

_RE_LABELED = re.compile(r"PROBABILIDAD_DE_MORA\s*:\s*([0-9]{1,3})")


def parse_prob_direct(text: str) -> float:
    """Extrae P(mora) 0-100 → [0,1] del texto directo (fuente: 06b._parse_prob).

    Elimina bloques ``<think>...</think>`` y toma el primer número válido 0-100.
    """
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for n in re.findall(r"\b([0-9]{1,3})\b", clean):
        val = int(n)
        if 0 <= val <= 100:
            return val / 100.0
    return float("nan")


def parse_prob_labeled(text: str) -> float:
    """Extrae P(mora) SOLO de la línea ``PROBABILIDAD_DE_MORA: NN`` (fuente: 09.parse_prob).

    Estricto a propósito: en modo razonamiento el modelo escribe muchos números;
    sin la línea canónica devuelve NaN (el caso se descarta, no contamina la métrica).
    Si hay varias líneas etiquetadas toma la última (la final).
    """
    matches = _RE_LABELED.findall(text)
    for val_str in reversed(matches):
        val = int(val_str)
        if 0 <= val <= 100:
            return val / 100.0
    return float("nan")


# ---------------------------------------------------------------------------
# Transportes LLM (model/think como argumentos; import lazy de requests/openai)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def check_ollama(model: str = DEFAULT_OLLAMA_MODEL, tags_url: str = DEFAULT_OLLAMA_TAGS_URL) -> None:
    """Verifica que Ollama esté corriendo y el modelo disponible (fuente: 06b.check_ollama)."""
    import requests  # lazy: importar el pipeline nunca contacta un servidor

    try:
        r = requests.get(tags_url, timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
    except requests.ConnectionError as exc:
        raise RuntimeError("Ollama no está corriendo. Iniciarlo con: ollama serve") from exc
    if model not in available:
        raise RuntimeError(
            f"Modelo '{model}' no encontrado en Ollama. Disponibles: {available}. "
            f"Instalar con: ollama pull {model}"
        )


def call_ollama(messages: list[dict], *, model: str = DEFAULT_OLLAMA_MODEL, think: bool = False,
                url: str = DEFAULT_OLLAMA_URL, timeout: int = 45, retries: int = 3) -> float:
    """Llama a Ollama (respuesta directa) y retorna P(mora) en [0,1] (fuente: 06b.call_ollama).

    ``model``/``think`` son argumentos (los scripts los tenían como globals de módulo).
    """
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": 0.0, "num_predict": 512 if think else 20},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            if not content.strip() and think:
                content = data["message"].get("thinking", "")
            return parse_prob_direct(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return float("nan")


def _extract_final_logprob(data: dict) -> float:
    """Logprob del último token numérico de la respuesta, si Ollama la expone (fuente: 09)."""
    lp = data.get("logprobs") or data.get("message", {}).get("logprobs")
    if not lp or not isinstance(lp, list):
        return float("nan")
    last = float("nan")
    for entry in lp:
        tok = str(entry.get("token", ""))
        if any(c.isdigit() for c in tok):
            val = entry.get("logprob")
            if val is not None:
                last = float(val)
    return last


def call_ollama_think(messages: list[dict], *, model: str = DEFAULT_OLLAMA_MODEL,
                      think_native: bool = True, url: str = DEFAULT_OLLAMA_URL,
                      timeout: int = 600, retries: int = 3) -> dict:
    """Llama a Ollama en modo razonamiento y devuelve todo el output (fuente: 09.call_ollama_think).

    ``think_native`` (True para qwen3, False para modelos que inducen razonamiento
    solo por prompt, p. ej. gemma) reemplaza al global ``_THINK_NATIVE``.
    Retorna ``{prob, content, thinking, logprob, eval_count}``.
    """
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think_native,
        "logprobs": True,
        "options": {"temperature": 0.0, "num_predict": 3072},
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            content = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            text_for_parse = content if content.strip() else thinking
            return {
                "prob": parse_prob_labeled(text_for_parse),
                "content": content,
                "thinking": thinking,
                "logprob": _extract_final_logprob(data),
                "eval_count": data.get("eval_count"),
            }
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"prob": float("nan"), "content": "", "thinking": "",
                        "logprob": float("nan"), "eval_count": None}


# ---------------------------------------------------------------------------
# Selección de ejemplos few-shot (random balanceado + kNN, anti-leakage)
# Fuente: 08 (random + anti-leakage) y 09 (kNN)
# ---------------------------------------------------------------------------

def select_few_shot_examples(train_df: pd.DataFrame, n_shots: int,
                             rng: np.random.Generator) -> pd.DataFrame:
    """n_shots ejemplos balanceados del train (n/2 default, n/2 no) — fuente: 08."""
    n_per_class = n_shots // 2
    defaults = train_df[train_df["target"] == 1]
    no_defaults = train_df[train_df["target"] == 0]
    sampled_def = defaults.sample(n=min(n_per_class, len(defaults)),
                                  random_state=int(rng.integers(1_000_000)))
    sampled_no = no_defaults.sample(n=min(n_per_class, len(no_defaults)),
                                    random_state=int(rng.integers(1_000_000)))
    examples = pd.concat([sampled_def, sampled_no]).sample(
        frac=1, random_state=int(rng.integers(1_000_000)))
    return examples.reset_index(drop=True)


def build_replacement_pool(train_df: pd.DataFrame, base_examples: pd.DataFrame,
                           rng: np.random.Generator) -> dict:
    """Pool de reemplazo por clase: filas del train fuera de los ejemplos base (fuente: 08)."""
    base_ids = set(base_examples["credito_id_anon"])
    pool = {}
    for cls in (0, 1):
        cand = train_df[(train_df["target"] == cls) &
                        (~train_df["credito_id_anon"].isin(base_ids))]
        pool[cls] = cand.sample(frac=1, random_state=int(rng.integers(1_000_000))
                                ).reset_index(drop=True)
    return pool


def examples_for_case(base_examples: pd.DataFrame, case_id: str, pool: dict) -> pd.DataFrame:
    """n_shots ejemplos que excluyen ``case_id`` (garantía anti-leakage) — fuente: 08."""
    ids = set(base_examples["credito_id_anon"])
    if case_id not in ids:
        return base_examples

    rows, used = [], set()
    for _, ex in base_examples.iterrows():
        if ex["credito_id_anon"] != case_id:
            rows.append(ex)
            continue
        cls = int(ex["target"])
        for _, cand in pool[cls].iterrows():
            cid = cand["credito_id_anon"]
            if cid == case_id or cid in ids or cid in used:
                continue
            used.add(cid)
            rows.append(cand)
            break
    return pd.DataFrame(rows).reset_index(drop=True)


def build_knn_space(train_df: pd.DataFrame):
    """(matriz estandarizada del train, medias, desvíos, features usadas) — fuente: 09."""
    feats = [f for f in KNN_FEATURES if f in train_df.columns]
    X = train_df[feats].apply(pd.to_numeric, errors="coerce")
    X = X.mask(X < 0)  # centinelas de buró → faltante para la distancia
    mu = X.mean()
    sd = X.std().replace(0, 1.0)
    Xz = ((X - mu) / sd).fillna(0.0).values
    return Xz, mu, sd, feats


def _standardize_case(row: pd.Series, mu, sd, feats) -> np.ndarray:
    x = pd.to_numeric(pd.Series({f: row.get(f) for f in feats}), errors="coerce")
    x = x.mask(x < 0)
    return ((x - mu) / sd).fillna(0.0).values


def knn_examples_for_case(case_row: pd.Series, case_id: str, train_df: pd.DataFrame,
                          knn_space, n_shots: int) -> pd.DataFrame:
    """n_shots ejemplos más similares al caso, balanceados por clase, sin el propio caso (fuente: 09)."""
    Xz, mu, sd, feats = knn_space
    n_per_class = n_shots // 2
    xz = _standardize_case(case_row, mu, sd, feats)
    dist = np.sqrt(((Xz - xz) ** 2).sum(axis=1))
    order = np.argsort(dist)

    picked, seen_ids = [], set()
    need = {0: n_per_class, 1: n_per_class}
    tgt = train_df["target"].values
    ids = train_df["credito_id_anon"].values
    idx = train_df.index.values
    for pos in order:
        cls = int(tgt[pos])
        cid = ids[pos]
        if cid == case_id or cid in seen_ids or need[cls] == 0:
            continue
        picked.append(idx[pos])
        seen_ids.add(cid)
        need[cls] -= 1
        if need[0] == 0 and need[1] == 0:
            break
    ex = train_df.loc[picked]
    return ex.sample(frac=1, random_state=SEED).reset_index(drop=True)


def json_safe(obj):
    """Convierte tipos numpy a nativos para serialización JSON estable."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
