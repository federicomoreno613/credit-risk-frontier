"""Dominio compartido del pipeline: variables, métricas, serialización y Ollama.

La entrada y la salida de archivos están a cargo de pipeline/contrato.py.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


SEED = 42

# Columnas que nunca pueden entrar como predictores de los modelos clásicos.
META = ["credito_id_anon", "fecha_desembolso", "target", "set"]
TEXT = [
    "subcategoria_texto",
    "descripcion_negocio",
    "otra_categoria_negocio",
    "tipo_credito",
]
LEAK_COLS = [
    "credits_amount_granted",
    "credits_fee_month_amount_granted",
    "credits_interest_amount",
    "credits_surety_bond_amount",
    "credits_digital_instrumentation_amount",
]
SCORES_INTERNOS = ["score_debets", "appusers_score"]
TEMPORAL_PROXY_COLS = ["antiguedad_cliente"]
DERIVED = ["n_tu_missing", "segmento"]

# El orden forma parte del contrato experimental y se valida antes de entrenar.
TU_VARS = [
    "agg308",
    "wd81",
    "agg2503",
    "utlmag04",
    "duemag01",
    "aepmag01",
    "bi21s",
    "lmd34s",
    "ri27s",
    "rle904",
    "tel32s",
    "tranbal09",
    "at104s",
    "sa21s",
    "at103s",
    "tel03s",
    "at34af",
    "g051s",
    "agg9316",
    "wd03",
]
FORM_DIRECT_VARS = [
    "appusers_age",
    "credits_dependants_amount",
    "credits_family_expenses",
    "shops_monthly_incomes",
    "shops_monthly_outcomes",
    "shops_daily_incomes",
    "shops_initial_capital",
    "shops_rent_amount",
    "shops_shop_age",
]

CORTE_ESPARSO = 6

# Descripciones usadas en la serialización que efectivamente recibe Qwen.
NUMERIC_DESC = {
    "appusers_age": "la edad del solicitante en años",
    "credits_dependants_amount": "el número de dependientes del solicitante",
    "credits_family_expenses": "los gastos familiares mensuales en pesos",
    "shops_monthly_incomes": "los ingresos mensuales del negocio en pesos",
    "shops_monthly_outcomes": "los egresos mensuales del negocio en pesos",
    "shops_daily_incomes": "los ingresos diarios del negocio en pesos",
    "shops_initial_capital": "el capital inicial del negocio en pesos",
    "shops_rent_amount": "el arriendo mensual del negocio en pesos",
    "shops_shop_age": "la antigüedad del negocio en años",
}
TU_DESC = {
    "g051s": "el porcentaje de obligaciones que alguna vez estuvo en mora",
    "wd81": "la mora ponderada en créditos financieros en el mes M=01 (índice)",
    "wd03": "la mora ponderada en las obligaciones en el mes M=06 (índice)",
    "at103s": "el porcentaje de obligaciones vigentes y al día del total de obligaciones",
    "duemag01": "la magnitud total de todas las obligaciones en los últimos 24 meses (índice 0–600)",
    "agg308": "el monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M=08",
    "tel03s": "el número de obligaciones de telecomunicaciones vigentes y al día",
    "agg9316": "el monto agregado en mora en el mes M=16",
    "utlmag04": "la magnitud de utilización de obligaciones retail en los últimos 24 meses (índice 0–600)",
    "aepmag01": "la magnitud del exceso de pago inferido agregado no hipotecario en 24 meses (índice 0–600)",
    "bi21s": "los meses desde la más reciente apertura bancaria en cuotas",
    "tranbal09": "el saldo asignado a obligaciones identificadas como transactor al mes 9",
    "agg2503": "el plazo inferido agregado (relación de saldo sobre cuota mínima) en el mes M=03",
    "at104s": "el porcentaje de obligaciones aperturadas en los últimos 24 meses sobre el total",
    "tel32s": "el saldo máximo en obligaciones de telecomunicaciones (últimos 12 meses) en pesos",
    "sa21s": "los meses desde la más reciente cuenta de ahorros aperturada",
    "ri27s": "el número de obligaciones retail vigentes y al día con 24 meses o más de antigüedad",
    "rle904": "el exceso de pago inferido en cuentas hipotecarias en los últimos 6 meses",
    "at34af": "la utilización de obligaciones vigentes en créditos financieros (últimos 12 meses)",
    "lmd34s": "la utilización de obligaciones bancarias sin garantía de mediano plazo (últimos 12 meses)",
}


def credit_metrics(y_true, y_prob) -> dict:
    """Calcula discriminación y calibración con ``target=1`` como mora."""
    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    y = np.asarray(y_true, dtype=float)
    probability = np.asarray(y_prob, dtype=float)
    bins = np.minimum((np.clip(probability, 0, 1) * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - probability[mask].mean())

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    calibration = LogisticRegression(C=1e6, solver="lbfgs").fit(
        logits.reshape(-1, 1), y
    )
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks,
        "Brier": brier,
        "PR_AUC": pr_auc,
        "ECE": float(ece),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def annotate_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Distingue historial de buró esparso a partir de códigos TU negativos."""
    missing = [column for column in TU_VARS if column not in frame.columns]
    if missing:
        raise ValueError(f"faltan variables TransUnion para segmentar: {missing}")
    result = frame.copy()
    result["n_tu_missing"] = (result[TU_VARS] < 0).sum(axis=1)
    result["segmento"] = result["n_tu_missing"].ge(CORTE_ESPARSO).map(
        {True: "esparso", False: "denso"}
    )
    return result


def _format_prompt_value(value, *, is_transunion: bool) -> str | None:
    if value is None or pd.isna(value):
        return None
    if is_transunion and value < 0:
        return None
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def serialize_intermediate_profile(
    row: pd.Series,
    feature_names: list[str],
    profile: str,
    compact: bool = False,
) -> str:
    """Serializa 20 o 29 variables y, cuando corresponde, una descripción libre."""
    if profile not in {"tu", "tu_form", "tu_form_description"}:
        raise ValueError(f"perfil de serialización desconocido: {profile!r}")
    expected = TU_VARS if profile == "tu" else TU_VARS + FORM_DIRECT_VARS
    if list(feature_names) != expected:
        if profile == "tu":
            raise ValueError(
                "La serialización TransUnion requiere exactamente las 20 "
                "variables canónicas del buró"
            )
        raise ValueError(
            "La serialización ampliada requiere exactamente las 20 variables "
            "TransUnion seguidas por las 9 declaraciones directas del formulario"
        )

    values = []
    descriptions = {**NUMERIC_DESC, **TU_DESC}
    for column in expected:
        formatted = _format_prompt_value(
            row.get(column), is_transunion=column in TU_VARS
        )
        if compact:
            if formatted is not None:
                values.append(f"{column}={formatted}")
        else:
            rendered = formatted if formatted is not None else "sin historial"
            values.append(f"{descriptions[column]}: {rendered}")

    parts = [", ".join(values)]
    if profile == "tu_form_description":
        description = row.get("descripcion_negocio")
        if description is not None and not pd.isna(description) and str(description).strip():
            parts.append(f"descripción del negocio: {str(description).strip()}")
    return ", ".join(part for part in parts if part)


INSTR_INTERMEDIATE = (
    "Analiza el caso y estima la probabilidad de que este crédito presente una mora "
    "mayor de 60 días dentro de los primeros 150 días de observación. Usa solamente "
    "los datos proporcionados. Razona brevemente y termina tu respuesta con una única "
    "línea con este formato EXACTO:\n"
    "PROBABILIDAD_DE_MORA: <entero de 0 a 100>"
)

_RE_LABELED = re.compile(r"PROBABILIDAD_DE_MORA\s*:\s*([0-9]{1,3})")


def parse_prob_labeled(text: str) -> float:
    """Extrae únicamente la última probabilidad que respeta el formato pedido."""
    for value in reversed(_RE_LABELED.findall(text)):
        integer = int(value)
        if 0 <= integer <= 100:
            return integer / 100.0
    return float("nan")


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


def check_ollama(
    model: str = DEFAULT_OLLAMA_MODEL,
    tags_url: str = DEFAULT_OLLAMA_TAGS_URL,
) -> None:
    """Comprueba que el servicio local y el modelo requerido estén disponibles."""
    import requests

    try:
        response = requests.get(tags_url, timeout=5)
        response.raise_for_status()
        available = [item["name"] for item in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise RuntimeError("Ollama no está disponible") from exc
    if model not in available:
        raise RuntimeError(
            f"Modelo {model!r} no encontrado en Ollama. Disponibles: {available}"
        )


def _extract_final_logprob(data: dict) -> float:
    probabilities = data.get("logprobs") or data.get("message", {}).get("logprobs")
    if not probabilities or not isinstance(probabilities, list):
        return float("nan")
    last = float("nan")
    for entry in probabilities:
        if any(character.isdigit() for character in str(entry.get("token", ""))):
            if entry.get("logprob") is not None:
                last = float(entry["logprob"])
    return last


def call_ollama_think(
    messages: list[dict],
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    think_native: bool = True,
    url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 600,
    retries: int = 3,
    num_predict: int = 1024,
    num_ctx: int = 40960,
    temperature: float = 0.0,
) -> dict:
    """Ejecuta Qwen localmente y devuelve la probabilidad y datos de auditoría."""
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think_native,
        "logprobs": True,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "") or ""
            thinking = message.get("thinking", "") or ""
            parsed_text = content if content.strip() else thinking
            return {
                "prob": parse_prob_labeled(parsed_text),
                "content": content,
                "thinking": thinking,
                "logprob": _extract_final_logprob(data),
                "eval_count": data.get("eval_count"),
            }
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                return {
                    "prob": float("nan"),
                    "content": "",
                    "thinking": "",
                    "logprob": float("nan"),
                    "eval_count": None,
                }


def build_knn_space(
    train_frame: pd.DataFrame,
    feature_names: list[str] | None = None,
):
    """Prepara el espacio de las variables del perfil para elegir ocho ejemplos."""
    features = list(feature_names) if feature_names is not None else TU_VARS + FORM_DIRECT_VARS
    missing = [column for column in features if column not in train_frame.columns]
    if missing:
        raise ValueError(f"faltan variables para construir el espacio kNN: {missing}")
    numeric = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    values = train_frame[numeric].apply(pd.to_numeric, errors="coerce")
    transunion = [column for column in numeric if column in TU_VARS]
    values.loc[:, transunion] = values[transunion].mask(values[transunion] < 0)
    means = values.mean()
    deviations = values.std().replace(0, 1.0)
    standardized = ((values - means) / deviations).fillna(0.0).values
    return standardized, means, deviations, numeric


def _standardize_case(
    row: pd.Series,
    means: pd.Series,
    deviations: pd.Series,
    features: list[str],
) -> np.ndarray:
    values = pd.to_numeric(
        pd.Series({feature: row.get(feature) for feature in features}),
        errors="coerce",
    )
    transunion = [feature for feature in features if feature in TU_VARS]
    values.loc[transunion] = values.loc[transunion].mask(values.loc[transunion] < 0)
    return ((values - means) / deviations).fillna(0.0).values


def knn_examples_for_case(
    case_row: pd.Series,
    case_id: str,
    train_frame: pd.DataFrame,
    knn_space,
    n_shots: int,
) -> pd.DataFrame:
    """Selecciona vecinos de entrenamiento balanceados entre las dos clases."""
    standardized, means, deviations, features = knn_space
    if n_shots % 2:
        raise ValueError("La cantidad de ejemplos debe ser par para balancear clases")
    per_class = n_shots // 2
    case = _standardize_case(case_row, means, deviations, features)
    order = np.argsort(np.sqrt(((standardized - case) ** 2).sum(axis=1)))

    selected = []
    seen_ids = set()
    needed = {0: per_class, 1: per_class}
    targets = train_frame["target"].values
    identifiers = train_frame["credito_id_anon"].values
    indices = train_frame.index.values
    for position in order:
        target = int(targets[position])
        identifier = identifiers[position]
        if identifier == case_id or identifier in seen_ids or needed[target] == 0:
            continue
        selected.append(indices[position])
        seen_ids.add(identifier)
        needed[target] -= 1
        if needed[0] == 0 and needed[1] == 0:
            break
    if needed[0] or needed[1]:
        raise ValueError("No fue posible seleccionar ejemplos balanceados suficientes")
    examples = train_frame.loc[selected]
    return examples.sample(frac=1, random_state=SEED).reset_index(drop=True)


def json_safe(value):
    """Convierte tipos de NumPy a valores nativos antes de guardar JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
