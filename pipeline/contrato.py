"""Contrato único del pipeline: rutas, 29 variables, split, modelos y costos.

Las definiciones de dominio viven en ``src/credit_risk_frontier/utils.py``.
Cada modelo escribe ``PRED_DIR/<nombre>.parquet`` con ``COLUMNAS_PREDICCION``.
QLoRA (modelo #8) sigue pendiente en ``07_finetuning/``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from credit_risk_frontier import utils  # noqa: E402  (contrato de dominio)

# ---------------------------------------------------------------------------
# Semilla y variables (el orden es parte del contrato)
# ---------------------------------------------------------------------------
SEED = utils.SEED
TU_VARS = list(utils.TU_VARS)                    # 20 atributos TransUnion
FORM_DIRECT_VARS = list(utils.FORM_DIRECT_VARS)  # 9 declaraciones del formulario
FEATURES_29 = TU_VARS + FORM_DIRECT_VARS
META = list(utils.META)                          # id, fecha, target, set
TEXT = list(utils.TEXT)
TEXTO_LIBRE = "descripcion_negocio"
# Cualitativas restantes (decisión 2026-08: entran al perfil full; se verificó
# que no hay otra columna de texto libre útil en el original de 1.146 cols).
# objetivo_credito viene del extracto original (paso 00) vía join opcional.
CUALITATIVAS = {
    "subcategoria_texto": "rubro del negocio",
    "tipo_credito": "tipo de crédito",
    "otra_categoria_negocio": "otra categoría declarada",
    "objetivo_credito": "objetivo declarado del crédito",
}
# Columnas de texto que se traen del extracto original (solo local, con PII):
EXTRACTO_ORIGINAL = ROOT / "data" / "00_original" / "original_cohorte.parquet"
TEXTO_DESDE_ORIGINAL = {"credits_credit_alternative_goal": "objetivo_credito"}
CORTE_ESPARSO = utils.CORTE_ESPARSO

# ---------------------------------------------------------------------------
# Desenlace y partición temporal — única declaración del split
# target=1: mora > 60 días dentro de los primeros 150 días.
# Partición por fecha_desembolso: 70% train, 15% val, 15% test (PLAN §2.3).
# ---------------------------------------------------------------------------
DESENLACE = {
    "horizon_days": 150,
    "default_dpd": 60,
    "good_dpd": 30,
    "observation_cutoff": None,  # se infiere del máximo fecha_pago local
    "train_fraction": 0.7,
    "validation_fraction": 0.15,
}

# ---------------------------------------------------------------------------
# Función de costos para decisiones (matrices de confusión, umbral óptimo).
# FN = aprobar un crédito que entra en mora (se pierde capital);
# FP = rechazar un buen pagador (se pierde el margen del crédito).
# La razón 5:1 es un placeholder razonable de microcréditos: calibrar con
# monto medio y margen reales antes de reportar costos absolutos en la tesis.
# ---------------------------------------------------------------------------
COSTOS = {"costo_fn": 5.0, "costo_fp": 1.0}

# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
QWEN_MODEL = "qwen3:8b"
GPT_MODEL = os.environ.get("TESIS_GPT_MODEL", "gpt-5.5")
# tu_form_description_full = 29 vars + descripción + CUALITATIVAS (config nueva;
# los dos primeros perfiles no cambian, sus caches siguen válidos).
PERFILES = ("tu_form", "tu_form_description", "tu_form_description_full")
SHOTS = (0, 8, 16)  # decisión 2026-08: 32 ejemplos eran demasiados
# v2 (2026-09): respuesta CLASE + PROBABILIDAD, ejemplos few-shot humanizados y
# sin anclaje 0/100. El nombre del cache incluye la variante: nunca se mezclan
# respuestas de prompts distintos en una misma corrida.
PROMPT_VARIANT = "clase_prob_v2"
QWEN_OPCIONES = {
    "think_native": True,
    "temperature": 0.0,
    "num_ctx": 40960,
    # v2: el prompt con consigna de granularidad alarga el thinking; 1024 se
    # quedaba corto y producia NaN evitables (verificado en el smoke 2026-09-06).
    "num_predict": 2048,
    "retry_num_predict": 3072,
    "request_retries": 3,
    "timeout_seconds": 600,
    "parallel_workers": 4,
}

# ---------------------------------------------------------------------------
# Rutas — entradas crudas, salidas del preprocesamiento y salidas por modelo
# ---------------------------------------------------------------------------
RAW_CREDITOS = ROOT / "data" / "crudo" / "credit_applications_anonymized.csv"
RAW_PAGOS = ROOT / "data" / "crudo" / "01_pagos_cuota.csv"
RAW_LEGACY = ROOT / "data" / "crudo" / "02_dataset_modelo.csv"

SALIDAS = ROOT / "data" / "pipeline"
BASE = SALIDAS / "01_base.parquet"
MANIFIESTO = SALIDAS / "01_manifiesto_particion.json"
VARIABLES = SALIDAS / "02_variables.parquet"
CONTRATO_VARIABLES = SALIDAS / "02_contrato_variables.json"
HUMANIZADO = SALIDAS / "03_humanizado.parquet"
MODELOS = SALIDAS / "modelos"          # artefactos entrenados (.joblib, estudios Optuna)
PRED_DIR = SALIDAS / "predicciones"    # un parquet por modelo
PREDICCIONES = SALIDAS / "predicciones_consolidadas.parquet"
RAZONAMIENTOS = SALIDAS / "razonamientos"  # JSONL: thinking Qwen / reasoning GPT
MONITOREO = SALIDAS / "monitoreo"

# Columnas obligatorias del parquet de predicciones de cada modelo
COLUMNAS_PREDICCION = [
    "modelo", "configuracion", "credito_id_anon", "segmento",
    "y_true", "probabilidad", "valida",
]

def cache_razonamientos(motor: str, etiqueta: str, shots: int):
    """Ruta del cache JSONL de una corrida LLM; incluye la variante de prompt."""
    return RAZONAMIENTOS / f"{motor}_{etiqueta}_few{shots}_{PROMPT_VARIANT}.jsonl"


# Funciones de dominio re-exportadas (una sola implementación, en utils)
credit_metrics = utils.credit_metrics
serializar_perfil = utils.serialize_intermediate_profile
parse_prob = utils.parse_prob_labeled
parse_clase = utils.parse_clase_labeled


def guardar_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(utils.json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def cargar_variables() -> pd.DataFrame:
    """Tabla de variables del paso 02, con la partición ya asignada."""
    if not VARIABLES.exists():
        raise FileNotFoundError(
            f"Falta {VARIABLES}. Correr primero pipeline/01_preprocesamiento/"
        )
    return pd.read_parquet(VARIABLES)


def dividir(frame: pd.DataFrame):
    """Todos los pasos parten de la MISMA partición: la columna `set` del paso 01."""
    train = frame.loc[frame["set"] == "train"].copy()
    val = frame.loc[frame["set"] == "val"].copy()
    test = frame.loc[frame["set"] == "test"].copy()
    return train, val, test


def preparar_numerico(frame: pd.DataFrame) -> pd.DataFrame:
    """Convierte las 29 variables a número; códigos TU negativos = ausencia."""
    result = frame[FEATURES_29].apply(pd.to_numeric, errors="coerce").copy()
    for column in TU_VARS:
        result.loc[result[column] < 0, column] = float("nan")
    return result


def guardar_predicciones(nombre: str, frame: pd.DataFrame) -> Path:
    """Valida el esquema común y escribe el parquet del modelo en PRED_DIR."""
    faltantes = [c for c in COLUMNAS_PREDICCION if c not in frame.columns]
    if faltantes:
        raise ValueError(f"Predicciones de {nombre!r} sin columnas: {faltantes}")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{nombre}.parquet"
    frame.loc[:, COLUMNAS_PREDICCION].to_parquet(path, index=False)
    return path
