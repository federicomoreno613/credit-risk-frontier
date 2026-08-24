"""Contrato compartido de los notebooks del plan final.

Los scripts en `nbs/` importan este módulo. No redefinir rutas, variables ni
métricas en cada archivo. Fuente de datos: catalog.yml → model_input_table.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from credit_risk_frontier import utils

ROOT = Path(__file__).resolve().parents[2]

SEED = utils.SEED
META = list(utils.META)
TEXT = list(utils.TEXT)
TU_VARS = list(utils.TU_VARS)
FORM_DIRECT_VARS = list(utils.FORM_DIRECT_VARS)
FEATURES_29 = TU_VARS + FORM_DIRECT_VARS
FREE_TEXT = "descripcion_negocio"
CORTE_ESPARSO = utils.CORTE_ESPARSO
QWEN_MODEL = "qwen3:8b"

MODEL_INPUT = ROOT / "data" / "05_model_input" / "model_input_table.parquet"
REPORTING = ROOT / "data" / "08_reporting" / "intermedia_20260714_redesign"
NOTEBOOK_OUT = ROOT / "data" / "08_reporting" / "notebooks"
FIGURES = ROOT / "figures" / "tesis"
PLAN = ROOT / "entregas" / "PLAN de tesis Federico Moreno.md"
ENTREGA = ROOT / "entregas" / "ENTREGA-INTERMEDIA-G1-MORENO-FEDERICO-2026.md"

credit_metrics = utils.credit_metrics
annotate_segments = utils.annotate_segments
serialize_intermediate_profile = utils.serialize_intermediate_profile
intermediate_feature_columns = utils.intermediate_feature_columns


def is_smoke() -> bool:
    return os.environ.get("TESIS_SMOKE", "").strip() in {"1", "true", "True", "yes"}


def load_model_input() -> pd.DataFrame:
    if not MODEL_INPUT.exists():
        raise FileNotFoundError(
            f"Falta {MODEL_INPUT}. Correr: poetry run kedro run --pipeline data_processing"
        )
    frame = pd.read_parquet(MODEL_INPUT)
    return utils.annotate_segments(frame)


def split_sets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["set"] == "train"].copy()
    val = frame.loc[frame["set"] == "val"].copy()
    test = frame.loc[frame["set"] == "test"].copy()
    return train, val, test


def smoke_frame(frame: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    if not is_smoke() or len(frame) <= n:
        return frame
    return frame.sample(n=n, random_state=SEED)


def structured_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.loc[:, FEATURES_29].copy(), frame["target"].astype(int)


def save_table(frame: pd.DataFrame, name: str) -> Path:
    NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_OUT / name
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def figures_dir() -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES
