"""Registro de pipelines (canónico: find_pipelines + __default__ offline).

``find_pipelines()`` descubre por carpeta cada pipeline bajo ``pipelines/`` con un
``create_pipeline()`` (data_processing, data_science, tabllm, reporting). Las
variantes namespaced viven DENTRO de cada ``create_pipeline()``, así que cada
carpeta es una sola clave de registry.

``__default__`` debe correr offline y en CPU: se excluye ``tabllm`` (necesita
transportes ollama/openai) y las variantes de ``data_science`` que requieren torch
o son opcionales (TabFM y los brazos de ablación) — para eso se reconstruye
``data_science`` con solo ``DEFAULT_VARIANTS`` (xgb, logreg) en el default.
"""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from credit_risk_frontier.pipelines import data_science


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines()

    # data_science offline = solo los clásicos citables (xgb, logreg no-leak).
    ds_offline = data_science.create_pipeline(variants=data_science.DEFAULT_VARIANTS)

    default_parts = [
        pipelines.get("data_processing", Pipeline([])),
        ds_offline,
        pipelines.get("reporting", Pipeline([])),
    ]
    pipelines["__default__"] = sum(default_parts, Pipeline([]))
    pipelines["public_repro"] = pipelines["__default__"]
    return pipelines
