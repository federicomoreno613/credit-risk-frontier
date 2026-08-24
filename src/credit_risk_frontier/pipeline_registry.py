"""Pipelines vigentes. `__default__` no dispara Qwen."""

from __future__ import annotations

from credit_risk_frontier.pipelines import data_processing, intermediate_delivery


def register_pipelines():
    data = data_processing.create_pipeline()
    classics = intermediate_delivery.create_classic_pipeline()
    evaluate = intermediate_delivery.create_evaluate_pipeline()
    reporting = intermediate_delivery.create_reporting_pipeline()
    return {
        "__default__": classics,
        "data_processing": data,
        "intermediate_classics": classics,
        "intermediate_evaluate": evaluate,
        "intermediate_reporting": reporting,
        "intermediate_delivery": classics + evaluate + reporting,
    }
