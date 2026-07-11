"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from credit_risk_frontier.pipelines import (
    classic_models,
    data_preparation,
    eda,
    llm_few_shot,
    llm_zero_shot,
    model_comparison,
    segment_analysis,
    thesis_reporting,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register safe public pipelines separately from local/LLM pipelines."""
    data_preparation_pipeline = data_preparation.create_pipeline()
    eda_pipeline = eda.create_pipeline()
    classic_models_pipeline = classic_models.create_pipeline()
    segment_analysis_pipeline = segment_analysis.create_pipeline()
    model_comparison_pipeline = model_comparison.create_pipeline()
    thesis_reporting_pipeline = thesis_reporting.create_pipeline()
    llm_zero_shot_pipeline = llm_zero_shot.create_pipeline()
    llm_few_shot_pipeline = llm_few_shot.create_pipeline()

    public_repro = (
        data_preparation_pipeline
        + eda_pipeline
        + classic_models_pipeline
        + segment_analysis_pipeline
        + model_comparison_pipeline
        + thesis_reporting_pipeline
    )
    llm_local = llm_zero_shot_pipeline + llm_few_shot_pipeline

    return {
        "__default__": public_repro,
        "public_repro": public_repro,
        "data_preparation": data_preparation_pipeline,
        "eda": eda_pipeline,
        "classic_models": classic_models_pipeline,
        "segment_analysis": segment_analysis_pipeline,
        "model_comparison": model_comparison_pipeline,
        "thesis_reporting": thesis_reporting_pipeline,
        "llm_zero_shot": llm_zero_shot_pipeline,
        "llm_few_shot": llm_few_shot_pipeline,
        "llm_local": llm_local,
        "full_local": public_repro + llm_local,
    }
