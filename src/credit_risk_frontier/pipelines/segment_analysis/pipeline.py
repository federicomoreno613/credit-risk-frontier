"""Temporal segment-analysis pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import run_segment_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_segment_analysis,
                inputs=["params:paths", "params:segment_analysis", "xgboost_metrics_kedro"],
                outputs="segment_metrics_kedro",
                name="run_segment_analysis_node",
                tags=["segments", "public"],
            )
        ]
    )
