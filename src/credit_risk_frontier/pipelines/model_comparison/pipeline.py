"""Model comparison pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import build_model_comparison


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_model_comparison,
                inputs=[
                    "xgboost_metrics_kedro",
                    "logreg_metrics_kedro",
                    "segment_metrics_kedro",
                    "params:paths",
                    "params:model_comparison",
                ],
                outputs="comparison_table_kedro",
                name="build_model_comparison_node",
                tags=["comparison", "public"],
            )
        ]
    )
