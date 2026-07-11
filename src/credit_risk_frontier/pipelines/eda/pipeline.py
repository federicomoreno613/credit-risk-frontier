"""EDA report pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import build_eda_report


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_eda_report,
                inputs=["credit_dataset", "params:eda"],
                outputs="eda_report",
                name="build_eda_report_node",
                tags=["eda", "public"],
            )
        ]
    )
