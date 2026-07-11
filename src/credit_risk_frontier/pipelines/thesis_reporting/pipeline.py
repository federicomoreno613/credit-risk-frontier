"""Thesis artifact validation pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import validate_thesis_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_thesis_artifacts,
                inputs=[
                    "comparison_table_kedro",
                    "dataset_validation_report",
                    "eda_report",
                    "params:thesis_reporting",
                ],
                outputs="thesis_artifact_report",
                name="validate_thesis_artifacts_node",
                tags=["reporting", "public"],
            )
        ]
    )
