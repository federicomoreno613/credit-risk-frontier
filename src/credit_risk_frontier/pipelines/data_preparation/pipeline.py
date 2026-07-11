"""Pipeline for validating the curated credit-risk dataset."""

from kedro.pipeline import Pipeline, node

from .nodes import validate_credit_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_credit_dataset,
                inputs=["credit_dataset", "params:dataset_contract"],
                outputs="dataset_validation_report",
                name="validate_credit_dataset_node",
                tags=["data", "public"],
            )
        ]
    )
