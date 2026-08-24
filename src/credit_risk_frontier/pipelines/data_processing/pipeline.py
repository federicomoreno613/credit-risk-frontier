"""Cohorte: validar → unir pagos → target → split temporal."""

from kedro.pipeline import node, pipeline

from .nodes import (
    build_bridge_coverage_report,
    build_credit_outcomes,
    build_exact_credit_bridge,
    build_outcome_sensitivity,
    build_split_manifest,
    create_model_input,
    validate_credit_dataset,
)


def create_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=validate_credit_dataset,
                inputs=["credit_dataset", "params:dataset_contract"],
                outputs="dataset_validation_report",
                name="validate_credit_dataset_node",
                tags=["data"],
            ),
            node(
                func=build_exact_credit_bridge,
                inputs=["legacy_model_dataset", "credit_dataset"],
                outputs="credit_bridge",
                name="build_exact_credit_bridge_node",
                tags=["data"],
            ),
            node(
                func=build_bridge_coverage_report,
                inputs=["credit_dataset", "credit_bridge"],
                outputs="bridge_coverage_report",
                name="build_bridge_coverage_report_node",
                tags=["data"],
            ),
            node(
                func=build_credit_outcomes,
                inputs=["credit_dataset", "payment_schedule", "credit_bridge", "params:outcome_definition"],
                outputs="credit_outcomes_150d",
                name="build_credit_outcomes_150d_node",
                tags=["data"],
            ),
            node(
                func=create_model_input,
                inputs=["credit_dataset", "credit_outcomes_150d", "params:outcome_definition"],
                outputs="model_input_table",
                name="create_model_input_node",
                tags=["data"],
            ),
            node(
                func=build_split_manifest,
                inputs=[
                    "credit_dataset",
                    "credit_bridge",
                    "credit_outcomes_150d",
                    "model_input_table",
                    "params:split_contract",
                ],
                outputs="split_manifest",
                name="build_split_manifest_node",
                tags=["data"],
            ),
            node(
                func=build_outcome_sensitivity,
                inputs=["credit_dataset", "payment_schedule", "credit_bridge", "params:outcome_definition"],
                outputs="outcome_sensitivity_report",
                name="build_outcome_sensitivity_node",
                tags=["data"],
            ),
        ]
    )
