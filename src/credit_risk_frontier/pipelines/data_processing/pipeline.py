"""Pipeline de procesamiento de datos (huella spaceflights: data_processing)."""

from kedro.pipeline import Pipeline, node

from .nodes import build_eda_report, create_model_input_table, validate_credit_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_credit_dataset,
                inputs=["credit_dataset", "params:dataset_contract"],
                outputs="dataset_validation_report",
                name="validate_credit_dataset_node",
                tags=["data", "public"],
            ),
            node(
                func=create_model_input_table,
                inputs=["credit_dataset", "params:dataset_contract"],
                outputs="model_input_table",
                name="create_model_input_table_node",
                tags=["data", "public"],
            ),
            node(
                func=build_eda_report,
                inputs=["credit_dataset", "params:eda"],
                outputs="eda_report",
                name="build_eda_report_node",
                tags=["eda", "public"],
            ),
        ]
    )
