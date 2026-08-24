"""Entrega intermedia: clásicos, evaluación con caches de Qwen, y figuras."""

from kedro.pipeline import node, pipeline

from . import figures, nodes


def create_classic_pipeline():
    """Logística y XGBoost. No llama a Qwen."""
    return pipeline(
        [
            node(
                func=nodes.run_intermediate_classic_experiments,
                inputs=[
                    "model_input_table",
                    "params:intermediate_classic_experiments",
                    "params:intermediate_delivery",
                ],
                outputs=[
                    "intermediate_classic_predictions",
                    "intermediate_classic_metrics",
                    "intermediate_logreg_coefficients",
                    "intermediate_xgb_shap_summary",
                ],
                name="run_intermediate_classic_experiments_node",
                tags=["classic"],
            ),
        ]
    )


def create_evaluate_pipeline():
    """Junta caches de Qwen, calcula métricas y deja el contrato materializado."""
    return pipeline(
        [
            node(
                func=nodes.build_intermediate_feature_contract,
                inputs=["model_input_table", "params:intermediate_delivery"],
                outputs="intermediate_feature_contract",
                name="build_intermediate_feature_contract_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.combine_intermediate_predictions,
                inputs=[
                    "intermediate_classic_predictions",
                    "qwen_intermediate_tu_form_zero_cache",
                    "qwen_intermediate_tu_form_few8_cache",
                    "qwen_intermediate_tu_form_description_zero_cache",
                    "qwen_intermediate_tu_form_description_few8_cache",
                ],
                outputs="intermediate_predictions",
                name="combine_intermediate_predictions_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.build_intermediate_metrics_table,
                inputs=["intermediate_predictions", "params:intermediate_reporting"],
                outputs="intermediate_metrics_table",
                name="build_intermediate_metrics_table_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.build_qwen_description_comparisons,
                inputs=["intermediate_predictions", "params:intermediate_reporting"],
                outputs="intermediate_qwen_description_comparisons",
                name="build_intermediate_qwen_description_comparisons_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.build_intermediate_results,
                inputs="intermediate_metrics_table",
                outputs="intermediate_results_table",
                name="build_intermediate_results_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.summarize_text_fields,
                inputs="model_input_table",
                outputs="intermediate_text_summary",
                name="summarize_intermediate_text_fields_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.build_transunion_dictionary,
                inputs="tu_dictionary_official",
                outputs="intermediate_tu_dictionary",
                name="build_intermediate_tu_dictionary_node",
                tags=["evaluate"],
            ),
            node(
                func=nodes.build_real_serialization_example,
                inputs=["model_input_table", "params:intermediate_delivery"],
                outputs="intermediate_real_serialization_example",
                name="build_intermediate_real_serialization_example_node",
                tags=["evaluate"],
            ),
        ]
    )


def create_reporting_pipeline():
    """Figuras de la entrega. Lee tablas ya persistidas."""
    plots = [
        (figures.plot_temporal_change, "model_input_table", "fig_intermediate_temporal", "plot_intermediate_temporal_node"),
        (figures.plot_target_distribution, "model_input_table", "fig_intermediate_target", "plot_intermediate_target_node"),
        (figures.plot_text_fields, "intermediate_text_summary", "fig_intermediate_text_fields", "plot_intermediate_text_fields_node"),
        (figures.plot_subcategory_rates, "model_input_table", "fig_intermediate_subcategory", "plot_intermediate_subcategory_node"),
        (figures.plot_bureau_bivariate, ["model_input_table", "intermediate_tu_dictionary"], "fig_intermediate_bureau", "plot_intermediate_bureau_node"),
        (figures.plot_training_associations, ["model_input_table", "intermediate_tu_dictionary"], "fig_intermediate_predictors", "plot_intermediate_predictors_node"),
        (figures.plot_financial_outliers, "model_input_table", "fig_intermediate_outliers", "plot_intermediate_outliers_node"),
        (figures.plot_history_missingness, "model_input_table", "fig_intermediate_history", "plot_intermediate_history_node"),
        (figures.plot_feature_missingness, ["model_input_table", "intermediate_tu_dictionary"], "fig_intermediate_feature_missingness", "plot_intermediate_feature_missingness_node"),
        (figures.plot_segment_rates, "model_input_table", "fig_intermediate_segment_rates", "plot_intermediate_segment_rates_node"),
        (figures.plot_logreg_coefficients, ["intermediate_logreg_coefficients", "intermediate_tu_dictionary"], "fig_intermediate_logreg_coefficients", "plot_intermediate_logreg_coefficients_node"),
        (figures.plot_xgb_shap_summary, ["intermediate_xgb_shap_summary", "intermediate_tu_dictionary"], "fig_intermediate_xgb_shap", "plot_intermediate_xgb_shap_node"),
        (figures.plot_roc_classics, "intermediate_predictions", "fig_intermediate_roc_classics", "plot_intermediate_roc_classics_node"),
        (figures.plot_roc_qwen, "intermediate_predictions", "fig_intermediate_roc_qwen", "plot_intermediate_roc_qwen_node"),
        (figures.plot_auc, "intermediate_results_table", "fig_intermediate_auc", "plot_intermediate_auc_node"),
        (figures.plot_text_delta, "intermediate_qwen_description_comparisons", "fig_intermediate_delta", "plot_intermediate_delta_node"),
    ]
    return pipeline(
        [
            node(func=func, inputs=inputs, outputs=outputs, name=name, tags=["report"])
            for func, inputs, outputs, name in plots
        ]
    )


def create_pipeline(**kwargs):
    return create_classic_pipeline() + create_evaluate_pipeline() + create_reporting_pipeline()
