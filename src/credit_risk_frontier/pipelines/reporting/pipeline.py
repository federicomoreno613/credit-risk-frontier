"""Pipeline de reporting (huella spaceflights: reporting).

Comparación + figuras + f1 posthoc + validación de artefactos. Consume las métricas
publicadas (evidencia frozen) desde el catálogo; las figuras retornan Figures que
Kedro persiste como PNG. Plano (no namespaced), como el reporting de spaceflights.
"""

from kedro.pipeline import Pipeline, node

from . import figures
from .nodes import (
    build_comparison_table,
    compute_f1_posthoc,
    validate_thesis_artifacts,
)


def create_pipeline(**kwargs) -> Pipeline:
    comparison = Pipeline(
        [
            node(
                func=build_comparison_table,
                inputs=["xgb.metrics", "logreg.metrics", "cv_seg_metrics",
                        "published_llm_nothink_metrics", "published_llm_fewshot_metrics",
                        "params:model_comparison"],
                outputs="comparison_table",
                name="build_comparison_table_node",
                tags=["public"],
            ),
        ]
    )

    figs = Pipeline(
        [
            node(
                func=figures.plot_comparison_segmentos,
                inputs="comparison_table",
                outputs="fig_22_comparacion_auc_segmentos",
                name="fig22_node",
            ),
            node(
                func=figures.plot_fig3_auc_ejemplos,
                inputs=["published_llm_think_zero_metrics", "published_llm_think_few16_metrics",
                        "published_llm_think_few32_metrics", "published_llm_fewshot_metrics",
                        "xgb.metrics"],
                outputs="fig3_auc_ejemplos",
                name="fig3_node",
            ),
            node(
                func=figures.plot_fig4_comparacion,
                inputs=["published_llm_think_few16_metrics", "xgb.metrics",
                        "logreg.metrics", "cv_seg_metrics"],
                outputs="fig4_comparacion_modelos",
                name="fig4_node",
            ),
        ],
        tags=["reporting.figures"],
    )

    artifacts = Pipeline(
        [
            node(
                func=validate_thesis_artifacts,
                inputs=["comparison_table", "dataset_validation_report", "eda_report",
                        "params:thesis_reporting"],
                outputs="thesis_artifact_report",
                name="validate_thesis_artifacts_node",
                tags=["public"],
            ),
        ]
    )

    posthoc = Pipeline(
        [
            node(
                func=compute_f1_posthoc,
                inputs=["gpt_zero.predictions", "params:f1_posthoc"],
                outputs="gpt_zero_f1",
                name="f1_gpt_zero_node",
                tags=["requires_caches"],
            ),
        ]
    )

    return comparison + figs + artifacts + posthoc
