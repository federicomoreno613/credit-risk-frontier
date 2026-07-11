"""Pipeline de modelado clásico + TabFM (huella spaceflights: data_science).

Un pipeline base ``split → train → evaluate`` instanciado por namespace (patrón
active/candidate de spaceflights: ``Pipeline(nodes=base, inputs=..., namespace=...)``).
Cada variante define su ``model_options`` en ``parameters_data_science.yml`` bajo
su clave de namespace. ``segment.cv`` es un nodo cross-model aparte.

``DEFAULT_VARIANTS`` (xgb, logreg) es lo que entra al ``__default__`` offline; el
resto (tabfm, brazos de ablación, variantes con leakage) se corre explícitamente
con ``kedro run --pipeline data_science``.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_estimator,
    run_segment_cv,
    split_by_feature_arm,
    train_estimator,
)

# Variantes offline seguras (CPU, sin torch) que reproducen la evidencia publicada.
DEFAULT_VARIANTS = ["xgb", "logreg"]

# Set completo: clásicos no-leak (citables), con-leak (para la tabla), TabFM, y los
# 7 brazos de la ablación ML de fuentes.
ALL_VARIANTS = DEFAULT_VARIANTS + [
    "xgb_leak", "logreg_leak", "tabfm",
    "arm_all_tu", "arm_all_sin_tu", "arm_all_full",
    "arm_llm_tu", "arm_llm_tu_sem", "arm_llm_sin_tu", "arm_llm_full",
]

# Tags por variante para excluir del default lo que necesita torch/servidor.
_NEURAL = {"tabfm"}


def _model_template() -> Pipeline:
    """Template base de un modelo: split → train → evaluate."""
    return Pipeline(
        [
            node(
                func=split_by_feature_arm,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "features"],
                name="split_data_node",
            ),
            node(
                func=train_estimator,
                inputs=["X_train", "y_train", "X_val", "y_val", "params:model_options"],
                outputs="classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_estimator,
                inputs=["classifier", "X_train", "y_train", "X_val", "y_val",
                        "X_test", "y_test", "params:model_options"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )


def _instance(variant: str) -> Pipeline:
    tags = ["classic"]
    if variant in _NEURAL:
        tags = ["neural", "requires_tabfm"]
    return pipeline(
        _model_template(),
        inputs={"model_input_table": "model_input_table"},
        namespace=variant,
        tags=tags,
    )


def create_pipeline(**kwargs) -> Pipeline:
    variants = kwargs.get("variants", ALL_VARIANTS)
    model_pipes = sum((_instance(v) for v in variants), Pipeline([]))
    # cv_seg (script 05) consume los best_params del XGBoost citable (no-leak), que
    # está en el __default__ offline. El valor se recalcula respecto del committeado
    # (que usaba best_params con leakage) — cambio intencional, coherente con la
    # decisión de baseline no-leak.
    seg = Pipeline(
        [
            node(
                func=run_segment_cv,
                inputs=["model_input_table", "xgb.metrics", "params:segment_cv"],
                outputs="cv_seg_metrics",
                name="run_segment_cv_node",
                tags=["classic", "public"],
            )
        ]
    )
    return model_pipes + seg
