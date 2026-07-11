"""Classic ML model pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import run_logreg_baseline, run_xgboost_baseline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_xgboost_baseline,
                inputs=["params:paths", "params:classic_models.xgboost"],
                outputs="xgboost_metrics_kedro",
                name="run_xgboost_baseline_node",
                tags=["models", "classic", "public"],
            ),
            node(
                func=run_logreg_baseline,
                inputs=["params:paths", "params:classic_models.logreg"],
                outputs="logreg_metrics_kedro",
                name="run_logreg_baseline_node",
                tags=["models", "classic", "public"],
            ),
        ]
    )
