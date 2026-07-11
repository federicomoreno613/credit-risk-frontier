"""Local LLM zero-shot pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import run_llm_zero_shot_nothink, run_llm_zero_shot_think


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_llm_zero_shot_nothink,
                inputs=["params:paths", "params:llm_zero_shot"],
                outputs="llm_zero_shot_nothink_metrics_kedro",
                name="run_llm_zero_shot_nothink_node",
                tags=["llm", "local"],
            ),
            node(
                func=run_llm_zero_shot_think,
                inputs=["params:paths", "params:llm_zero_shot"],
                outputs="llm_zero_shot_think_metrics_kedro",
                name="run_llm_zero_shot_think_node",
                tags=["llm", "local"],
            ),
        ]
    )
