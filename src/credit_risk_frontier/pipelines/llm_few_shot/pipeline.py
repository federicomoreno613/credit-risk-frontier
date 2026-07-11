"""Local LLM few-shot pipeline."""

from kedro.pipeline import Pipeline, node

from .nodes import run_llm_few_shot_suite


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_llm_few_shot_suite,
                inputs=["params:paths", "params:llm_few_shot"],
                outputs="llm_few_shot_metrics_kedro",
                name="run_llm_few_shot_suite_node",
                tags=["llm", "local"],
            )
        ]
    )
