"""Pipeline tabllm — 18 instancias namespaced de un template base (cache-first).

Igual que el data_science: un template ``select → score → canonicalize → evaluate``
instanciado por namespace (patrón active/candidate de spaceflights). Cada variante
define su ``variant`` en ``parameters_tabllm.yml`` y su ``cache`` frozen en el
catálogo (entrada explícita a la ruta legacy del parquet).

Excluido del ``__default__`` offline: aunque con ``allow_infer: false`` no contacta
la red, agrupa la familia LLM que se corre a demanda (``kedro run --pipeline tabllm``).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    canonicalize_predictions,
    evaluate_predictions,
    score_llm,
    select_target_rows,
)

# Las 18 variantes. El backend/config de cada una vive en parameters_tabllm.yml;
# su cache frozen en catalog.yml (entrada explícita <namespace>.cache).
VARIANTS = [
    "qwen_nothink", "qwen_few8", "qwen_few16", "qwen_few32",
    "qwen_think_zero", "qwen_think_few16", "qwen_think_few32",
    "gemma_zero", "gemma_few16",
    "gemma_abl_tu", "gemma_abl_tusem", "gemma_abl_sintu",
    "gpt_zero", "gpt_few16",
    "prompt_v0", "prompt_v2", "prompt_v3",
]

_REQUIRES_OPENAI = {"gpt_zero", "gpt_few16"}


def _tabllm_template() -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_target_rows,
                inputs=["model_input_table", "cache", "params:variant"],
                outputs="target_rows",
                name="select_sample_node",
            ),
            node(
                func=score_llm,
                inputs=["target_rows", "cache", "model_input_table", "params:variant"],
                outputs="scored",
                name="score_llm_node",
            ),
            node(
                func=canonicalize_predictions,
                inputs=["scored", "model_input_table"],
                outputs="predictions",
                name="canonicalize_node",
            ),
            node(
                func=evaluate_predictions,
                inputs=["predictions", "params:variant"],
                outputs="metrics",
                name="evaluate_node",
            ),
        ]
    )


def _instance(variant: str) -> Pipeline:
    backend_tag = "requires_openai" if variant in _REQUIRES_OPENAI else "requires_ollama"
    return pipeline(
        _tabllm_template(),
        inputs={"model_input_table": "model_input_table"},
        namespace=variant,
        tags=["tabllm", backend_tag],
    )


def create_pipeline(**kwargs) -> Pipeline:
    variants = kwargs.get("variants", VARIANTS)
    return sum((_instance(v) for v in variants), Pipeline([]))
