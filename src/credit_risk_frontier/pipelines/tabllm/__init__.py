"""Pipeline tabllm (LLM TabLLM: zero/few-shot, gemma, gpt, ablaciones)."""

from .pipeline import VARIANTS, create_pipeline

__all__ = ["create_pipeline", "VARIANTS"]
