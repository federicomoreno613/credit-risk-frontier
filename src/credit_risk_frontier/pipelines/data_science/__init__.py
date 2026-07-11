"""Pipeline de modelado clásico + TabFM."""

from .pipeline import ALL_VARIANTS, DEFAULT_VARIANTS, create_pipeline

__all__ = ["create_pipeline", "DEFAULT_VARIANTS", "ALL_VARIANTS"]
