"""
ChronoMoE Integration for swiss-ai/MoE.

Provides lifecycle-aware MoE layers with fixed-width routing.
"""

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.expert_registry import (
    ExpertRegistry,
    ExpertState,
    ProbationConfig,
    ExpertInfo,
)
from chronomoe_integration.fixed_width_router import FixedWidthRouter

__all__ = [
    "ChronoMoE",
    "ExpertRegistry",
    "ExpertState",
    "ProbationConfig",
    "ExpertInfo",
    "FixedWidthRouter",
]
