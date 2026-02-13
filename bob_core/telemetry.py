"""
DecisionTrace: the atomic artifact. Every decision produces one trace.

The trace is constructed in bob_core from adapter raw ingredients.
The adapter provides snapshots; Bob constructs meaning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DecisionTrace:
    """
    Full trace of one routing decision. THE unit of truth.

    Every field logged for falsifiability. If the system gets cheaper
    but quality drops, traces will show it. If it gets cheaper and
    quality holds, we have the artifact.
    """
    step: int
    context_class: int
    governance_state: str
    path: str                          # "cheap" or "full"
    expert_ids: Tuple[int, ...]        # Which experts actually fired
    expert_invocations: int            # THE cost metric
    tokens_processed: int
    loss: float                        # THE quality metric

    # Gate signals (for falsifiability)
    routing_stability: float
    debt_level: float
    motif_survival: float
    gate_passed: bool
    stability_passed: bool
    debt_passed: bool
    survival_passed: bool

    # Motif info
    motif_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "context_class": self.context_class,
            "governance_state": self.governance_state,
            "path": self.path,
            "expert_ids": list(self.expert_ids),
            "expert_invocations": self.expert_invocations,
            "tokens_processed": self.tokens_processed,
            "loss": round(self.loss, 6),
            "routing_stability": round(self.routing_stability, 4),
            "debt_level": round(self.debt_level, 4),
            "motif_survival": round(self.motif_survival, 4),
            "gate_passed": self.gate_passed,
            "stability_passed": self.stability_passed,
            "debt_passed": self.debt_passed,
            "survival_passed": self.survival_passed,
            "motif_id": self.motif_id,
        }
