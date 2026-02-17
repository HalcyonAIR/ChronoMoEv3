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

    # Governor fields (all Optional, backward compatible)
    governor_decision: Optional[str] = None
    governor_reasons: Optional[List[str]] = None
    forced_exploration: bool = False
    medium_activation: Optional[float] = None
    scar_debt: Optional[float] = None
    cost_cheap_fraction: Optional[float] = None
    commitment_id: Optional[int] = None
    identity_weight: Optional[float] = None

    # Geometry fields (backward compatible)
    router_entropy: Optional[float] = None       # Mean per-token entropy across MoE layers
    churn: Optional[float] = None                # Raw Jaccard distance from previous step
    scar_hit: Optional[bool] = None              # Was this region in a scar neighborhood?
    baseline_loss: Optional[float] = None        # Expensive-path running avg for this class
    neff: Optional[float] = None                 # Effective number of experts (1/Herfindahl)
    flipflop_ema: Optional[float] = None         # Medium clock flipflop EMA
    routing_weights_top: Optional[List[float]] = None  # Mean routing weights for top-k slots

    def to_dict(self) -> Dict:
        d = {
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
        # Governor fields: include when set
        if self.governor_decision is not None:
            d["governor_decision"] = self.governor_decision
        if self.governor_reasons is not None:
            d["governor_reasons"] = self.governor_reasons
        if self.forced_exploration:
            d["forced_exploration"] = True
        if self.medium_activation is not None:
            d["medium_activation"] = round(self.medium_activation, 4)
        if self.scar_debt is not None:
            d["scar_debt"] = round(self.scar_debt, 4)
        if self.cost_cheap_fraction is not None:
            d["cost_cheap_fraction"] = round(self.cost_cheap_fraction, 4)
        if self.commitment_id is not None:
            d["commitment_id"] = self.commitment_id
        if self.identity_weight is not None:
            d["identity_weight"] = round(self.identity_weight, 4)
        # Geometry fields
        if self.router_entropy is not None:
            d["router_entropy"] = round(self.router_entropy, 4)
        if self.churn is not None:
            d["churn"] = round(self.churn, 4)
        if self.scar_hit is not None:
            d["scar_hit"] = self.scar_hit
        if self.baseline_loss is not None:
            d["baseline_loss"] = round(self.baseline_loss, 4)
        if self.neff is not None:
            d["neff"] = round(self.neff, 2)
        if self.flipflop_ema is not None:
            d["flipflop_ema"] = round(self.flipflop_ema, 4)
        if self.routing_weights_top is not None:
            d["routing_wts"] = [round(w, 4) for w in self.routing_weights_top]
        return d
