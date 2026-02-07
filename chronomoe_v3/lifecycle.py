"""
Lifecycle coordinator: Detects prune candidates based on coherence feedback.

This is a dry-run implementation - detects and logs decisions without executing.
Proves that lifecycle can read all three state containers correctly.

Phase 2 scope: Prune detection only
Phase 3+: Split, merge, spawn
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from torch import Tensor

from .coherence import CoherenceState


@dataclass
class PruneDecision:
    """
    Proposed prune action.

    Dry-run only: logged but not executed.
    """

    expert_id: str
    reason: str  # "low_coherence" or "low_observability"
    phi_slow: float
    phi_fast: float
    phi_delta: float
    total_tokens_seen: int
    step: int

    def to_dict(self) -> Dict:
        """Serialize to dict for logging."""
        return {
            "expert_id": self.expert_id,
            "reason": self.reason,
            "phi_slow": self.phi_slow,
            "phi_fast": self.phi_fast,
            "phi_delta": self.phi_delta,
            "total_tokens_seen": self.total_tokens_seen,
            "step": self.step,
        }


class LifecycleCoordinator:
    """
    Lifecycle coordinator: Reads coherence state, proposes prune decisions.

    Phase 2 scope: Prune detection only (dry-run)
    - Detects experts with low phi_slow
    - Checks starvation before proposing prune
    - Logs decisions, does NOT execute

    The coordinator is a READER, not a WRITER.
    It reads CoherenceState and RouterState, writes only decision logs.
    """

    def __init__(
        self,
        prune_threshold: float = 0.3,
        min_tokens_threshold: int = 1000,
        starvation_threshold: float = 0.5,
    ):
        """
        Initialize lifecycle coordinator.

        Args:
            prune_threshold: phi_slow below this → prune candidate
            min_tokens_threshold: Min tokens seen before prune consideration
            starvation_threshold: Layer coherence below this → don't prune
        """
        self.prune_threshold = prune_threshold
        self.min_tokens_threshold = min_tokens_threshold
        self.starvation_threshold = starvation_threshold

        # Decision log (dry-run)
        self.decisions: List[PruneDecision] = []

    def evaluate_layer(
        self,
        layer_id: int,
        coherence_snapshot: Dict[int, CoherenceState],
        step: int,
    ) -> List[PruneDecision]:
        """
        Evaluate prune candidates for one layer.

        Dry-run only: Returns proposed decisions, does NOT execute.

        Args:
            layer_id: Layer to evaluate
            coherence_snapshot: {expert_id: CoherenceState}
            step: Current training step

        Returns:
            List of PruneDecision proposals
        """
        candidates = []

        # Check if layer is starving (low overall coherence)
        layer_coherence = self._compute_layer_coherence(coherence_snapshot)
        layer_is_starving = layer_coherence < self.starvation_threshold

        if layer_is_starving:
            # Don't prune if layer is already struggling
            return []

        # Check each expert
        for expert_id, coh_state in coherence_snapshot.items():
            # Skip if not observed enough
            if coh_state.total_tokens_seen < self.min_tokens_threshold:
                continue

            # Criterion 1: Low coherence (decoherent)
            if coh_state.phi_slow < self.prune_threshold:
                decision = PruneDecision(
                    expert_id=coh_state.expert_id,
                    reason="low_coherence",
                    phi_slow=coh_state.phi_slow,
                    phi_fast=coh_state.phi_fast,
                    phi_delta=coh_state.phi_delta,
                    total_tokens_seen=coh_state.total_tokens_seen,
                    step=step,
                )
                candidates.append(decision)
                self.decisions.append(decision)

        return candidates

    def _compute_layer_coherence(
        self, snapshot: Dict[int, CoherenceState]
    ) -> float:
        """
        Compute weighted average coherence for layer.

        Uses phi_slow, weighted by observation count.

        Args:
            snapshot: {expert_id: CoherenceState}

        Returns:
            Layer coherence (weighted average)
        """
        if not snapshot:
            return 0.0

        phi_values = []
        weights = []

        for coh_state in snapshot.values():
            if coh_state.total_tokens_seen > 0:
                phi_values.append(coh_state.phi_slow)
                weights.append(float(coh_state.total_tokens_seen))

        if not phi_values:
            return 0.0

        phi_tensor = torch.tensor(phi_values)
        weight_tensor = torch.tensor(weights)
        weight_tensor = weight_tensor / weight_tensor.sum()

        return (phi_tensor * weight_tensor).sum().item()

    def get_statistics(self) -> Dict:
        """
        Get coordinator statistics.

        Returns:
            Dict with decision counts and breakdown
        """
        return {
            "total_decisions": len(self.decisions),
            "by_reason": self._count_by_reason(),
        }

    def _count_by_reason(self) -> Dict[str, int]:
        """Count decisions by reason."""
        counts = {}
        for decision in self.decisions:
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return counts

    def clear_decisions(self):
        """Clear decision log."""
        self.decisions.clear()


def compute_neff(routing_probs: Tensor) -> float:
    """
    Compute effective number of experts (Neff).

    Neff = exp(entropy) where entropy = -sum(p * log(p))

    This is a starvation proxy: low Neff means routing is concentrated
    on few experts.

    Args:
        routing_probs: [B×S, E] - routing probabilities

    Returns:
        Neff: Effective number of experts
    """
    # Average routing distribution
    p_avg = routing_probs.mean(dim=0)  # [E]

    # Compute entropy
    entropy = -(p_avg * (p_avg + 1e-9).log()).sum()

    # Neff = exp(entropy)
    return torch.exp(entropy).item()


def compute_saturation(routing_probs: Tensor) -> float:
    """
    Compute routing saturation: max expert's average mass.

    High saturation (>0.5) means one expert is dominating.

    Args:
        routing_probs: [B×S, E] - routing probabilities

    Returns:
        saturation: Max expert mass in [0, 1]
    """
    p_avg = routing_probs.mean(dim=0)  # [E]
    return p_avg.max().item()
