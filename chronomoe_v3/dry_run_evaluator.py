"""
Dry run evaluator: test edits without committing.

Answers ONE question: If we apply this edit, do Phase 1-4 signals improve?

This keeps F_l a sensor, not an objective. We don't minimize F_l directly.
We check if proposed changes make the diagnostics better.

Key principle: Evidence must persist. Proposed at N, executed at N+1 only
if improvement signal holds. This prevents "clever one-window hacks" from
becoming architecture.

CRITICAL: "Doing nothing" is a valid outcome. Edits must clear minimum
improvement thresholds or they are rejected with reason "insufficient_benefit".
This prevents edit churn under noise.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any

import torch
from torch import Tensor

from .edit_proposals import EditEvidence, EditProposal


# Global minimum improvement thresholds
# Edits that don't clear these are rejected as "insufficient_benefit"
MIN_DELTA_F = 0.01  # Minimum F_l reduction to justify edit
MIN_DELTA_PSI = 0.02  # Minimum coherence improvement to justify edit


@dataclass
class DryRunResult:
    """
    Result of dry-run evaluation.

    Compares predicted improvement vs actual improvement after simulating edit.
    """

    # Predictions (from proposal evidence)
    predicted_delta_f_l: float
    predicted_delta_psi: float

    # Actual (from dry run simulation)
    actual_delta_f_l: float
    actual_delta_psi: float
    actual_delta_neff: Optional[float] = None

    # Verdict
    improvement_confirmed: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "predicted_delta_f_l": self.predicted_delta_f_l,
            "predicted_delta_psi": self.predicted_delta_psi,
            "actual_delta_f_l": self.actual_delta_f_l,
            "actual_delta_psi": self.actual_delta_psi,
            "actual_delta_neff": self.actual_delta_neff,
            "improvement_confirmed": self.improvement_confirmed,
            "reason": self.reason,
        }


class DryRunEvaluator:
    """
    Evaluate edit proposals by simulation.

    Does NOT modify model state. Only simulates what would happen.
    """

    def __init__(
        self,
        min_f_l_improvement: float = MIN_DELTA_F,
        min_psi_improvement: float = MIN_DELTA_PSI,
    ):
        """
        Initialize evaluator.

        Args:
            min_f_l_improvement: Minimum ΔF_l to consider "improvement" (default: MIN_DELTA_F)
            min_psi_improvement: Minimum ΔPsi to consider "improvement" (default: MIN_DELTA_PSI)
        """
        self.min_f_l_improvement = min_f_l_improvement
        self.min_psi_improvement = min_psi_improvement

    def evaluate_spawn(
        self,
        proposal: EditProposal,
        current_f_l: float,
        current_psi: float,
        current_neff: Optional[float],
        simulate_fn: Callable[[EditProposal], tuple[float, float, Optional[float]]],
    ) -> DryRunResult:
        """
        Evaluate a spawn proposal by simulation.

        Args:
            proposal: The spawn proposal to evaluate
            current_f_l: Current free energy
            current_psi: Current layer coherence
            current_neff: Current effective expert count
            simulate_fn: Function that simulates spawn and returns (f_l, psi, neff)

        Returns:
            DryRunResult with verdict
        """
        # Get predictions from evidence
        if proposal.evidence is None:
            return DryRunResult(
                predicted_delta_f_l=0.0,
                predicted_delta_psi=0.0,
                actual_delta_f_l=0.0,
                actual_delta_psi=0.0,
                improvement_confirmed=False,
                reason="no_evidence",
            )

        predicted_delta_f_l = proposal.evidence.delta_f_l
        predicted_delta_psi = proposal.evidence.psi_predicted - proposal.evidence.psi_before

        # Simulate the edit
        simulated_f_l, simulated_psi, simulated_neff = simulate_fn(proposal)

        # Compute actual deltas
        actual_delta_f_l = simulated_f_l - current_f_l
        actual_delta_psi = simulated_psi - current_psi
        actual_delta_neff = (
            simulated_neff - current_neff if simulated_neff is not None and current_neff is not None else None
        )

        # Check if improvement confirmed
        f_l_improved = actual_delta_f_l < -self.min_f_l_improvement
        psi_improved = actual_delta_psi > self.min_psi_improvement

        if f_l_improved and psi_improved:
            improvement_confirmed = True
            reason = "both_f_l_and_psi_improved"
        elif f_l_improved:
            improvement_confirmed = True
            reason = "f_l_improved"
        elif psi_improved:
            improvement_confirmed = True
            reason = "psi_improved"
        else:
            improvement_confirmed = False
            reason = f"insufficient_benefit: ΔF_l={actual_delta_f_l:.3f} (need <-{self.min_f_l_improvement}), ΔPsi={actual_delta_psi:.3f} (need >{self.min_psi_improvement})"

        return DryRunResult(
            predicted_delta_f_l=predicted_delta_f_l,
            predicted_delta_psi=predicted_delta_psi,
            actual_delta_f_l=actual_delta_f_l,
            actual_delta_psi=actual_delta_psi,
            actual_delta_neff=actual_delta_neff,
            improvement_confirmed=improvement_confirmed,
            reason=reason,
        )

    def evaluate_proposal(
        self,
        proposal: EditProposal,
        current_state: dict,
        simulate_fn: Callable[[EditProposal], dict],
    ) -> DryRunResult:
        """
        Generic evaluation for any proposal type.

        Args:
            proposal: The edit proposal
            current_state: Current diagnostic state (f_l, psi, neff, etc.)
            simulate_fn: Function that simulates edit and returns new state

        Returns:
            DryRunResult with verdict
        """
        if proposal.edit_type == "SPAWN":
            return self.evaluate_spawn(
                proposal,
                current_state["f_l"],
                current_state["psi"],
                current_state.get("neff"),
                lambda p: (
                    simulate_fn(p)["f_l"],
                    simulate_fn(p)["psi"],
                    simulate_fn(p).get("neff"),
                ),
            )
        else:
            raise NotImplementedError(f"Dry run for {proposal.edit_type} not implemented yet")


def create_spawn_evidence(
    f_l_before: float,
    components_before: Any,  # FreeEnergyComponents
    psi_before: float,
    neff_before: Optional[float],
    # After spawn predictions
    num_experts_after: int,
    predicted_psi_improvement: float = 0.1,
    lambda_complexity: float = 0.01,
) -> EditEvidence:
    """
    Create evidence for spawn proposal.

    Predictions:
    - Misfit decreases (more capacity → better coherence)
    - Complexity increases (one more expert)
    - Net F_l should decrease if spawn justified

    Args:
        f_l_before: Current free energy
        components_before: Current free energy components
        psi_before: Current layer coherence
        neff_before: Current effective expert count
        num_experts_after: Number of experts after spawn
        predicted_psi_improvement: Expected coherence improvement
        lambda_complexity: Complexity weight

    Returns:
        EditEvidence with predictions
    """
    # Predict improvement in coherence
    psi_predicted = min(1.0, psi_before + predicted_psi_improvement)

    # Predict new misfit
    misfit_predicted = 1.0 - psi_predicted

    # Predict new complexity (one more expert)
    complexity_predicted = lambda_complexity * num_experts_after

    # Other components stay same
    redundancy_predicted = components_before.redundancy
    instability_predicted = components_before.instability

    # Predict new F_l
    f_l_predicted = (
        misfit_predicted + complexity_predicted + redundancy_predicted + instability_predicted
    )

    # Predict Neff improvement (more experts → higher Neff, but modest)
    neff_predicted = neff_before + 0.5 if neff_before is not None else None

    return EditEvidence(
        f_l_before=f_l_before,
        f_l_predicted=f_l_predicted,
        delta_f_l=f_l_predicted - f_l_before,
        misfit_before=components_before.misfit,
        misfit_predicted=misfit_predicted,
        complexity_before=components_before.complexity,
        complexity_predicted=complexity_predicted,
        redundancy_before=components_before.redundancy,
        redundancy_predicted=redundancy_predicted,
        instability_before=components_before.instability,
        instability_predicted=instability_predicted,
        psi_before=psi_before,
        psi_predicted=psi_predicted,
        neff_before=neff_before,
        neff_predicted=neff_predicted,
        trigger="layer_starving",
    )


def create_prune_evidence(
    f_l_before: float,
    components_before: Any,  # FreeEnergyComponents
    psi_before: float,
    neff_before: Optional[float],
    # After prune predictions
    num_experts_after: int,
    target_expert_coherence: float,
    target_expert_utilization: float,
    predicted_psi_delta: float = 0.0,
    lambda_complexity: float = 0.01,
) -> EditEvidence:
    """
    Create evidence for prune proposal.

    Predictions:
    - Misfit increases slightly (less capacity) OR stays same if expert was useless
    - Complexity decreases (one fewer expert)
    - Redundancy improves (removing a redundant expert)
    - Net F_l should decrease if prune justified

    Args:
        f_l_before: Current free energy
        components_before: Current free energy components
        psi_before: Current layer coherence
        neff_before: Current effective expert count
        num_experts_after: Number of experts after prune
        target_expert_coherence: phi_slow of expert being pruned
        target_expert_utilization: Utilization of expert being pruned
        predicted_psi_delta: Expected coherence change (usually 0 or small negative)
        lambda_complexity: Complexity weight

    Returns:
        EditEvidence with predictions
    """
    # Predict coherence change
    # If expert is decoherent (low phi_slow), removing it has minimal impact
    # If expert is low-utilization, removing it has minimal impact
    psi_predicted = max(0.0, min(1.0, psi_before + predicted_psi_delta))

    # Predict new misfit
    misfit_predicted = 1.0 - psi_predicted

    # Predict new complexity (one fewer expert)
    complexity_predicted = lambda_complexity * num_experts_after

    # Predict redundancy improvement (removing a decoherent expert reduces redundancy)
    # If expert was decoherent, it was likely redundant or useless
    redundancy_improvement = 0.05 if target_expert_coherence < 0.5 else 0.02
    redundancy_predicted = max(0.0, components_before.redundancy - redundancy_improvement)

    # Predict instability improvement (removing unstable expert helps)
    instability_improvement = 0.03 if target_expert_coherence < 0.3 else 0.01
    instability_predicted = max(0.0, components_before.instability - instability_improvement)

    # Predict new F_l
    f_l_predicted = (
        misfit_predicted + complexity_predicted + redundancy_predicted + instability_predicted
    )

    # Predict Neff change (removing expert → lower Neff, but modest if expert was low-util)
    neff_predicted = None
    if neff_before is not None:
        # If expert had low utilization, Neff barely changes
        neff_delta = -0.2 if target_expert_utilization > 100 else -0.05
        neff_predicted = max(1.0, neff_before + neff_delta)

    return EditEvidence(
        f_l_before=f_l_before,
        f_l_predicted=f_l_predicted,
        delta_f_l=f_l_predicted - f_l_before,
        misfit_before=components_before.misfit,
        misfit_predicted=misfit_predicted,
        complexity_before=components_before.complexity,
        complexity_predicted=complexity_predicted,
        redundancy_before=components_before.redundancy,
        redundancy_predicted=redundancy_predicted,
        instability_before=components_before.instability,
        instability_predicted=instability_predicted,
        psi_before=psi_before,
        psi_predicted=psi_predicted,
        neff_before=neff_before,
        neff_predicted=neff_predicted,
        trigger="expert_decoherent",
    )


def create_split_evidence(
    f_l_before: float,
    components_before: Any,  # FreeEnergyComponents
    psi_before: float,
    neff_before: Optional[float],
    # After split predictions
    num_experts_after: int,
    source_expert_bimodality: float,
    source_expert_utilization: float,
    predicted_psi_improvement: float = 0.05,
    lambda_complexity: float = 0.01,
    kappa_instability: float = 0.1,
) -> EditEvidence:
    """
    Create evidence for split proposal.

    Predictions:
    - Misfit decreases slightly (two coherent experts better than one bimodal)
    - Complexity increases (one more expert: N → N+1)
    - Instability decreases (bimodal expert → two unimodal experts)
    - Net F_l should decrease if instability reduction beats complexity cost

    Args:
        f_l_before: Current free energy
        components_before: Current free energy components
        psi_before: Current layer coherence
        neff_before: Current effective expert count
        num_experts_after: Number of experts after split (N+1)
        source_expert_bimodality: Bimodality score of expert being split
        source_expert_utilization: Utilization of expert being split
        predicted_psi_improvement: Expected coherence improvement
        lambda_complexity: Complexity weight
        kappa_instability: Instability weight

    Returns:
        EditEvidence with predictions
    """
    # Predict coherence improvement (two coherent experts better than one bimodal)
    psi_predicted = min(1.0, psi_before + predicted_psi_improvement)

    # Predict new misfit
    misfit_predicted = 1.0 - psi_predicted

    # Predict new complexity (one more expert)
    complexity_predicted = lambda_complexity * num_experts_after

    # Other components stay same initially
    redundancy_predicted = components_before.redundancy

    # Predict instability improvement (removing bimodal expert)
    # The bimodal expert contributes: source_bimodality * (utilization / total_utilization)
    # After split, two unimodal experts (bimodality ~0) replace it
    # Assuming bimodality contributes to instability via utilization-weighted mean
    instability_reduction = kappa_instability * source_expert_bimodality * 0.5  # Conservative estimate
    instability_predicted = max(0.0, components_before.instability - instability_reduction)

    # Predict new F_l
    f_l_predicted = (
        misfit_predicted + complexity_predicted + redundancy_predicted + instability_predicted
    )

    # Predict Neff improvement (splitting increases effective expert count)
    neff_predicted = None
    if neff_before is not None:
        # Split increases Neff if source expert had high utilization
        neff_delta = 0.5 if source_expert_utilization > 100 else 0.2
        neff_predicted = neff_before + neff_delta

    return EditEvidence(
        f_l_before=f_l_before,
        f_l_predicted=f_l_predicted,
        delta_f_l=f_l_predicted - f_l_before,
        misfit_before=components_before.misfit,
        misfit_predicted=misfit_predicted,
        complexity_before=components_before.complexity,
        complexity_predicted=complexity_predicted,
        redundancy_before=components_before.redundancy,
        redundancy_predicted=redundancy_predicted,
        instability_before=components_before.instability,
        instability_predicted=instability_predicted,
        psi_before=psi_before,
        psi_predicted=psi_predicted,
        neff_before=neff_before,
        neff_predicted=neff_predicted,
        trigger="expert_bimodal",
    )


def create_merge_evidence(
    f_l_before: float,
    components_before: Any,  # FreeEnergyComponents
    psi_before: float,
    neff_before: Optional[float],
    # After merge predictions
    num_experts_after: int,
    source_a_utilization: float,
    source_b_utilization: float,
    similarity: float,
    predicted_psi_delta: float = -0.02,
    lambda_complexity: float = 0.01,
    rho_redundancy: float = 0.1,
) -> EditEvidence:
    """
    Create evidence for merge proposal.

    Predictions:
    - Misfit increases slightly (less capacity) OR stays same if experts truly redundant
    - Complexity decreases (one fewer expert: N → N-1)
    - Redundancy decreases (removing duplicate experts)
    - Net F_l should decrease if redundancy + complexity reduction beats capacity loss

    WARNING: MERGE is destructive and NOT reversible. Evidence must be strong.

    Args:
        f_l_before: Current free energy
        components_before: Current free energy components
        psi_before: Current layer coherence
        neff_before: Current effective expert count
        num_experts_after: Number of experts after merge (N-1)
        source_a_utilization: Utilization of first expert
        source_b_utilization: Utilization of second expert
        similarity: Similarity between experts (0-1, higher = more similar)
        predicted_psi_delta: Expected coherence change (usually small negative)
        lambda_complexity: Complexity weight
        rho_redundancy: Redundancy weight

    Returns:
        EditEvidence with predictions
    """
    # Predict coherence change
    # If experts are truly redundant (high similarity), removing one has minimal impact
    # If similarity is high, psi_delta should be near zero
    if similarity > 0.9:
        predicted_psi_delta = max(predicted_psi_delta, -0.01)  # Very small impact
    psi_predicted = max(0.0, min(1.0, psi_before + predicted_psi_delta))

    # Predict new misfit
    misfit_predicted = 1.0 - psi_predicted

    # Predict new complexity (one fewer expert)
    complexity_predicted = lambda_complexity * num_experts_after

    # Predict redundancy improvement (removing duplicate experts)
    # High similarity = high redundancy contribution
    redundancy_reduction = rho_redundancy * similarity * 0.1  # Conservative
    redundancy_predicted = max(0.0, components_before.redundancy - redundancy_reduction)

    # Instability stays same (merging doesn't affect bimodality)
    instability_predicted = components_before.instability

    # Predict new F_l
    f_l_predicted = (
        misfit_predicted + complexity_predicted + redundancy_predicted + instability_predicted
    )

    # Predict Neff change (merging reduces effective expert count)
    neff_predicted = None
    if neff_before is not None:
        # If both experts had low utilization, Neff barely changes
        total_util = source_a_utilization + source_b_utilization
        neff_delta = -0.5 if total_util > 200 else -0.1
        neff_predicted = max(1.0, neff_before + neff_delta)

    return EditEvidence(
        f_l_before=f_l_before,
        f_l_predicted=f_l_predicted,
        delta_f_l=f_l_predicted - f_l_before,
        misfit_before=components_before.misfit,
        misfit_predicted=misfit_predicted,
        complexity_before=components_before.complexity,
        complexity_predicted=complexity_predicted,
        redundancy_before=components_before.redundancy,
        redundancy_predicted=redundancy_predicted,
        instability_before=components_before.instability,
        instability_predicted=instability_predicted,
        psi_before=psi_before,
        psi_predicted=psi_predicted,
        neff_before=neff_before,
        neff_predicted=neff_predicted,
        trigger="experts_redundant",
    )
