"""
Edit proposals: what, why, when, and with what evidence.

Every structural change must:
1. Be proposed with evidence (not just executed)
2. Pass gates (evidence + calmness)
3. Be logged with full audit trail
4. Be evaluated for actual improvement

This prevents "clever one-window hacks" from becoming architecture.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any
from datetime import datetime


EditType = Literal["SPAWN", "PRUNE", "SPLIT", "MERGE"]
EditStatus = Literal["PROPOSED", "APPROVED", "EXECUTED", "REJECTED", "REVERTED"]


@dataclass
class EditEvidence:
    """
    Evidence supporting an edit proposal.

    Answers: Why should we do this? What will improve?
    """

    # Free energy prediction
    f_l_before: float
    f_l_predicted: float
    delta_f_l: float

    # Component-level predictions
    misfit_before: float
    misfit_predicted: float
    complexity_before: float
    complexity_predicted: float
    redundancy_before: float
    redundancy_predicted: float
    instability_before: float
    instability_predicted: float

    # Diagnostic predictions (Phase 1-3 signals)
    psi_before: float
    psi_predicted: float
    neff_before: Optional[float] = None
    neff_predicted: Optional[float] = None
    saturation_before: Optional[float] = None
    saturation_predicted: Optional[float] = None

    # Trigger reason
    trigger: str = ""  # e.g., "layer_starving", "expert_decoherent", "expert_bimodal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "f_l_before": self.f_l_before,
            "f_l_predicted": self.f_l_predicted,
            "delta_f_l": self.delta_f_l,
            "misfit_delta": self.misfit_predicted - self.misfit_before,
            "complexity_delta": self.complexity_predicted - self.complexity_before,
            "redundancy_delta": self.redundancy_predicted - self.redundancy_before,
            "instability_delta": self.instability_predicted - self.instability_before,
            "psi_before": self.psi_before,
            "psi_predicted": self.psi_predicted,
            "trigger": self.trigger,
        }


@dataclass
class EditProposal:
    """
    A proposed structural edit.

    Two-step commit:
    1. Proposed at step N (evidence gathered)
    2. Executed at step N+1 (if improvement signal holds and gates pass)
    """

    # Identity
    proposal_id: str
    layer_id: int
    edit_type: EditType
    status: EditStatus = "PROPOSED"

    # Timing
    proposed_at_step: int = 0
    approved_at_step: Optional[int] = None
    executed_at_step: Optional[int] = None

    # Evidence
    evidence: Optional[EditEvidence] = None

    # Gate state at proposal time
    gates_at_proposal: Optional[Dict[str, Any]] = None

    # Edit-specific details
    details: Dict[str, Any] = field(default_factory=dict)

    # Results (filled in after execution)
    actual_delta_f_l: Optional[float] = None
    actual_psi_delta: Optional[float] = None
    reverted: bool = False
    revert_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit log."""
        return {
            "proposal_id": self.proposal_id,
            "layer_id": self.layer_id,
            "edit_type": self.edit_type,
            "status": self.status,
            "proposed_at_step": self.proposed_at_step,
            "approved_at_step": self.approved_at_step,
            "executed_at_step": self.executed_at_step,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "gates_at_proposal": self.gates_at_proposal,
            "details": self.details,
            "actual_delta_f_l": self.actual_delta_f_l,
            "actual_psi_delta": self.actual_psi_delta,
            "reverted": self.reverted,
            "revert_reason": self.revert_reason,
        }


@dataclass
class SpawnProposal(EditProposal):
    """
    Spawn a new expert by cloning the best one.

    Why SPAWN is clean:
    - Doesn't destroy information (only adds capacity)
    - Reversible (can prune if it doesn't help)
    - Evidence: layer starving (high misfit, experts coherent but insufficient)
    """

    # Override to set default
    edit_type: EditType = "SPAWN"

    @property
    def parent_expert_id(self) -> Optional[int]:
        """Expert to clone."""
        return self.details.get("parent_expert_id")

    @property
    def child_expert_id(self) -> Optional[int]:
        """Newly spawned expert."""
        return self.details.get("child_expert_id")

    @property
    def perturbation_scale(self) -> float:
        """Noise scale for clone perturbation."""
        return self.details.get("perturbation_scale", 0.01)


@dataclass
class AuditLogEntry:
    """
    Single entry in edit audit log.

    Every proposal, approval, execution, rejection, revert is logged.
    """

    timestamp: str
    step: int
    event_type: Literal["PROPOSED", "APPROVED", "EXECUTED", "REJECTED", "REVERTED"]
    proposal_id: str
    layer_id: int
    edit_type: EditType

    # Full proposal state
    proposal: Dict[str, Any]

    # Context at event time
    f_l: float
    stress_band: str
    time_in_comfort: int
    gates_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "event_type": self.event_type,
            "proposal_id": self.proposal_id,
            "layer_id": self.layer_id,
            "edit_type": self.edit_type,
            "proposal": self.proposal,
            "context": {
                "f_l": self.f_l,
                "stress_band": self.stress_band,
                "time_in_comfort": self.time_in_comfort,
                "gates_state": self.gates_state,
            },
        }


@dataclass
class PruneProposal(EditProposal):
    """
    Prune a decoherent expert.

    Why PRUNE is careful:
    - Destroys information (not reversible)
    - Evidence: expert persistently decoherent (phi_slow < threshold)
    - Must check starvation: removing expert won't collapse layer
    - Requires sustained calm (identity change)
    """

    # Override to set default
    edit_type: EditType = "PRUNE"

    @property
    def target_expert_id(self) -> Optional[int]:
        """Expert to remove."""
        return self.details.get("target_expert_id")

    @property
    def phi_slow_at_proposal(self) -> Optional[float]:
        """Coherence at proposal time."""
        return self.details.get("phi_slow")

    @property
    def utilization_at_proposal(self) -> Optional[float]:
        """Utilization at proposal time."""
        return self.details.get("utilization")


def create_spawn_proposal(
    proposal_id: str,
    layer_id: int,
    step: int,
    parent_expert_id: int,
    evidence: EditEvidence,
    gates_state: Dict[str, Any],
    perturbation_scale: float = 0.01,
) -> SpawnProposal:
    """
    Create a spawn proposal.

    Args:
        proposal_id: Unique identifier
        layer_id: Which layer to spawn in
        step: Current training step
        parent_expert_id: Expert to clone
        evidence: Why spawn is beneficial
        gates_state: Current gate state
        perturbation_scale: Noise scale for clone

    Returns:
        SpawnProposal ready for evaluation
    """
    return SpawnProposal(
        proposal_id=proposal_id,
        layer_id=layer_id,
        proposed_at_step=step,
        evidence=evidence,
        gates_at_proposal=gates_state,
        details={
            "parent_expert_id": parent_expert_id,
            "perturbation_scale": perturbation_scale,
        },
    )


def create_prune_proposal(
    proposal_id: str,
    layer_id: int,
    step: int,
    target_expert_id: int,
    evidence: EditEvidence,
    gates_state: Dict[str, Any],
    phi_slow: float,
    utilization: float,
) -> PruneProposal:
    """
    Create a prune proposal.

    Args:
        proposal_id: Unique identifier
        layer_id: Which layer
        step: Current step
        target_expert_id: Expert to remove
        evidence: Why prune is beneficial
        gates_state: Current gate state
        phi_slow: Expert coherence at proposal
        utilization: Expert utilization at proposal

    Returns:
        PruneProposal ready for evaluation
    """
    return PruneProposal(
        proposal_id=proposal_id,
        layer_id=layer_id,
        proposed_at_step=step,
        evidence=evidence,
        gates_at_proposal=gates_state,
        details={
            "target_expert_id": target_expert_id,
            "phi_slow": phi_slow,
            "utilization": utilization,
        },
    )


@dataclass
class SplitProposal(EditProposal):
    """
    Split a bimodal expert into two coherent experts.

    Why SPLIT is reversible:
    - Doesn't destroy information (redistributes capacity)
    - Reversible (can merge if doesn't help)
    - Evidence: expert bimodal (high bimodality score, suggests serving two modes)
    - Improves instability by creating two coherent experts
    """

    # Override to set default
    edit_type: EditType = "SPLIT"

    @property
    def source_expert_id(self) -> Optional[int]:
        """Expert to split."""
        return self.details.get("source_expert_id")

    @property
    def child_a_id(self) -> Optional[int]:
        """First new expert."""
        return self.details.get("child_a_id")

    @property
    def child_b_id(self) -> Optional[int]:
        """Second new expert."""
        return self.details.get("child_b_id")

    @property
    def bimodality_at_proposal(self) -> Optional[float]:
        """Bimodality score at proposal time."""
        return self.details.get("bimodality")

    @property
    def split_strategy(self) -> str:
        """How to split: 'kmeans', 'gradient', 'random'."""
        return self.details.get("split_strategy", "kmeans")


def create_split_proposal(
    proposal_id: str,
    layer_id: int,
    step: int,
    source_expert_id: int,
    evidence: EditEvidence,
    gates_state: Dict[str, Any],
    bimodality: float,
    split_strategy: str = "kmeans",
) -> SplitProposal:
    """
    Create a split proposal.

    Args:
        proposal_id: Unique identifier
        layer_id: Which layer
        step: Current step
        source_expert_id: Expert to split
        evidence: Why split is beneficial
        gates_state: Current gate state
        bimodality: Bimodality score at proposal
        split_strategy: How to split ('kmeans', 'gradient', 'random')

    Returns:
        SplitProposal ready for evaluation
    """
    return SplitProposal(
        proposal_id=proposal_id,
        layer_id=layer_id,
        proposed_at_step=step,
        evidence=evidence,
        gates_at_proposal=gates_state,
        details={
            "source_expert_id": source_expert_id,
            "bimodality": bimodality,
            "split_strategy": split_strategy,
        },
    )


@dataclass
class MergeProposal(EditProposal):
    """
    Merge two redundant experts into one.

    Why MERGE is dangerous (MOST CAREFUL):
    - Destroys information (NOT reversible)
    - Evidence: experts highly redundant (high similarity, low diversity)
    - Must verify: neither expert is critical to layer function
    - Requires EXTRA sustained calm (destructive identity change)

    This is where you accidentally delete a personality and call it compression.
    """

    # Override to set default
    edit_type: EditType = "MERGE"

    @property
    def source_a_id(self) -> Optional[int]:
        """First expert to merge."""
        return self.details.get("source_a_id")

    @property
    def source_b_id(self) -> Optional[int]:
        """Second expert to merge."""
        return self.details.get("source_b_id")

    @property
    def merged_expert_id(self) -> Optional[int]:
        """Resulting merged expert."""
        return self.details.get("merged_expert_id")

    @property
    def similarity_at_proposal(self) -> Optional[float]:
        """Similarity between experts at proposal time."""
        return self.details.get("similarity")

    @property
    def merge_strategy(self) -> str:
        """How to merge: 'average', 'weighted_average', 'keep_dominant'."""
        return self.details.get("merge_strategy", "average")


def create_merge_proposal(
    proposal_id: str,
    layer_id: int,
    step: int,
    source_a_id: int,
    source_b_id: int,
    evidence: EditEvidence,
    gates_state: Dict[str, Any],
    similarity: float,
    merge_strategy: str = "average",
) -> MergeProposal:
    """
    Create a merge proposal.

    Args:
        proposal_id: Unique identifier
        layer_id: Which layer
        step: Current step
        source_a_id: First expert to merge
        source_b_id: Second expert to merge
        evidence: Why merge is beneficial
        gates_state: Current gate state
        similarity: Similarity between experts at proposal
        merge_strategy: How to merge ('average', 'weighted_average', 'keep_dominant')

    Returns:
        MergeProposal ready for evaluation
    """
    return MergeProposal(
        proposal_id=proposal_id,
        layer_id=layer_id,
        proposed_at_step=step,
        evidence=evidence,
        gates_at_proposal=gates_state,
        details={
            "source_a_id": source_a_id,
            "source_b_id": source_b_id,
            "similarity": similarity,
            "merge_strategy": merge_strategy,
        },
    )
