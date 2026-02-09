"""
Edit executor: gate-checked, two-step commit, fully logged.

Implements SPAWN only (for now). Other edits (prune/split/merge) come later.

Two-step commit protocol:
1. Proposed at step N (evidence gathered, gates checked)
2. Executed at step N+1 (if improvement holds and gates still pass)

This prevents "clever one-window hacks" from becoming architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
import json

import torch
from torch import Tensor
import torch.nn.functional as F

from .edit_proposals import (
    EditProposal,
    SpawnProposal,
    PruneProposal,
    SplitProposal,
    MergeProposal,
    EditEvidence,
    AuditLogEntry,
    create_spawn_proposal,
    create_prune_proposal,
    create_split_proposal,
    create_merge_proposal,
)
from .dry_run_evaluator import DryRunEvaluator, DryRunResult
from .lifecycle_gates import LifecycleGates, GateViolation
from .stress_bands import StressBandsState


@dataclass
class ExpertSpawnResult:
    """
    Result of spawning a new expert.

    Contains everything needed to integrate the new expert into the model.
    """

    child_expert_id: int
    parent_expert_id: int
    child_params: Dict[str, Tensor]  # Cloned + perturbed parameters
    success: bool
    reason: str = ""


@dataclass
class ExpertSplitResult:
    """
    Result of splitting a bimodal expert.

    Contains everything needed to integrate two new experts from one source.
    """

    child_a_id: int
    child_b_id: int
    source_expert_id: int
    child_a_params: Dict[str, Tensor]  # Cloned + perturbed (variant A)
    child_b_params: Dict[str, Tensor]  # Cloned + perturbed (variant B)
    success: bool
    reason: str = ""


@dataclass
class ExpertMergeResult:
    """
    Result of merging two redundant experts.

    Contains everything needed to integrate the merged expert.
    WARNING: Destructive operation, NOT reversible.
    """

    merged_expert_id: int
    source_a_id: int
    source_b_id: int
    merged_params: Dict[str, Tensor]  # Averaged or blended parameters
    similarity: float  # Similarity at merge time
    success: bool
    reason: str = ""


class EditExecutor:
    """
    Executes structural edits with full gate checking and logging.

    Currently implements: SPAWN only.

    Design principles:
    - Non-bypassable gates (raises GateViolation if blocked)
    - Two-step commit (propose → evaluate → execute)
    - Full audit trail (every proposal/execution logged)
    - Reversible (can undo if doesn't improve)
    """

    def __init__(
        self,
        audit_log_path: Optional[str] = None,
        dry_run_evaluator: Optional[DryRunEvaluator] = None,
    ):
        """
        Initialize executor.

        Args:
            audit_log_path: Path to audit log file (JSONL)
            dry_run_evaluator: Evaluator for testing edits before commit
        """
        self.audit_log_path = audit_log_path
        self.dry_run_evaluator = dry_run_evaluator or DryRunEvaluator()

        # Pending proposals (two-step commit)
        self.pending_proposals: Dict[str, EditProposal] = {}

        # Audit log (in-memory)
        self.audit_log: List[AuditLogEntry] = []

    def propose_spawn(
        self,
        layer_id: int,
        step: int,
        parent_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        perturbation_scale: float = 0.01,
    ) -> Optional[SpawnProposal]:
        """
        Propose spawning a new expert.

        Step 1 of two-step commit. Checks gates, creates proposal, logs it.

        Args:
            layer_id: Which layer
            step: Current training step
            parent_expert_id: Expert to clone
            evidence: Why spawn is beneficial
            gates: Current lifecycle gates
            perturbation_scale: Noise scale for clone

        Returns:
            SpawnProposal if gates pass, None if blocked

        Raises:
            GateViolation: If calmness requirement not met
        """
        # Check gates (raises GateViolation if blocked)
        gates.require_edit_allowed()

        # Create proposal
        proposal_id = f"spawn_L{layer_id}_S{step}"
        proposal = create_spawn_proposal(
            proposal_id=proposal_id,
            layer_id=layer_id,
            step=step,
            parent_expert_id=parent_expert_id,
            evidence=evidence,
            gates_state={
                "allow_edit": gates.check_edit_allowed(),
                "current_band": gates.stress_state.current_band,
                "time_in_comfort": gates.stress_state.time_in_comfort,
            },
            perturbation_scale=perturbation_scale,
        )

        # Add to pending
        self.pending_proposals[proposal_id] = proposal

        # Log proposal
        self._log_event(
            event_type="PROPOSED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
        )

        return proposal

    def approve_spawn(
        self,
        proposal_id: str,
        step: int,
        dry_run_result: DryRunResult,
        gates: LifecycleGates,
    ) -> bool:
        """
        Approve spawn proposal after dry-run evaluation.

        Step 1.5: Check if improvement signal holds.

        Args:
            proposal_id: Which proposal
            step: Current step
            dry_run_result: Result of dry-run simulation
            gates: Current gates (checked again)

        Returns:
            True if approved, False if rejected
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None:
            return False

        # Check gates again (may have changed since proposal)
        if not gates.check_edit_allowed():
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "gates_changed", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Check if improvement confirmed
        if not dry_run_result.improvement_confirmed:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "no_improvement", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Approve
        proposal.status = "APPROVED"
        proposal.approved_at_step = step

        self._log_event(
            event_type="APPROVED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"dry_run": dry_run_result.to_dict()},
        )

        return True

    def execute_spawn(
        self,
        proposal_id: str,
        step: int,
        parent_params: Dict[str, Tensor],
        child_expert_id: int,
        gates: LifecycleGates,
    ) -> Optional[ExpertSpawnResult]:
        """
        Execute approved spawn proposal.

        Step 2 of two-step commit. Actually creates the new expert.

        Args:
            proposal_id: Which proposal
            step: Current step
            parent_params: Parent expert parameters to clone
            child_expert_id: ID for new expert
            gates: Current gates (final check)

        Returns:
            ExpertSpawnResult if successful, None if blocked
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None or proposal.status != "APPROVED":
            return None

        # Final gate check
        try:
            gates.require_edit_allowed()
        except GateViolation as e:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": f"final_gate_check_failed: {e}"},
            )
            return None

        # Clone parent parameters
        child_params = {}
        perturbation_scale = proposal.details.get("perturbation_scale", 0.01)

        for key, param in parent_params.items():
            # Clone and add small perturbation
            cloned = param.clone()
            noise = torch.randn_like(cloned) * perturbation_scale
            child_params[key] = cloned + noise

        # Mark as executed
        proposal.status = "EXECUTED"
        proposal.executed_at_step = step
        proposal.details["child_expert_id"] = child_expert_id

        # Log execution
        self._log_event(
            event_type="EXECUTED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"child_expert_id": child_expert_id},
        )

        # Remove from pending
        del self.pending_proposals[proposal_id]

        return ExpertSpawnResult(
            child_expert_id=child_expert_id,
            parent_expert_id=proposal.details["parent_expert_id"],
            child_params=child_params,
            success=True,
            reason="spawned_successfully",
        )

    def spawn_expert_full_pipeline(
        self,
        layer_id: int,
        step: int,
        parent_expert_id: int,
        parent_params: Dict[str, Tensor],
        child_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        dry_run_fn: Optional[Callable[[EditProposal], Dict[str, Any]]] = None,
        perturbation_scale: float = 0.01,
    ) -> Optional[ExpertSpawnResult]:
        """
        Full spawn pipeline: propose → evaluate → approve → execute.

        Convenience method that runs all steps if you have everything ready.

        Args:
            layer_id: Which layer
            step: Current step
            parent_expert_id: Expert to clone
            parent_params: Parent parameters
            child_expert_id: ID for new expert
            evidence: Why spawn is beneficial
            gates: Lifecycle gates
            dry_run_fn: Optional simulation function for dry-run
            perturbation_scale: Noise scale

        Returns:
            ExpertSpawnResult if successful, None if blocked at any step
        """
        # Step 1: Propose
        try:
            proposal = self.propose_spawn(
                layer_id=layer_id,
                step=step,
                parent_expert_id=parent_expert_id,
                evidence=evidence,
                gates=gates,
                perturbation_scale=perturbation_scale,
            )
        except GateViolation:
            return None

        if proposal is None:
            return None

        # Step 1.5: Dry-run evaluation (if function provided)
        if dry_run_fn is not None:
            current_state = {
                "f_l": evidence.f_l_before,
                "psi": evidence.psi_before,
                "neff": evidence.neff_before,
            }
            dry_run_result = self.dry_run_evaluator.evaluate_proposal(
                proposal=proposal,
                current_state=current_state,
                simulate_fn=dry_run_fn,
            )

            # Step 2: Approve (or reject)
            approved = self.approve_spawn(
                proposal_id=proposal.proposal_id,
                step=step,
                dry_run_result=dry_run_result,
                gates=gates,
            )

            if not approved:
                return None
        else:
            # No dry-run: auto-approve based on evidence alone
            proposal.status = "APPROVED"
            proposal.approved_at_step = step

        # Step 3: Execute
        return self.execute_spawn(
            proposal_id=proposal.proposal_id,
            step=step,
            parent_params=parent_params,
            child_expert_id=child_expert_id,
            gates=gates,
        )

    # ========================================================================
    # PRUNE methods
    # ========================================================================

    def propose_prune(
        self,
        layer_id: int,
        step: int,
        target_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        phi_slow: float,
        utilization: float,
    ) -> Optional[PruneProposal]:
        """
        Propose pruning a decoherent expert.

        Step 1 of two-step commit. Checks gates, creates proposal, logs it.

        Args:
            layer_id: Which layer
            step: Current training step
            target_expert_id: Expert to remove
            evidence: Why prune is beneficial
            gates: Current lifecycle gates
            phi_slow: Expert coherence at proposal
            utilization: Expert utilization at proposal

        Returns:
            PruneProposal if gates pass, None if blocked

        Raises:
            GateViolation: If calmness requirement not met
        """
        # Check gates (raises GateViolation if blocked)
        # PRUNE requires edit permission (identity change)
        gates.require_edit_allowed()

        # Create proposal
        proposal_id = f"prune_L{layer_id}_E{target_expert_id}_S{step}"
        proposal = create_prune_proposal(
            proposal_id=proposal_id,
            layer_id=layer_id,
            step=step,
            target_expert_id=target_expert_id,
            evidence=evidence,
            gates_state={
                "allow_edit": gates.check_edit_allowed(),
                "current_band": gates.stress_state.current_band,
                "time_in_comfort": gates.stress_state.time_in_comfort,
            },
            phi_slow=phi_slow,
            utilization=utilization,
        )

        # Add to pending
        self.pending_proposals[proposal_id] = proposal

        # Log proposal
        self._log_event(
            event_type="PROPOSED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
        )

        return proposal

    def approve_prune(
        self,
        proposal_id: str,
        step: int,
        dry_run_result: DryRunResult,
        gates: LifecycleGates,
    ) -> bool:
        """
        Approve prune proposal after dry-run evaluation.

        Step 1.5: Check if improvement signal holds.

        Args:
            proposal_id: Which proposal
            step: Current step
            dry_run_result: Result of dry-run simulation
            gates: Current gates (checked again)

        Returns:
            True if approved, False if rejected
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None:
            return False

        # Check gates again (may have changed since proposal)
        if not gates.check_edit_allowed():
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "gates_changed", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Check if improvement confirmed
        if not dry_run_result.improvement_confirmed:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "no_improvement", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Approve
        proposal.status = "APPROVED"
        proposal.approved_at_step = step

        self._log_event(
            event_type="APPROVED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"dry_run": dry_run_result.to_dict()},
        )

        return True

    def execute_prune(
        self,
        proposal_id: str,
        step: int,
        gates: LifecycleGates,
    ) -> Optional[int]:
        """
        Execute approved prune proposal.

        Step 2 of two-step commit. Returns expert ID to remove.

        Args:
            proposal_id: Which proposal
            step: Current step
            gates: Current gates (final check)

        Returns:
            Expert ID to remove if successful, None if blocked
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None or proposal.status != "APPROVED":
            return None

        # Final gate check
        try:
            gates.require_edit_allowed()
        except GateViolation as e:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": f"final_gate_check_failed: {e}"},
            )
            return None

        # Mark as executed
        proposal.status = "EXECUTED"
        proposal.executed_at_step = step

        target_expert_id = proposal.details.get("target_expert_id")

        # Log execution
        self._log_event(
            event_type="EXECUTED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"target_expert_id": target_expert_id},
        )

        # Remove from pending
        del self.pending_proposals[proposal_id]

        return target_expert_id

    def prune_expert_full_pipeline(
        self,
        layer_id: int,
        step: int,
        target_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        phi_slow: float,
        utilization: float,
        dry_run_fn: Optional[Callable[[EditProposal], Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """
        Full prune pipeline: propose → evaluate → approve → execute.

        Convenience method that runs all steps if you have everything ready.

        Args:
            layer_id: Which layer
            step: Current step
            target_expert_id: Expert to remove
            evidence: Why prune is beneficial
            gates: Lifecycle gates
            phi_slow: Expert coherence
            utilization: Expert utilization
            dry_run_fn: Optional simulation function for dry-run

        Returns:
            Expert ID to remove if successful, None if blocked at any step
        """
        # Step 1: Propose
        try:
            proposal = self.propose_prune(
                layer_id=layer_id,
                step=step,
                target_expert_id=target_expert_id,
                evidence=evidence,
                gates=gates,
                phi_slow=phi_slow,
                utilization=utilization,
            )
        except GateViolation:
            return None

        if proposal is None:
            return None

        # Step 1.5: Dry-run evaluation (if function provided)
        if dry_run_fn is not None:
            current_state = {
                "f_l": evidence.f_l_before,
                "psi": evidence.psi_before,
                "neff": evidence.neff_before,
            }
            dry_run_result = self.dry_run_evaluator.evaluate_proposal(
                proposal=proposal,
                current_state=current_state,
                simulate_fn=dry_run_fn,
            )

            # Step 2: Approve (or reject)
            approved = self.approve_prune(
                proposal_id=proposal.proposal_id,
                step=step,
                dry_run_result=dry_run_result,
                gates=gates,
            )

            if not approved:
                return None
        else:
            # No dry-run: auto-approve based on evidence alone
            proposal.status = "APPROVED"
            proposal.approved_at_step = step

        # Step 3: Execute
        return self.execute_prune(
            proposal_id=proposal.proposal_id,
            step=step,
            gates=gates,
        )

    # ========================================================================
    # SPLIT methods
    # ========================================================================

    def propose_split(
        self,
        layer_id: int,
        step: int,
        source_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        bimodality: float,
        split_strategy: str = "kmeans",
    ) -> Optional[SplitProposal]:
        """
        Propose splitting a bimodal expert.

        Step 1 of two-step commit. Checks gates, creates proposal, logs it.

        Args:
            layer_id: Which layer
            step: Current training step
            source_expert_id: Expert to split
            evidence: Why split is beneficial
            gates: Current lifecycle gates
            bimodality: Bimodality score at proposal
            split_strategy: How to split ('kmeans', 'gradient', 'random')

        Returns:
            SplitProposal if gates pass, None if blocked

        Raises:
            GateViolation: If calmness requirement not met
        """
        # Check gates (raises GateViolation if blocked)
        # SPLIT requires edit permission (identity change)
        gates.require_edit_allowed()

        # Create proposal
        proposal_id = f"split_L{layer_id}_E{source_expert_id}_S{step}"
        proposal = create_split_proposal(
            proposal_id=proposal_id,
            layer_id=layer_id,
            step=step,
            source_expert_id=source_expert_id,
            evidence=evidence,
            gates_state={
                "allow_edit": gates.check_edit_allowed(),
                "current_band": gates.stress_state.current_band,
                "time_in_comfort": gates.stress_state.time_in_comfort,
            },
            bimodality=bimodality,
            split_strategy=split_strategy,
        )

        # Add to pending
        self.pending_proposals[proposal_id] = proposal

        # Log proposal
        self._log_event(
            event_type="PROPOSED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
        )

        return proposal

    def approve_split(
        self,
        proposal_id: str,
        step: int,
        dry_run_result: DryRunResult,
        gates: LifecycleGates,
    ) -> bool:
        """
        Approve split proposal after dry-run evaluation.

        Step 1.5: Check if improvement signal holds.

        Args:
            proposal_id: Which proposal
            step: Current step
            dry_run_result: Result of dry-run simulation
            gates: Current gates (checked again)

        Returns:
            True if approved, False if rejected
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None:
            return False

        # Check gates again (may have changed since proposal)
        if not gates.check_edit_allowed():
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "gates_changed", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Check if improvement confirmed
        if not dry_run_result.improvement_confirmed:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "no_improvement", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Approve
        proposal.status = "APPROVED"
        proposal.approved_at_step = step

        self._log_event(
            event_type="APPROVED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"dry_run": dry_run_result.to_dict()},
        )

        return True

    def execute_split(
        self,
        proposal_id: str,
        step: int,
        source_params: Dict[str, Tensor],
        child_a_id: int,
        child_b_id: int,
        gates: LifecycleGates,
        perturbation_scale: float = 0.02,
    ) -> Optional[ExpertSplitResult]:
        """
        Execute approved split proposal.

        Step 2 of two-step commit. Creates two new experts from source.

        Args:
            proposal_id: Which proposal
            step: Current step
            source_params: Source expert parameters to split
            child_a_id: ID for first new expert
            child_b_id: ID for second new expert
            gates: Current gates (final check)
            perturbation_scale: Noise scale for variants

        Returns:
            ExpertSplitResult if successful, None if blocked
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None or proposal.status != "APPROVED":
            return None

        # Final gate check
        try:
            gates.require_edit_allowed()
        except GateViolation as e:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": f"final_gate_check_failed: {e}"},
            )
            return None

        # Clone source parameters twice with different perturbations
        child_a_params = {}
        child_b_params = {}

        for key, param in source_params.items():
            # Clone A with positive perturbation
            cloned_a = param.clone()
            noise_a = torch.randn_like(cloned_a) * perturbation_scale
            child_a_params[key] = cloned_a + noise_a

            # Clone B with negative perturbation (orthogonal direction)
            cloned_b = param.clone()
            noise_b = torch.randn_like(cloned_b) * perturbation_scale
            child_b_params[key] = cloned_b - noise_b

        # Mark as executed
        proposal.status = "EXECUTED"
        proposal.executed_at_step = step
        proposal.details["child_a_id"] = child_a_id
        proposal.details["child_b_id"] = child_b_id

        source_expert_id = proposal.details.get("source_expert_id")

        # Log execution
        self._log_event(
            event_type="EXECUTED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={
                "source_expert_id": source_expert_id,
                "child_a_id": child_a_id,
                "child_b_id": child_b_id,
            },
        )

        # Remove from pending
        del self.pending_proposals[proposal_id]

        return ExpertSplitResult(
            child_a_id=child_a_id,
            child_b_id=child_b_id,
            source_expert_id=source_expert_id,
            child_a_params=child_a_params,
            child_b_params=child_b_params,
            success=True,
            reason="split_successfully",
        )

    def split_expert_full_pipeline(
        self,
        layer_id: int,
        step: int,
        source_expert_id: int,
        source_params: Dict[str, Tensor],
        child_a_id: int,
        child_b_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        bimodality: float,
        split_strategy: str = "kmeans",
        dry_run_fn: Optional[Callable[[EditProposal], Dict[str, Any]]] = None,
        perturbation_scale: float = 0.02,
    ) -> Optional[ExpertSplitResult]:
        """
        Full split pipeline: propose → evaluate → approve → execute.

        Convenience method that runs all steps if you have everything ready.

        Args:
            layer_id: Which layer
            step: Current step
            source_expert_id: Expert to split
            source_params: Source parameters
            child_a_id: ID for first new expert
            child_b_id: ID for second new expert
            evidence: Why split is beneficial
            gates: Lifecycle gates
            bimodality: Bimodality score
            split_strategy: How to split ('kmeans', 'gradient', 'random')
            dry_run_fn: Optional simulation function for dry-run
            perturbation_scale: Noise scale

        Returns:
            ExpertSplitResult if successful, None if blocked at any step
        """
        # Step 1: Propose
        try:
            proposal = self.propose_split(
                layer_id=layer_id,
                step=step,
                source_expert_id=source_expert_id,
                evidence=evidence,
                gates=gates,
                bimodality=bimodality,
                split_strategy=split_strategy,
            )
        except GateViolation:
            return None

        if proposal is None:
            return None

        # Step 1.5: Dry-run evaluation (if function provided)
        if dry_run_fn is not None:
            current_state = {
                "f_l": evidence.f_l_before,
                "psi": evidence.psi_before,
                "neff": evidence.neff_before,
            }
            dry_run_result = self.dry_run_evaluator.evaluate_proposal(
                proposal=proposal,
                current_state=current_state,
                simulate_fn=dry_run_fn,
            )

            # Step 2: Approve (or reject)
            approved = self.approve_split(
                proposal_id=proposal.proposal_id,
                step=step,
                dry_run_result=dry_run_result,
                gates=gates,
            )

            if not approved:
                return None
        else:
            # No dry-run: auto-approve based on evidence alone
            proposal.status = "APPROVED"
            proposal.approved_at_step = step

        # Step 3: Execute
        return self.execute_split(
            proposal_id=proposal.proposal_id,
            step=step,
            source_params=source_params,
            child_a_id=child_a_id,
            child_b_id=child_b_id,
            gates=gates,
            perturbation_scale=perturbation_scale,
        )

    # ========================================================================
    # MERGE methods (MOST DANGEROUS - destructive, NOT reversible)
    # ========================================================================

    def propose_merge(
        self,
        layer_id: int,
        step: int,
        source_a_id: int,
        source_b_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        similarity: float,
        merge_strategy: str = "average",
    ) -> Optional[MergeProposal]:
        """
        Propose merging two redundant experts.

        Step 1 of two-step commit. Checks gates, creates proposal, logs it.

        WARNING: MERGE is destructive and NOT reversible.

        Args:
            layer_id: Which layer
            step: Current training step
            source_a_id: First expert to merge
            source_b_id: Second expert to merge
            evidence: Why merge is beneficial
            gates: Current lifecycle gates
            similarity: Similarity between experts (0-1)
            merge_strategy: How to merge ('average', 'weighted_average', 'keep_dominant')

        Returns:
            MergeProposal if gates pass, None if blocked

        Raises:
            GateViolation: If calmness requirement not met
        """
        # Check gates (raises GateViolation if blocked)
        # MERGE requires edit permission (destructive identity change)
        gates.require_edit_allowed()

        # Create proposal
        proposal_id = f"merge_L{layer_id}_E{source_a_id}+E{source_b_id}_S{step}"
        proposal = create_merge_proposal(
            proposal_id=proposal_id,
            layer_id=layer_id,
            step=step,
            source_a_id=source_a_id,
            source_b_id=source_b_id,
            evidence=evidence,
            gates_state={
                "allow_edit": gates.check_edit_allowed(),
                "current_band": gates.stress_state.current_band,
                "time_in_comfort": gates.stress_state.time_in_comfort,
            },
            similarity=similarity,
            merge_strategy=merge_strategy,
        )

        # Add to pending
        self.pending_proposals[proposal_id] = proposal

        # Log proposal
        self._log_event(
            event_type="PROPOSED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
        )

        return proposal

    def approve_merge(
        self,
        proposal_id: str,
        step: int,
        dry_run_result: DryRunResult,
        gates: LifecycleGates,
    ) -> bool:
        """
        Approve merge proposal after dry-run evaluation.

        Step 1.5: Check if improvement signal holds.

        WARNING: MERGE is destructive. Approval must be careful.

        Args:
            proposal_id: Which proposal
            step: Current step
            dry_run_result: Result of dry-run simulation
            gates: Current gates (checked again)

        Returns:
            True if approved, False if rejected
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None:
            return False

        # Check gates again (may have changed since proposal)
        if not gates.check_edit_allowed():
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "gates_changed", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Check if improvement confirmed
        if not dry_run_result.improvement_confirmed:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": "no_improvement", "dry_run": dry_run_result.to_dict()},
            )
            return False

        # Approve
        proposal.status = "APPROVED"
        proposal.approved_at_step = step

        self._log_event(
            event_type="APPROVED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={"dry_run": dry_run_result.to_dict()},
        )

        return True

    def execute_merge(
        self,
        proposal_id: str,
        step: int,
        source_a_params: Dict[str, Tensor],
        source_b_params: Dict[str, Tensor],
        merged_expert_id: int,
        gates: LifecycleGates,
    ) -> Optional[ExpertMergeResult]:
        """
        Execute approved merge proposal.

        Step 2 of two-step commit. Merges two experts into one.

        WARNING: DESTRUCTIVE operation, NOT reversible.

        Args:
            proposal_id: Which proposal
            step: Current step
            source_a_params: First expert parameters
            source_b_params: Second expert parameters
            merged_expert_id: ID for merged expert
            gates: Current gates (final check)

        Returns:
            ExpertMergeResult if successful, None if blocked
        """
        proposal = self.pending_proposals.get(proposal_id)
        if proposal is None or proposal.status != "APPROVED":
            return None

        # Final gate check
        try:
            gates.require_edit_allowed()
        except GateViolation as e:
            proposal.status = "REJECTED"
            self._log_event(
                event_type="REJECTED",
                proposal=proposal,
                step=step,
                stress_state=gates.stress_state,
                gates=gates,
                extra={"reason": f"final_gate_check_failed: {e}"},
            )
            return None

        # Merge parameters (simple average)
        merged_params = {}
        merge_strategy = proposal.details.get("merge_strategy", "average")

        if merge_strategy == "average":
            # Simple average
            for key in source_a_params.keys():
                if key in source_b_params:
                    merged_params[key] = (source_a_params[key] + source_b_params[key]) / 2.0
                else:
                    merged_params[key] = source_a_params[key].clone()
        else:
            # Default to average if unknown strategy
            for key in source_a_params.keys():
                if key in source_b_params:
                    merged_params[key] = (source_a_params[key] + source_b_params[key]) / 2.0
                else:
                    merged_params[key] = source_a_params[key].clone()

        # Mark as executed
        proposal.status = "EXECUTED"
        proposal.executed_at_step = step
        proposal.details["merged_expert_id"] = merged_expert_id

        source_a_id = proposal.details.get("source_a_id")
        source_b_id = proposal.details.get("source_b_id")
        similarity = proposal.details.get("similarity")

        # Log execution
        self._log_event(
            event_type="EXECUTED",
            proposal=proposal,
            step=step,
            stress_state=gates.stress_state,
            gates=gates,
            extra={
                "source_a_id": source_a_id,
                "source_b_id": source_b_id,
                "merged_expert_id": merged_expert_id,
                "similarity": similarity,
            },
        )

        # Remove from pending
        del self.pending_proposals[proposal_id]

        return ExpertMergeResult(
            merged_expert_id=merged_expert_id,
            source_a_id=source_a_id,
            source_b_id=source_b_id,
            merged_params=merged_params,
            similarity=similarity,
            success=True,
            reason="merged_successfully",
        )

    def merge_experts_full_pipeline(
        self,
        layer_id: int,
        step: int,
        source_a_id: int,
        source_b_id: int,
        source_a_params: Dict[str, Tensor],
        source_b_params: Dict[str, Tensor],
        merged_expert_id: int,
        evidence: EditEvidence,
        gates: LifecycleGates,
        similarity: float,
        merge_strategy: str = "average",
        dry_run_fn: Optional[Callable[[EditProposal], Dict[str, Any]]] = None,
    ) -> Optional[ExpertMergeResult]:
        """
        Full merge pipeline: propose → evaluate → approve → execute.

        Convenience method that runs all steps if you have everything ready.

        WARNING: MERGE is destructive and NOT reversible.

        Args:
            layer_id: Which layer
            step: Current step
            source_a_id: First expert to merge
            source_b_id: Second expert to merge
            source_a_params: First expert parameters
            source_b_params: Second expert parameters
            merged_expert_id: ID for merged expert
            evidence: Why merge is beneficial
            gates: Lifecycle gates
            similarity: Similarity between experts
            merge_strategy: How to merge ('average', 'weighted_average', 'keep_dominant')
            dry_run_fn: Optional simulation function for dry-run

        Returns:
            ExpertMergeResult if successful, None if blocked at any step
        """
        # Step 1: Propose
        try:
            proposal = self.propose_merge(
                layer_id=layer_id,
                step=step,
                source_a_id=source_a_id,
                source_b_id=source_b_id,
                evidence=evidence,
                gates=gates,
                similarity=similarity,
                merge_strategy=merge_strategy,
            )
        except GateViolation:
            return None

        if proposal is None:
            return None

        # Step 1.5: Dry-run evaluation (if function provided)
        if dry_run_fn is not None:
            current_state = {
                "f_l": evidence.f_l_before,
                "psi": evidence.psi_before,
                "neff": evidence.neff_before,
            }
            dry_run_result = self.dry_run_evaluator.evaluate_proposal(
                proposal=proposal,
                current_state=current_state,
                simulate_fn=dry_run_fn,
            )

            # Step 2: Approve (or reject)
            approved = self.approve_merge(
                proposal_id=proposal.proposal_id,
                step=step,
                dry_run_result=dry_run_result,
                gates=gates,
            )

            if not approved:
                return None
        else:
            # No dry-run: auto-approve based on evidence alone
            proposal.status = "APPROVED"
            proposal.approved_at_step = step

        # Step 3: Execute
        return self.execute_merge(
            proposal_id=proposal.proposal_id,
            step=step,
            source_a_params=source_a_params,
            source_b_params=source_b_params,
            merged_expert_id=merged_expert_id,
            gates=gates,
        )

    def _log_event(
        self,
        event_type: str,
        proposal: EditProposal,
        step: int,
        stress_state: StressBandsState,
        gates: LifecycleGates,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log an event to audit trail.

        Args:
            event_type: PROPOSED, APPROVED, EXECUTED, REJECTED, REVERTED
            proposal: The proposal
            step: Current step
            stress_state: Current stress bands state
            gates: Current gates
            extra: Optional extra data
        """
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            step=step,
            event_type=event_type,
            proposal_id=proposal.proposal_id,
            layer_id=proposal.layer_id,
            edit_type=proposal.edit_type,
            proposal=proposal.to_dict(),
            f_l=stress_state.current_f,
            stress_band=stress_state.current_band,
            time_in_comfort=stress_state.time_in_comfort,
            gates_state={
                "allow_scar": gates.check_scar_allowed(),
                "allow_crystallize": gates.check_crystallize_allowed(),
                "allow_edit": gates.check_edit_allowed(),
            },
        )

        # Add extra data if provided
        if extra:
            entry_dict = entry.to_dict()
            entry_dict["extra"] = extra
            self.audit_log.append(entry)  # Store original entry
            # Write enriched version to file
            if self.audit_log_path:
                with open(self.audit_log_path, "a") as f:
                    f.write(json.dumps(entry_dict) + "\n")
        else:
            self.audit_log.append(entry)
            if self.audit_log_path:
                with open(self.audit_log_path, "a") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")

    def get_audit_log(self) -> List[AuditLogEntry]:
        """Get full audit log."""
        return self.audit_log

    def get_pending_proposals(self) -> Dict[str, EditProposal]:
        """Get all pending proposals (not yet executed)."""
        return self.pending_proposals.copy()
