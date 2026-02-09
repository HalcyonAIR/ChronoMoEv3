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
    EditEvidence,
    AuditLogEntry,
    create_spawn_proposal,
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
