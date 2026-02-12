"""
ChronoMoE Controller: The ONLY interface between swiss-ai/MoE and ChronoMoEv3 logic.

This module enforces a strict boundary:
- swiss-ai/MoE code only imports from THIS module
- ChronoMoEv3 signal processing stays behind this boundary
- API surface is minimal: observe() → decide() → apply()

Philosophy: Thin interface layer, thick implementation behind it.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch

from .coherence import (
    CoherenceState,
    compute_coherence,
    compute_expert_utilization,
    update_coherence_ema,
    compute_layer_coherence,
)
from .bimodality import (
    BimodalityState,
    update_bimodality,
    compute_layer_bimodality,
)
from .free_energy import (
    FreeEnergyState,
    create_free_energy_state,
)


@dataclass
class ObservationSnapshot:
    """
    One timestep's signals from the MoE layer.

    This is the data contract: everything the controller needs from swiss-ai/MoE.
    """
    step: int
    layer_id: int

    # Router state
    router_probs: torch.Tensor  # [B*T, num_experts] - post-softmax probabilities
    selected_experts: torch.Tensor  # [B*T, top_k] - selected expert indices

    # Expert state
    expert_outputs: Optional[torch.Tensor]  # [num_experts, B*T, d_model] - per-expert outputs
    mixture_output: torch.Tensor  # [B*T, d_model] - final mixture output

    # Utilization
    utilization: torch.Tensor  # [num_experts] - token counts per expert

    # Training context
    loss: Optional[float] = None  # Current step loss (for stress bands)

    def validate(self) -> None:
        """Sanity check dimensions."""
        assert self.router_probs.ndim == 2, f"router_probs must be [B*T, E], got {self.router_probs.shape}"
        assert self.mixture_output.ndim == 2, f"mixture_output must be [B*T, D], got {self.mixture_output.shape}"
        if self.expert_outputs is not None:
            assert self.expert_outputs.ndim == 3, f"expert_outputs must be [E, B*T, D], got {self.expert_outputs.shape}"


@dataclass
class EditProposal:
    """
    A proposed lifecycle operation.

    The controller returns these; swiss-ai/MoE decides whether to execute.
    """
    edit_type: str  # "spawn", "prune", "split", "merge"
    expert_id: int  # Target expert (or parent for spawn)
    reason: str  # Human-readable justification
    evidence: Dict[str, Any]  # Metrics supporting the decision

    # Execution metadata
    calm_credit_required: int = 200  # Steps in COMFORT needed
    delta_f_l: Optional[float] = None  # Predicted free energy change


@dataclass
class EditResult:
    """
    Result of executing an edit.

    swiss-ai/MoE reports back to the controller what happened.
    """
    edit_type: str
    success: bool
    expert_id: int
    new_expert_id: Optional[int] = None  # For spawn (child_a for split)
    reason: str = ""  # Why it failed (if not success)
    metadata: Optional[Dict[str, Any]] = None  # Additional data (e.g., child_b_id for split)


@dataclass
class PendingSplitLatch:
    """
    Pending SPLIT proposal with TTL.

    When SPLIT is proposed but blocked (e.g., in STRAIN), the evidence is latched
    and can be executed later in COMFORT once calm credit is met, provided:
    - TTL hasn't expired (evidence not too stale)
    - Evidence hasn't been contradicted (bimodality still valid)
    """
    expert_id: int
    evidence: Dict[str, Any]  # Original bimodality evidence
    proposed_at_step: int
    ttl_steps: int  # Time-to-live (e.g., 500 steps)
    calm_credit_required: int
    delta_f_l: float

    def is_expired(self, current_step: int) -> bool:
        """Check if latch has expired."""
        return current_step >= self.proposed_at_step + self.ttl_steps

    def is_valid(self, current_bimodality_score: float, threshold: float) -> bool:
        """
        Check if evidence is still valid (not contradicted).

        Evidence is contradicted if bimodality drops to near-zero, indicating
        the expert's behavior fundamentally changed. Small drops below threshold
        are not contradictions - just noise/variance.
        """
        # Use a relaxed threshold: evidence is only contradicted if bimodality
        # drops significantly (e.g., below 20% of original threshold)
        relaxed_threshold = threshold * 0.2
        return current_bimodality_score >= relaxed_threshold


class ChronoController:
    """
    The ONLY way swiss-ai/MoE talks to ChronoMoEv3 decision logic.

    Usage:
        controller = ChronoController(layer_id=0, max_experts=32)

        # Every forward pass
        snapshot = ObservationSnapshot(...)
        controller.observe(snapshot)

        # Periodically (e.g., every 10 steps)
        proposals = controller.decide()
        for proposal in proposals:
            result = execute_edit(proposal)  # swiss-ai/MoE code
            controller.apply(result)
    """

    def __init__(
        self,
        layer_id: int,
        max_experts: int,
        initial_active: int,
        config: Optional[Dict[str, Any]] = None,
        autonomous_mode: bool = False,
    ):
        self.layer_id = layer_id
        self.max_experts = max_experts
        self.initial_active = initial_active

        # Mode: DIAGNOSTIC (Milestones A-C) vs AUTONOMOUS (Milestone D+)
        # DIAGNOSTIC: decide() returns empty list (signals only)
        # AUTONOMOUS: decide() generates proposals (signals → actions)
        self.autonomous_mode = autonomous_mode

        # Config consolidation (one place for all signal thresholds)
        # Merge user config with defaults
        self.config = self._default_config()
        if config:
            self._merge_config(config)

        # Internal state (stays behind the boundary)
        # Milestone A: Coherence tracking
        self.coherence_states: Dict[int, CoherenceState] = {}
        for expert_id in range(initial_active):
            self.coherence_states[expert_id] = CoherenceState(
                expert_id=expert_id,
                layer_id=layer_id,
            )

        # Milestone B: Bimodality tracking
        self.bimodality_states: Dict[int, BimodalityState] = {}
        # Note: Bimodality states created on-demand (need d_model from first observation)

        # Milestone C: Free energy tracking
        self.free_energy_state: Optional[FreeEnergyState] = None
        # Note: Created on first observation with full coherence + bimodality data

        # Observation history (ring buffer)
        self.observation_history: List[ObservationSnapshot] = []
        self.max_history = 100  # Keep last 100 observations

        # Edit audit trail
        self.edit_log: List[Dict[str, Any]] = []

        # Pending SPLIT latch (Milestone E)
        # When SPLIT is proposed but blocked, latch the evidence with TTL
        self.pending_split_latch: Optional[PendingSplitLatch] = None

        # Split lineage cooldown (Milestone E enhancement)
        # Prevents repeated splits on transient bimodality signals
        # Maps expert_id -> step when expert or its parent was split
        self.split_lineage_history: Dict[int, int] = {}

        # Expert penalties for suppression trials (Milestone F Phase 1.5)
        # Before MERGE execution, suppress one expert to test redundancy
        # If system adapts (quality maintained), redundancy is real
        # If quality drops or router thrashes, experts were not redundant
        self.expert_penalties: Dict[int, float] = {}  # expert_id -> penalty magnitude
        self.expert_cooldowns: Dict[int, int] = {}    # expert_id -> step when cooldown ends

        # Track edits count
        self.edits_count: int = 0

    def observe(self, snapshot: ObservationSnapshot) -> None:
        """
        Process one timestep's signals.

        This is where signal processing happens:
        - Update coherence EMAs (Milestone A)
        - Update bimodality centroids (Milestone B)
        - Compute free energy (Milestone C)

        Args:
            snapshot: Current step's router/expert state
        """
        snapshot.validate()

        # Store in ring buffer
        self.observation_history.append(snapshot)
        if len(self.observation_history) > self.max_history:
            self.observation_history.pop(0)

        # Milestone A: Update coherence state
        if snapshot.expert_outputs is not None:
            self._update_coherence(snapshot)

        # Milestone B: Update bimodality state
        if snapshot.expert_outputs is not None:
            self._update_bimodality(snapshot)

        # Milestone C: Compute free energy
        if snapshot.expert_outputs is not None:
            self._update_free_energy(snapshot)

    def decide(self) -> List[EditProposal]:
        """
        Propose lifecycle operations based on accumulated signals.

        Mode behavior:
        - DIAGNOSTIC (autonomous_mode=False): Returns empty list (Milestones A-C)
        - AUTONOMOUS (autonomous_mode=True): Generates proposals (Milestone D+)

        Decision logic (AUTONOMOUS mode only):
        - Evaluate evidence (ΔF_l threshold, MIN_DELTA_F)
        - Propose spawn if layer inefficient (high F_l, need capacity)
        - Propose prune if expert decoherent (low phi_slow) or redundant

        NOTE: swiss-ai/MoE layer enforces stress bands and calm gates.
        Controller only generates proposals based on signals.

        Returns:
            List of proposed edits (may be empty)
        """
        proposals = []

        # Check mode: DIAGNOSTIC returns empty (backwards compatibility)
        if not self.autonomous_mode:
            return proposals  # Milestones A-C: signals only, no triggers

        # AUTONOMOUS mode: Generate proposals (Milestone D+)

        # Check if we have sufficient signal data
        if not self.free_energy_state:
            return proposals  # Need free energy state to make decisions

        if not self.coherence_states:
            return proposals  # Need coherence states for PRUNE decisions

        # SPAWN trigger
        spawn_proposal = self._try_propose_spawn()
        if spawn_proposal:
            proposals.append(spawn_proposal)

        # PRUNE trigger
        prune_proposal = self._try_propose_prune()
        if prune_proposal:
            proposals.append(prune_proposal)

        # SPLIT trigger (Milestone E)
        # Check for pending latched SPLIT first
        split_proposal = self._check_pending_split_latch()
        if not split_proposal:
            # No latched SPLIT ready, try proposing new one
            split_proposal = self._try_propose_split()

        if split_proposal:
            proposals.append(split_proposal)

        # MERGE trigger (Milestone F Phase 1: diagnostic-only)
        merge_proposal = self._try_propose_merge()
        if merge_proposal:
            proposals.append(merge_proposal)

        return proposals

    def apply(self, result: EditResult) -> None:
        """
        Update internal state after edit execution.

        Args:
            result: What happened when swiss-ai/MoE tried to execute the edit
        """
        # Log the result
        self.edit_log.append({
            "step": len(self.observation_history),
            "edit_type": result.edit_type,
            "success": result.success,
            "expert_id": result.expert_id,
            "new_expert_id": result.new_expert_id,
            "reason": result.reason,
        })

        # Update internal state if needed
        if result.success and result.edit_type == "split":
            # Parent expert was pruned, two children were created
            current_step = len(self.observation_history)

            # Reset bimodality state for parent (now inactive)
            if result.expert_id in self.bimodality_states:
                del self.bimodality_states[result.expert_id]

            # Clear pending split latch (SPLIT executed successfully)
            if self.pending_split_latch and self.pending_split_latch.expert_id == result.expert_id:
                self.pending_split_latch = None

            # Track split lineage for cooldown enforcement
            # Record parent and both children to prevent re-splitting too soon
            parent_id = result.expert_id
            child_a_id = result.new_expert_id
            child_b_id = result.metadata.get("child_b_id") if result.metadata else None

            self.split_lineage_history[parent_id] = current_step
            if child_a_id is not None:
                self.split_lineage_history[child_a_id] = current_step
            if child_b_id is not None:
                self.split_lineage_history[child_b_id] = current_step

            # Note: new children will initialize bimodality states on first observation

        self.edits_count += 1

    def get_routing_adjustments(self, current_step: int) -> tuple[set[int], dict[int, float]]:
        """
        Get routing adjustments for suppression trials.

        Returns two separate channels:
        - hard_block_ids: Set of expert IDs that must be masked out entirely (cooldown)
        - soft_penalties: Dict of expert_id -> penalty to subtract from logits (decay phase)

        This separation prevents accidentally hard-blocking an expert when you meant
        to apply a soft penalty, and makes the audit log clearer.

        Layer applies:
        1. Hard mask: logits[hard_block_ids] = -inf (before softmax)
        2. Soft penalty: logits[expert_id] -= penalty (before softmax)
        """
        hard_block_ids = set()
        soft_penalties = {}

        for expert_id in list(self.expert_penalties.keys()):
            # Check cooldown (hard block phase)
            if expert_id in self.expert_cooldowns:
                cooldown_end = self.expert_cooldowns[expert_id]
                if current_step < cooldown_end:
                    # Hard cooldown: expert is unavailable
                    hard_block_ids.add(expert_id)
                    continue
                else:
                    # Cooldown expired, remove it
                    del self.expert_cooldowns[expert_id]

            # Apply decaying penalty (soft block phase)
            penalty = self.expert_penalties[expert_id]
            if penalty > 0.01:  # Threshold for cleanup
                soft_penalties[expert_id] = penalty
            else:
                # Penalty decayed to negligible, remove
                del self.expert_penalties[expert_id]

        return hard_block_ids, soft_penalties

    def update_penalties(self) -> None:
        """
        Decay penalties each step.

        Called by layer after forward pass.
        """
        decay_rate = self.config["suppression"]["decay_rate"]

        for expert_id in list(self.expert_penalties.keys()):
            # Skip decay during cooldown
            if expert_id in self.expert_cooldowns:
                current_step = len(self.observation_history)
                if current_step < self.expert_cooldowns[expert_id]:
                    continue  # No decay during hard cooldown

            # Apply decay
            self.expert_penalties[expert_id] *= (1.0 - decay_rate)

            # Cleanup negligible penalties
            if self.expert_penalties[expert_id] < 0.01:
                del self.expert_penalties[expert_id]

    def compute_adaptive_penalty(self, logit_std: float) -> float:
        """
        Compute scale-aware penalty magnitude.

        Penalty should be large enough to reliably flip top-k routing decisions.
        Use k * logit_std with a minimum floor to handle edge cases.

        Args:
            logit_std: Standard deviation of router logits for current batch/window

        Returns:
            Penalty magnitude (subtracted from logits)
        """
        k = self.config["suppression"].get("penalty_scale_factor", 3.0)
        floor = self.config["suppression"].get("penalty_floor", 5.0)

        # Scale to logit distribution, with floor for stability
        penalty = max(k * logit_std, floor)

        return penalty

    def suppress_expert(
        self,
        expert_id: int,
        duration_steps: int = 0,
        penalty_override: Optional[float] = None,
    ) -> None:
        """
        Suppress expert with penalty (for redundancy trial).

        Args:
            expert_id: Expert to suppress
            duration_steps: Cooldown duration (0 = penalty only, no hard cooldown)
            penalty_override: Override penalty magnitude (for tests or manual control)
                             If None, uses config penalty_magnitude (fixed)
                             For adaptive penalties, caller should compute via
                             compute_adaptive_penalty() and pass as override
        """
        current_step = len(self.observation_history)

        # Use override if provided, else fall back to config
        if penalty_override is not None:
            penalty_magnitude = penalty_override
        else:
            penalty_magnitude = self.config["suppression"]["penalty_magnitude"]

        # Set penalty
        self.expert_penalties[expert_id] = penalty_magnitude

        # Set cooldown if duration specified
        if duration_steps > 0:
            self.expert_cooldowns[expert_id] = current_step + duration_steps

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Export current state for logging/debugging.

        Returns:
            Dictionary with signal values, band state, recent edits, etc.
        """
        # Milestone A: Coherence stats
        coherence_by_expert = {
            expert_id: state.to_dict()
            for expert_id, state in self.coherence_states.items()
        }

        layer_coherence_fast = compute_layer_coherence(
            self.coherence_states, timescale="fast"
        )
        layer_coherence_mid = compute_layer_coherence(
            self.coherence_states, timescale="mid"
        )
        layer_coherence_slow = compute_layer_coherence(
            self.coherence_states, timescale="slow"
        )

        # Milestone B: Bimodality stats
        bimodality_by_expert = {
            expert_id: state.to_dict()
            for expert_id, state in self.bimodality_states.items()
        }

        layer_bimodality = compute_layer_bimodality(
            self.bimodality_states,
            min_observations=self.config["bimodality"]["min_observations"],
        )

        # Milestone F: Merge candidate diagnostics (Phase 1: diagnostic-only)
        merge_diagnostics = None
        if self.config["merge"]["enabled"]:
            merge_proposal = self._try_propose_merge()
            if merge_proposal:
                merge_diagnostics = {
                    "candidate_found": True,
                    "expert_a": merge_proposal.expert_id,
                    "expert_b": merge_proposal.evidence.get("expert_b_id"),
                    "similarity": round(merge_proposal.evidence.get("similarity_score", 0.0), 4),
                    "utilization_a": round(merge_proposal.evidence.get("utilization_a", 0.0), 4),
                    "utilization_b": round(merge_proposal.evidence.get("utilization_b", 0.0), 4),
                    "predicted_delta_f": round(merge_proposal.delta_f_l, 4),
                }
            else:
                merge_diagnostics = {"candidate_found": False}

        current_step = len(self.observation_history)

        return {
            "layer_id": self.layer_id,
            "max_experts": self.max_experts,
            "observations_count": current_step,
            "edits_count": len(self.edit_log),
            # Milestone A: Coherence diagnostics
            "coherence": {
                "layer_coherence_fast": round(layer_coherence_fast, 4),
                "layer_coherence_mid": round(layer_coherence_mid, 4),
                "layer_coherence_slow": round(layer_coherence_slow, 4),
                "by_expert": coherence_by_expert,
            },
            # Milestone B: Bimodality diagnostics
            "bimodality": {
                "layer_bimodality": round(layer_bimodality, 4),
                "by_expert": bimodality_by_expert,
            },
            # Milestone C: Free energy diagnostics
            "free_energy": (
                self.free_energy_state.to_dict() if self.free_energy_state else None
            ),
            # Milestone F: Merge diagnostics (Phase 1: diagnostic-only)
            "merge": merge_diagnostics,
            # Milestone F Phase 1.5: Suppression trials (two-channel)
            "suppression": {
                "hard_blocks": list(self.expert_cooldowns.keys()),
                "soft_penalties": {
                    expert_id: round(penalty, 4)
                    for expert_id, penalty in self.expert_penalties.items()
                    if expert_id not in self.expert_cooldowns  # Only show soft penalties not in cooldown
                },
                "cooldowns": {
                    expert_id: cooldown_end - current_step
                    for expert_id, cooldown_end in self.expert_cooldowns.items()
                    if current_step < cooldown_end
                },
            },
        }

    def _try_propose_spawn(self) -> Optional[EditProposal]:
        """
        Try to propose SPAWN if layer needs more capacity.

        Milestone D: SPAWN trigger logic.

        Conditions:
        - F_l is high (layer inefficient)
        - Not at max capacity (num_active < max_experts)
        - Predicted ΔF_l < MIN_DELTA_F (edit will help)

        Returns:
            EditProposal if conditions met, None otherwise
        """
        if not self.free_energy_state:
            return None

        # Get config thresholds
        min_delta_f = self.config["triggers"]["min_delta_f"]

        # Check capacity limit
        num_active = self.free_energy_state.num_active_experts
        if num_active >= self.max_experts:
            return None  # Already at max capacity

        # Check if F_l is high enough to justify spawn
        f_l_current = self.free_energy_state.components.total
        f_l_threshold = 0.01  # TEST-CALIBRATED: For 8-expert layers with partial F_l

        if f_l_current < f_l_threshold:
            return None  # F_l not high enough

        # Predict ΔF_l from spawning (rough estimate)
        # SPAWN adds capacity, should reduce complexity term
        complexity_delta = -self.config["free_energy"]["lambda_complexity"] / self.max_experts
        predicted_delta_f = complexity_delta  # Simplified prediction

        # Filter by MIN_DELTA_F
        if predicted_delta_f >= min_delta_f:
            return None  # Edit won't help enough

        # Find best parent expert (highest coherence)
        best_parent_id = max(
            self.coherence_states.keys(),
            key=lambda eid: self.coherence_states[eid].phi_slow
        )

        # Create proposal
        return EditProposal(
            edit_type="spawn",
            expert_id=best_parent_id,  # Parent expert to clone from
            reason=f"Layer inefficient (F_l={f_l_current:.4f}), spawning to add capacity",
            evidence={
                "f_l_current": f_l_current,
                "predicted_delta_f": predicted_delta_f,
                "num_active": num_active,
                "max_experts": self.max_experts,
                "complexity_current": self.free_energy_state.components.complexity,
            },
            calm_credit_required=self.config["triggers"]["spawn_calm_steps"],
            delta_f_l=predicted_delta_f,
        )

    def _try_propose_prune(self) -> Optional[EditProposal]:
        """
        Try to propose PRUNE if expert is decoherent or redundant.

        Milestone D: PRUNE trigger logic.

        Conditions:
        - Expert has low coherence (phi_slow < threshold)
        - OR expert has high redundancy with another expert
        - Predicted ΔF_l < MIN_DELTA_F (edit will help)

        Returns:
            EditProposal if conditions met, None otherwise
        """
        if not self.free_energy_state or not self.coherence_states:
            return None

        # Get config thresholds
        min_delta_f = self.config["triggers"]["min_delta_f"]
        prune_coherence_threshold = 0.3  # TEST-CALIBRATED: phi_slow threshold for decoherence

        # Find decoherent experts
        decoherent_experts = [
            eid
            for eid, state in self.coherence_states.items()
            if state.phi_slow < prune_coherence_threshold and state.total_tokens_seen > 100
        ]

        if not decoherent_experts:
            return None  # No decoherent experts

        # Pick worst expert (lowest coherence)
        target_expert_id = min(
            decoherent_experts,
            key=lambda eid: self.coherence_states[eid].phi_slow
        )

        # Predict ΔF_l from pruning (rough estimate)
        # PRUNE reduces complexity term
        complexity_delta = -self.config["free_energy"]["lambda_complexity"] / self.max_experts
        predicted_delta_f = complexity_delta  # Simplified prediction

        # Filter by MIN_DELTA_F
        if predicted_delta_f >= min_delta_f:
            return None  # Edit won't help enough

        # Create proposal
        target_state = self.coherence_states[target_expert_id]
        return EditProposal(
            edit_type="prune",
            expert_id=target_expert_id,
            reason=f"Expert {target_expert_id} decoherent (phi_slow={target_state.phi_slow:.4f})",
            evidence={
                "phi_slow": target_state.phi_slow,
                "phi_fast": target_state.phi_fast,
                "phi_delta": target_state.phi_delta,
                "predicted_delta_f": predicted_delta_f,
                "f_l_current": self.free_energy_state.components.total,
            },
            calm_credit_required=self.config["triggers"]["prune_calm_steps"],
            delta_f_l=predicted_delta_f,
        )

    def _check_pending_split_latch(self) -> Optional[EditProposal]:
        """
        Check if there's a pending latched SPLIT that can be executed now.

        Milestone E: Pending split latch with TTL.

        When SPLIT is proposed but blocked (e.g., in STRAIN), the evidence is latched.
        This method checks if the latched SPLIT can now be executed:
        - TTL hasn't expired
        - Evidence hasn't been contradicted (bimodality still valid)
        - Returns proposal if ready, None otherwise

        Returns:
            EditProposal if latched SPLIT is ready, None otherwise
        """
        if self.pending_split_latch is None:
            return None  # No latched SPLIT

        current_step = len(self.observation_history)
        latch = self.pending_split_latch

        # Check if latch has expired
        if latch.is_expired(current_step):
            self.pending_split_latch = None  # Clear expired latch
            return None

        # Check if evidence is still valid (re-check bimodality)
        expert_id = latch.expert_id
        if expert_id not in self.bimodality_states:
            # Expert no longer exists or not tracked
            self.pending_split_latch = None
            return None

        state = self.bimodality_states[expert_id]
        current_score = state.compute_bimodality_score()
        split_threshold = self.config["bimodality"]["split_threshold"]

        if not latch.is_valid(current_score, split_threshold):
            # Evidence contradicted (bimodality dropped below threshold)
            self.pending_split_latch = None
            return None

        # Latch is still valid, return the proposal
        # Note: Caller (swiss-ai/MoE layer) will check stress bands and calm credit
        # If executed, caller will call apply() and we'll clear the latch
        return EditProposal(
            edit_type="split",
            expert_id=latch.expert_id,
            reason=f"Expert {latch.expert_id} latched SPLIT (from step {latch.proposed_at_step}, bimodality={current_score:.2f})",
            evidence=latch.evidence,
            calm_credit_required=latch.calm_credit_required,
            delta_f_l=latch.delta_f_l,
        )

    def _try_propose_split(self) -> Optional[EditProposal]:
        """
        Try to propose SPLIT if expert exhibits high bimodality.

        Milestone E: SPLIT trigger logic.

        Conditions:
        - Expert has high bimodality score (> split_threshold)
        - Capacity available for 2 new experts (net +1 after parent pruned)
        - Sufficient observations (min_observations threshold)
        - Predicted ΔF_l < MIN_DELTA_F (edit will help)

        NOTE: split_threshold is TEST-CALIBRATED for 8-expert layers.
        Real training may need different thresholds.
        """
        # Check if we have free energy state
        if self.free_energy_state is None:
            return None

        # Check capacity: need 2 slots (parent will be pruned, net +1)
        capacity_needed = 2
        num_active = self.free_energy_state.num_active_experts
        capacity_available = self.max_experts - num_active

        if capacity_available < capacity_needed:
            return None  # Not enough capacity

        # Get config thresholds
        split_threshold = self.config["bimodality"]["split_threshold"]  # 0.5
        min_observations = self.config["bimodality"]["min_observations"]  # 100
        min_delta_f = self.config["triggers"]["min_delta_f"]  # -0.001

        # Find expert with highest bimodality
        best_candidate = None
        best_score = split_threshold  # Must exceed threshold
        best_state = None

        for expert_id, state in self.bimodality_states.items():
            total_obs = state.count_a + state.count_b
            if total_obs < min_observations:
                continue  # Need more observations

            score = state.compute_bimodality_score()
            if score > best_score:
                best_score = score
                best_candidate = expert_id
                best_state = state

        if best_candidate is None:
            return None  # No bimodal experts above threshold

        # Check lineage cooldown (prevent repeated splits on transient signals)
        current_step = len(self.observation_history)
        cooldown_window = self.config["triggers"]["split_lineage_cooldown"]

        if best_candidate in self.split_lineage_history:
            last_split_step = self.split_lineage_history[best_candidate]
            steps_since_split = current_step - last_split_step

            if steps_since_split < cooldown_window:
                # Expert or its lineage was recently split, skip
                return None

        # Predict ΔF_l from splitting
        # Splitting a bimodal expert should reduce redundancy
        # (expert currently serving two incompatible modes)
        rho = self.config["free_energy"]["rho_redundancy"]
        predicted_delta_f = -rho * 0.5  # Estimate: 50% redundancy reduction

        # Filter by MIN_DELTA_F threshold
        if predicted_delta_f >= min_delta_f:
            return None  # Edit won't help enough

        # Create evidence dict
        evidence = {
            "bimodality_score": best_score,
            "separation": best_state.compute_separation(),
            "balance": best_state.compute_balance(),
            "count_a": best_state.count_a,
            "count_b": best_state.count_b,
            "predicted_delta_f": predicted_delta_f,
        }

        # Create latch if it doesn't exist yet (preserve evidence across band transitions)
        if self.pending_split_latch is None or self.pending_split_latch.expert_id != best_candidate:
            current_step = len(self.observation_history)
            ttl_steps = self.config["triggers"]["split_latch_ttl"]

            self.pending_split_latch = PendingSplitLatch(
                expert_id=best_candidate,
                evidence=evidence,
                proposed_at_step=current_step,
                ttl_steps=ttl_steps,
                calm_credit_required=self.config["triggers"]["split_calm_steps"],
                delta_f_l=predicted_delta_f,
            )

        # Create proposal
        return EditProposal(
            edit_type="split",
            expert_id=best_candidate,
            reason=f"Expert {best_candidate} bimodal (score={best_score:.2f}), splitting for specialization",
            evidence=evidence,
            calm_credit_required=self.config["triggers"]["split_calm_steps"],  # 300
            delta_f_l=predicted_delta_f,
        )

    def _try_propose_merge(self) -> Optional[EditProposal]:
        """
        Try to propose MERGE if two experts are redundant (Milestone F Phase 1).

        Diagnostic-only mode: Proposals logged but not executed.

        Conditions:
        - Both experts have low utilization (< utilization_threshold)
        - High output similarity (cosine similarity > similarity_threshold)
        - Sufficient observations (min_observations threshold)
        - Predicted ΔF_l < MIN_DELTA_F (merge will reduce redundancy)

        NOTE: merge thresholds are TEST-CALIBRATED. Real training may need tuning.
        """
        # Check if merge detection is enabled (diagnostic mode)
        if not self.config["merge"]["enabled"]:
            return None

        # Check if we have free energy state
        if self.free_energy_state is None:
            return None

        # Get config thresholds
        similarity_threshold = self.config["merge"]["similarity_threshold"]  # 0.8
        utilization_threshold = self.config["merge"]["utilization_threshold"]  # 0.1
        min_observations = self.config["merge"]["min_observations"]  # 100
        min_delta_f = self.config["triggers"]["min_delta_f"]  # -0.001

        # Find expert pairs with low utilization and high similarity
        best_pair = None
        best_score = similarity_threshold  # Must exceed threshold
        best_evidence = None

        expert_ids = list(self.bimodality_states.keys())
        num_active = self.free_energy_state.num_active_experts

        # Only consider merge if we have at least 2 experts
        if num_active < 2:
            return None

        # Consider all pairs of experts
        for i in range(len(expert_ids)):
            for j in range(i + 1, len(expert_ids)):
                expert_a = expert_ids[i]
                expert_b = expert_ids[j]

                # Check minimum observations for both experts
                state_a = self.bimodality_states[expert_a]
                state_b = self.bimodality_states[expert_b]
                obs_a = state_a.count_a + state_a.count_b
                obs_b = state_b.count_a + state_b.count_b

                if obs_a < min_observations or obs_b < min_observations:
                    continue

                # Check utilization (from most recent observation)
                if expert_a not in self.coherence_states or expert_b not in self.coherence_states:
                    continue

                # Compute utilization from recent observation
                # Use total_tokens_seen as proxy (higher = more utilized)
                # Normalize by observation count for fair comparison
                if not self.observation_history:
                    continue

                last_obs = self.observation_history[-1]
                total_tokens = last_obs.utilization.sum().item()
                if total_tokens == 0:
                    continue

                util_a = last_obs.utilization[expert_a].item() / total_tokens
                util_b = last_obs.utilization[expert_b].item() / total_tokens

                if util_a > utilization_threshold or util_b > utilization_threshold:
                    continue  # At least one expert is well-utilized

                # Compute cosine similarity between centroids
                # Both centroids from mode_a (primary mode) since we're checking overall similarity
                centroid_a = state_a.centroid_a
                centroid_b = state_b.centroid_a

                if centroid_a is None or centroid_b is None:
                    continue  # Not enough data yet

                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    centroid_a.unsqueeze(0),
                    centroid_b.unsqueeze(0),
                    dim=1
                ).item()

                if sim > best_score:
                    best_score = sim
                    best_pair = (expert_a, expert_b)
                    best_evidence = {
                        "expert_a_id": expert_a,
                        "expert_b_id": expert_b,
                        "similarity_score": sim,
                        "utilization_a": util_a,
                        "utilization_b": util_b,
                    }

        if best_pair is None:
            return None  # No merge candidates

        expert_a, expert_b = best_pair

        # Predict ΔF_l from merging
        # Merging two redundant experts should reduce redundancy term
        rho = self.config["free_energy"]["rho_redundancy"]
        predicted_delta_f = -rho * 0.3  # Estimate: 30% redundancy reduction

        # Filter by MIN_DELTA_F threshold
        if predicted_delta_f >= min_delta_f:
            return None  # Merge won't help enough

        best_evidence["predicted_delta_f"] = predicted_delta_f

        # Create proposal
        return EditProposal(
            edit_type="merge",
            expert_id=expert_a,  # Primary expert (will absorb expert_b)
            reason=f"Experts {expert_a} and {expert_b} redundant (similarity={best_score:.2f}, util={best_evidence['utilization_a']:.2f}/{best_evidence['utilization_b']:.2f})",
            evidence=best_evidence,
            calm_credit_required=self.config["triggers"]["merge_calm_steps"],  # 400
            delta_f_l=predicted_delta_f,
        )

    def _update_coherence(self, snapshot: ObservationSnapshot) -> None:
        """
        Update coherence states from observation snapshot.

        Milestone A: Coherence tracking only (no triggers, no edits).

        Args:
            snapshot: Current step's observation
        """
        # Get config values
        config_coherence = self.config["coherence"]
        alpha_fast = config_coherence["alpha_fast"]
        alpha_mid = config_coherence["alpha_mid"]
        alpha_slow = config_coherence["alpha_slow"]

        # Compute coherence for all experts
        # Need to build active_mask from utilization (proxy)
        utilization = snapshot.utilization
        active_mask = utilization > 0  # Experts that processed tokens

        # Compute raw coherence scores
        phi_raw = compute_coherence(
            expert_outputs=snapshot.expert_outputs,
            mixture_output=snapshot.mixture_output,
            router_probs=snapshot.router_probs,
            active_mask=active_mask,
        )

        # Update EMA states for each expert
        for expert_id in range(len(phi_raw)):
            if not active_mask[expert_id]:
                continue  # Skip inactive experts

            # Create state if doesn't exist (for spawned experts)
            if expert_id not in self.coherence_states:
                self.coherence_states[expert_id] = CoherenceState(
                    expert_id=expert_id,
                    layer_id=self.layer_id,
                )

            # Update with new measurement
            update_coherence_ema(
                state=self.coherence_states[expert_id],
                phi_raw=phi_raw[expert_id].item(),
                alpha_fast=alpha_fast,
                alpha_mid=alpha_mid,
                alpha_slow=alpha_slow,
                step=snapshot.step,
                num_tokens=int(utilization[expert_id].item()),
            )

    def _update_bimodality(self, snapshot: ObservationSnapshot) -> None:
        """
        Update bimodality states from observation snapshot.

        Milestone B: Bimodality tracking only (no triggers, no edits).

        Args:
            snapshot: Current step's observation
        """
        # Get config values
        config_bimodality = self.config["bimodality"]
        alpha = config_bimodality["ema_alpha"]

        # Get active experts
        utilization = snapshot.utilization
        active_mask = utilization > 0

        # Get d_model from expert outputs
        d_model = snapshot.expert_outputs.shape[2]

        # Update bimodality for each active expert
        for expert_id in range(len(active_mask)):
            if not active_mask[expert_id]:
                continue  # Skip inactive experts

            # Create state if doesn't exist (for spawned experts or first observation)
            if expert_id not in self.bimodality_states:
                self.bimodality_states[expert_id] = BimodalityState(
                    expert_id=expert_id,
                    layer_id=self.layer_id,
                    d_model=d_model,
                    centroid_a=torch.zeros(d_model, device=snapshot.expert_outputs.device),
                    centroid_b=torch.zeros(d_model, device=snapshot.expert_outputs.device),
                    alpha=alpha,
                )

            # Update with expert output for this batch
            expert_output = snapshot.expert_outputs[expert_id]  # [B*T, d_model]
            update_bimodality(
                state=self.bimodality_states[expert_id],
                expert_output=expert_output,
            )

    def _update_free_energy(self, snapshot: ObservationSnapshot) -> None:
        """
        Update free energy state from observation snapshot.

        Milestone C: Free energy tracking only (no triggers, no edits).

        Args:
            snapshot: Current step's observation
        """
        # Get config values
        config_fe = self.config["free_energy"]
        lambda_complexity = config_fe["lambda_complexity"]
        rho_redundancy = config_fe["rho_redundancy"]
        kappa_instability = config_fe["kappa_instability"]
        min_tokens_active = config_fe["min_tokens_active"]

        # Build active mask from utilization
        utilization = snapshot.utilization
        active_mask = utilization >= min_tokens_active
        num_active = active_mask.sum().item()

        # Build coherence tensors from coherence states
        num_experts = len(self.coherence_states)
        coherence_fast = torch.zeros(num_experts)
        coherence_slow = torch.zeros(num_experts)

        for expert_id, state in self.coherence_states.items():
            coherence_fast[expert_id] = state.phi_fast
            coherence_slow[expert_id] = state.phi_slow

        # Create free energy state
        self.free_energy_state = create_free_energy_state(
            layer_id=self.layer_id,
            step=snapshot.step,
            num_active_experts=num_active,
            max_experts=self.max_experts,
            expert_outputs=snapshot.expert_outputs,
            coherence_fast=coherence_fast,
            coherence_slow=coherence_slow,
            active_mask=active_mask,
            lambda_complexity=lambda_complexity,
            rho_redundancy=rho_redundancy,
            kappa_instability=kappa_instability,
            misfit=None,  # Leave as None (partial F_l) until canonical misfit proxy exists
        )

    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """
        Merge user config into default config.

        Allows partial config overrides without breaking defaults.
        """
        for section, values in user_config.items():
            if section in self.config:
                self.config[section].update(values)
            else:
                self.config[section] = values

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """
        Default configuration for signal processing.

        IMPORTANT: This is the ONE place for all thresholds and weights.
        If you change a value, it should be obvious in a diff.
        """
        return {
            # Coherence tracking (Milestone A)
            "coherence": {
                "alpha_fast": 0.1,  # ~10 steps half-life
                "alpha_mid": 0.01,  # ~100 steps half-life
                "alpha_slow": 0.001,  # ~1000 steps half-life
                "degrade_threshold": -0.02,  # phi_delta threshold for degradation
            },

            # Bimodality detection (Milestone B)
            "bimodality": {
                "ema_alpha": 0.95,  # ~20 steps half-life for centroid updates
                "min_observations": 100,  # Minimum observations before reporting
                "split_threshold": 0.5,  # Bimodality score threshold (future use)
            },

            # Merge detection (Milestone F Phase 1)
            # NOTE: Diagnostic-only mode (proposals logged, not executed)
            "merge": {
                "enabled": False,  # Phase 1: diagnostic only, Phase 2: execution
                "similarity_threshold": 0.8,  # Cosine similarity > 0.8 = merge candidate
                "utilization_threshold": 0.1,  # Both experts < 10% utilization
                "min_observations": 100,  # Minimum observations before considering
            },

            # Suppression trials (Milestone F Phase 1.5)
            # Before MERGE, suppress one expert to test redundancy
            "suppression": {
                "penalty_magnitude": 10.0,  # Fixed penalty (for tests or manual override)
                "penalty_scale_factor": 3.0,  # Adaptive: penalty = k * logit_std
                "penalty_floor": 5.0,  # Minimum penalty (when logits are very flat)
                "cooldown_steps": 100,  # Steps to suppress expert (0 = use penalty only)
                "decay_rate": 0.1,  # Penalty decay per step (0 = no decay until cooldown ends)
            },

            # Free energy (Milestone C)
            "free_energy": {
                "lambda_complexity": 0.01,  # Weight for complexity term
                "rho_redundancy": 0.01,  # Weight for redundancy term
                "kappa_instability": 0.01,  # Weight for instability term
                "min_tokens_active": 1,  # Minimum tokens to count as active
                "min_tokens_redundancy": 100,  # Stricter threshold for redundancy
            },

            # Evidence thresholds (Milestone D)
            # NOTE: These are TEST-CALIBRATED defaults, not universal constants
            # Real training runs may need different thresholds based on:
            # - Layer width (more experts → different F_l scale)
            # - Dataset characteristics (clean vs noisy data)
            # - Training regime (batch size, learning rate)
            "triggers": {
                "min_delta_f": -0.001,  # ΔF_l threshold (test-calibrated for 8-expert layers)
                "spawn_calm_steps": 200,  # Calm credit required for spawn
                "prune_calm_steps": 500,  # Calm credit required for prune (stricter than spawn)
                "split_calm_steps": 300,  # NEW (Milestone E): Calm credit for split (between spawn and prune)
                "split_latch_ttl": 500,  # Time-to-live for pending SPLIT latch (steps)
                "split_lineage_cooldown": 500,  # Prevent re-splitting children or siblings (steps)
                "merge_calm_steps": 400,  # NEW (Milestone F): Calm credit for merge (between split and prune)
            },

            # Stress bands (already integrated)
            "stress_bands": {
                "comfort_ceiling": 1.0,
                "strain_ceiling": 2.0,
                "hysteresis_margin": 0.05,
            },
        }


# ============================================================================
# Helper Functions (Module-Level Interface)
# ============================================================================

def create_controller(
    layer_id: int,
    max_experts: int,
    initial_active: int,
    config: Optional[Dict[str, Any]] = None,
    autonomous_mode: bool = False,
) -> ChronoController:
    """
    Factory function for controller creation.

    This is the primary entry point from swiss-ai/MoE.

    Args:
        layer_id: Layer index
        max_experts: Maximum experts in layer
        initial_active: Number of initially active experts
        config: Optional config overrides
        autonomous_mode: If True, enable autonomous triggers (Milestone D+)
                        If False, diagnostic mode only (Milestones A-C)
    """
    return ChronoController(
        layer_id=layer_id,
        max_experts=max_experts,
        initial_active=initial_active,
        config=config,
        autonomous_mode=autonomous_mode,
    )


def load_config_from_file(path: str) -> Dict[str, Any]:
    """
    Load controller config from JSON/YAML file.

    Enables versioned config without code changes.
    """
    import json
    from pathlib import Path

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(config_path) as f:
        return json.load(f)
