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
    new_expert_id: Optional[int] = None  # For spawn
    reason: str = ""  # Why it failed (if not success)


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
    ):
        self.layer_id = layer_id
        self.max_experts = max_experts
        self.initial_active = initial_active

        # Config consolidation (one place for all signal thresholds)
        self.config = config or self._default_config()

        # Internal state (stays behind the boundary)
        # Milestone A: Coherence tracking
        self.coherence_states: Dict[int, CoherenceState] = {}
        for expert_id in range(initial_active):
            self.coherence_states[expert_id] = CoherenceState(
                expert_id=expert_id,
                layer_id=layer_id,
            )

        # Milestone B: Add bimodality_state here
        # Milestone C: Add free_energy_state here

        # Observation history (ring buffer)
        self.observation_history: List[ObservationSnapshot] = []
        self.max_history = 100  # Keep last 100 observations

        # Edit audit trail
        self.edit_log: List[Dict[str, Any]] = []

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
        # self._update_bimodality(snapshot)

        # Milestone C: Compute free energy
        # self._compute_free_energy(snapshot)

    def decide(self) -> List[EditProposal]:
        """
        Propose lifecycle operations based on accumulated signals.

        Returns empty list until Milestone D (autonomous triggers).

        Decision logic:
        - Check calm gates (stress bands)
        - Evaluate evidence (ΔF_l threshold)
        - Propose spawn if layer starving (high misfit)
        - Propose prune if expert decoherent (low phi_slow)

        Returns:
            List of proposed edits (may be empty)
        """
        proposals = []

        # Milestone D: Enable autonomous triggers here
        # if self._should_propose_spawn():
        #     proposals.append(self._create_spawn_proposal())
        # if self._should_propose_prune():
        #     proposals.append(self._create_prune_proposal())

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
        # (e.g., reset coherence state for spawned expert)

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

        return {
            "layer_id": self.layer_id,
            "max_experts": self.max_experts,
            "observations_count": len(self.observation_history),
            "edits_count": len(self.edit_log),
            # Milestone A: Coherence diagnostics
            "coherence": {
                "layer_coherence_fast": round(layer_coherence_fast, 4),
                "layer_coherence_mid": round(layer_coherence_mid, 4),
                "layer_coherence_slow": round(layer_coherence_slow, 4),
                "by_expert": coherence_by_expert,
            },
            # Milestone B: Add bimodality stats
            # Milestone C: Add free energy values
        }

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
                "ema_alpha": 0.1,
                "min_observations": 100,  # Centroid initialization threshold
                "split_threshold": 0.5,  # Bimodality score threshold
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
            "triggers": {
                "min_delta_f": -0.05,  # ΔF_l threshold for proposals
                "spawn_calm_steps": 200,  # Calm credit required for spawn
                "prune_calm_steps": 500,  # Calm credit required for prune
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
) -> ChronoController:
    """
    Factory function for controller creation.

    This is the primary entry point from swiss-ai/MoE.
    """
    return ChronoController(
        layer_id=layer_id,
        max_experts=max_experts,
        initial_active=initial_active,
        config=config,
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
