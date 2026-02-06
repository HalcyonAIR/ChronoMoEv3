"""
Phase coherence computation and tracking.

This is the foundation of ChronoMoEv3. If phi_e doesn't track functional
participation, nothing else matters.

Coherence measures: "Is the expert's output directionally aligned with what
the layer actually produced under the router's mixture?"

    phi_e(t) = cosine(y_bar_e(t), y_bar_mix(t))

where:
    y_bar_e(t) = mean expert output over tokens it processed
    y_bar_mix(t) = mean mixture output over all tokens
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class MoETrace:
    """
    Coherence-ready trace from one MoE layer forward pass.

    This is the stable interface between router/dispatch and coherence tracking,
    independent of which MoE backend (Mixtral-style loop or Switch-style batch).
    """

    # Layer-level outputs
    mixture: Tensor  # [B×S, d_model] - raw weighted mixture, pre-residual

    # Router state (optional, for bridge detection and routing analysis)
    router_logits_clean: Optional[Tensor] = None  # [B×S, E] - pre-softmax, no beta
    router_logits_biased: Optional[Tensor] = None  # [B×S, E] - pre-softmax, with beta
    router_probs: Optional[Tensor] = None  # [B×S, E] - full softmax distribution
    slow_bias: Optional[Tensor] = None  # [E] - beta vector for this layer

    # Per-expert state (aligned by expert_id)
    active_expert_ids: List[int] = field(default_factory=list)  # [num_active_experts]
    expert_mean_outputs: List[Tensor] = field(default_factory=list)  # List of [d_model]
    token_row_indices: List[Tensor] = field(default_factory=list)  # List of [n_e]
    gate_weights: List[Tensor] = field(default_factory=list)  # List of [n_e]

    @property
    def num_active_experts(self) -> int:
        """Number of experts that processed at least one token."""
        return len(self.active_expert_ids)

    @property
    def d_model(self) -> int:
        """Hidden dimension."""
        return self.mixture.shape[-1]

    def get_expert_utilization(self) -> Tensor:
        """
        Number of tokens routed to each active expert.

        Returns:
            utilization: [num_active_experts] - token count per expert
        """
        return torch.tensor(
            [len(idx) for idx in self.token_row_indices],
            device=self.mixture.device,
        )

    def compute_coherence(self) -> Tensor:
        """
        Compute phi_e = cosine(y_bar_e, y_bar_mix) for all active experts.

        Returns:
            phi: [num_active_experts] - coherence score per expert in [-1, 1]
        """
        if self.num_active_experts == 0:
            return torch.zeros(0, device=self.mixture.device)

        # Mixture mean direction
        y_bar_mix = self.mixture.mean(dim=0)  # [d_model]

        # Per-expert coherence
        phi = torch.zeros(self.num_active_experts, device=self.mixture.device)
        for i in range(self.num_active_experts):
            y_bar_e = self.expert_mean_outputs[i]  # [d_model]
            phi[i] = F.cosine_similarity(
                y_bar_e.unsqueeze(0), y_bar_mix.unsqueeze(0), dim=-1
            )

        return phi

    def compute_weighted_coherence(self) -> Tensor:
        """
        Compute gate-weighted coherence: sum(g_i * phi_i) / sum(g_i).

        This gives more weight to experts that handle more routing mass.

        Returns:
            weighted_phi: [num_active_experts]
        """
        phi = self.compute_coherence()
        utilization = self.get_expert_utilization().float()

        # Weight by number of tokens (proxy for routing mass)
        weights = utilization / utilization.sum()

        return phi, weights

    def compute_layer_coherence(self) -> float:
        """
        Layer-wide coherence: weighted average over active experts.

        Returns:
            Psi: scalar - layer coherence score
        """
        phi, weights = self.compute_weighted_coherence()
        return (phi * weights).sum().item()


@dataclass
class CoherenceState:
    """
    Per-expert coherence state with three-timescale EMA tracking.

    This is the core state variable of ChronoMoEv3. Everything else
    (lifecycle decisions, slow bias, bimodality) derives from these EMAs.
    """

    expert_id: str  # e.g., "L2_E5"
    layer_id: int
    d_model: int

    # Three persistence filters on the same signal
    phi_fast: float = 0.0  # Fast clock: ~10 steps half-life
    phi_mid: float = 0.0  # Mid clock: ~100 steps half-life
    phi_slow: float = 0.0  # Slow clock: ~1000 steps half-life

    # Role vector (what this expert tends to produce)
    role_vector: Optional[Tensor] = None  # [d_model] - EMA of y_bar_e

    # Tracking
    born_step: int = 0
    last_update_step: int = 0
    total_tokens_seen: int = 0

    def __post_init__(self):
        """Initialize role vector if not provided."""
        if self.role_vector is None:
            self.role_vector = torch.zeros(self.d_model)

    @property
    def phi_delta(self) -> float:
        """Fast - Slow coherence delta. Negative = degradation in progress."""
        return self.phi_fast - self.phi_slow

    @property
    def is_degrading(self) -> bool:
        """Is this expert's fast coherence below its slow baseline?"""
        return self.phi_delta < 0

    def to_dict(self) -> Dict:
        """Serialize to dict (for logging)."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "phi_fast": self.phi_fast,
            "phi_mid": self.phi_mid,
            "phi_slow": self.phi_slow,
            "phi_delta": self.phi_delta,
            "total_tokens_seen": self.total_tokens_seen,
            "last_update_step": self.last_update_step,
        }


def compute_coherence(trace: MoETrace) -> Tensor:
    """
    Compute coherence for all active experts in a trace.

    This is a convenience wrapper around trace.compute_coherence().

    Args:
        trace: MoETrace from forward pass

    Returns:
        phi: [num_active_experts] - coherence scores
    """
    return trace.compute_coherence()


def update_coherence_ema(
    state: CoherenceState,
    phi_raw: float,
    y_bar_e: Tensor,
    alpha_fast: float,
    alpha_mid: float,
    alpha_slow: float,
    alpha_role: float = 0.99,
    step: int = 0,
    num_tokens: int = 0,
) -> CoherenceState:
    """
    Update coherence state with new measurement using three-timescale EMA.

    Args:
        state: Current coherence state
        phi_raw: Raw coherence measurement for this step
        y_bar_e: Expert mean output this step [d_model]
        alpha_fast: Fast clock decay rate (e.g., 0.9)
        alpha_mid: Mid clock decay rate (e.g., 0.99)
        alpha_slow: Slow clock decay rate (e.g., 0.999)
        alpha_role: Role vector decay rate (default 0.99)
        step: Current training step
        num_tokens: Number of tokens this expert processed this step

    Returns:
        Updated CoherenceState (mutates in place and returns)
    """
    # Update three-timescale EMAs
    state.phi_fast = alpha_fast * state.phi_fast + (1 - alpha_fast) * phi_raw
    state.phi_mid = alpha_mid * state.phi_mid + (1 - alpha_mid) * phi_raw
    state.phi_slow = alpha_slow * state.phi_slow + (1 - alpha_slow) * phi_raw

    # Update role vector (what this expert typically outputs)
    state.role_vector = alpha_role * state.role_vector + (1 - alpha_role) * y_bar_e

    # Tracking
    state.last_update_step = step
    state.total_tokens_seen += num_tokens

    return state


def batch_update_coherence(
    states: Dict[str, CoherenceState],
    trace: MoETrace,
    alpha_fast: float,
    alpha_mid: float,
    alpha_slow: float,
    step: int,
    layer_id: int,
) -> Dict[str, CoherenceState]:
    """
    Update coherence states for all active experts from a trace.

    Args:
        states: Dict of expert_id -> CoherenceState
        trace: MoETrace from forward pass
        alpha_fast: Fast clock decay rate
        alpha_mid: Mid clock decay rate
        alpha_slow: Slow clock decay rate
        step: Current training step
        layer_id: Layer index

    Returns:
        Updated states dict (mutates in place and returns)
    """
    phi_raw = trace.compute_coherence()  # [num_active_experts]

    for i, expert_idx in enumerate(trace.active_expert_ids):
        expert_id = f"L{layer_id}_E{expert_idx}"

        # Create state if doesn't exist
        if expert_id not in states:
            states[expert_id] = CoherenceState(
                expert_id=expert_id,
                layer_id=layer_id,
                d_model=trace.d_model,
                born_step=step,
            )

        # Update with new measurement
        update_coherence_ema(
            state=states[expert_id],
            phi_raw=phi_raw[i].item(),
            y_bar_e=trace.expert_mean_outputs[i],
            alpha_fast=alpha_fast,
            alpha_mid=alpha_mid,
            alpha_slow=alpha_slow,
            step=step,
            num_tokens=len(trace.token_row_indices[i]),
        )

    return states


def compute_layer_coherence(
    states: Dict[str, CoherenceState], layer_id: int, timescale: str = "slow"
) -> float:
    """
    Compute layer-wide coherence Psi_l from expert states.

    Args:
        states: Dict of expert_id -> CoherenceState
        layer_id: Which layer to compute for
        timescale: Which clock to use ('fast', 'mid', or 'slow')

    Returns:
        Psi_l: Weighted coherence score for this layer
    """
    layer_experts = [s for s in states.values() if s.layer_id == layer_id]

    if not layer_experts:
        return 0.0

    # Get coherence at specified timescale
    phi_attr = f"phi_{timescale}"
    phi_values = torch.tensor([getattr(s, phi_attr) for s in layer_experts])

    # Weight by utilization (experts that process more tokens count more)
    weights = torch.tensor(
        [s.total_tokens_seen for s in layer_experts], dtype=torch.float32
    )
    weights = weights / weights.sum()

    return (phi_values * weights).sum().item()
