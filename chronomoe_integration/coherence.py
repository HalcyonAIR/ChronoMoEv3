"""
Phase 1 coherence tracking (simplified for swiss-ai/MoE integration).

Ported from ChronoMoEv3/chronomoe_v3/coherence.py.

Coherence measures: "Is the expert's output directionally aligned with what
the layer actually produced under the router's mixture?"

    phi_e(t) = cosine(y_bar_e(t), y_bar_mix(t))

where:
    y_bar_e(t) = mean expert output over tokens it processed
    y_bar_mix(t) = mean mixture output over all tokens
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class CoherenceState:
    """
    Per-expert coherence state with three-timescale EMA tracking.

    This is the signal layer: diagnostic only, not causal.
    """

    expert_id: int
    layer_id: int

    # Three persistence filters on the same signal
    phi_fast: float = 0.0  # Fast clock: ~10 steps half-life
    phi_mid: float = 0.0  # Mid clock: ~100 steps half-life
    phi_slow: float = 0.0  # Slow clock: ~1000 steps half-life

    # Tracking
    last_update_step: int = 0
    total_tokens_seen: int = 0

    @property
    def phi_delta(self) -> float:
        """Fast - Slow coherence delta. Negative = degradation in progress."""
        return self.phi_fast - self.phi_slow

    @property
    def is_degrading(self) -> bool:
        """Is this expert's fast coherence below its slow baseline?"""
        return self.phi_delta < -0.02  # Threshold from config

    def to_dict(self) -> Dict:
        """Serialize to dict (for logging)."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "phi_fast": round(self.phi_fast, 4),
            "phi_mid": round(self.phi_mid, 4),
            "phi_slow": round(self.phi_slow, 4),
            "phi_delta": round(self.phi_delta, 4),
            "total_tokens_seen": self.total_tokens_seen,
            "is_degrading": self.is_degrading,
        }


def compute_coherence(
    expert_outputs: Tensor,  # [num_experts, B*T, d_model]
    mixture_output: Tensor,  # [B*T, d_model]
    router_probs: Tensor,  # [B*T, num_experts]
    active_mask: Tensor,  # [num_experts]
) -> Tensor:
    """
    Compute phi_e = cosine(y_bar_e, y_bar_mix) for all active experts.

    Args:
        expert_outputs: Per-expert outputs (includes inactive experts)
        mixture_output: Final mixture output
        router_probs: Router probabilities (post-softmax)
        active_mask: Boolean mask indicating which experts are active

    Returns:
        phi: [num_experts] - coherence score per expert in [-1, 1]
                            (inactive experts get 0.0)
    """
    num_experts = expert_outputs.shape[0]
    device = expert_outputs.device

    # Mixture mean direction
    y_bar_mix = mixture_output.mean(dim=0)  # [d_model]

    # Per-expert coherence
    phi = torch.zeros(num_experts, device=device)

    for expert_id in range(num_experts):
        if not active_mask[expert_id]:
            continue  # Inactive experts get phi=0

        # Expert mean output
        y_bar_e = expert_outputs[expert_id].mean(dim=0)  # [d_model]

        # Cosine similarity
        phi[expert_id] = F.cosine_similarity(
            y_bar_e.unsqueeze(0), y_bar_mix.unsqueeze(0), dim=-1
        ).item()

    return phi


def compute_expert_utilization(
    router_probs: Tensor,  # [B*T, num_experts]
    active_mask: Tensor,  # [num_experts]
) -> Tensor:
    """
    Compute token counts per expert (utilization).

    Args:
        router_probs: Router probabilities (post-softmax)
        active_mask: Boolean mask indicating which experts are active

    Returns:
        utilization: [num_experts] - token count per expert
    """
    num_experts = router_probs.shape[1]

    # Count tokens where expert was selected (argmax)
    selected_experts = router_probs.argmax(dim=1)  # [B*T]
    utilization = torch.zeros(num_experts, device=router_probs.device)

    for expert_id in range(num_experts):
        if not active_mask[expert_id]:
            continue
        utilization[expert_id] = (selected_experts == expert_id).sum().item()

    return utilization


def update_coherence_ema(
    state: CoherenceState,
    phi_raw: float,
    alpha_fast: float,
    alpha_mid: float,
    alpha_slow: float,
    step: int,
    num_tokens: int,
) -> CoherenceState:
    """
    Update coherence state with new measurement using three-timescale EMA.

    Args:
        state: Current coherence state
        phi_raw: Raw coherence measurement for this step
        alpha_fast: Fast clock decay rate (e.g., 0.9 = 10 steps half-life)
        alpha_mid: Mid clock decay rate (e.g., 0.99 = 100 steps half-life)
        alpha_slow: Slow clock decay rate (e.g., 0.999 = 1000 steps half-life)
        step: Current training step
        num_tokens: Number of tokens this expert processed this step

    Returns:
        Updated CoherenceState (mutates in place and returns)
    """
    # Update three-timescale EMAs
    state.phi_fast = alpha_fast * state.phi_fast + (1 - alpha_fast) * phi_raw
    state.phi_mid = alpha_mid * state.phi_mid + (1 - alpha_mid) * phi_raw
    state.phi_slow = alpha_slow * state.phi_slow + (1 - alpha_slow) * phi_raw

    # Tracking
    state.last_update_step = step
    state.total_tokens_seen += num_tokens

    return state


def compute_layer_coherence(
    states: Dict[int, CoherenceState], timescale: str = "slow"
) -> float:
    """
    Compute layer-wide coherence Psi_l from expert states.

    Args:
        states: Dict of expert_id -> CoherenceState
        timescale: Which clock to use ('fast', 'mid', or 'slow')

    Returns:
        Psi_l: Weighted coherence score for this layer
    """
    if not states:
        return 0.0

    # Get coherence at specified timescale
    phi_attr = f"phi_{timescale}"
    phi_values = torch.tensor([getattr(s, phi_attr) for s in states.values()])

    # Weight by utilization (experts that process more tokens count more)
    weights = torch.tensor(
        [s.total_tokens_seen for s in states.values()], dtype=torch.float32
    )

    if weights.sum() == 0:
        return 0.0

    weights = weights / weights.sum()

    return (phi_values * weights).sum().item()
