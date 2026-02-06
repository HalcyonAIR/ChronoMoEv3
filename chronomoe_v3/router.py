"""
Router infrastructure with slow bias (beta) mechanism.

The locus lives here: beta_coeff driven by phi_slow feedback.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RouterState:
    """
    Routing infrastructure for one layer.

    Manages beta (slow bias) and routing health metrics.

    Owner: Router system
    Influences: Routing (via beta), crisis detection (via disagreement)
    """

    layer_id: int
    num_experts: int

    # Scale-free bias coefficients (THE LOCUS)
    beta_coeff: Tensor  # [num_experts], device='cuda'
    k_max: float = 0.3

    # Logit scale tracking (for scale-free beta)
    logit_std_ema: float = 1.0
    logit_std_alpha: float = 0.99

    # Disagreement metrics (logged each step)
    disagreement_js: float = 0.0
    disagreement_flip: float = 0.0

    # Temperature (for debugging)
    temperature: float = 1.0

    def compute_beta_eff(self) -> Tensor:
        """
        Effective bias: beta_coeff * logit_std.

        Scale-free: same k produces same impact across regimes.
        """
        k = self.beta_coeff.clamp(-self.k_max, self.k_max)
        return k * self.logit_std_ema

    def update_logit_std(self, z_clean: Tensor):
        """
        Update logit scale EMA.

        Args:
            z_clean: [B×S, E] - clean logits
        """
        current_std = z_clean.std().item()
        self.logit_std_ema = (
            self.logit_std_alpha * self.logit_std_ema
            + (1 - self.logit_std_alpha) * current_std
        )

    def is_in_crisis(self, threshold: float = 0.45) -> bool:
        """Is clean/biased disagreement catastrophic?"""
        return self.disagreement_js > threshold

    def needs_beta_decay(self, threshold: float = 0.30) -> bool:
        """Should we decay beta due to disagreement?"""
        return self.disagreement_js > threshold


def compute_js_divergence(p: Tensor, q: Tensor) -> float:
    """
    Jensen-Shannon divergence between two distributions.

    Args:
        p: [B×S, E] - first distribution
        q: [B×S, E] - second distribution

    Returns:
        JS divergence (scalar)
    """
    p = p + 1e-9
    q = q + 1e-9
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean().item()


def compute_flip_rate(p: Tensor, q: Tensor, top_k: int) -> float:
    """
    Top-1 flip rate between two distributions.

    Args:
        p: [B×S, E] - first distribution
        q: [B×S, E] - second distribution
        top_k: Number of experts to route to

    Returns:
        Flip rate (fraction where top-1 changed)
    """
    top_k_p = p.topk(top_k, dim=-1).indices
    top_k_q = q.topk(top_k, dim=-1).indices

    # Check if top-1 changed
    flip = (top_k_p[:, 0] != top_k_q[:, 0]).float().mean().item()
    return flip


def compute_relevance(router_state: RouterState, threshold: float = 0.2) -> float:
    """
    Compute routing relevance based on clean/biased agreement.

    r = 1.0 if disagreement is low (beta is aligned with input)
    r → 0.0 if disagreement is high (beta is fighting input)

    This is the "bridge detector veto" - prevents Krypto-from-nowhere.

    Args:
        router_state: RouterState with disagreement metrics
        threshold: JS divergence threshold for full relevance

    Returns:
        r: Relevance scalar in [0, 1]
    """
    js = router_state.disagreement_js

    if js < threshold:
        return 1.0  # Full relevance
    elif js < threshold * 3:
        # Linear decay
        return 1.0 - (js - threshold) / (threshold * 2)
    else:
        return 0.0  # No relevance (crisis)


class ChronoRouter(nn.Module):
    """
    Router with dual distribution tracking.

    Computes both clean (z_clean) and biased (z_biased) logits.
    Routes using biased, but logs disagreement between them.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        layer_id: int = 0,
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.layer_id = layer_id

        # Gate weights
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Router state
        self.router_state = RouterState(
            layer_id=layer_id,
            num_experts=num_experts,
            beta_coeff=torch.zeros(num_experts, device=device),
        )

    def forward(
        self, hidden_states: Tensor, top_k: int = 2, use_relevance: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Router with dual distribution tracking.

        Args:
            hidden_states: [B×S, d_model] - input hidden states
            top_k: Number of experts to route to
            use_relevance: Whether to use bridge detector veto

        Returns:
            top_k_probs: [B×S, top_k] - normalized routing weights
            top_k_indices: [B×S, top_k] - expert indices
            z_clean: [B×S, E] - clean logits
            z_biased: [B×S, E] - biased logits
        """
        # 1. Compute clean logits
        z_clean = self.gate(hidden_states)  # [B×S, E]

        # 2. Update logit scale
        self.router_state.update_logit_std(z_clean)

        # 3. Compute biased logits
        beta_eff = self.router_state.compute_beta_eff()  # [E]

        # 4. Apply relevance modulation (bridge detector)
        if use_relevance:
            r = compute_relevance(self.router_state, threshold=0.2)
            beta_eff = r * beta_eff

        z_biased = z_clean + beta_eff.unsqueeze(0)  # [B×S, E]

        # 5. Compute both distributions
        p_clean = F.softmax(z_clean / self.router_state.temperature, dim=-1)
        p_biased = F.softmax(z_biased / self.router_state.temperature, dim=-1)

        # 6. Compute disagreement (BEFORE top-k)
        self.router_state.disagreement_js = compute_js_divergence(p_clean, p_biased)
        self.router_state.disagreement_flip = compute_flip_rate(
            p_clean, p_biased, top_k
        )

        # 7. Route using biased distribution
        top_k_probs, top_k_indices = p_biased.topk(top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_probs, top_k_indices, z_clean, z_biased


def update_beta(
    router_state: RouterState,
    coherence_snapshot: dict,
    eta: float = 0.01,
    tau: float = 0.5,
):
    """
    Update beta coefficients based on coherence feedback.

    PROMOTION PRIOR: High phi_slow → increase beta (earn routing advantage)

    Simple rule:
    - If phi_slow < tau: reduce beta (expert not earning routing advantage)
    - If phi_slow > tau: increase beta (expert earning routing advantage)

    Args:
        router_state: RouterState to update
        coherence_snapshot: {expert_id: CoherenceState}
        eta: Learning rate
        tau: Coherence threshold (target)
    """
    for expert_id, coh_state in coherence_snapshot.items():
        # Skip if not observed recently
        if not coh_state.is_being_observed:
            continue

        # Compute delta (PROMOTION prior)
        delta = eta * (coh_state.phi_slow - tau)

        # Update beta_coeff (scale-free)
        router_state.beta_coeff[expert_id] += delta

    # Clamp to prevent saturation
    router_state.beta_coeff.clamp_(-router_state.k_max, router_state.k_max)
