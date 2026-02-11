"""
Free energy sensor for MoE layers (Phase 3).

Computes F_l = [misfit] + lambda * complexity + rho * redundancy + kappa * instability

Where:
- misfit: OPTIONAL (None for partial F_l until canonical proxy exists)
- complexity: num_active / max_experts (wasteful capacity)
- redundancy: expert output similarity (duplicate work)
- instability: coherence variance (degradation volatility)

Ported from ChronoMoEv3/chronomoe_v3/free_energy.py (simplified for swiss-ai/MoE).

IMPORTANT: Milestone C is sensor-only. This module ONLY computes and logs F_l.
NO triggers, NO edits, NO causal effects.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class FreeEnergyComponents:
    """
    Decomposed free energy components for one layer.

    Breaking down F_l into constituent terms allows:
    - Understanding what drives lifecycle decisions (future)
    - Debugging pathological behavior
    - Logging evidence for structural changes
    """

    misfit: Optional[float]  # (1 - Psi_l) - layer incoherence (OPTIONAL)
    complexity: float  # lambda * (N_active / N_max) - capacity waste
    redundancy: float  # rho * R_l - duplicate expert penalty
    instability: float  # kappa * I_l - coherence variance penalty

    @property
    def total(self) -> float:
        """
        Total free energy F_l.

        If misfit is None, returns partial F_l (sum of other components).
        """
        base = self.complexity + self.redundancy + self.instability
        if self.misfit is not None:
            return self.misfit + base
        return base

    @property
    def is_partial(self) -> bool:
        """True if misfit term is missing (partial F_l)."""
        return self.misfit is None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = {
            "complexity": self.complexity,
            "redundancy": self.redundancy,
            "instability": self.instability,
            "total": self.total,
            "is_partial": self.is_partial,
        }
        if self.misfit is not None:
            result["misfit"] = self.misfit
        return result


@dataclass
class FreeEnergyState:
    """
    Per-layer free energy state.

    Tracks all components needed to compute F_l.
    """

    layer_id: int
    step: int

    # Current free energy
    components: FreeEnergyComponents

    # Raw inputs (for diagnostics)
    num_active_experts: int
    max_experts: int
    redundancy_score: float  # R_l (before weight)
    instability_score: float  # I_l (before weight)

    def to_dict(self) -> Dict:
        """Serialize state to dict (for logging)."""
        return {
            "layer_id": self.layer_id,
            "step": self.step,
            "num_active_experts": self.num_active_experts,
            "max_experts": self.max_experts,
            "redundancy_score": round(self.redundancy_score, 4),
            "instability_score": round(self.instability_score, 4),
            "components": self.components.to_dict(),
        }


def compute_complexity_term(
    num_active: int, max_experts: int, lambda_weight: float
) -> float:
    """
    Compute complexity term lambda * (N_active / N_max).

    Penalizes wasteful capacity allocation. Normalized by max_experts to keep
    term in [0, lambda] regardless of layer width.

    Args:
        num_active: Number of active experts (utilization > min_tokens)
        max_experts: Maximum experts in layer
        lambda_weight: Complexity tax weight

    Returns:
        Complexity penalty in [0, lambda_weight]
    """
    if max_experts == 0:
        return 0.0

    # Normalized by max capacity
    capacity_ratio = num_active / max_experts
    return lambda_weight * capacity_ratio


def compute_redundancy_term(
    expert_outputs: Tensor,
    active_mask: Tensor,
    rho_weight: float,
    min_separation: float = 0.3,
) -> tuple[float, float]:
    """
    Compute redundancy term rho * R_l.

    Detects near-duplicate experts by measuring pairwise cosine similarity
    of expert mean outputs. High similarity = redundant work.

    R_l = mean pairwise similarity (1 = all experts identical)

    Args:
        expert_outputs: [num_experts, B*T, d_model] - per-expert outputs
        active_mask: [num_experts] - which experts are active
        rho_weight: Redundancy penalty weight
        min_separation: Minimum mean cosine distance for "diverse" (default 0.3)

    Returns:
        (weighted_redundancy, raw_redundancy):
            - weighted_redundancy: rho * R_l
            - raw_redundancy: R_l (before weight, for diagnostics)
    """
    # Filter to active experts only
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    num_active = len(active_indices)

    if num_active < 2:
        # No pairs to compare
        return 0.0, 0.0

    # Compute mean output per expert [num_active, d_model]
    expert_means = expert_outputs[active_indices].mean(dim=1)  # Average over tokens

    # Normalize for cosine similarity
    expert_means_normed = F.normalize(expert_means, dim=-1)

    # Pairwise cosine similarity [num_active, num_active]
    similarity = torch.mm(expert_means_normed, expert_means_normed.t())

    # Extract upper triangle (exclude diagonal)
    # We want mean of all pairwise similarities
    triu_indices = torch.triu_indices(num_active, num_active, offset=1)
    pairwise_sims = similarity[triu_indices[0], triu_indices[1]]

    # R_l = mean pairwise similarity (higher = more redundant)
    r_l = pairwise_sims.mean().item()

    # Apply weight
    weighted = rho_weight * r_l

    return weighted, r_l


def compute_instability_term(
    coherence_fast: Tensor,
    coherence_slow: Tensor,
    active_mask: Tensor,
    kappa_weight: float,
) -> tuple[float, float]:
    """
    Compute instability term kappa * I_l.

    Measures coherence variance (fast - slow). High variance indicates
    expert is degrading or oscillating.

    I_l = variance of (phi_fast - phi_slow) across active experts

    Args:
        coherence_fast: [num_experts] - fast coherence per expert
        coherence_slow: [num_experts] - slow coherence per expert
        active_mask: [num_experts] - which experts are active
        kappa_weight: Instability penalty weight

    Returns:
        (weighted_instability, raw_instability):
            - weighted_instability: kappa * I_l
            - raw_instability: I_l (before weight, for diagnostics)
    """
    # Filter to active experts
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    num_active = len(active_indices)

    if num_active == 0:
        return 0.0, 0.0

    # Compute phi_delta = fast - slow
    phi_delta = coherence_fast[active_indices] - coherence_slow[active_indices]

    # I_l = variance of phi_delta (absolute value for scale-invariance)
    # Need at least 2 samples for variance
    if num_active < 2:
        i_l = 0.0
    else:
        i_l = phi_delta.abs().var().item()

    # Apply weight
    weighted = kappa_weight * i_l

    return weighted, i_l


def compute_free_energy(
    num_active_experts: int,
    max_experts: int,
    expert_outputs: Tensor,
    coherence_fast: Tensor,
    coherence_slow: Tensor,
    active_mask: Tensor,
    lambda_complexity: float,
    rho_redundancy: float,
    kappa_instability: float,
    misfit: Optional[float] = None,
) -> tuple[FreeEnergyComponents, float]:
    """
    Compute full free energy F_l for one layer.

    F_l = [misfit] + lambda * complexity + rho * redundancy + kappa * instability

    If misfit is None, computes partial F_l (sum of other components).

    Args:
        num_active_experts: Number of active experts
        max_experts: Maximum experts in layer
        expert_outputs: [num_experts, B*T, d_model] - per-expert outputs
        coherence_fast: [num_experts] - fast coherence per expert
        coherence_slow: [num_experts] - slow coherence per expert
        active_mask: [num_experts] - which experts are active
        lambda_complexity: Complexity tax weight
        rho_redundancy: Redundancy penalty weight
        kappa_instability: Instability penalty weight
        misfit: Optional misfit term (None for partial F_l)

    Returns:
        (components, free_energy):
            - components: FreeEnergyComponents with all terms
            - free_energy: F_l scalar (sum of all components)
    """
    # 1. Complexity term
    complexity = compute_complexity_term(num_active_experts, max_experts, lambda_complexity)

    # 2. Redundancy term
    redundancy, _ = compute_redundancy_term(
        expert_outputs, active_mask, rho_redundancy
    )

    # 3. Instability term
    instability, _ = compute_instability_term(
        coherence_fast, coherence_slow, active_mask, kappa_instability
    )

    # 4. Assemble components
    components = FreeEnergyComponents(
        misfit=misfit,
        complexity=complexity,
        redundancy=redundancy,
        instability=instability,
    )

    # Total free energy
    free_energy = components.total

    return components, free_energy


def create_free_energy_state(
    layer_id: int,
    step: int,
    num_active_experts: int,
    max_experts: int,
    expert_outputs: Tensor,
    coherence_fast: Tensor,
    coherence_slow: Tensor,
    active_mask: Tensor,
    lambda_complexity: float,
    rho_redundancy: float,
    kappa_instability: float,
    misfit: Optional[float] = None,
) -> FreeEnergyState:
    """
    Create FreeEnergyState snapshot for one layer.

    Computes all free energy components and stores raw inputs for diagnostics.

    Args:
        layer_id: Layer index
        step: Current training step
        num_active_experts: Number of active experts
        max_experts: Maximum experts in layer
        expert_outputs: [num_experts, B*T, d_model] - per-expert outputs
        coherence_fast: [num_experts] - fast coherence per expert
        coherence_slow: [num_experts] - slow coherence per expert
        active_mask: [num_experts] - which experts are active
        lambda_complexity: Complexity tax weight
        rho_redundancy: Redundancy penalty weight
        kappa_instability: Instability penalty weight
        misfit: Optional misfit term (None for partial F_l)

    Returns:
        FreeEnergyState with all components computed
    """
    # Compute free energy
    components, _ = compute_free_energy(
        num_active_experts=num_active_experts,
        max_experts=max_experts,
        expert_outputs=expert_outputs,
        coherence_fast=coherence_fast,
        coherence_slow=coherence_slow,
        active_mask=active_mask,
        lambda_complexity=lambda_complexity,
        rho_redundancy=rho_redundancy,
        kappa_instability=kappa_instability,
        misfit=misfit,
    )

    # Compute raw scores (before weighting)
    _, redundancy_raw = compute_redundancy_term(
        expert_outputs, active_mask, rho_weight=1.0  # No weight for raw score
    )

    _, instability_raw = compute_instability_term(
        coherence_fast, coherence_slow, active_mask, kappa_weight=1.0  # No weight
    )

    return FreeEnergyState(
        layer_id=layer_id,
        step=step,
        components=components,
        num_active_experts=num_active_experts,
        max_experts=max_experts,
        redundancy_score=redundancy_raw,
        instability_score=instability_raw,
    )
