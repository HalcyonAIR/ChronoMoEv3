"""
Free energy objective for MoE lifecycle.

Single objective that unifies spawn/prune/split/merge decisions:

    F_l = (1 - Psi_l) + lambda * N_l + rho * R_l + kappa * I_l

Where:
- Psi_l: Layer coherence (weighted average of expert slow coherence)
- N_l: Number of active experts (complexity tax)
- R_l: Redundancy (output similarity + co-activation overlap)
- I_l: Instability (sum of bimodality scores)

Design principle: The slow clock acts only when it can reduce F_l enough to
justify structural disruption. This replaces the entire rule bag with one
principled objective.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class FreeEnergyComponents:
    """
    Decomposed free energy components for one layer.

    Breaking down F_l into its constituent terms allows:
    - Understanding what drives lifecycle decisions
    - Debugging pathological behavior
    - Logging evidence for structural changes
    """

    misfit: float  # (1 - Psi_l) - layer incoherence
    complexity: float  # lambda * N_l - expert count penalty
    redundancy: float  # rho * R_l - duplicate expert penalty
    instability: float  # kappa * I_l - bimodality penalty

    @property
    def total(self) -> float:
        """Total free energy F_l."""
        return self.misfit + self.complexity + self.redundancy + self.instability

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "misfit": self.misfit,
            "complexity": self.complexity,
            "redundancy": self.redundancy,
            "instability": self.instability,
            "total": self.total,
        }


@dataclass
class FreeEnergyState:
    """
    Per-layer free energy state.

    Tracks all components needed to compute F_l and predict edits.
    """

    layer_id: int
    step: int

    # Current free energy
    components: FreeEnergyComponents

    # Raw inputs (for edit prediction)
    layer_coherence: float  # Psi_l
    num_active_experts: int  # N_l
    redundancy_score: float  # R_l (before weight)
    instability_score: float  # I_l (before weight)

    # Expert-level detail (for targeting edits)
    expert_coherence: Tensor  # [E] - phi_slow per expert
    expert_utilization: Tensor  # [E] - routing share per expert
    expert_bimodality: Optional[Tensor] = None  # [E] - bimodality score per expert
    expert_similarity: Optional[Tensor] = None  # [E, E] - pairwise output similarity


def compute_layer_coherence(
    phi_slow: Tensor,
    utilization: Tensor,
    weight_by_utilization: bool = True,
    mask: Optional[Tensor] = None,
    min_tokens: int = 1,
) -> float:
    """
    Compute layer coherence Psi_l.

    Psi_l = weighted average of expert slow coherence, where weights are
    routing shares (how much each expert is used).

    Args:
        phi_slow: [E] - slow coherence per expert
        utilization: [E] - tokens routed to each expert
        weight_by_utilization: If True, weight by routing share. If False, uniform.
        mask: [E] - optional pre-computed active mask. If None, computed from min_tokens.
        min_tokens: Minimum tokens to consider expert (used only if mask is None)

    Returns:
        Psi_l: Layer coherence in [0, 1] (higher = more coherent), clamped to valid range
    """
    # Use provided mask or compute from min_tokens
    if mask is None:
        mask = utilization >= min_tokens

    if not mask.any():
        # No experts with sufficient data
        return 0.0

    phi_filtered = phi_slow[mask]
    utilization_filtered = utilization[mask]

    if weight_by_utilization:
        # Weighted average by routing share
        weights = utilization_filtered / utilization_filtered.sum()
        psi = (phi_filtered * weights).sum().item()
    else:
        # Uniform average
        psi = phi_filtered.mean().item()

    # Clamp to [0, 1] to prevent weird edge cases from upstream noise
    return max(0.0, min(1.0, psi))


def compute_misfit_term(psi: float) -> float:
    """
    Compute misfit term (1 - Psi_l).

    High misfit = layer is incoherent = experts not aligned with mixture.

    Args:
        psi: Layer coherence Psi_l in [0, 1]

    Returns:
        Misfit in [0, 1] (higher = worse)
    """
    return 1.0 - psi


def compute_complexity_term(num_experts: int, lambda_weight: float) -> float:
    """
    Compute complexity term lambda * N_l.

    Penalizes having too many experts. Encourages parsimony.

    Args:
        num_experts: Number of active experts in layer
        lambda_weight: Complexity tax weight

    Returns:
        Complexity penalty (higher = more experts)
    """
    return lambda_weight * num_experts


def compute_redundancy_term(
    role_vectors: Tensor,
    utilization: Tensor,
    rho_weight: float,
    similarity_threshold: float = 0.9,
    mask: Optional[Tensor] = None,
    min_tokens: int = 100,
) -> tuple[float, Tensor]:
    """
    Compute redundancy term rho * R_l.

    Detects near-duplicate experts:
    - High output-direction similarity (role vectors nearly parallel)
    - Both experts actually used (not just theoretically similar)

    R_l = fraction of expert pairs that are redundant

    Args:
        role_vectors: [E, d_model] - mean output per expert (normalized)
        utilization: [E] - tokens routed to each expert
        rho_weight: Redundancy penalty weight
        similarity_threshold: Cosine similarity threshold for "redundant" (default 0.9)
        mask: [E] - optional pre-computed active mask. If None, computed from min_tokens.
        min_tokens: Minimum tokens to consider expert (used only if mask is None)

    Returns:
        (weighted_redundancy, similarity_matrix):
            - weighted_redundancy: rho * R_l
            - similarity_matrix: [E, E] - pairwise cosine similarities (always computed)
    """
    num_experts = role_vectors.shape[0]

    # Compute pairwise cosine similarity (always, for debugging)
    role_normed = F.normalize(role_vectors, dim=-1)
    similarity = torch.mm(role_normed, role_normed.t())  # [E, E]

    if num_experts < 2:
        # No pairs to compare, but return similarity for debugging
        return 0.0, similarity

    # Use provided mask or compute from min_tokens
    if mask is None:
        mask = utilization >= min_tokens

    num_valid = mask.sum().item()

    if num_valid < 2:
        # Not enough observed experts, but return similarity for debugging
        return 0.0, similarity

    # Count redundant pairs (upper triangle, excluding diagonal)
    # Only among experts that have been observed enough
    redundant_count = 0
    total_pairs = 0

    for i in range(num_experts):
        if not mask[i]:
            continue
        for j in range(i + 1, num_experts):
            if not mask[j]:
                continue
            total_pairs += 1
            if similarity[i, j] > similarity_threshold:
                redundant_count += 1

    # R_l = fraction of pairs that are redundant
    if total_pairs == 0:
        r_l = 0.0
    else:
        r_l = redundant_count / total_pairs

    return rho_weight * r_l, similarity


def compute_instability_term(
    bimodality_scores: Tensor,
    kappa_weight: float,
    utilization: Tensor,
    mask: Optional[Tensor] = None,
    min_tokens: int = 100,
) -> float:
    """
    Compute instability term kappa * I_l.

    Aggregates bimodality across experts. High bimodality = expert serving
    incompatible modes = unstable = should split or prune.

    I_l = utilization-weighted mean of bimodality
    This prevents the penalty from scaling with number of experts.

    Args:
        bimodality_scores: [E] - bimodality score per expert
        kappa_weight: Instability penalty weight
        utilization: [E] - tokens per expert (required for weighting)
        mask: [E] - optional pre-computed active mask. If None, computed from min_tokens.
        min_tokens: Minimum tokens to consider expert (used only if mask is None)

    Returns:
        Instability penalty (higher = more bimodal experts)
    """
    # Use provided mask or compute from min_tokens
    if mask is None:
        mask = utilization >= min_tokens

    if not mask.any():
        return 0.0

    bimodality_filtered = bimodality_scores[mask]
    utilization_filtered = utilization[mask]

    # Utilization-weighted mean (prevents scaling with expert count)
    weights = utilization_filtered / utilization_filtered.sum()
    i_l = (bimodality_filtered * weights).sum().item()

    return kappa_weight * i_l


def compute_free_energy(
    phi_slow: Tensor,
    utilization: Tensor,
    role_vectors: Tensor,
    bimodality_scores: Optional[Tensor],
    lambda_complexity: float,
    rho_redundancy: float,
    kappa_instability: float,
    weight_by_utilization: bool = True,
    min_tokens: int = 1,
    redundancy_min_tokens: int = 100,
    similarity_threshold: float = 0.9,
) -> tuple[FreeEnergyComponents, Tensor, float]:
    """
    Compute full free energy F_l for one layer.

    F_l = (1 - Psi_l) + lambda * N_active + rho * R_l + kappa * I_l

    Args:
        phi_slow: [E] - slow coherence per expert
        utilization: [E] - tokens routed to each expert
        role_vectors: [E, d_model] - mean output per expert (normalized)
        bimodality_scores: [E] or None - bimodality score per expert
        lambda_complexity: Complexity tax weight
        rho_redundancy: Redundancy penalty weight
        kappa_instability: Instability penalty weight
        weight_by_utilization: Weight coherence by routing share
        min_tokens: Minimum tokens to consider expert active (for coherence, instability, count)
        redundancy_min_tokens: Minimum tokens for redundancy detection (stricter)
        similarity_threshold: Cosine similarity threshold for redundancy

    Returns:
        (components, similarity_matrix, free_energy):
            - components: FreeEnergyComponents with all four terms
            - similarity_matrix: [E, E] - pairwise expert similarities
            - free_energy: F_l scalar (sum of all components)
    """
    # Compute canonical active mask once (used for coherence, instability, count)
    active_mask = utilization >= min_tokens
    num_active_experts = active_mask.sum().item()

    # Redundancy uses stricter threshold (needs more samples for reliable similarity)
    redundancy_mask = utilization >= redundancy_min_tokens

    # 1. Layer coherence (uses active_mask)
    psi = compute_layer_coherence(
        phi_slow, utilization, weight_by_utilization, mask=active_mask
    )

    # 2. Misfit term
    misfit = compute_misfit_term(psi)

    # 3. Complexity term (uses active_mask count)
    complexity = compute_complexity_term(num_active_experts, lambda_complexity)

    # 4. Redundancy term (uses stricter redundancy_mask)
    redundancy, similarity = compute_redundancy_term(
        role_vectors,
        utilization,
        rho_redundancy,
        similarity_threshold,
        mask=redundancy_mask,
    )

    # 5. Instability term (uses active_mask, utilization-weighted mean)
    if bimodality_scores is not None:
        instability = compute_instability_term(
            bimodality_scores,
            kappa_instability,
            utilization,
            mask=active_mask,
        )
    else:
        instability = 0.0

    components = FreeEnergyComponents(
        misfit=misfit,
        complexity=complexity,
        redundancy=redundancy,
        instability=instability,
    )

    # Total free energy
    free_energy = components.total

    return components, similarity, free_energy


def create_free_energy_state(
    layer_id: int,
    step: int,
    phi_slow: Tensor,
    utilization: Tensor,
    role_vectors: Tensor,
    bimodality_scores: Optional[Tensor],
    lambda_complexity: float,
    rho_redundancy: float,
    kappa_instability: float,
    weight_by_utilization: bool = True,
    min_tokens: int = 1,
    redundancy_min_tokens: int = 100,
) -> FreeEnergyState:
    """
    Create FreeEnergyState snapshot for one layer.

    Computes all free energy components and stores raw inputs for edit prediction.

    Args:
        layer_id: Layer index
        step: Current training step
        phi_slow: [E] - slow coherence per expert
        utilization: [E] - tokens routed to each expert
        role_vectors: [E, d_model] - mean output per expert
        bimodality_scores: [E] or None - bimodality score per expert
        lambda_complexity: Complexity tax weight
        rho_redundancy: Redundancy penalty weight
        kappa_instability: Instability penalty weight
        weight_by_utilization: Weight coherence by routing share
        min_tokens: Minimum tokens to consider expert active
        redundancy_min_tokens: Minimum tokens for redundancy detection (stricter)

    Returns:
        FreeEnergyState with all components computed
    """
    # Compute free energy
    components, similarity, _ = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=lambda_complexity,
        rho_redundancy=rho_redundancy,
        kappa_instability=kappa_instability,
        weight_by_utilization=weight_by_utilization,
        min_tokens=min_tokens,
        redundancy_min_tokens=redundancy_min_tokens,
    )

    # Compute raw scores (before weighting)
    active_mask = utilization >= min_tokens
    psi = compute_layer_coherence(
        phi_slow, utilization, weight_by_utilization, mask=active_mask
    )

    # Count active experts
    num_active = active_mask.sum().item()

    # Redundancy score (before weight)
    if rho_redundancy > 0:
        redundancy_raw = components.redundancy / rho_redundancy
    else:
        redundancy_raw = 0.0

    # Instability score (before weight)
    if kappa_instability > 0 and bimodality_scores is not None:
        instability_raw = components.instability / kappa_instability
    else:
        instability_raw = 0.0

    return FreeEnergyState(
        layer_id=layer_id,
        step=step,
        components=components,
        layer_coherence=psi,
        num_active_experts=num_active,
        redundancy_score=redundancy_raw,
        instability_score=instability_raw,
        expert_coherence=phi_slow.clone(),
        expert_utilization=utilization.clone(),
        expert_bimodality=bimodality_scores.clone() if bimodality_scores is not None else None,
        expert_similarity=similarity,
    )
