"""
Collapse detection: unified survival signal for stress bands.

Answers: "Did the system survive this window without internal collapse?"

Not task performance. Not loss. Internal functional integrity.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class CollapseThresholds:
    """
    Thresholds for what counts as "collapse".

    These are conservative: only flag collapse when things are clearly broken,
    not just suboptimal. The system should be allowed to be stressed without
    being declared collapsed.
    """

    # Layer coherence below this = layer incoherent (experts not aligned with mixture)
    psi_critical: float = 0.1

    # Effective expert count below this = routing collapsed to single expert
    neff_critical: float = 1.5

    # Saturation above this = one expert hogging all load
    saturation_critical: float = 0.9

    # Mean bimodality above this = many experts serving incompatible basins
    bimodality_critical: float = 0.7


@dataclass
class CollapseSignals:
    """
    Raw signals used for collapse detection.

    All optional because not all contexts have all signals.
    """

    psi: Optional[float] = None  # Layer coherence (from compute_layer_coherence)
    neff: Optional[float] = None  # Effective expert count (from compute_neff)
    saturation: Optional[float] = None  # Max expert load (from compute_saturation)
    mean_bimodality: Optional[float] = None  # Average bimodality across experts


def compute_neff(probs: Tensor, min_prob: float = 1e-8) -> float:
    """
    Effective number of experts (entropy-based).

    Neff = exp(H(p)) where H is Shannon entropy.
    Neff ≈ N when uniform, Neff ≈ 1 when collapsed to single expert.

    Args:
        probs: [E] - routing probabilities per expert (summed over tokens)
        min_prob: Minimum probability to avoid log(0)

    Returns:
        Effective expert count in [1, E]
    """
    probs = torch.clamp(probs, min=min_prob)
    probs = probs / probs.sum()  # Ensure normalized

    # Shannon entropy
    entropy = -(probs * torch.log(probs)).sum()

    # Convert to effective count
    neff = torch.exp(entropy).item()

    return neff


def compute_saturation(utilization: Tensor) -> float:
    """
    Maximum expert load as fraction of total.

    Saturation = max(util_e) / sum(util_e)

    High saturation = one expert hogging all tokens.

    Args:
        utilization: [E] - tokens routed to each expert

    Returns:
        Saturation in [0, 1]
    """
    total = utilization.sum()
    if total == 0:
        return 0.0

    saturation = (utilization.max() / total).item()
    return saturation


def check_survival(
    signals: CollapseSignals,
    thresholds: CollapseThresholds,
) -> tuple[bool, str]:
    """
    Check if system survived this window without collapse.

    Args:
        signals: Raw collapse signals (layer coherence, Neff, etc.)
        thresholds: Thresholds defining what counts as collapse

    Returns:
        (survived, reason):
            - survived: True if no collapse detected
            - reason: Human-readable explanation (for logging)
    """
    # Check each signal if available
    if signals.psi is not None and signals.psi < thresholds.psi_critical:
        return False, f"coherence_collapse: Psi={signals.psi:.3f} < {thresholds.psi_critical}"

    if signals.neff is not None and signals.neff < thresholds.neff_critical:
        return False, f"routing_collapse: Neff={signals.neff:.3f} < {thresholds.neff_critical}"

    if signals.saturation is not None and signals.saturation > thresholds.saturation_critical:
        return False, f"saturation_collapse: sat={signals.saturation:.3f} > {thresholds.saturation_critical}"

    if signals.mean_bimodality is not None and signals.mean_bimodality > thresholds.bimodality_critical:
        return False, f"instability_collapse: bimodality={signals.mean_bimodality:.3f} > {thresholds.bimodality_critical}"

    # No collapse detected
    return True, "survived"


def collapse_signals_from_free_energy(
    psi: float,
    utilization: Tensor,
    router_probs: Optional[Tensor] = None,
    bimodality_scores: Optional[Tensor] = None,
) -> CollapseSignals:
    """
    Convenience helper: extract collapse signals from free energy inputs.

    Args:
        psi: Layer coherence (from compute_layer_coherence)
        utilization: [E] - tokens per expert
        router_probs: [E] - routing probabilities (optional, for Neff)
        bimodality_scores: [E] - bimodality per expert (optional)

    Returns:
        CollapseSignals with all available fields populated
    """
    signals = CollapseSignals(psi=psi)

    # Compute Neff if router probs available
    if router_probs is not None:
        signals.neff = compute_neff(router_probs)

    # Compute saturation from utilization
    signals.saturation = compute_saturation(utilization)

    # Compute mean bimodality if available
    if bimodality_scores is not None:
        # Only consider active experts (utilization > 0)
        active_mask = utilization > 0
        if active_mask.any():
            signals.mean_bimodality = bimodality_scores[active_mask].mean().item()

    return signals
