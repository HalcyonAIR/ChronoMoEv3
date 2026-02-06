"""
ChronoMoEv3 configuration.

Single source of truth for all hyperparameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChronoConfig:
    """
    ChronoMoEv3 configuration.

    Controls three clocks, slow bias, free energy weights, and lifecycle policies.
    """

    # ============================================================================
    # Clock decay constants
    # ============================================================================

    alpha_fast: float = 0.9  # Half-life ~7 steps
    alpha_mid: float = 0.99  # Half-life ~69 steps
    alpha_slow: float = 0.999  # Half-life ~693 steps

    # Role vector EMA (what expert typically outputs)
    alpha_role: float = 0.99

    # ============================================================================
    # Slow bias (locus mechanism)
    # ============================================================================

    # Slow influence bias on router logits
    eta_beta: float = 0.01  # Learning rate for beta updates
    tau: float = 0.5  # Coherence threshold for beta growth
    beta_min: float = -1.0  # Minimum bias
    beta_max: float = 1.0  # Maximum bias

    # ============================================================================
    # Free energy weights (slow clock objective)
    # ============================================================================

    lambda_complexity: float = 0.01  # Complexity tax per expert
    rho_redundancy: float = 0.1  # Redundancy penalty
    kappa_instability: float = 0.1  # Instability penalty (bimodality)

    # ============================================================================
    # Edit selection (lifecycle)
    # ============================================================================

    min_improvement: float = 0.01  # Minimum delta-F to justify an edit
    max_edits_per_cycle: int = 1  # Maximum structural changes per slow eval
    cooldown_steps: int = 500  # Minimum steps between edits in same layer

    # ============================================================================
    # Expert bounds
    # ============================================================================

    min_experts_per_layer: int = 2
    max_experts_per_layer: int = 16

    # ============================================================================
    # Spawning
    # ============================================================================

    spawn_strategy: str = "clone_and_perturb"  # or "random_init"
    split_perturbation_scale: float = 0.01  # Noise scale for split

    # ============================================================================
    # Bimodality detector
    # ============================================================================

    alpha_centroid: float = 0.95  # EMA rate for centroid updates
    bimodality_threshold: float = 0.5  # Separation * balance threshold for split

    # ============================================================================
    # Timing
    # ============================================================================

    cooling_period: int = 100  # Steps before new expert eligible for slow eval
    lifecycle_eval_interval: int = 100  # Steps between slow clock evaluations

    # ============================================================================
    # Basin tracking (for interpretability and merge detection)
    # ============================================================================

    basin_window_size: int = 50  # History length for basin tracking

    # ============================================================================
    # Coherence computation
    # ============================================================================

    # Whether to weight coherence by routing mass (vs uniform)
    weight_coherence_by_utilization: bool = True

    # Minimum tokens for coherence measurement (skip if expert saw fewer)
    min_tokens_for_coherence: int = 1

    # ============================================================================
    # Logging
    # ============================================================================

    log_decisions: bool = True  # Log lifecycle decisions to JSONL
    log_interval: int = 100  # Steps between coherence state logs

    def validate(self) -> None:
        """Validate configuration constraints."""
        # Clock ordering
        assert (
            self.alpha_fast < self.alpha_mid < self.alpha_slow
        ), "Clocks must be ordered: fast < mid < slow"

        # Bounds
        assert 0 < self.alpha_fast < 1, "alpha_fast must be in (0,1)"
        assert 0 < self.alpha_mid < 1, "alpha_mid must be in (0,1)"
        assert 0 < self.alpha_slow < 1, "alpha_slow must be in (0,1)"

        assert self.beta_min < self.beta_max, "beta_min must be < beta_max"

        assert (
            self.min_experts_per_layer >= 1
        ), "Must have at least 1 expert per layer"
        assert (
            self.max_experts_per_layer >= self.min_experts_per_layer
        ), "max_experts must be >= min_experts"

        assert self.min_improvement > 0, "min_improvement must be positive"
        assert (
            self.max_edits_per_cycle >= 0
        ), "max_edits_per_cycle must be non-negative"

    def __post_init__(self):
        """Validate after initialization."""
        self.validate()
