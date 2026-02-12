#!/usr/bin/env python3
"""
Convergence Detection: Operational Definition

Convergence is a detectable system state, not a feeling.

State-level convergence:
  A sustained condition where:
  1. Bob and Maniac agree on high-impact claims
  2. No successful downgrades for N consecutive evaluations
  3. Routing diversity below regime-specific threshold

Convergence event:
  Transition from non-converged → converged state

Regime-aware:
  Diversity thresholds vary by deformation regime (exploration/transition/saturation)

Persistence-required:
  Must hold for K intervals to avoid transient alignment spikes
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np


class DeformationRegime(Enum):
    """
    Deformation regime based on total scar strength.

    From Phase 1 monotonicity validation:
    - Exploration: 0.0-2.0 (0-25% contraction, shallow slope)
    - Transition: 2.0-4.0 (25-85% contraction, steep slope)
    - Saturation: 4.0+ (85-96% contraction, flat slope)
    """
    EXPLORATION = "exploration"  # scar < 2.0
    TRANSITION = "transition"    # 2.0 <= scar < 4.0
    SATURATION = "saturation"    # scar >= 4.0


@dataclass
class ConvergenceThresholds:
    """
    Operational thresholds for convergence detection.

    IMPORTANT: These should be derived from baseline distributions,
    not intuition. Use percentiles from natural behavior.
    """
    # Agreement window (consecutive high-impact agreements)
    W_min: int = 10

    # Downgrade silence (cycles since last successful downgrade)
    S_min: int = 100

    # Diversity thresholds (regime-specific)
    D_exploration: float = 0.3   # Strict (low diversity suspicious in exploration)
    D_transition: float = 0.5    # Moderate (low diversity expected)
    D_saturation: float = 0.7    # Lenient (diversity already collapsed)

    # Persistence requirement (intervals convergence must hold)
    K_persistence: int = 5

    # High-impact claim threshold (what qualifies as "high impact")
    impact_threshold: float = 0.7


@dataclass
class ConvergenceState:
    """
    Current convergence state of the system.

    State, not event. This is the sustained condition.
    """
    is_converged: bool

    # Evidence
    agreement_window: int       # Current consecutive agreements
    downgrade_silence: int      # Cycles since last downgrade
    diversity_current: float    # Current motif diversity
    diversity_normalized: float # Normalized by regime baseline

    # Context
    regime: DeformationRegime
    scar_strength_total: float

    # Persistence
    persistence_count: int      # How many intervals has this state held?

    # Timestamps
    step: int
    converged_since: Optional[int] = None  # Step when convergence began


@dataclass
class ConvergenceEvent:
    """
    Convergence event: Transition from non-converged → converged.

    Event, not state. This is the moment of transition.
    """
    step: int
    regime: DeformationRegime
    scar_strength: float

    # Evidence at transition
    agreement_window: int
    downgrade_silence: int
    diversity: float

    # Context
    domain_volatility: Optional[float] = None
    performance_stability: Optional[float] = None


@dataclass
class DivergenceEvent:
    """
    Divergence event: Successful downgrade after convergence.

    Indicates convergence was premature.
    """
    step: int
    converged_duration: int  # How long was it converged before divergence?

    # State at divergence
    scar_strength: float
    diversity: float
    domain_volatility: Optional[float] = None


def compute_deformation_regime(scar_strength_total: float) -> DeformationRegime:
    """
    Determine deformation regime from total scar strength.

    From Phase 1 validation:
    - Exploration: 0.0-2.0
    - Transition: 2.0-4.0
    - Saturation: 4.0+

    Args:
        scar_strength_total: Sum of all active scar penalties

    Returns:
        DeformationRegime
    """
    if scar_strength_total < 2.0:
        return DeformationRegime.EXPLORATION
    elif scar_strength_total < 4.0:
        return DeformationRegime.TRANSITION
    else:
        return DeformationRegime.SATURATION


def get_diversity_threshold(regime: DeformationRegime, thresholds: ConvergenceThresholds) -> float:
    """
    Get regime-specific diversity threshold.

    Exploration: Strict (low diversity is suspicious)
    Transition: Moderate (low diversity expected but not decisive)
    Saturation: Lenient (diversity already collapsed by design)
    """
    if regime == DeformationRegime.EXPLORATION:
        return thresholds.D_exploration
    elif regime == DeformationRegime.TRANSITION:
        return thresholds.D_transition
    else:  # SATURATION
        return thresholds.D_saturation


def check_convergence_conditions(
    agreement_window: int,
    downgrade_silence: int,
    diversity_normalized: float,
    regime: DeformationRegime,
    thresholds: ConvergenceThresholds,
) -> bool:
    """
    Check if convergence conditions hold.

    All three must be true:
    1. W >= W_min (sustained agreement)
    2. S >= S_min (no recent downgrades)
    3. D <= D_threshold(regime) (low diversity for regime)

    Args:
        agreement_window: Consecutive high-impact agreements
        downgrade_silence: Cycles since last successful downgrade
        diversity_normalized: Motif diversity normalized by regime baseline
        regime: Current deformation regime
        thresholds: Convergence thresholds

    Returns:
        True if all three conditions met
    """
    # Condition 1: Agreement window
    agreement_met = agreement_window >= thresholds.W_min

    # Condition 2: Downgrade silence
    silence_met = downgrade_silence >= thresholds.S_min

    # Condition 3: Diversity (regime-specific threshold)
    diversity_threshold = get_diversity_threshold(regime, thresholds)
    diversity_met = diversity_normalized <= diversity_threshold

    # All three must hold
    return agreement_met and silence_met and diversity_met


class ConvergenceDetector:
    """
    Convergence detector with regime awareness and persistence requirements.

    Usage:
        detector = ConvergenceDetector(thresholds)

        for step in training:
            state = detector.update(
                step=step,
                bob_agreed_with_maniac=True/False,
                claim_impact=0.8,
                maniac_downgraded=False,
                diversity_current=0.25,
                scar_strength_total=3.5,
            )

            if state.is_converged:
                print(f"Converged at step {state.converged_since}")
    """

    def __init__(
        self,
        thresholds: ConvergenceThresholds,
        baseline_diversity: Dict[DeformationRegime, float],
    ):
        """
        Initialize convergence detector.

        Args:
            thresholds: Convergence thresholds (operational definition)
            baseline_diversity: Baseline motif diversity per regime (for normalization)
        """
        self.thresholds = thresholds
        self.baseline_diversity = baseline_diversity

        # Current state tracking
        self.agreement_window = 0
        self.downgrade_silence = 0
        self.last_downgrade_step = -1

        # Convergence state
        self.converged = False
        self.converged_since = None
        self.persistence_count = 0

        # Event log
        self.convergence_events: List[ConvergenceEvent] = []
        self.divergence_events: List[DivergenceEvent] = []

    def update(
        self,
        step: int,
        bob_agreed_with_maniac: bool,
        claim_impact: float,
        maniac_downgraded: bool,
        diversity_current: float,
        scar_strength_total: float,
    ) -> ConvergenceState:
        """
        Update convergence detector with new observation.

        Args:
            step: Current training step
            bob_agreed_with_maniac: Did Bob and maniac agree on this claim?
            claim_impact: Impact score of claim (0.0-1.0)
            maniac_downgraded: Did maniac successfully downgrade Bob's claim?
            diversity_current: Current motif diversity
            scar_strength_total: Total scar strength (sum of penalties)

        Returns:
            ConvergenceState (current state after update)
        """
        # Determine regime
        regime = compute_deformation_regime(scar_strength_total)

        # Update agreement window
        if bob_agreed_with_maniac and claim_impact >= self.thresholds.impact_threshold:
            self.agreement_window += 1
        else:
            self.agreement_window = 0  # Reset on disagreement

        # Update downgrade silence
        if maniac_downgraded:
            # Divergence event (if currently converged)
            if self.converged and self.converged_since is not None:
                divergence = DivergenceEvent(
                    step=step,
                    converged_duration=step - self.converged_since,
                    scar_strength=scar_strength_total,
                    diversity=diversity_current,
                )
                self.divergence_events.append(divergence)

            self.downgrade_silence = 0
            self.last_downgrade_step = step
        else:
            self.downgrade_silence = step - self.last_downgrade_step if self.last_downgrade_step >= 0 else step

        # Normalize diversity by regime baseline
        baseline = self.baseline_diversity.get(regime, 1.0)
        diversity_normalized = diversity_current / baseline if baseline > 0 else 0.0

        # Check convergence conditions
        conditions_met = check_convergence_conditions(
            self.agreement_window,
            self.downgrade_silence,
            diversity_normalized,
            regime,
            self.thresholds,
        )

        # Update persistence
        if conditions_met:
            self.persistence_count += 1
        else:
            self.persistence_count = 0

        # State transition (with persistence requirement)
        was_converged = self.converged

        if not was_converged and self.persistence_count >= self.thresholds.K_persistence:
            # Transition: non-converged → converged
            self.converged = True
            self.converged_since = step - self.thresholds.K_persistence  # When persistence started

            # Log convergence event
            event = ConvergenceEvent(
                step=self.converged_since,
                regime=regime,
                scar_strength=scar_strength_total,
                agreement_window=self.agreement_window,
                downgrade_silence=self.downgrade_silence,
                diversity=diversity_current,
            )
            self.convergence_events.append(event)

        elif was_converged and not conditions_met:
            # Transition: converged → non-converged (lost persistence)
            self.converged = False
            self.converged_since = None

        # Build current state
        state = ConvergenceState(
            is_converged=self.converged,
            agreement_window=self.agreement_window,
            downgrade_silence=self.downgrade_silence,
            diversity_current=diversity_current,
            diversity_normalized=diversity_normalized,
            regime=regime,
            scar_strength_total=scar_strength_total,
            persistence_count=self.persistence_count,
            step=step,
            converged_since=self.converged_since,
        )

        return state

    def get_convergence_statistics(self) -> Dict:
        """
        Get statistics about convergence events.

        Returns:
            Dict with convergence/divergence counts, durations, regime distribution
        """
        if not self.convergence_events:
            return {
                "num_convergence_events": 0,
                "num_divergence_events": len(self.divergence_events),
                "convergence_by_regime": {
                    "exploration": 0,
                    "transition": 0,
                    "saturation": 0,
                },
                "scar_strength_at_convergence": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "divergence_durations": {
                    "mean": 0.0,
                    "median": 0.0,
                },
            }

        # Convergence by regime
        regime_counts = {
            DeformationRegime.EXPLORATION: 0,
            DeformationRegime.TRANSITION: 0,
            DeformationRegime.SATURATION: 0,
        }

        for event in self.convergence_events:
            regime_counts[event.regime] += 1

        # Scar strength at convergence
        scar_strengths = [e.scar_strength for e in self.convergence_events]

        # Divergence durations
        divergence_durations = [e.converged_duration for e in self.divergence_events]

        return {
            "num_convergence_events": len(self.convergence_events),
            "num_divergence_events": len(self.divergence_events),
            "convergence_by_regime": {
                "exploration": regime_counts[DeformationRegime.EXPLORATION],
                "transition": regime_counts[DeformationRegime.TRANSITION],
                "saturation": regime_counts[DeformationRegime.SATURATION],
            },
            "scar_strength_at_convergence": {
                "mean": np.mean(scar_strengths) if scar_strengths else 0.0,
                "std": np.std(scar_strengths) if scar_strengths else 0.0,
                "min": min(scar_strengths) if scar_strengths else 0.0,
                "max": max(scar_strengths) if scar_strengths else 0.0,
            },
            "divergence_durations": {
                "mean": np.mean(divergence_durations) if divergence_durations else 0.0,
                "median": np.median(divergence_durations) if divergence_durations else 0.0,
            },
        }


def derive_thresholds_from_baseline(
    baseline_data: Dict,
    percentile: float = 90.0,
) -> ConvergenceThresholds:
    """
    Derive convergence thresholds from baseline distributions.

    DO NOT use intuition. Use percentiles from natural behavior.

    Args:
        baseline_data: Dict with baseline distributions:
            - agreement_lengths: List[int] (agreement window lengths)
            - silence_periods: List[int] (downgrade silence durations)
            - diversity_by_regime: Dict[regime, List[float]]
        percentile: Percentile to use for threshold (default 90th)

    Returns:
        ConvergenceThresholds derived from data
    """
    agreement_lengths = baseline_data.get("agreement_lengths", [10])
    silence_periods = baseline_data.get("silence_periods", [100])
    diversity_by_regime = baseline_data.get("diversity_by_regime", {})

    # W_min: 90th percentile of natural agreement lengths
    W_min = int(np.percentile(agreement_lengths, percentile))

    # S_min: 90th percentile of natural silence periods
    S_min = int(np.percentile(silence_periods, percentile))

    # D_threshold: 10th percentile (low diversity) per regime
    D_exploration = np.percentile(
        diversity_by_regime.get("exploration", [0.3]),
        100 - percentile  # Low end of distribution
    )
    D_transition = np.percentile(
        diversity_by_regime.get("transition", [0.5]),
        100 - percentile
    )
    D_saturation = np.percentile(
        diversity_by_regime.get("saturation", [0.7]),
        100 - percentile
    )

    return ConvergenceThresholds(
        W_min=W_min,
        S_min=S_min,
        D_exploration=float(D_exploration),
        D_transition=float(D_transition),
        D_saturation=float(D_saturation),
        K_persistence=5,  # Fixed: require 5 intervals
        impact_threshold=0.7,  # Fixed: 70% impact for "high impact"
    )
