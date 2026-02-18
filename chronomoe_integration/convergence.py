#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
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

from dataclasses import dataclass, field
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
        K_persistence=5,  # Production: 5 intervals (synthetic validation used 3)
        impact_threshold=0.7,  # Fixed: 70% impact for "high impact"
    )


# ---------------------------------------------------------------------------
# Three-State Convergence Detector (Phase 5-8 validated)
#
# States:
#   SETTLEMENT  — gradient collapse near zero, clean mastery
#   EQUILIBRIUM — gradient plateau at noise floor
#   DRIFT       — gradients rising or actively changing
#
# Detection method:
#   1. Slope: linear regression of grad_norms over a rolling window,
#      normalized by mean.  |slope| < threshold → plateau.
#   2. Magnitude: grad_norm / baseline.  < 1% → SETTLEMENT, else EQUILIBRIUM.
#   3. If not plateau → DRIFT.
#
# Validated:
#   - Clean → SETTLEMENT (67%, max run 1369)
#   - 1-20% noise → EQUILIBRIUM (95-100%)
#   - Domain shift → DRIFT in 0-3 steps
#   - Covariate drift → DRIFT at 3-13% contamination
#   - Window=200 confirmed default
#   - Cross-seed consistent (seeds 42, 123)
#
# All thresholds are TEST-CALIBRATED defaults for 8-expert MoE layers.
# ---------------------------------------------------------------------------


class TrainingConvergenceState(Enum):
    """Three-state convergence ontology."""
    DRIFT = "drift"
    EQUILIBRIUM = "equilibrium"
    SETTLEMENT = "settlement"


@dataclass
class TrainingConvergenceSnapshot:
    """
    Structured telemetry from a single detector update.

    Returned by ThreeStateDetector.update() for governance logging.
    """
    state: TrainingConvergenceState
    slope: Optional[float]        # Normalized gradient slope (None before window fills)
    grad_ratio: Optional[float]   # grad_norm / baseline (None before baseline calibrated)
    baseline: Optional[float]     # Calibrated baseline (P75 of warmup grad norms)
    window_size: int              # Slope window used for this classification
    step: int                     # Training step


class ThreeStateDetector:
    """
    Three-state convergence detector: SETTLEMENT / EQUILIBRIUM / DRIFT.

    Self-contained.  Caller provides (loss, grad_norm, step) on each
    training step; the detector maintains its own rolling window, calibrates
    its own baseline, and returns a TrainingConvergenceSnapshot.

    Usage::

        detector = ThreeStateDetector()

        for step in range(total_steps):
            ...
            snap = detector.update(loss=loss_val, grad_norm=grad_norm, step=step)
            if snap.state == TrainingConvergenceState.SETTLEMENT:
                save_priors()

    Parameters (all have validated defaults — do not tune without re-running
    the Phase 5-8 test suite):

        slope_window:      Rolling window length for slope regression (200).
        slope_threshold:   Max |relative_slope| for plateau detection (0.002).
        warmup_steps:      Steps reserved for baseline calibration (200).
        baseline_percentile: Percentile of warmup grad norms used as baseline (75).
        magnitude_ratio:   Threshold separating SETTLEMENT from EQUILIBRIUM (0.01).
    """

    def __init__(
        self,
        slope_window: int = 200,
        slope_threshold: float = 0.002,
        warmup_steps: int = 200,
        baseline_percentile: float = 75.0,
        magnitude_ratio: float = 0.01,
    ):
        # Config (frozen after init)
        self._slope_window = slope_window
        self._slope_threshold = slope_threshold
        self._warmup_steps = warmup_steps
        self._baseline_percentile = baseline_percentile
        self._magnitude_ratio = magnitude_ratio

        # Rolling buffers
        self._grad_norms: List[float] = []
        self._losses: List[float] = []

        # Calibrated baseline (set once at end of warmup)
        self._baseline: Optional[float] = None

        # Last classification
        self._last_state = TrainingConvergenceState.DRIFT
        self._last_step: int = -1

    # -- public API ----------------------------------------------------------

    @property
    def baseline(self) -> Optional[float]:
        """Calibrated baseline grad norm (None until warmup completes)."""
        return self._baseline

    @property
    def state(self) -> TrainingConvergenceState:
        """Most recent classification."""
        return self._last_state

    def update(
        self,
        loss: float,
        grad_norm: float,
        step: int,
    ) -> TrainingConvergenceSnapshot:
        """
        Ingest one training step and return the current convergence state.

        Args:
            loss:      Scalar training loss for this step.
            grad_norm: L2 norm of gradients for this step.
            step:      Training step index (0-based).

        Returns:
            TrainingConvergenceSnapshot with state, slope, grad_ratio,
            baseline, window_size, and step.
        """
        self._grad_norms.append(grad_norm)
        self._losses.append(loss)
        self._last_step = step

        # --- Baseline calibration (end of warmup) ---------------------------
        if (
            self._baseline is None
            and step == self._warmup_steps - 1
            and len(self._grad_norms) >= self._warmup_steps
        ):
            self._baseline = float(
                np.percentile(
                    self._grad_norms[: self._warmup_steps],
                    self._baseline_percentile,
                )
            )

        # --- Classification --------------------------------------------------
        min_history = self._warmup_steps + self._slope_window
        if step < min_history or self._baseline is None:
            # Not enough data yet — DRIFT by default
            self._last_state = TrainingConvergenceState.DRIFT
            return TrainingConvergenceSnapshot(
                state=TrainingConvergenceState.DRIFT,
                slope=None,
                grad_ratio=None,
                baseline=self._baseline,
                window_size=self._slope_window,
                step=step,
            )

        slope = self._compute_slope(self._grad_norms, self._slope_window)
        grad_ratio = grad_norm / (self._baseline + 1e-8)
        state = self._classify(slope, grad_ratio)

        self._last_state = state
        return TrainingConvergenceSnapshot(
            state=state,
            slope=slope,
            grad_ratio=grad_ratio,
            baseline=self._baseline,
            window_size=self._slope_window,
            step=step,
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _compute_slope(values: List[float], window: int) -> float:
        """
        Normalized slope of *values* over the last *window* entries.

        Returns  slope / mean(window_values).  A value of -0.01 means
        the signal is declining at ~1% per step.
        """
        y = np.array(values[-window:])
        mean_y = float(np.mean(y))
        if abs(mean_y) < 1e-10:
            return 0.0
        x = np.arange(len(y), dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope / mean_y

    def _classify(
        self,
        slope: float,
        grad_ratio: float,
    ) -> TrainingConvergenceState:
        """
        Map (slope, magnitude_ratio) → state.

        Decision tree (exactly as validated in Phase 7):
            if |slope| >= slope_threshold  →  DRIFT
            elif grad_ratio < magnitude_ratio  →  SETTLEMENT
            else  →  EQUILIBRIUM
        """
        if abs(slope) >= self._slope_threshold:
            return TrainingConvergenceState.DRIFT
        if grad_ratio < self._magnitude_ratio:
            return TrainingConvergenceState.SETTLEMENT
        return TrainingConvergenceState.EQUILIBRIUM


# ---------------------------------------------------------------------------
# Governance State Machine (conditional irreversible commitment)
#
# Wraps ThreeStateDetector and adds a FROZEN state:
#
#   DRIFT ←→ EQUILIBRIUM ←→ SETTLEMENT → (sustained N steps) → FROZEN
#                                              ↑                    |
#                                              |    (DRIFT M steps) |
#                                              +--------------------+
#
# FROZEN triggers:
#   - Save priors to disk (artifact)
#   - Freeze topology (no SPAWN / PRUNE / SPLIT)
#   - Weight updates continue
#
# FROZEN is NOT terminal.  If DRIFT is sustained for M consecutive steps
# while frozen, the system unfreezes and returns to DRIFT.
#
# N = 25 (derived from data: 10 seeds, 125 SETTLEMENT runs.
#         Max transient = 16, min terminal = 1214.
#         N=25 filters 100% of transients with 56% margin.
#         Leave-one-out cross-validated across all 10 seeds.)
#
# M = 5  (drift latency tests: 0 steps from settlement, 1-3 from
#         equilibrium.  M=5 gives ~2x margin over worst observed.)
# ---------------------------------------------------------------------------


class GovernanceState(Enum):
    """Four-state governance ontology (extends TrainingConvergenceState)."""
    DRIFT = "drift"
    EQUILIBRIUM = "equilibrium"
    SETTLEMENT = "settlement"
    FROZEN = "frozen"


@dataclass
class GovernanceEvent:
    """A governance state transition event for telemetry."""
    event_type: str       # "entered_settlement", "entered_frozen",
                          # "exited_frozen", "entered_drift", "entered_equilibrium"
    step: int
    from_state: GovernanceState
    to_state: GovernanceState
    details: Dict         # Event-specific metadata


@dataclass
class GovernanceSnapshot:
    """
    Structured telemetry from a single governance update.

    Extends TrainingConvergenceSnapshot with governance-level state.
    """
    governance_state: GovernanceState
    detector_state: TrainingConvergenceState  # Underlying detector classification
    slope: Optional[float]
    grad_ratio: Optional[float]
    baseline: Optional[float]
    window_size: int
    step: int

    # Governance counters
    settlement_streak: int    # Consecutive SETTLEMENT steps
    frozen_duration: int      # Steps in FROZEN (0 if not frozen)
    drift_streak_in_frozen: int  # Consecutive DRIFT steps while frozen

    # Lifecycle gates
    allow_spawn: bool
    allow_prune: bool
    allow_split: bool

    # Events emitted this step (usually 0 or 1)
    events: List[GovernanceEvent] = field(default_factory=list)


class GovernanceStateMachine:
    """
    Governance state machine with conditional irreversible commitment.

    Wraps ThreeStateDetector.  Adds FROZEN state entered after N
    consecutive SETTLEMENT steps.  FROZEN is exited after M consecutive
    DRIFT steps.

    Usage::

        gov = GovernanceStateMachine()

        for step in range(total_steps):
            ...
            snap = gov.update(loss=loss_val, grad_norm=grad_norm, step=step)

            if not snap.allow_spawn:
                # Topology frozen
                pass

            for event in snap.events:
                logger.info(event)

    Args:
        freeze_threshold:  N — consecutive SETTLEMENT steps to enter FROZEN.
            Default 25, derived from 10-seed clean training (see
            derive_freeze_threshold_v2.py).  Max observed transient = 16,
            N = 25 gives 56% margin.
        unfreeze_threshold:  M — consecutive DRIFT steps in FROZEN to exit.
            Default 5, derived from drift latency tests (0-3 step latency,
            M = 5 gives ~2x margin).
        **kwargs:  Forwarded to ThreeStateDetector.
    """

    def __init__(
        self,
        freeze_threshold: int = 25,
        unfreeze_threshold: int = 5,
        **detector_kwargs,
    ):
        self._detector = ThreeStateDetector(**detector_kwargs)

        # Config
        self._freeze_N = freeze_threshold
        self._unfreeze_M = unfreeze_threshold

        # Governance state
        self._gov_state = GovernanceState.DRIFT
        self._settlement_streak: int = 0
        self._frozen_since: Optional[int] = None
        self._drift_streak_in_frozen: int = 0

        # Event log (append-only)
        self._events: List[GovernanceEvent] = []

    # -- public API ----------------------------------------------------------

    @property
    def governance_state(self) -> GovernanceState:
        """Current governance state."""
        return self._gov_state

    @property
    def detector(self) -> ThreeStateDetector:
        """Underlying three-state detector (read-only access)."""
        return self._detector

    @property
    def events(self) -> List[GovernanceEvent]:
        """Full event log (append-only, newest last)."""
        return self._events

    @property
    def is_frozen(self) -> bool:
        return self._gov_state == GovernanceState.FROZEN

    def update(
        self,
        loss: float,
        grad_norm: float,
        step: int,
    ) -> GovernanceSnapshot:
        """
        Ingest one training step and return governance snapshot.

        Internally updates the ThreeStateDetector, then runs the
        governance state machine transitions.
        """
        # Step 1: underlying detector
        det_snap = self._detector.update(loss=loss, grad_norm=grad_norm, step=step)
        det_state = det_snap.state

        # Step 2: governance transitions
        step_events: List[GovernanceEvent] = []
        prev_gov = self._gov_state

        if self._gov_state == GovernanceState.FROZEN:
            # --- Currently FROZEN ---
            if det_state == TrainingConvergenceState.DRIFT:
                self._drift_streak_in_frozen += 1
                if self._drift_streak_in_frozen >= self._unfreeze_M:
                    # Unfreeze
                    frozen_duration = step - self._frozen_since if self._frozen_since is not None else 0
                    event = GovernanceEvent(
                        event_type="exited_frozen",
                        step=step,
                        from_state=GovernanceState.FROZEN,
                        to_state=GovernanceState.DRIFT,
                        details={
                            "freeze_duration": frozen_duration,
                            "break_reason": f"DRIFT sustained for {self._unfreeze_M} steps",
                            "frozen_since": self._frozen_since,
                        },
                    )
                    step_events.append(event)
                    self._events.append(event)

                    self._gov_state = GovernanceState.DRIFT
                    self._frozen_since = None
                    self._drift_streak_in_frozen = 0
                    self._settlement_streak = 0
            else:
                # Non-DRIFT resets the unfreeze counter
                self._drift_streak_in_frozen = 0

        else:
            # --- Not FROZEN: track settlement streak ---
            if det_state == TrainingConvergenceState.SETTLEMENT:
                self._settlement_streak += 1

                if self._settlement_streak >= self._freeze_N:
                    # Enter FROZEN
                    event = GovernanceEvent(
                        event_type="entered_frozen",
                        step=step,
                        from_state=prev_gov,
                        to_state=GovernanceState.FROZEN,
                        details={
                            "settlement_streak": self._settlement_streak,
                            "freeze_threshold": self._freeze_N,
                        },
                    )
                    step_events.append(event)
                    self._events.append(event)

                    self._gov_state = GovernanceState.FROZEN
                    self._frozen_since = step
                    self._drift_streak_in_frozen = 0
            else:
                # Non-SETTLEMENT resets the streak
                if self._settlement_streak > 0:
                    self._settlement_streak = 0

            # Track governance state for non-FROZEN
            if self._gov_state != GovernanceState.FROZEN:
                new_gov = GovernanceState(det_state.value)
                if new_gov != prev_gov:
                    event = GovernanceEvent(
                        event_type=f"entered_{new_gov.value}",
                        step=step,
                        from_state=prev_gov,
                        to_state=new_gov,
                        details={},
                    )
                    step_events.append(event)
                    self._events.append(event)

                self._gov_state = new_gov

        # Step 3: lifecycle gates
        frozen = self._gov_state == GovernanceState.FROZEN
        allow_spawn = not frozen
        allow_prune = not frozen
        allow_split = not frozen

        # Step 4: build snapshot
        frozen_duration = (
            step - self._frozen_since + 1
            if self._frozen_since is not None and frozen
            else 0
        )

        return GovernanceSnapshot(
            governance_state=self._gov_state,
            detector_state=det_state,
            slope=det_snap.slope,
            grad_ratio=det_snap.grad_ratio,
            baseline=det_snap.baseline,
            window_size=det_snap.window_size,
            step=step,
            settlement_streak=self._settlement_streak,
            frozen_duration=frozen_duration,
            drift_streak_in_frozen=self._drift_streak_in_frozen,
            allow_spawn=allow_spawn,
            allow_prune=allow_prune,
            allow_split=allow_split,
            events=step_events,
        )
