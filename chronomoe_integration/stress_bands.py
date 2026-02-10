"""
Stress bands for lifecycle calm gating.

Ported from ChronoMoEv3/chronomoe_v3/stress_bands.py.

Three bands based on free energy (loss or other stress metric):
- COMFORT: Normal operation, all lifecycle operations allowed (with calm credit)
- STRAIN: Elevated stress, spawn/graduate allowed, prune frozen
- PANIC: Preservation mode, all lifecycle operations frozen

Key principle: Pressure tunes behavior. Calm commits identity.
"""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any
from enum import Enum


class Band(str, Enum):
    """Stress band classification."""
    COMFORT = "comfort"
    STRAIN = "strain"
    PANIC = "panic"


@dataclass
class EMA:
    """Exponential moving average."""
    value: float = 0.0
    alpha: float = 0.01
    initialized: bool = False

    def update(self, x: float) -> float:
        """Update EMA with new value."""
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x
        return self.value


@dataclass
class StressBandsConfig:
    """
    Configuration for stress bands.

    Stress is measured by a metric (typically loss or free energy).
    Bands are defined by ceilings that can adapt over time.
    """
    # Initial thresholds (can be learned/adapted)
    comfort_ceiling_init: float = 1.0
    strain_ceiling_init: float = 2.0

    # Hysteresis prevents thrashing between bands
    hysteresis_margin: float = 0.05  # 5% of ceiling

    # Calm credit requirements (steps in comfort before allowing irreversibles)
    spawn_calm_steps: int = 200       # Minimum calm before spawn allowed
    prune_calm_steps: int = 500       # Minimum calm before prune allowed
    graduate_calm_steps: int = 200    # Minimum calm before probation graduation

    # EMA smoothing of stress signal
    stress_ema_alpha: float = 0.05

    # Safety bounds for adaptive ceilings (if implemented)
    comfort_min: float = 0.05
    comfort_max: float = 1e6
    strain_min: float = 1.0
    strain_max: float = 1e6


@dataclass
class StressBandsState:
    """
    Runtime state for stress bands.

    Tracks current band, stress level, and calm credit.
    """
    # Adaptive ceilings (initialized from config, can learn)
    comfort_ceiling: float
    strain_ceiling: float

    # Smoothed stress signal
    stress_ema: EMA = field(default_factory=lambda: EMA(alpha=0.05))

    # Current state
    current_stress: float = 0.0
    current_band: Band = Band.COMFORT

    # Time tracking (for calm credit)
    time_in_comfort: int = 0
    time_in_strain: int = 0
    time_in_panic: int = 0
    total_steps: int = 0

    def band_counts(self) -> Dict[str, int]:
        """Get time spent in each band."""
        return {
            "comfort": self.time_in_comfort,
            "strain": self.time_in_strain,
            "panic": self.time_in_panic,
        }


@dataclass
class LifecycleGates:
    """
    Calm gates for lifecycle operations.

    Based on stress band and calm credit (time in comfort).
    """
    allow_spawn: bool
    allow_prune: bool
    allow_graduate: bool
    reason: str


def init_stress_bands(config: StressBandsConfig) -> StressBandsState:
    """Initialize stress bands state from config."""
    state = StressBandsState(
        comfort_ceiling=config.comfort_ceiling_init,
        strain_ceiling=config.strain_ceiling_init,
    )
    state.stress_ema.alpha = config.stress_ema_alpha
    return state


def classify_band(
    stress: float,
    comfort_ceiling: float,
    strain_ceiling: float,
    prev_band: Band,
    hysteresis: float,
) -> Band:
    """
    Classify stress level into band with hysteresis.

    Hysteresis rule:
    - Entering a worse band: happens at raw threshold
    - Exiting a worse band: requires threshold - hysteresis margin

    This prevents thrashing when stress hovers near boundaries.
    """
    # Thresholds to enter worse states
    enter_strain = comfort_ceiling
    enter_panic = strain_ceiling

    # Thresholds to exit worse states (with hysteresis)
    exit_strain = comfort_ceiling * (1.0 - hysteresis)
    exit_panic = strain_ceiling * (1.0 - hysteresis)

    if prev_band == Band.PANIC:
        if stress < exit_panic:
            # Can improve to strain or comfort
            return Band.COMFORT if stress < exit_strain else Band.STRAIN
        return Band.PANIC

    if prev_band == Band.STRAIN:
        if stress >= enter_panic:
            return Band.PANIC
        if stress < exit_strain:
            return Band.COMFORT
        return Band.STRAIN

    # prev_band == COMFORT
    if stress >= enter_panic:
        return Band.PANIC
    if stress >= enter_strain:
        return Band.STRAIN
    return Band.COMFORT


def lifecycle_gates(
    state: StressBandsState,
    config: StressBandsConfig,
) -> LifecycleGates:
    """
    Determine which lifecycle operations are allowed.

    Rules:
    - PANIC: All lifecycle operations frozen (preservation mode)
    - STRAIN: Spawn/graduate allowed (with calm), prune frozen
    - COMFORT: All operations allowed (with calm credit)

    Calm credit = time_in_comfort must exceed threshold for each operation.
    """
    if state.current_band == Band.PANIC:
        return LifecycleGates(
            allow_spawn=False,
            allow_prune=False,
            allow_graduate=False,
            reason="panic: preservation mode, all lifecycle frozen",
        )

    if state.current_band == Band.STRAIN:
        # During strain: can spawn/graduate if calm credit sufficient
        # Prune is frozen (don't remove capacity under pressure)
        allow_spawn = state.time_in_comfort >= config.spawn_calm_steps
        allow_graduate = state.time_in_comfort >= config.graduate_calm_steps

        return LifecycleGates(
            allow_spawn=allow_spawn,
            allow_prune=False,
            allow_graduate=allow_graduate,
            reason=f"strain: prune frozen, spawn/graduate need calm ({state.time_in_comfort} steps)",
        )

    # COMFORT band: all operations allowed if calm credit sufficient
    allow_spawn = state.time_in_comfort >= config.spawn_calm_steps
    allow_prune = state.time_in_comfort >= config.prune_calm_steps
    allow_graduate = state.time_in_comfort >= config.graduate_calm_steps

    return LifecycleGates(
        allow_spawn=allow_spawn,
        allow_prune=allow_prune,
        allow_graduate=allow_graduate,
        reason=f"comfort: calm credit {state.time_in_comfort} steps",
    )


@dataclass
class StressStepResult:
    """Result of stress bands update."""
    stress_raw: float
    stress_smoothed: float
    band_prev: Band
    band_now: Band
    gates: LifecycleGates
    band_counts: Dict[str, int]


def step_stress_bands(
    state: StressBandsState,
    config: StressBandsConfig,
    stress: float,
) -> StressStepResult:
    """
    Update stress bands for one step.

    Args:
        state: Current stress bands state (modified in-place)
        config: Stress bands configuration
        stress: Current stress metric (e.g., loss, free energy)

    Returns:
        StressStepResult with band classification and gates
    """
    band_prev = state.current_band

    # Smooth stress signal with EMA
    stress_smoothed = state.stress_ema.update(float(stress))
    state.current_stress = stress_smoothed

    # Classify into band with hysteresis
    state.current_band = classify_band(
        stress=stress_smoothed,
        comfort_ceiling=state.comfort_ceiling,
        strain_ceiling=state.strain_ceiling,
        prev_band=state.current_band,
        hysteresis=config.hysteresis_margin,
    )

    # Update time tracking
    state.total_steps += 1
    if state.current_band == Band.COMFORT:
        state.time_in_comfort += 1
    else:
        state.time_in_comfort = 0  # Reset calm credit when leaving comfort

    if state.current_band == Band.STRAIN:
        state.time_in_strain += 1
    if state.current_band == Band.PANIC:
        state.time_in_panic += 1

    # Compute lifecycle gates
    gates = lifecycle_gates(state, config)

    return StressStepResult(
        stress_raw=stress,
        stress_smoothed=stress_smoothed,
        band_prev=band_prev,
        band_now=state.current_band,
        gates=gates,
        band_counts=state.band_counts(),
    )
