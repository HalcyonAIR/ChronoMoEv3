"""
Stress bands: autonomic regulation wrapped around constitutional core.

F_l is not an objective to minimize. It's a physiological signal.

Three learned bands:
- Comfort: normal operation, irreversibles allowed with evidence + calmness
- Strain: behavior changes, identity doesn't. Irreversible thresholds GO UP.
- Panic: preservation mode. Zero irreversibles.

Key principle: Pressure tunes behavior. Calm commits identity.

Stress is allowed to exist. It's not allowed to rewrite identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any
import math


Band = Literal["comfort", "strain", "panic"]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class EMA:
    value: float = 0.0
    alpha: float = 0.01
    initialized: bool = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x
        return self.value


@dataclass
class StressBandsConfig:
    # Initial ceilings. Learned over time.
    comfort_ceiling_init: float = 1.0
    strain_ceiling_init: float = 2.0

    # Hysteresis prevents thrash: entering a worse band is easier than exiting it.
    hysteresis_margin: float = 0.05  # fraction of ceiling

    # Time constants (steps) for calmness gating.
    scar_calm_steps: int = 200
    crystallize_calm_steps: int = 2000
    edit_calm_steps: int = 500

    # Default target distribution. Used gently as a long-run attractor.
    target_p_comfort: float = 0.80
    target_p_strain: float = 0.15
    target_p_panic: float = 0.05

    # Learning rates for boundary drift.
    lr_widen: float = 0.0005
    lr_narrow: float = 0.0005
    lr_dist: float = 0.001

    # Safety bounds so learning can't explode.
    comfort_min: float = 0.05
    comfort_max: float = 1e6
    strain_min_ratio: float = 1.05  # strain_ceiling >= comfort_ceiling * ratio
    strain_max: float = 1e6

    # Optional smoothing of the incoming stress sensor.
    f_ema_alpha: float = 0.05


@dataclass
class StressBandsState:
    comfort_ceiling: float
    strain_ceiling: float

    f_ema: EMA = field(default_factory=lambda: EMA(alpha=0.05))

    current_f: float = 0.0
    current_band: Band = "comfort"

    time_in_comfort: int = 0
    time_in_strain: int = 0
    time_in_panic: int = 0

    # "Survival" EMAs: did we operate without collapse while in that band?
    comfort_survival: EMA = field(default_factory=lambda: EMA(alpha=0.01))
    strain_survival: EMA = field(default_factory=lambda: EMA(alpha=0.01))
    panic_survival: EMA = field(default_factory=lambda: EMA(alpha=0.01))

    # For distribution learning
    total_steps: int = 0

    def band_counts(self) -> Dict[Band, int]:
        return {
            "comfort": self.time_in_comfort,
            "strain": self.time_in_strain,
            "panic": self.time_in_panic,
        }


def init_stress_bands(cfg: StressBandsConfig) -> StressBandsState:
    s = StressBandsState(
        comfort_ceiling=cfg.comfort_ceiling_init,
        strain_ceiling=max(cfg.strain_ceiling_init, cfg.comfort_ceiling_init * cfg.strain_min_ratio),
    )
    s.f_ema.alpha = cfg.f_ema_alpha
    return s


def classify_band(f: float, comfort: float, strain: float, prev: Band, h: float) -> Band:
    """
    Hysteresis rule:
    - Moving to a worse band happens at the raw thresholds.
    - Moving to a better band requires clearing a slightly lower threshold.
    """
    # thresholds to enter worse states
    enter_strain = comfort
    enter_panic = strain

    # thresholds to exit worse states
    exit_strain = comfort * (1.0 - h)
    exit_panic = strain * (1.0 - h)

    if prev == "panic":
        if f < exit_panic:
            # can improve to strain or comfort depending on comfort threshold
            return "comfort" if f < exit_strain else "strain"
        return "panic"

    if prev == "strain":
        if f >= enter_panic:
            return "panic"
        if f < exit_strain:
            return "comfort"
        return "strain"

    # prev == comfort
    if f >= enter_panic:
        return "panic"
    if f >= enter_strain:
        return "strain"
    return "comfort"


@dataclass
class IrreversibleGates:
    allow_scar: bool
    allow_crystallize: bool
    allow_structural_edits: bool
    reason: str


def irreversible_gates(state: StressBandsState, cfg: StressBandsConfig) -> IrreversibleGates:
    """
    Double gate: evidence gate happens elsewhere.
    This is the calmness gate only.
    """
    if state.current_band == "panic":
        return IrreversibleGates(False, False, False, "panic: preservation mode")

    if state.current_band == "strain":
        # During strain: you may still form scars (reactive), but only if you have recent calm history.
        # Crystallisation and structural edits are frozen.
        allow_scar = state.time_in_comfort >= cfg.scar_calm_steps
        return IrreversibleGates(allow_scar, False, False, "strain: freeze crystallise/edits, scar needs calm credit")

    # comfort band
    allow_scar = state.time_in_comfort >= cfg.scar_calm_steps
    allow_crys = state.time_in_comfort >= cfg.crystallize_calm_steps
    allow_edits = state.time_in_comfort >= cfg.edit_calm_steps
    return IrreversibleGates(allow_scar, allow_crys, allow_edits, "comfort: calm gates satisfied by time_in_comfort")


def update_boundaries(
    state: StressBandsState,
    cfg: StressBandsConfig,
    survived: bool,
) -> None:
    """
    Learn ceilings slowly. Two forces:
    - Local survival: if you survive comfortably, you can widen tolerance slightly.
    - Target distribution: nudge ceilings so most time is comfort, panic rare.
    """
    # Update survival EMAs for the current band.
    x = 1.0 if survived else 0.0
    if state.current_band == "comfort":
        state.comfort_survival.update(x)
    elif state.current_band == "strain":
        state.strain_survival.update(x)
    else:
        state.panic_survival.update(x)

    # Local rule: widen if surviving, narrow if failing, but only in the band you're in.
    if state.current_band == "comfort":
        if survived:
            state.comfort_ceiling *= (1.0 + cfg.lr_widen)
        else:
            state.comfort_ceiling *= (1.0 - cfg.lr_narrow)

    elif state.current_band == "strain":
        if survived:
            state.strain_ceiling *= (1.0 + cfg.lr_widen)
        else:
            state.strain_ceiling *= (1.0 - cfg.lr_narrow)

    else:  # panic
        # In panic, "survived" means we didn't collapse further; allow slight widening of strain ceiling.
        if survived:
            state.strain_ceiling *= (1.0 + cfg.lr_widen)
        else:
            state.strain_ceiling *= (1.0 - cfg.lr_narrow)

    # Distribution rule: match long-run band occupancy ratios gently.
    total = max(1, state.total_steps)
    p_comfort = state.time_in_comfort / total
    p_panic = state.time_in_panic / total

    # If panic too frequent, lower strain ceiling so we enter panic earlier? No. That's trauma logic.
    # Instead: if panic too frequent, *raise* strain ceiling so we get more room before panic,
    # but rely on irreversible freezes to prevent damage while strained.
    # If comfort too rare, widen comfort band (raise comfort_ceiling).
    # If comfort too frequent and strain never seen, narrow comfort slightly so the system learns.
    comfort_err = cfg.target_p_comfort - p_comfort
    panic_err = cfg.target_p_panic - p_panic

    state.comfort_ceiling *= (1.0 + cfg.lr_dist * comfort_err)
    state.strain_ceiling *= (1.0 - cfg.lr_dist * panic_err)  # if panic too high => panic_err negative => strain_ceiling up

    # Enforce bounds and ordering
    state.comfort_ceiling = clamp(state.comfort_ceiling, cfg.comfort_min, cfg.comfort_max)
    min_strain = max(cfg.strain_min_ratio * state.comfort_ceiling, cfg.comfort_min * cfg.strain_min_ratio)
    state.strain_ceiling = clamp(state.strain_ceiling, min_strain, cfg.strain_max)


@dataclass
class StressStepResult:
    f_raw: float
    f_used: float
    band_prev: Band
    band_now: Band
    gates: IrreversibleGates
    ceilings: Dict[str, float]
    debug: Dict[str, Any]


def step_stress_bands(
    state: StressBandsState,
    cfg: StressBandsConfig,
    f_l: float,
    survived: bool,
) -> StressStepResult:
    """
    One update per step (or per window if you operate in windows).
    f_l is the stress sensor, not an objective.
    survived is your collapse signal for this step/window.
    """
    band_prev = state.current_band

    # Smooth stress signal
    f_used = state.f_ema.update(float(f_l))
    state.current_f = f_used

    # Band transition with hysteresis
    state.current_band = classify_band(
        f_used,
        comfort=state.comfort_ceiling,
        strain=state.strain_ceiling,
        prev=state.current_band,
        h=cfg.hysteresis_margin,
    )

    # Update counters
    state.total_steps += 1
    if state.current_band == "comfort":
        state.time_in_comfort += 1
    elif state.current_band == "strain":
        state.time_in_strain += 1
    else:
        state.time_in_panic += 1

    gates = irreversible_gates(state, cfg)

    # Learn boundaries after updating band/time, so learning reflects current mode.
    update_boundaries(state, cfg, survived=survived)

    return StressStepResult(
        f_raw=float(f_l),
        f_used=float(f_used),
        band_prev=band_prev,
        band_now=state.current_band,
        gates=gates,
        ceilings={"comfort": state.comfort_ceiling, "strain": state.strain_ceiling},
        debug={
            "time_in_comfort": state.time_in_comfort,
            "time_in_strain": state.time_in_strain,
            "time_in_panic": state.time_in_panic,
            "comfort_survival": state.comfort_survival.value,
            "strain_survival": state.strain_survival.value,
            "panic_survival": state.panic_survival.value,
        },
    )
