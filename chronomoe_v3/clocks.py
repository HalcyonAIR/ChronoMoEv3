"""
Three-timescale clock definitions.

Each clock is the same state variable with a different retention rate.
Fast trails ~10 steps, mid trails ~100 steps, slow trails ~1000 steps.

The clocks don't "remember" — they just haven't let go yet.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ClockConfig:
    """
    Configuration for one clock's exponential moving average.

    Half-life determines how long information persists:
        alpha = exp(-ln(2) / half_life)

    For example:
        - half_life=10 → alpha≈0.933 (fast: immediate continuity)
        - half_life=100 → alpha≈0.993 (mid: conversational context)
        - half_life=1000 → alpha≈0.9993 (slow: structural trajectory)
    """

    name: str
    alpha: Optional[float] = None  # Decay constant (0 < alpha < 1)
    half_life: Optional[int] = None  # Steps until half decay

    def __post_init__(self):
        """Compute alpha from half_life or vice versa."""
        if self.alpha is None and self.half_life is None:
            raise ValueError("Must specify either alpha or half_life")

        if self.alpha is None:
            # Compute from half_life: alpha = exp(-ln(2) / half_life)
            self.alpha = math.exp(-math.log(2) / self.half_life)
        elif self.half_life is None:
            # Compute from alpha: half_life = -ln(2) / ln(alpha)
            self.half_life = int(-math.log(2) / math.log(self.alpha))

        # Validate
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")

    @property
    def retention(self) -> float:
        """Retention constant (alias for alpha)."""
        return self.alpha

    @property
    def decay_rate(self) -> float:
        """Decay rate per step: 1 - alpha."""
        return 1 - self.alpha

    def ema_update(self, old_value: float, new_value: float) -> float:
        """
        Apply exponential moving average update.

        new_state = alpha * old_state + (1 - alpha) * new_measurement
        """
        return self.alpha * old_value + (1 - self.alpha) * new_value

    def __repr__(self) -> str:
        return (
            f"ClockConfig(name={self.name}, "
            f"alpha={self.alpha:.4f}, "
            f"half_life={self.half_life})"
        )


class ThreeClockEMA:
    """
    Three exponential moving averages on the same signal.

    This is the core temporal structure of ChronoMoEv3:
        - Fast clock: What's happening now (~10 steps)
        - Mid clock: What's been happening (~100 steps)
        - Slow clock: What has persisted (~1000 steps)

    Usage:
        clocks = ThreeClockEMA()
        clocks.update(phi_raw)
        print(clocks.fast, clocks.mid, clocks.slow)
    """

    def __init__(
        self,
        alpha_fast: float = 0.9,
        alpha_mid: float = 0.99,
        alpha_slow: float = 0.999,
    ):
        """
        Initialize three clocks with default retention rates.

        Args:
            alpha_fast: Fast clock decay (default: 0.9, half-life ~7 steps)
            alpha_mid: Mid clock decay (default: 0.99, half-life ~69 steps)
            alpha_slow: Slow clock decay (default: 0.999, half-life ~693 steps)
        """
        self.config_fast = ClockConfig(name="fast", alpha=alpha_fast)
        self.config_mid = ClockConfig(name="mid", alpha=alpha_mid)
        self.config_slow = ClockConfig(name="slow", alpha=alpha_slow)

        # State
        self.fast = 0.0
        self.mid = 0.0
        self.slow = 0.0

    def update(self, new_value: float) -> None:
        """
        Update all three clocks with new measurement.

        Args:
            new_value: New measurement (e.g., phi_raw)
        """
        self.fast = self.config_fast.ema_update(self.fast, new_value)
        self.mid = self.config_mid.ema_update(self.mid, new_value)
        self.slow = self.config_slow.ema_update(self.slow, new_value)

    @property
    def delta_fast_slow(self) -> float:
        """
        Fast - Slow delta.

        Negative value indicates degradation in progress.
        """
        return self.fast - self.slow

    @property
    def delta_mid_slow(self) -> float:
        """
        Mid - Slow delta.

        Sustained negative value at mid timescale warrants lens intervention.
        """
        return self.mid - self.slow

    def reset(self, value: float = 0.0) -> None:
        """Reset all clocks to a value."""
        self.fast = self.mid = self.slow = value

    def __repr__(self) -> str:
        return (
            f"ThreeClockEMA("
            f"fast={self.fast:.3f}, "
            f"mid={self.mid:.3f}, "
            f"slow={self.slow:.3f}, "
            f"Δ={self.delta_fast_slow:.3f})"
        )


# Default clock configurations
DEFAULT_FAST_CLOCK = ClockConfig(name="fast", half_life=10)
DEFAULT_MID_CLOCK = ClockConfig(name="mid", half_life=100)
DEFAULT_SLOW_CLOCK = ClockConfig(name="slow", half_life=1000)


def compute_alpha_from_half_life(half_life: int) -> float:
    """
    Compute EMA alpha from desired half-life in steps.

    Args:
        half_life: Number of steps for signal to decay to 50%

    Returns:
        alpha: Retention constant for EMA update
    """
    return math.exp(-math.log(2) / half_life)


def compute_half_life_from_alpha(alpha: float) -> int:
    """
    Compute half-life from EMA alpha.

    Args:
        alpha: Retention constant (0 < alpha < 1)

    Returns:
        half_life: Number of steps for 50% decay
    """
    return int(-math.log(2) / math.log(alpha))
