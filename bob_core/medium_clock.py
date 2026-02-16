"""
Medium clock: stateful filter that outputs instability signals.

Medium clock is the stabilizer across a run — not across a single step
and not across the whole lifetime. It detects instability (oscillation,
dithering, repeated undoing), not just failure.

Medium clock is dynamics, not distance. Distance is slow clock (identity pressure).
Dynamics is stability pressure.

Explicitly separate file — not buried in governor.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MediumClockState:
    """Small vector updated each step."""
    churn_ema: float = 0.0         # EMA of routing-change magnitude (Jaccard distance)
    flipflop_ema: float = 0.0      # EMA of cheap/full alternation
    outcome_var_ema: float = 0.0   # EMA of |loss_change| between same-class steps
    escalation_ema: float = 0.0    # EMA of governor blocks
    provisional_active: bool = False
    provisional_ttl: int = 0


class MediumClock:
    """Stateful filter that outputs instability signals.

    Detects oscillation, dithering, repeated undoing across a run.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        instability_threshold: float = 0.5,
        churn_weight: float = 0.3,
        flipflop_weight: float = 0.3,
        outcome_weight: float = 0.2,
        escalation_weight: float = 0.2,
    ):
        self.ema_alpha = ema_alpha
        self.instability_threshold = instability_threshold
        self._weights = (churn_weight, flipflop_weight, outcome_weight, escalation_weight)
        self._state = MediumClockState()
        self._last_path: Optional[str] = None

    def tick(
        self,
        prev_expert_ids: Optional[Tuple[int, ...]],
        curr_expert_ids: Tuple[int, ...],
        prev_loss: Optional[float],
        curr_loss: float,
        was_blocked: bool,
        path: str,
    ) -> MediumClockState:
        """Update all EMAs from this step's observations.

        churn_ema: Jaccard distance between prev/curr expert sets
        flipflop_ema: 1.0 if path alternated from last step, else 0.0
        outcome_var_ema: |curr_loss - prev_loss|
        escalation_ema: 1.0 if was_blocked, else 0.0
        """
        alpha = self.ema_alpha

        # Churn: Jaccard distance between expert sets
        if prev_expert_ids is not None and curr_expert_ids:
            prev_set = set(prev_expert_ids)
            curr_set = set(curr_expert_ids)
            union = prev_set | curr_set
            intersection = prev_set & curr_set
            churn = 1.0 - (len(intersection) / len(union)) if union else 0.0
        else:
            churn = 0.0

        # Flipflop: did the path alternate?
        if self._last_path is not None and path != self._last_path:
            flipflop = 1.0
        else:
            flipflop = 0.0
        self._last_path = path

        # Outcome variance: |loss change|
        if prev_loss is not None:
            outcome_var = abs(curr_loss - prev_loss)
        else:
            outcome_var = 0.0

        # Escalation: governor blocked
        escalation = 1.0 if was_blocked else 0.0

        # Update EMAs
        self._state.churn_ema = (1 - alpha) * self._state.churn_ema + alpha * churn
        self._state.flipflop_ema = (1 - alpha) * self._state.flipflop_ema + alpha * flipflop
        self._state.outcome_var_ema = (1 - alpha) * self._state.outcome_var_ema + alpha * outcome_var
        self._state.escalation_ema = (1 - alpha) * self._state.escalation_ema + alpha * escalation

        # Tick provisional
        if self._state.provisional_active:
            self._state.provisional_ttl -= 1
            if self._state.provisional_ttl <= 0:
                self._state.provisional_active = False

        return self._state

    @property
    def activation(self) -> float:
        """0-1 instability signal. Weighted sum of EMAs, clipped."""
        w = self._weights
        raw = (
            w[0] * self._state.churn_ema
            + w[1] * self._state.flipflop_ema
            + w[2] * self._state.outcome_var_ema
            + w[3] * self._state.escalation_ema
        )
        return max(0.0, min(1.0, raw))

    @property
    def state(self) -> MediumClockState:
        return self._state

    def propose_provisional(self, ttl: int = 20) -> None:
        """Start provisional commitment monitoring."""
        self._state.provisional_active = True
        self._state.provisional_ttl = ttl

    def check_provisional(self, loss_trend: float, debt_slope: float) -> str:
        """Returns 'promote', 'kill', or 'continue'.

        loss_trend: negative = improving, positive = degrading
        debt_slope: positive = debt growing, negative = healing
        """
        if not self._state.provisional_active:
            return "continue"

        if self._state.provisional_ttl <= 0:
            # TTL expired — promote if stable, kill if not
            if loss_trend <= 0 and debt_slope <= 0:
                return "promote"
            return "kill"

        # Still active — check for early kill
        if self.activation > self.instability_threshold * 1.5:
            return "kill"

        return "continue"
