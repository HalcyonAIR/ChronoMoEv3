"""
Governor: sits ABOVE compound gate. Governs commit authorization.

Deliberately boring. The contribution is not cleverness. It's that
the rule exists, is measurable, and sits above routing.

Minimal deterministic rule set:
- COMMIT requires: fast < Tf AND medium < Tm AND debt < Td AND not in scar
- If any fail: BLOCK -> substitute single exploratory step
- If hard violation active: ESCALATE regardless

Guardrail: minimum commit rate prevents governor from "winning" by never
committing. If authorized commits in last W steps < min_commits, thresholds
are temporarily relaxed.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from bob_core.ledgers import BobCore, CostSignal
from bob_core.medium_clock import MediumClock
from bob_core.motifs import GateResult


class GovernorDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


@dataclass
class GovernorVerdict:
    """Full decision with reasons — every decision logged."""
    decision: GovernorDecision
    reasons: List[str]
    cost_signal: CostSignal
    debt_level: float
    medium_activation: float
    forced_exploration: bool = False
    thresholds_relaxed: bool = False


class BobGovernor:
    """Sits ABOVE compound gate. Governs commit authorization.

    The gate says "cheap path is structurally viable."
    The governor says "AND I authorize this commit."

    ALLOW: gate passed AND governor conditions met
    BLOCK: gate passed BUT governor says no -> single forced exploration
    ESCALATE: hard violation detected

    Guardrail: if authorized_commits in last commit_rate_window steps
    < min_commits_per_window, temporarily relax thresholds by +0.1 each
    until commits resume.
    """

    def __init__(
        self,
        bob_core: BobCore,
        medium_clock: MediumClock,
        fast_threshold: float = 0.8,
        medium_threshold: float = 0.5,
        debt_threshold: float = 0.7,
        commit_rate_window: int = 200,
        min_commits_per_window: int = 10,
        relaxation_amount: float = 0.1,
    ):
        self.bob_core = bob_core
        self.medium_clock = medium_clock
        self.fast_threshold = fast_threshold
        self.medium_threshold = medium_threshold
        self.debt_threshold = debt_threshold
        self._force_exploration_next: bool = False
        self._decisions: List[GovernorVerdict] = []

        # Minimum commit rate guardrail
        self.commit_rate_window = commit_rate_window
        self.min_commits_per_window = min_commits_per_window
        self.relaxation_amount = relaxation_amount
        # Track recent decisions in a sliding window
        self._recent_allows: deque = deque(maxlen=commit_rate_window)

    def _effective_thresholds(self) -> Tuple[float, float, float, bool]:
        """Return (medium_thresh, debt_thresh, scar_thresh, relaxed).

        If authorized commits are below minimum, relax thresholds.
        """
        recent_allows = sum(self._recent_allows)
        if (len(self._recent_allows) >= self.commit_rate_window
                and recent_allows < self.min_commits_per_window):
            # Starving for commits — relax
            return (
                self.medium_threshold + self.relaxation_amount,
                self.debt_threshold + self.relaxation_amount,
                0.3 + self.relaxation_amount,  # scar neighborhood threshold
                True,
            )
        return (self.medium_threshold, self.debt_threshold, 0.3, False)

    def evaluate_commit(
        self,
        context_class: int,
        expert_ids: Tuple[int, ...],
        gate_result: GateResult,
        step: int,
    ) -> GovernorVerdict:
        """Should Bob authorize this commit (cheap path)?

        Called only when gate_result.passed is True.
        If BLOCK: sets forced exploration for next step.
        """
        routing_region = tuple(sorted(expert_ids))
        reasons: List[str] = []

        cost_signal = self.bob_core.costs.get_signal()
        debt = self.bob_core.get_debt_for_region(routing_region, step)
        medium_act = self.medium_clock.activation

        medium_thresh, debt_thresh, scar_thresh, relaxed = self._effective_thresholds()

        # Check scar neighborhood (with effective threshold)
        in_scar = self.bob_core.scars.is_in_scar_neighborhood(
            routing_region, step, threshold=scar_thresh
        )
        if in_scar:
            reasons.append(f"scar_neighborhood: debt={debt:.3f}")

        # Check medium clock (with effective threshold)
        if medium_act > medium_thresh:
            reasons.append(f"medium_unstable: {medium_act:.3f} > {medium_thresh:.2f}")

        # Check debt (with effective threshold)
        if debt > debt_thresh:
            reasons.append(f"debt_high: {debt:.3f} > {debt_thresh:.2f}")

        # Check escalation rate
        if cost_signal.escalation_rate > 0.5:
            reasons.append(f"escalation_rate_high: {cost_signal.escalation_rate:.3f}")

        if reasons:
            # At least one condition failed
            if len(reasons) >= 3 or debt > 0.9:
                decision = GovernorDecision.ESCALATE
            else:
                decision = GovernorDecision.BLOCK
                self._force_exploration_next = True
        else:
            decision = GovernorDecision.ALLOW

        # Track for commit rate guardrail
        self._recent_allows.append(1 if decision == GovernorDecision.ALLOW else 0)

        verdict = GovernorVerdict(
            decision=decision,
            reasons=reasons,
            cost_signal=cost_signal,
            debt_level=debt,
            medium_activation=medium_act,
            forced_exploration=(decision == GovernorDecision.BLOCK),
            thresholds_relaxed=relaxed,
        )
        self._decisions.append(verdict)
        return verdict

    def consume_forced_exploration(self) -> bool:
        """If prior BLOCK set forced exploration, return True and clear.

        Single-step, not N-step escalation.
        """
        if self._force_exploration_next:
            self._force_exploration_next = False
            return True
        return False

    @property
    def decisions(self) -> List[GovernorVerdict]:
        return self._decisions

    @property
    def blocks_count(self) -> int:
        return sum(
            1 for d in self._decisions
            if d.decision in (GovernorDecision.BLOCK, GovernorDecision.ESCALATE)
        )

    @property
    def allows_count(self) -> int:
        return sum(
            1 for d in self._decisions
            if d.decision == GovernorDecision.ALLOW
        )
