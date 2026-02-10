"""
Expert Registry for lifecycle management.

Ported from ChronoMoEv3. Tracks expert states, probation, and active mask.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional
import torch


class ExpertState(Enum):
    """Expert lifecycle states."""
    ACTIVE = "active"
    PROBATION = "probation"
    COOLING = "cooling"
    ARCHIVED = "archived"


@dataclass
class ProbationConfig:
    """Probation mechanism configuration."""
    enabled: bool = True
    duration_steps: int = 30  # Validated parameter from nanoMoE tuning
    min_tokens: int = 1500    # Validated parameter from nanoMoE tuning
    initial_boost: float = 1.0  # Validated parameter from nanoMoE tuning
    decay_type: str = "linear"
    share_cap: float = 0.05  # Not yet enforced
    require_comfort_band: bool = True

    # Spawn strategy enforcement
    allow_clone_spawn: bool = True  # If False, only blank spawning allowed
    warn_on_clone: bool = True      # Log warning when clone strategy used

    @staticmethod
    def default() -> 'ProbationConfig':
        """Default probation config (balanced)."""
        return ProbationConfig(
            enabled=True,
            duration_steps=30,
            min_tokens=1500,
            initial_boost=1.0,
            decay_type="linear",
        )

    @staticmethod
    def proof_of_failure() -> 'ProbationConfig':
        """
        Probation config that demonstrates failures can occur.

        Validated to produce ~2 failures out of ~13 spawns in nanoMoE.
        Use this preset to verify probation is not a guaranteed graduation.
        """
        return ProbationConfig(
            enabled=True,
            duration_steps=30,
            min_tokens=1500,
            initial_boost=1.0,
            decay_type="linear",
        )

    @staticmethod
    def generous() -> 'ProbationConfig':
        """Generous probation (most experts graduate)."""
        return ProbationConfig(
            enabled=True,
            duration_steps=50,
            min_tokens=1000,
            initial_boost=2.0,
            decay_type="linear",
        )

    @staticmethod
    def blank_only() -> 'ProbationConfig':
        """
        Enforce blank-only spawning (no clone allowed).

        Use this to ensure all spawns use random initialization + probation,
        preventing any clone-seeded spawning.
        """
        return ProbationConfig(
            enabled=True,
            duration_steps=30,
            min_tokens=1500,
            initial_boost=1.0,
            decay_type="linear",
            allow_clone_spawn=False,  # Enforce blank-only
            warn_on_clone=True,
        )


@dataclass
class ExpertInfo:
    """Per-expert metadata."""
    expert_id: int
    state: ExpertState
    spawned_step: int
    parent_id: Optional[int] = None
    strategy: str = "blank"

    # Probation tracking
    probation_started_step: Optional[int] = None
    probation_tokens_accumulated: int = 0

    # Performance tracking
    total_tokens_processed: int = 0
    utilization_history: List[float] = field(default_factory=list)


class ExpertRegistry:
    """
    Manages expert lifecycle states and probation.

    Fixed-width design: max_experts slots pre-allocated.
    Only active experts are selectable (via active_mask).
    """

    def __init__(
        self,
        layer_id: int,
        max_experts: int,
        initial_active: int,
        probation_config: Optional[ProbationConfig] = None,
    ):
        self.layer_id = layer_id
        self.max_experts = max_experts
        self.next_expert_id = initial_active  # Next available slot

        self.probation_config = probation_config or ProbationConfig()

        # Active mask: True for selectable experts
        self.active_mask = torch.zeros(max_experts, dtype=torch.bool)
        self.active_mask[:initial_active] = True

        # Expert metadata
        self.experts: Dict[int, ExpertInfo] = {}
        for i in range(initial_active):
            self.experts[i] = ExpertInfo(
                expert_id=i,
                state=ExpertState.ACTIVE,
                spawned_step=0,
            )

    @property
    def num_active(self) -> int:
        """Count of currently active experts (ACTIVE or PROBATION)."""
        return int(self.active_mask.sum().item())

    @property
    def capacity_remaining(self) -> int:
        """Remaining spawn capacity (based on next_expert_id, not num_active)."""
        return self.max_experts - self.next_expert_id

    def status_summary(self) -> str:
        """Human-readable status."""
        states = [info.state.value for info in self.experts.values()]
        state_counts = {s: states.count(s) for s in set(states)}
        return f"Layer {self.layer_id}: {self.num_active}/{self.max_experts} active, {state_counts}"

    def register_expert(
        self,
        expert_id: int,
        parent_id: Optional[int],
        strategy: str,
        current_step: int,
    ) -> None:
        """Register a newly spawned expert in PROBATION state."""
        assert expert_id < self.max_experts, f"Expert ID {expert_id} exceeds max {self.max_experts}"
        assert expert_id == self.next_expert_id, \
            f"Must spawn sequentially: expected {self.next_expert_id}, got {expert_id}"

        self.experts[expert_id] = ExpertInfo(
            expert_id=expert_id,
            state=ExpertState.PROBATION,
            spawned_step=current_step,
            parent_id=parent_id,
            strategy=strategy,
            probation_started_step=current_step,
        )

        # Activate in mask
        self.active_mask[expert_id] = True
        self.next_expert_id += 1

    def get_probation_boost(self, expert_id: int, current_step: int) -> float:
        """
        Get probation boost for expert at current step.

        Returns decaying boost value (starts at initial_boost, decays to 0).
        """
        if not self.probation_config.enabled:
            return 0.0

        info = self.experts.get(expert_id)
        if not info or info.state != ExpertState.PROBATION:
            return 0.0

        steps_in_probation = current_step - info.probation_started_step
        duration_steps = self.probation_config.duration_steps

        if steps_in_probation >= duration_steps:
            return 0.0

        if self.probation_config.decay_type == "linear":
            decay_factor = max(0.0, 1.0 - steps_in_probation / duration_steps)
        else:
            decay_factor = 1.0  # Constant boost (no decay)

        return self.probation_config.initial_boost * decay_factor

    def update_probation_tokens(self, expert_id: int, num_tokens: int) -> None:
        """Update token count for expert in probation."""
        info = self.experts.get(expert_id)
        if info and info.state == ExpertState.PROBATION:
            info.probation_tokens_accumulated += num_tokens

    def check_probation_status(
        self,
        current_step: int,
        in_comfort_band: bool = True,
    ) -> List[Tuple[int, str]]:
        """
        Check all probation experts for graduation/failure.

        Returns list of (expert_id, status) where status is "graduate" or "fail".
        """
        results = []

        for expert_id, info in self.experts.items():
            if info.state != ExpertState.PROBATION:
                continue

            steps_in_probation = current_step - info.probation_started_step
            duration_complete = steps_in_probation >= self.probation_config.duration_steps

            if not duration_complete:
                continue

            # Check graduation criteria
            tokens_ok = info.probation_tokens_accumulated >= self.probation_config.min_tokens
            comfort_ok = (not self.probation_config.require_comfort_band) or in_comfort_band

            if tokens_ok and comfort_ok:
                results.append((expert_id, "graduate"))
            else:
                results.append((expert_id, "fail"))

        return results

    def graduate_from_probation(self, expert_id: int, current_step: int) -> None:
        """Graduate expert from probation to active state."""
        info = self.experts.get(expert_id)
        if info and info.state == ExpertState.PROBATION:
            info.state = ExpertState.ACTIVE

    def prune_expert(self, expert_id: int) -> None:
        """
        Prune expert (deactivate in mask, mark as ARCHIVED).

        Does NOT physically remove from ModuleList (fixed-width design).
        """
        info = self.experts.get(expert_id)
        if info:
            info.state = ExpertState.ARCHIVED
            self.active_mask[expert_id] = False
