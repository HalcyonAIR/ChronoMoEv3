"""
Expert Registry: Fixed-width router with lifecycle management.

Phase 6 of ChronoMoEv3. Manages expert capacity with fixed-width routing,
enabling structural edits without retraining the router.

Key design:
- Router dimension fixed at max_experts (e.g., 32 per layer)
- Active mask filters which experts are routable
- Structural edits modify mask, not router weights
- Optimizer state management on add/remove/merge

This solves the critical integration blocker: router dimension changes
requiring retraining after every structural edit.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
import torch
from torch import Tensor
import torch.nn as nn


class ExpertState(Enum):
    """
    Expert lifecycle states.

    - ACTIVE: Expert is routable and participating in forward pass
    - COOLING: Expert was pruned but kept for potential reactivation
    - ARCHIVED: Expert permanently removed (slot can be reused)
    """
    ACTIVE = "active"
    COOLING = "cooling"
    ARCHIVED = "archived"


@dataclass
class ExpertInfo:
    """
    Per-expert metadata tracked by registry.

    Tracks state, utilization history, and lifecycle events.
    """
    expert_id: int
    layer_id: int
    state: ExpertState

    # Lifecycle tracking
    created_at_step: int = 0
    last_active_step: int = 0
    spawn_parent_id: Optional[int] = None  # If spawned, which parent

    # Utilization tracking (for reactivation decisions)
    utilization_ema: float = 0.0  # EMA of tokens routed to this expert
    coherence_ema: float = 0.0    # EMA of phi_slow for this expert

    # Cooling state (if COOLING)
    cooling_since_step: Optional[int] = None
    cooling_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/checkpoints."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "state": self.state.value,
            "created_at_step": self.created_at_step,
            "last_active_step": self.last_active_step,
            "spawn_parent_id": self.spawn_parent_id,
            "utilization_ema": self.utilization_ema,
            "coherence_ema": self.coherence_ema,
            "cooling_since_step": self.cooling_since_step,
            "cooling_reason": self.cooling_reason,
        }


class ExpertRegistry:
    """
    Fixed-width expert registry for one MoE layer.

    Manages expert lifecycle with fixed router capacity.

    Key operations:
    - spawn_expert: Add new expert (activate unused slot)
    - prune_expert: Deactivate expert (move to cooling)
    - merge_experts: Combine two experts into one
    - split_expert: Divide one expert into two

    Optimizer state management:
    - Registers new experts in optimizer param groups
    - Cleans up optimizer state on prune/merge
    - Handles momentum/variance buffers correctly
    """

    def __init__(
        self,
        layer_id: int,
        max_experts: int = 32,
        initial_active: int = 8,
        cooling_steps: int = 1000,
    ):
        """
        Initialize expert registry for one layer.

        Args:
            layer_id: Which layer this registry manages
            max_experts: Maximum expert capacity (router output dimension)
            initial_active: Number of experts active at initialization
            cooling_steps: Steps to keep pruned experts in cooling before archiving
        """
        self.layer_id = layer_id
        self.max_experts = max_experts
        self.cooling_steps = cooling_steps

        # Expert metadata
        self.experts: Dict[int, ExpertInfo] = {}

        # Active expert mask [max_experts] - True if routable
        self.active_mask = torch.zeros(max_experts, dtype=torch.bool)

        # Initialize first N experts as active
        for i in range(initial_active):
            self.experts[i] = ExpertInfo(
                expert_id=i,
                layer_id=layer_id,
                state=ExpertState.ACTIVE,
                created_at_step=0,
            )
            self.active_mask[i] = True

        # Next expert ID to allocate
        self.next_expert_id = initial_active

        # Lifecycle event log
        self.events: List[Dict[str, Any]] = []

    @property
    def num_active(self) -> int:
        """Number of currently active experts."""
        return self.active_mask.sum().item()

    @property
    def num_cooling(self) -> int:
        """Number of experts in cooling state."""
        return sum(1 for e in self.experts.values() if e.state == ExpertState.COOLING)

    @property
    def num_archived(self) -> int:
        """Number of archived experts."""
        return sum(1 for e in self.experts.values() if e.state == ExpertState.ARCHIVED)

    @property
    def capacity_remaining(self) -> int:
        """Number of unused expert slots."""
        return self.max_experts - self.num_active

    @property
    def active_expert_ids(self) -> List[int]:
        """List of active expert IDs."""
        return [
            expert_id
            for expert_id, info in self.experts.items()
            if info.state == ExpertState.ACTIVE
        ]

    def get_active_mask(self) -> Tensor:
        """
        Get boolean mask for active experts.

        Returns:
            mask: [max_experts] - True for active experts
        """
        return self.active_mask.clone()

    def get_expert_info(self, expert_id: int) -> Optional[ExpertInfo]:
        """Get metadata for an expert."""
        return self.experts.get(expert_id)

    def spawn_expert(
        self,
        step: int,
        parent_id: int,
        expert_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        param_group_idx: int = 0,
    ) -> int:
        """
        Spawn a new expert by activating an unused slot.

        Args:
            step: Current training step
            parent_id: Parent expert ID (for cloning)
            expert_module: The new expert's nn.Module
            optimizer: Training optimizer (to register new params)
            param_group_idx: Which optimizer param group to add to

        Returns:
            new_expert_id: ID of spawned expert

        Raises:
            RuntimeError: If no capacity remaining
        """
        if self.capacity_remaining == 0:
            raise RuntimeError(
                f"Cannot spawn: layer {self.layer_id} at max capacity "
                f"({self.max_experts} experts)"
            )

        # Allocate new expert ID
        new_expert_id = self.next_expert_id
        self.next_expert_id += 1

        # Create expert info
        self.experts[new_expert_id] = ExpertInfo(
            expert_id=new_expert_id,
            layer_id=self.layer_id,
            state=ExpertState.ACTIVE,
            created_at_step=step,
            last_active_step=step,
            spawn_parent_id=parent_id,
        )

        # Activate in mask
        self.active_mask[new_expert_id] = True

        # Register parameters in optimizer
        # Add all parameters from the expert module to the optimizer's param group
        for param in expert_module.parameters():
            optimizer.param_groups[param_group_idx]['params'].append(param)

        # Log event
        self.events.append({
            "step": step,
            "event": "SPAWN",
            "expert_id": new_expert_id,
            "parent_id": parent_id,
            "num_active": self.num_active,
        })

        return new_expert_id

    def prune_expert(
        self,
        step: int,
        expert_id: int,
        reason: str,
        optimizer: torch.optim.Optimizer,
        expert_module: nn.Module,
    ) -> bool:
        """
        Prune an expert (move to cooling state).

        Deactivates expert and cleans up optimizer state.

        Args:
            step: Current training step
            expert_id: Which expert to prune
            reason: Why pruning (for forensics)
            optimizer: Training optimizer (to clean up state)
            expert_module: The expert's nn.Module (to remove params)

        Returns:
            True if pruned, False if expert not found or not active
        """
        info = self.experts.get(expert_id)
        if info is None or info.state != ExpertState.ACTIVE:
            return False

        # Deactivate in mask
        self.active_mask[expert_id] = False

        # Move to cooling
        info.state = ExpertState.COOLING
        info.cooling_since_step = step
        info.cooling_reason = reason

        # Remove parameters from optimizer
        # Clean up optimizer state for this expert's parameters
        expert_params = list(expert_module.parameters())

        for group in optimizer.param_groups:
            # Filter out expert params (use identity comparison)
            group['params'] = [
                p for p in group['params']
                if not any(p is ep for ep in expert_params)
            ]

        # Clean up optimizer state dict (momentum, variance, etc.)
        for param in expert_params:
            if param in optimizer.state:
                del optimizer.state[param]

        # Log event
        self.events.append({
            "step": step,
            "event": "PRUNE",
            "expert_id": expert_id,
            "reason": reason,
            "num_active": self.num_active,
        })

        return True

    def merge_experts(
        self,
        step: int,
        source_a_id: int,
        source_b_id: int,
        merged_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        source_a_module: nn.Module,
        source_b_module: nn.Module,
        keep_expert_id: Optional[int] = None,
        optimizer_state_strategy: str = "reset",
    ) -> Optional[int]:
        """
        Merge two experts into one.

        DESTRUCTIVE operation. Removes source_b, keeps source_a (or keep_expert_id).

        Args:
            step: Current training step
            source_a_id: First expert to merge
            source_b_id: Second expert to merge
            merged_module: The merged expert's nn.Module
            optimizer: Training optimizer
            source_a_module: Source A's nn.Module (for state cleanup)
            source_b_module: Source B's nn.Module (for state cleanup)
            keep_expert_id: Which ID to keep (default: source_a_id)
            optimizer_state_strategy: How to handle optimizer state
                - "reset": Zero momentum/variance for merged expert
                - "keep_a": Keep source_a optimizer state
                - "keep_b": Keep source_b optimizer state
                - "average": Average momentum/variance (experimental)

        Returns:
            merged_expert_id if successful, None if failed
        """
        info_a = self.experts.get(source_a_id)
        info_b = self.experts.get(source_b_id)

        if info_a is None or info_b is None:
            return None
        if info_a.state != ExpertState.ACTIVE or info_b.state != ExpertState.ACTIVE:
            return None

        # Determine which expert ID to keep
        merged_id = keep_expert_id if keep_expert_id is not None else source_a_id
        removed_id = source_b_id if merged_id == source_a_id else source_a_id

        # Update kept expert's info
        info_merged = info_a if merged_id == source_a_id else info_b
        info_merged.last_active_step = step

        # Remove the other expert
        info_removed = info_b if removed_id == source_b_id else info_a
        info_removed.state = ExpertState.ARCHIVED
        self.active_mask[removed_id] = False

        # Handle optimizer state
        if optimizer_state_strategy == "reset":
            # Remove old states, let optimizer reinitialize on next step
            source_a_params = list(source_a_module.parameters())
            source_b_params = list(source_b_module.parameters())

            for param in source_a_params:
                if param in optimizer.state:
                    del optimizer.state[param]
            for param in source_b_params:
                if param in optimizer.state:
                    del optimizer.state[param]

            # Merged module params will be initialized fresh on next optimizer step

        elif optimizer_state_strategy == "keep_a" and merged_id == source_a_id:
            # Keep source_a state, remove source_b state
            source_b_params = list(source_b_module.parameters())
            for param in source_b_params:
                if param in optimizer.state:
                    del optimizer.state[param]

        elif optimizer_state_strategy == "keep_b" and merged_id == source_b_id:
            # Keep source_b state, remove source_a state
            source_a_params = list(source_a_module.parameters())
            for param in source_a_params:
                if param in optimizer.state:
                    del optimizer.state[param]

        # Note: "average" strategy would require deep knowledge of optimizer internals
        # (Adam has exp_avg, exp_avg_sq; SGD might have momentum_buffer, etc.)
        # Defer to future work - for now, reset is safest

        # Log event
        self.events.append({
            "step": step,
            "event": "MERGE",
            "source_a_id": source_a_id,
            "source_b_id": source_b_id,
            "merged_id": merged_id,
            "removed_id": removed_id,
            "optimizer_strategy": optimizer_state_strategy,
            "num_active": self.num_active,
        })

        return merged_id

    def archive_cooling_experts(self, step: int) -> List[int]:
        """
        Archive experts that have been cooling long enough.

        Called periodically to clean up cooling experts.

        Args:
            step: Current training step

        Returns:
            List of expert IDs that were archived
        """
        archived = []

        for expert_id, info in self.experts.items():
            if info.state == ExpertState.COOLING:
                if info.cooling_since_step is not None:
                    cooling_duration = step - info.cooling_since_step
                    if cooling_duration >= self.cooling_steps:
                        info.state = ExpertState.ARCHIVED
                        archived.append(expert_id)

        if archived:
            self.events.append({
                "step": step,
                "event": "ARCHIVE",
                "expert_ids": archived,
                "num_archived": len(archived),
            })

        return archived

    def update_utilization(
        self,
        step: int,
        expert_id: int,
        utilization: float,
        coherence: float,
        alpha: float = 0.95,
    ):
        """
        Update expert's utilization and coherence EMAs.

        Called after each forward pass to track expert health.

        Args:
            step: Current training step
            expert_id: Which expert
            utilization: Current token count for this expert
            coherence: Current phi_slow for this expert
            alpha: EMA decay factor
        """
        info = self.experts.get(expert_id)
        if info is None:
            return

        info.last_active_step = step
        info.utilization_ema = alpha * info.utilization_ema + (1 - alpha) * utilization
        info.coherence_ema = alpha * info.coherence_ema + (1 - alpha) * coherence

    def to_dict(self) -> Dict[str, Any]:
        """
        Export registry state for checkpointing.

        Returns:
            Dictionary containing full registry state
        """
        return {
            "layer_id": self.layer_id,
            "max_experts": self.max_experts,
            "next_expert_id": self.next_expert_id,
            "active_mask": self.active_mask.tolist(),
            "experts": {
                expert_id: info.to_dict()
                for expert_id, info in self.experts.items()
            },
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "ExpertRegistry":
        """
        Restore registry from checkpoint.

        Args:
            state: Dictionary from to_dict()

        Returns:
            Restored ExpertRegistry
        """
        registry = cls(
            layer_id=state["layer_id"],
            max_experts=state["max_experts"],
            initial_active=0,  # Will be restored from experts dict
        )

        registry.next_expert_id = state["next_expert_id"]
        registry.active_mask = torch.tensor(state["active_mask"], dtype=torch.bool)

        # Restore experts
        for expert_id_str, expert_dict in state["experts"].items():
            expert_id = int(expert_id_str)
            registry.experts[expert_id] = ExpertInfo(
                expert_id=expert_dict["expert_id"],
                layer_id=expert_dict["layer_id"],
                state=ExpertState(expert_dict["state"]),
                created_at_step=expert_dict["created_at_step"],
                last_active_step=expert_dict["last_active_step"],
                spawn_parent_id=expert_dict["spawn_parent_id"],
                utilization_ema=expert_dict["utilization_ema"],
                coherence_ema=expert_dict["coherence_ema"],
                cooling_since_step=expert_dict["cooling_since_step"],
                cooling_reason=expert_dict["cooling_reason"],
            )

        registry.events = state["events"]

        return registry

    def status_summary(self) -> str:
        """Human-readable status summary."""
        return (
            f"ExpertRegistry(layer={self.layer_id}, "
            f"active={self.num_active}/{self.max_experts}, "
            f"cooling={self.num_cooling}, archived={self.num_archived})"
        )
