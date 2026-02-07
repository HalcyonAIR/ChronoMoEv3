"""
GPU-resident coherence buffer for fast updates.

Coherence state stays on GPU and updates every step. Snapshots to CPU
CoherenceState only on eval intervals (e.g., every 100 steps) to avoid
synchronization bottlenecks.
"""

from typing import Dict, Optional

import torch
from torch import Tensor

from .coherence import CoherenceState


class CoherenceBuffer:
    """
    GPU-resident coherence state for one layer.

    Updates happen entirely on GPU. CPU snapshots are created only when
    explicitly requested (e.g., for lifecycle evaluation every 100 steps).

    This eliminates the CPU sync bottleneck from the training loop.
    """

    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        d_model: int,
        device: str = "cuda",
        alpha_fast: float = 0.9,
        alpha_mid: float = 0.99,
        alpha_slow: float = 0.999,
    ):
        """
        Initialize GPU-resident coherence buffer.

        Args:
            layer_id: Layer index
            num_experts: Number of experts in this layer
            d_model: Hidden dimension
            device: Device to place tensors on
            alpha_fast: Fast clock decay rate (~10 steps)
            alpha_mid: Mid clock decay rate (~100 steps)
            alpha_slow: Slow clock decay rate (~1000 steps)
        """
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device

        # Three-timescale EMAs (on GPU)
        self.phi_fast = torch.zeros(num_experts, device=device)
        self.phi_mid = torch.zeros(num_experts, device=device)
        self.phi_slow = torch.zeros(num_experts, device=device)

        # Observation tracking (on GPU)
        self.total_tokens_seen = torch.zeros(
            num_experts, dtype=torch.long, device=device
        )
        self.last_update_step = torch.zeros(
            num_experts, dtype=torch.long, device=device
        )

        # Clock config
        self.alpha_fast = alpha_fast
        self.alpha_mid = alpha_mid
        self.alpha_slow = alpha_slow

    def update(
        self,
        phi_raw: Tensor,
        active_expert_ids: Tensor,
        step: int,
        num_tokens: Tensor,
    ):
        """
        Update coherence EMAs on GPU.

        Args:
            phi_raw: [num_active_experts] - raw coherence this step
            active_expert_ids: [num_active_experts] - which experts (indices)
            step: current training step
            num_tokens: [num_active_experts] - tokens per expert
        """
        # Three-clock EMA update (all on GPU)
        self.phi_fast[active_expert_ids] = (
            self.alpha_fast * self.phi_fast[active_expert_ids]
            + (1 - self.alpha_fast) * phi_raw
        )

        self.phi_mid[active_expert_ids] = (
            self.alpha_mid * self.phi_mid[active_expert_ids]
            + (1 - self.alpha_mid) * phi_raw
        )

        self.phi_slow[active_expert_ids] = (
            self.alpha_slow * self.phi_slow[active_expert_ids]
            + (1 - self.alpha_slow) * phi_raw
        )

        # Observation tracking
        self.total_tokens_seen[active_expert_ids] += num_tokens
        self.last_update_step[active_expert_ids] = step

    def snapshot(self, step: int, current_step: Optional[int] = None) -> Dict[int, CoherenceState]:
        """
        Sync to CPU and export CoherenceState snapshot.

        Called every N steps for lifecycle evaluation. This is the only
        time we pay the GPU→CPU transfer cost.

        Args:
            step: Step at which snapshot was taken (for metadata)
            current_step: Current training step (for is_being_observed check)

        Returns:
            Dict mapping expert_id (int) to CoherenceState
        """
        if current_step is None:
            current_step = step

        states = {}
        for expert_idx in range(self.num_experts):
            expert_id = expert_idx  # Use int as key for GPU buffer

            state = CoherenceState(
                expert_id=f"L{self.layer_id}_E{expert_idx}",
                layer_id=self.layer_id,
                d_model=self.d_model,
                phi_fast=self.phi_fast[expert_idx].item(),
                phi_mid=self.phi_mid[expert_idx].item(),
                phi_slow=self.phi_slow[expert_idx].item(),
                last_update_step=self.last_update_step[expert_idx].item(),
                total_tokens_seen=self.total_tokens_seen[expert_idx].item(),
            )

            states[expert_id] = state

        return states

    def get_phi_slow(self) -> Tensor:
        """
        Get slow coherence for all experts (stays on GPU).

        Useful for beta update without full CPU snapshot.

        Returns:
            phi_slow: [num_experts] on GPU
        """
        return self.phi_slow

    def get_phi_delta(self) -> Tensor:
        """
        Get phi_delta (fast - slow) for all experts (stays on GPU).

        Useful for degradation detection without CPU snapshot.

        Returns:
            phi_delta: [num_experts] on GPU
        """
        return self.phi_fast - self.phi_slow

    def reset_expert(self, expert_idx: int):
        """
        Reset coherence state for one expert.

        Used after lifecycle events (prune/spawn/split/merge).

        Args:
            expert_idx: Expert index to reset
        """
        self.phi_fast[expert_idx] = 0.0
        self.phi_mid[expert_idx] = 0.0
        self.phi_slow[expert_idx] = 0.0
        self.total_tokens_seen[expert_idx] = 0
        self.last_update_step[expert_idx] = 0

    def get_memory_usage(self) -> int:
        """
        Estimate memory usage in bytes.

        Returns:
            Memory usage in bytes
        """
        # 3 float tensors + 2 long tensors
        float_bytes = 3 * self.num_experts * 4  # float32
        long_bytes = 2 * self.num_experts * 8  # int64
        return float_bytes + long_bytes

    def to_dict(self) -> Dict:
        """
        Export state to dict (for checkpointing).

        Returns:
            State dict with all tensors moved to CPU
        """
        return {
            "layer_id": self.layer_id,
            "num_experts": self.num_experts,
            "d_model": self.d_model,
            "phi_fast": self.phi_fast.cpu(),
            "phi_mid": self.phi_mid.cpu(),
            "phi_slow": self.phi_slow.cpu(),
            "total_tokens_seen": self.total_tokens_seen.cpu(),
            "last_update_step": self.last_update_step.cpu(),
            "alpha_fast": self.alpha_fast,
            "alpha_mid": self.alpha_mid,
            "alpha_slow": self.alpha_slow,
        }

    def load_from_dict(self, state_dict: Dict):
        """
        Load state from dict (for checkpointing).

        Args:
            state_dict: State dict from to_dict()
        """
        self.layer_id = state_dict["layer_id"]
        self.num_experts = state_dict["num_experts"]
        self.d_model = state_dict["d_model"]
        self.phi_fast = state_dict["phi_fast"].to(self.device)
        self.phi_mid = state_dict["phi_mid"].to(self.device)
        self.phi_slow = state_dict["phi_slow"].to(self.device)
        self.total_tokens_seen = state_dict["total_tokens_seen"].to(self.device)
        self.last_update_step = state_dict["last_update_step"].to(self.device)
        self.alpha_fast = state_dict["alpha_fast"]
        self.alpha_mid = state_dict["alpha_mid"]
        self.alpha_slow = state_dict["alpha_slow"]


class MultiLayerCoherenceBuffer:
    """
    Manages coherence buffers for multiple layers.

    Convenience wrapper for models with multiple MoE layers.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts_per_layer: int,
        d_model: int,
        device: str = "cuda",
        alpha_fast: float = 0.9,
        alpha_mid: float = 0.99,
        alpha_slow: float = 0.999,
    ):
        """
        Initialize multi-layer coherence buffer.

        Args:
            num_layers: Number of MoE layers
            num_experts_per_layer: Experts per layer
            d_model: Hidden dimension
            device: Device to place tensors on
            alpha_fast: Fast clock decay rate
            alpha_mid: Mid clock decay rate
            alpha_slow: Slow clock decay rate
        """
        self.num_layers = num_layers
        self.num_experts_per_layer = num_experts_per_layer
        self.d_model = d_model
        self.device = device

        # Create buffer for each layer
        self.buffers = [
            CoherenceBuffer(
                layer_id=layer_id,
                num_experts=num_experts_per_layer,
                d_model=d_model,
                device=device,
                alpha_fast=alpha_fast,
                alpha_mid=alpha_mid,
                alpha_slow=alpha_slow,
            )
            for layer_id in range(num_layers)
        ]

    def update(
        self,
        layer_id: int,
        phi_raw: Tensor,
        active_expert_ids: Tensor,
        step: int,
        num_tokens: Tensor,
    ):
        """Update coherence for one layer."""
        self.buffers[layer_id].update(phi_raw, active_expert_ids, step, num_tokens)

    def snapshot(self, step: int) -> Dict[str, CoherenceState]:
        """
        Snapshot all layers to CPU.

        Returns:
            Dict mapping expert_id (str "L{layer}_E{expert}") to CoherenceState
        """
        all_states = {}
        for layer_id, buffer in enumerate(self.buffers):
            layer_states = buffer.snapshot(step)
            # Convert int keys to string keys with layer prefix
            for expert_idx, state in layer_states.items():
                all_states[state.expert_id] = state
        return all_states

    def get_memory_usage(self) -> int:
        """Total memory usage across all layers."""
        return sum(buf.get_memory_usage() for buf in self.buffers)
