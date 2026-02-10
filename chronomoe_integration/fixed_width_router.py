"""
Fixed-width router for swiss-ai/MoE.

Ported from ChronoMoEv3/nanoMoE integration.
The router outputs max_experts logits from initialization, not moe_num_experts.
Lifecycle operations change masks, not tensor dimensions.
"""

import torch
import torch.nn as nn


class FixedWidthRouter(nn.Module):
    """
    Fixed-width router wrapper.

    Takes a standard router (nn.Linear(n_embd, moe_num_experts)) and extends it
    to output max_experts logits. Initial moe_num_experts weights are copied,
    remaining slots initialized randomly.

    This ensures the router action space is fixed from step 0, and lifecycle
    events only change masks and state, never tensor shapes mid-flight.
    """

    def __init__(self, base_router: nn.Linear, max_experts: int):
        super().__init__()

        # Router that outputs max_experts logits (not moe_num_experts)
        n_embd = base_router.in_features
        initial_n_exp = base_router.out_features

        assert max_experts >= initial_n_exp, \
            f"max_experts ({max_experts}) must be >= initial experts ({initial_n_exp})"

        self.max_experts = max_experts
        self.w_g = nn.Linear(n_embd, max_experts, bias=False)

        # Copy initial weights, initialize rest randomly
        with torch.no_grad():
            self.w_g.weight[:initial_n_exp].copy_(base_router.weight)
            # Remaining weights already random from Linear init

        print(f"  [FixedWidthRouter] Extended router from {initial_n_exp} → {max_experts} logits")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute router logits for all max_experts.

        Args:
            x: Input tensor [B, T, n_embd] or [B*T, n_embd]

        Returns:
            Router logits [B*T, max_experts]
        """
        return self.w_g(x)
