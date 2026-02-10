"""
ChronoMoE Layer for swiss-ai/MoE.

Wraps the standard MoE class with lifecycle capabilities:
- Fixed-width routing
- Pre-allocated expert slots
- Probation mechanism
- Hard mask for inactive experts

Ported from ChronoMoEv3/nanoMoE integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from chronomoe_integration.expert_registry import ExpertRegistry, ProbationConfig, ExpertState
from chronomoe_integration.fixed_width_router import FixedWidthRouter


class ChronoMoE(nn.Module):
    """
    MoE layer with lifecycle capabilities.

    Fixed-width design:
    - Router outputs max_experts logits from initialization
    - Experts pre-allocated as max_experts slots in ModuleList
    - Lifecycle operations change active_mask, not tensor shapes
    """

    def __init__(
        self,
        config,
        mlp,
        layer_id: int,
        max_experts: Optional[int] = None,
        probation_config: Optional[ProbationConfig] = None,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.config = config
        self.top_k = config.moe_num_experts_per_tok
        self.softmax_order = config.moe_softmax_order

        # Fixed-width: max_experts (default to 2x initial)
        initial_experts = config.moe_num_experts
        self.max_experts = max_experts or (initial_experts * 2)

        assert self.max_experts >= initial_experts

        # Pre-allocate ALL expert slots (fixed-width)
        self.experts = nn.ModuleList([
            mlp(config=config) for _ in range(self.max_experts)
        ])

        # Create fixed-width router
        base_router = nn.Linear(config.n_embd, initial_experts, bias=False)
        self.router = FixedWidthRouter(base_router, max_experts=self.max_experts)

        # Expert registry for lifecycle management
        self.registry = ExpertRegistry(
            layer_id=layer_id,
            max_experts=self.max_experts,
            initial_active=initial_experts,
            probation_config=probation_config,
        )

        # Current step (updated externally before forward)
        self.current_step = 0

        print(f"  [ChronoMoE Layer {layer_id}] {initial_experts} active → {self.max_experts} max experts")

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with lifecycle-aware routing.

        Order of operations:
        1. Compute router logits for all max_experts
        2. Apply probation boosts
        3. Apply hard mask to inactive experts (-inf logits)
        4. Softmax + top-k
        5. Dispatch to experts

        Args:
            inputs: [B, T, n_embd]

        Returns:
            (output, metadata) where metadata contains router_logits, selected_experts, utilization
        """
        # [batch_size * sequence_length, n_embd]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        num_tokens = inputs_squashed.shape[0]

        # 1. Compute router logits for ALL max_experts
        router_logits = self.router(inputs_squashed)  # [B*T, max_experts]

        # 2. Apply probation boosts (BEFORE masking)
        active_mask = self.registry.active_mask.to(router_logits.device)
        for expert_id in range(self.max_experts):
            if active_mask[expert_id]:
                boost = self.registry.get_probation_boost(expert_id, self.current_step)
                if boost > 0:
                    router_logits[:, expert_id] += boost

        # 3. Hard mask inactive experts to -inf (BEFORE softmax)
        inactive_mask = ~active_mask
        router_logits[:, inactive_mask] = float('-inf')

        # 4. Softmax + top-k (standard swiss-ai logic)
        if self.softmax_order == "softmax_topk":
            all_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
            weights, selected_experts = torch.topk(all_probs, self.top_k)
        elif self.softmax_order == "topk_softmax":
            weights, selected_experts = torch.topk(router_logits, self.top_k)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown softmax_order: {self.softmax_order}")

        # 5. Dispatch to experts (standard swiss-ai logic)
        results = torch.zeros_like(inputs_squashed)
        expert_utilization = torch.zeros(self.max_experts, dtype=torch.long, device=inputs.device)

        for i, expert in enumerate(self.experts):
            if not active_mask[i]:
                continue  # Skip inactive experts

            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                continue

            output, _ = expert(inputs_squashed[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output

            # Track utilization (number of tokens processed)
            expert_utilization[i] = len(batch_idx)

        # Update probation token counts
        for expert_id in range(self.max_experts):
            if active_mask[expert_id]:
                num_tokens = int(expert_utilization[expert_id].item())
                self.registry.update_probation_tokens(expert_id, num_tokens)

        # Return output and metadata
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "expert_utilization": expert_utilization,
        }

    def spawn_expert(
        self,
        parent_id: int,
        strategy: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """
        Spawn a new expert from parent.

        Args:
            parent_id: Parent expert ID (for cloning)
            strategy: "blank" or "clone"
            optimizer: Optimizer to register new parameters

        Returns:
            New expert ID
        """
        assert self.registry.capacity_remaining > 0, \
            f"Layer {self.layer_id}: No capacity remaining"

        new_expert_id = self.registry.next_expert_id

        # Initialize expert (pre-existing slot)
        if strategy == "clone":
            # Clone parent weights
            parent_expert = self.experts[parent_id]
            new_expert = self.experts[new_expert_id]
            new_expert.load_state_dict(parent_expert.state_dict())
        elif strategy == "blank":
            # Already randomly initialized
            pass
        else:
            raise ValueError(f"Unknown spawn strategy: {strategy}")

        # Register in registry (activates in active_mask)
        self.registry.register_expert(
            expert_id=new_expert_id,
            parent_id=parent_id,
            strategy=strategy,
            current_step=self.current_step,
        )

        # Add to optimizer if provided
        if optimizer is not None:
            optimizer.add_param_group({
                'params': self.experts[new_expert_id].parameters()
            })

        print(f"  [ChronoMoE Layer {self.layer_id}] SPAWN: Expert {new_expert_id} ({strategy} from {parent_id})")

        return new_expert_id

    def prune_expert(self, expert_id: int) -> None:
        """
        Prune expert (deactivate in mask).

        Does NOT physically remove from ModuleList (fixed-width).
        """
        # Skip probation experts
        info = self.registry.experts.get(expert_id)
        if info and info.state == ExpertState.PROBATION:
            return

        self.registry.prune_expert(expert_id)
        print(f"  [ChronoMoE Layer {self.layer_id}] PRUNE: Expert {expert_id}")

    def check_probation_graduations(self, in_comfort_band: bool = True) -> None:
        """
        Check probation experts for graduation/failure.

        Should be called after forward pass at appropriate intervals.
        """
        results = self.registry.check_probation_status(
            current_step=self.current_step,
            in_comfort_band=in_comfort_band,
        )

        for expert_id, status in results:
            if status == "graduate":
                self.registry.graduate_from_probation(expert_id, self.current_step)
                tokens = self.registry.experts[expert_id].probation_tokens_accumulated
                print(f"  [ChronoMoE Layer {self.layer_id}] GRADUATE: Expert {expert_id} ({tokens} tokens)")
            elif status == "fail":
                self.prune_expert(expert_id)
                tokens = self.registry.experts[expert_id].probation_tokens_accumulated
                print(f"  [ChronoMoE Layer {self.layer_id}] PROBATION FAILURE: Expert {expert_id} ({tokens} tokens)")
