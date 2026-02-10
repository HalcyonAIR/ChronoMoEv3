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
from chronomoe_integration.stress_bands import (
    StressBandsState,
    StressBandsConfig,
    init_stress_bands,
    step_stress_bands,
    Band,
)


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
        stress_bands_config: Optional[StressBandsConfig] = None,
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

        # Stress bands for calm gating
        self.stress_bands_config = stress_bands_config or StressBandsConfig()
        self.stress_bands = init_stress_bands(self.stress_bands_config)

        # Current step (updated externally before forward)
        self.current_step = 0

        print(f"  [ChronoMoE Layer {layer_id}] {initial_experts} active → {self.max_experts} max experts (stress bands: {self.stress_bands_config.comfort_ceiling_init:.2f}/{self.stress_bands_config.strain_ceiling_init:.2f})")

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
        strategy: str = "blank",  # DEFAULT TO BLANK
        optimizer: Optional[torch.optim.Optimizer] = None,
        check_calm_gate: bool = True,
    ) -> Optional[int]:
        """
        Spawn a new expert from parent.

        Default strategy is "blank" (random init + probation). This is the
        recommended approach as probation mechanism ensures new experts get
        sufficient training signal.

        Clone spawning is available as explicit opt-in for special cases where
        inheriting parent weights is necessary.

        Args:
            parent_id: Parent expert ID (for cloning, if strategy="clone")
            strategy: Spawn strategy (default: "blank")
                - "blank": Random initialization + probation boost (recommended)
                - "clone": Copy parent weights + probation boost (opt-in only)
            optimizer: Optimizer to register new parameters
            check_calm_gate: If True, check stress bands calm gate before spawning

        Returns:
            New expert ID, or None if spawn blocked by calm gate
        """
        # Check calm gate
        if check_calm_gate:
            from chronomoe_integration.stress_bands import lifecycle_gates
            gates = lifecycle_gates(self.stress_bands, self.stress_bands_config)
            if not gates.allow_spawn:
                print(f"  [ChronoMoE Layer {self.layer_id}] SPAWN BLOCKED: {gates.reason}")
                return None

        assert self.registry.capacity_remaining > 0, \
            f"Layer {self.layer_id}: No capacity remaining"

        new_expert_id = self.registry.next_expert_id

        # Validate and enforce strategy
        if strategy == "clone":
            # Clone spawning is opt-in only
            if not self.registry.probation_config.allow_clone_spawn:
                raise ValueError(
                    f"Clone spawning is disabled (probation_config.allow_clone_spawn=False). "
                    f"Use strategy='blank' or enable clone spawning in config."
                )
            if self.registry.probation_config.warn_on_clone:
                print(f"  [ChronoMoE Layer {self.layer_id}] WARNING: Clone spawn used (strategy='clone'). "
                      f"Blank spawn is recommended as default.")

        # Initialize expert (pre-existing slot)
        if strategy == "clone":
            # Clone parent weights
            parent_expert = self.experts[parent_id]
            new_expert = self.experts[new_expert_id]
            new_expert.load_state_dict(parent_expert.state_dict())
        elif strategy == "blank":
            # Already randomly initialized (no action needed)
            pass
        else:
            raise ValueError(f"Unknown spawn strategy: {strategy}. Use 'blank' (default) or 'clone' (opt-in).")

        # Register in registry (activates in active_mask)
        self.registry.register_expert(
            expert_id=new_expert_id,
            parent_id=parent_id,
            strategy=strategy,
            current_step=self.current_step,
        )

        # Add to optimizer if provided
        # Note: In fixed-width design, experts are pre-allocated, so parameters
        # are already in optimizer. Only add if using dynamic allocation.
        # For now, skip adding (already in optimizer from model creation)
        if optimizer is not None and False:  # Disabled for pre-allocated experts
            optimizer.add_param_group({
                'params': self.experts[new_expert_id].parameters()
            })

        print(f"  [ChronoMoE Layer {self.layer_id}] SPAWN: Expert {new_expert_id} ({strategy} from {parent_id})")

        return new_expert_id

    def prune_expert(self, expert_id: int, check_calm_gate: bool = True) -> bool:
        """
        Prune expert (deactivate in mask).

        Does NOT physically remove from ModuleList (fixed-width).

        Args:
            expert_id: Expert to prune
            check_calm_gate: If True, check stress bands calm gate before pruning

        Returns:
            True if pruned, False if blocked
        """
        # Check calm gate
        if check_calm_gate:
            from chronomoe_integration.stress_bands import lifecycle_gates
            gates = lifecycle_gates(self.stress_bands, self.stress_bands_config)
            if not gates.allow_prune:
                print(f"  [ChronoMoE Layer {self.layer_id}] PRUNE BLOCKED: Expert {expert_id}, {gates.reason}")
                return False

        # Skip probation experts
        info = self.registry.experts.get(expert_id)
        if info and info.state == ExpertState.PROBATION:
            return False

        self.registry.prune_expert(expert_id)
        print(f"  [ChronoMoE Layer {self.layer_id}] PRUNE: Expert {expert_id}")
        return True

    def update_stress_bands(self, stress: float) -> None:
        """
        Update stress bands based on current stress metric (e.g., loss).

        Should be called once per step, typically after computing loss.

        Args:
            stress: Current stress metric (loss, free energy, etc.)
        """
        result = step_stress_bands(
            state=self.stress_bands,
            config=self.stress_bands_config,
            stress=stress,
        )

        # Log band transitions
        if result.band_now != result.band_prev:
            print(f"  [ChronoMoE Layer {self.layer_id}] Stress band: {result.band_prev.value} → {result.band_now.value} (stress: {result.stress_smoothed:.4f})")

    def check_probation_graduations(self, check_calm_gate: bool = True) -> None:
        """
        Check probation experts for graduation/failure.

        Should be called after forward pass at appropriate intervals.

        Args:
            check_calm_gate: If True, check stress bands calm gate before graduating
        """
        # Check calm gate for graduation
        if check_calm_gate:
            from chronomoe_integration.stress_bands import lifecycle_gates
            gates = lifecycle_gates(self.stress_bands, self.stress_bands_config)
            in_comfort_band = gates.allow_graduate
        else:
            in_comfort_band = True

        results = self.registry.check_probation_status(
            current_step=self.current_step,
            in_comfort_band=in_comfort_band,
        )

        for expert_id, status in results:
            if status == "graduate":
                if not in_comfort_band and check_calm_gate:
                    print(f"  [ChronoMoE Layer {self.layer_id}] GRADUATION BLOCKED: Expert {expert_id} (calm gate)")
                    continue

                self.registry.graduate_from_probation(expert_id, self.current_step)
                tokens = self.registry.experts[expert_id].probation_tokens_accumulated
                print(f"  [ChronoMoE Layer {self.layer_id}] GRADUATE: Expert {expert_id} ({tokens} tokens)")
            elif status == "fail":
                # Probation failure triggers prune (skip calm gate check for failures)
                self.prune_expert(expert_id, check_calm_gate=False)
                tokens = self.registry.experts[expert_id].probation_tokens_accumulated
                print(f"  [ChronoMoE Layer {self.layer_id}] PROBATION FAILURE: Expert {expert_id} ({tokens} tokens)")
