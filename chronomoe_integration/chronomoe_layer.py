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
from chronomoe_integration.controller import (
    ChronoController,
    ObservationSnapshot,
    create_controller,
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
        autonomous_mode: bool = False,
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

        # Milestone A-D: Controller for signal processing and autonomous triggers
        # DIAGNOSTIC mode (default): signals only, no autonomous proposals
        # AUTONOMOUS mode (explicit): generates autonomous SPAWN/PRUNE proposals
        self.controller = create_controller(
            layer_id=layer_id,
            max_experts=self.max_experts,
            initial_active=initial_experts,
            autonomous_mode=autonomous_mode,
        )
        self.autonomous_mode = autonomous_mode

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

        # Milestone A: Capture per-expert outputs for coherence tracking
        # [max_experts, B*T, d_model] - filled with expert outputs
        expert_outputs = torch.zeros(
            self.max_experts, num_tokens, inputs_squashed.shape[-1],
            device=inputs.device, dtype=inputs_squashed.dtype
        )

        for i, expert in enumerate(self.experts):
            if not active_mask[i]:
                continue  # Skip inactive experts

            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                continue

            output, _ = expert(inputs_squashed[batch_idx])

            # Milestone A: Store per-expert output (before mixing)
            expert_outputs[i, batch_idx] = output

            # Mix into results
            results[batch_idx] += weights[batch_idx, nth_expert, None] * output

            # Track utilization (number of tokens processed)
            expert_utilization[i] = len(batch_idx)

        # Update probation token counts
        for expert_id in range(self.max_experts):
            if active_mask[expert_id]:
                num_tokens = int(expert_utilization[expert_id].item())
                self.registry.update_probation_tokens(expert_id, num_tokens)

        # Milestone A: Send observation to controller (coherence tracking)
        snapshot = ObservationSnapshot(
            step=self.current_step,
            layer_id=self.layer_id,
            router_probs=all_probs if self.softmax_order == "softmax_topk" else F.softmax(router_logits, dim=1),
            selected_experts=selected_experts,
            expert_outputs=expert_outputs,
            mixture_output=results,
            utilization=expert_utilization.float(),
            loss=None,  # Will be set by training loop if needed
        )
        self.controller.observe(snapshot)

        # Return output and metadata
        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "expert_utilization": expert_utilization,
            # Milestone A: Per-expert outputs for coherence tracking
            "expert_outputs": expert_outputs,  # [max_experts, B*T, d_model]
            "mixture_output": results,  # [B*T, d_model]
            "router_probs": all_probs if self.softmax_order == "softmax_topk" else F.softmax(router_logits, dim=1),
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

    def split_expert(
        self,
        parent_id: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        check_calm_gate: bool = True,
    ) -> Optional[Tuple[int, int]]:
        """
        Split expert into two specialized children (Milestone E).

        Creates two new experts by cloning parent, then prunes parent.
        Both children start in PROBATION state.

        Args:
            parent_id: Expert to split
            optimizer: Optional optimizer (for gradient state management)
            check_calm_gate: Whether to check stress band gates

        Returns:
            (child_a_id, child_b_id) if successful, None if blocked

        NOTE: SPLIT requires 2 free slots (net +1 after parent pruned).
        """
        # Check calm gate
        if check_calm_gate:
            from chronomoe_integration.stress_bands import lifecycle_gates
            gates = lifecycle_gates(self.stress_bands, self.stress_bands_config)
            if not gates.allow_split:
                print(f"  [ChronoMoE Layer {self.layer_id}] SPLIT BLOCKED: {gates.reason}")
                return None

        # Check capacity: need 2 slots (net +1 after parent pruned)
        assert self.registry.capacity_remaining >= 2, \
            f"Layer {self.layer_id}: Need 2 slots for split, only {self.registry.capacity_remaining} available"

        # Verify parent exists and is active
        assert parent_id in self.registry.experts, f"Parent expert {parent_id} not found"
        assert self.registry.experts[parent_id].state == ExpertState.ACTIVE, \
            f"Can only split ACTIVE experts (parent {parent_id} is {self.registry.experts[parent_id].state})"

        # Get two new expert IDs
        child_a_id = self.registry.next_expert_id
        child_b_id = self.registry.next_expert_id + 1

        # Clone parent weights to both children
        parent_expert = self.experts[parent_id]
        child_a_expert = self.experts[child_a_id]
        child_b_expert = self.experts[child_b_id]

        child_a_expert.load_state_dict(parent_expert.state_dict())
        child_b_expert.load_state_dict(parent_expert.state_dict())

        # Register both children (PROBATION state)
        self.registry.register_expert(
            expert_id=child_a_id,
            parent_id=parent_id,
            strategy="split_a",  # New strategy marker
            current_step=self.current_step,
        )

        self.registry.register_expert(
            expert_id=child_b_id,
            parent_id=parent_id,
            strategy="split_b",  # New strategy marker
            current_step=self.current_step,
        )

        # Prune parent
        self.registry.prune_expert(parent_id)

        print(f"  [ChronoMoE Layer {self.layer_id}] SPLIT: Expert {parent_id} → "
              f"[{child_a_id}, {child_b_id}] (probation)")

        return (child_a_id, child_b_id)

    def process_controller_proposals(self, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """
        Process autonomous proposals from controller (Milestone D).

        Non-bypassable enforcement:
        - Checks stress bands (COMFORT/STRAIN/PANIC)
        - Checks calm credit requirements
        - Only executes in COMFORT with sufficient calm credit
        - Logs all decisions (propose, queue, reject, execute)

        This is the TWO-STEP COMMIT boundary:
        1. Controller proposes (based on signals)
        2. Layer decides (based on stress bands + calm gates)

        Args:
            optimizer: Optional optimizer for spawn operations

        Returns:
            Dict with execution results and logs
        """
        from chronomoe_integration.stress_bands import lifecycle_gates
        from chronomoe_integration.controller import EditResult

        # Get proposals from controller (AUTONOMOUS mode must be enabled)
        proposals = self.controller.decide()

        if not proposals:
            return {"proposals": 0, "executed": 0, "rejected": 0, "queued": 0, "log": []}

        # Check current stress band and calm gates
        gates = lifecycle_gates(self.stress_bands, self.stress_bands_config)
        current_band = self.stress_bands.current_band
        time_in_comfort = self.stress_bands.time_in_comfort

        results = {
            "proposals": len(proposals),
            "executed": 0,
            "rejected": 0,
            "queued": 0,
            "log": [],
        }

        for proposal in proposals:
            log_entry = {
                "step": self.current_step,
                "type": proposal.edit_type,
                "expert_id": proposal.expert_id,
                "reason": proposal.reason,
                "delta_f_l": proposal.delta_f_l,
                "calm_required": proposal.calm_credit_required,
                "calm_actual": time_in_comfort,
                "band": current_band.value,
            }

            # NON-BYPASSABLE GATE ENFORCEMENT
            # Check 1: Must be in COMFORT band
            if current_band != Band.COMFORT:
                log_entry["action"] = "REJECTED"
                log_entry["block_reason"] = f"Not in COMFORT (current: {current_band.value})"
                results["rejected"] += 1
                results["log"].append(log_entry)
                print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL REJECTED: {proposal.edit_type} expert {proposal.expert_id} - {log_entry['block_reason']}")

                # Report to controller
                self.controller.apply(EditResult(
                    edit_type=proposal.edit_type,
                    success=False,
                    expert_id=proposal.expert_id,
                    reason=log_entry["block_reason"],
                ))
                continue

            # Check 2: Must have sufficient calm credit
            if time_in_comfort < proposal.calm_credit_required:
                log_entry["action"] = "QUEUED"
                log_entry["block_reason"] = f"Insufficient calm credit ({time_in_comfort} < {proposal.calm_credit_required})"
                results["queued"] += 1
                results["log"].append(log_entry)
                print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL QUEUED: {proposal.edit_type} expert {proposal.expert_id} - {log_entry['block_reason']}")

                # Report to controller
                self.controller.apply(EditResult(
                    edit_type=proposal.edit_type,
                    success=False,
                    expert_id=proposal.expert_id,
                    reason=log_entry["block_reason"],
                ))
                continue

            # Gates passed - EXECUTE
            log_entry["action"] = "EXECUTED"

            if proposal.edit_type == "spawn":
                new_expert_id = self.spawn_expert(
                    parent_id=proposal.expert_id,
                    strategy="blank",  # Default to blank
                    optimizer=optimizer,
                    check_calm_gate=False,  # Already checked above
                )
                if new_expert_id is not None:
                    log_entry["new_expert_id"] = new_expert_id
                    results["executed"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL EXECUTED: SPAWN expert {new_expert_id} (parent: {proposal.expert_id})")

                    # Report success to controller
                    self.controller.apply(EditResult(
                        edit_type="spawn",
                        success=True,
                        expert_id=proposal.expert_id,
                        new_expert_id=new_expert_id,
                    ))
                else:
                    log_entry["action"] = "REJECTED"
                    log_entry["block_reason"] = "Spawn failed (capacity?)"
                    results["rejected"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL FAILED: SPAWN - {log_entry['block_reason']}")

                    # Report failure to controller
                    self.controller.apply(EditResult(
                        edit_type="spawn",
                        success=False,
                        expert_id=proposal.expert_id,
                        reason=log_entry["block_reason"],
                    ))

            elif proposal.edit_type == "prune":
                success = self.prune_expert(
                    expert_id=proposal.expert_id,
                    check_calm_gate=False,  # Already checked above
                )
                if success:
                    results["executed"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL EXECUTED: PRUNE expert {proposal.expert_id}")

                    # Report success to controller
                    self.controller.apply(EditResult(
                        edit_type="prune",
                        success=True,
                        expert_id=proposal.expert_id,
                    ))
                else:
                    log_entry["action"] = "REJECTED"
                    log_entry["block_reason"] = "Prune failed (probation?)"
                    results["rejected"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL FAILED: PRUNE - {log_entry['block_reason']}")

                    # Report failure to controller
                    self.controller.apply(EditResult(
                        edit_type="prune",
                        success=False,
                        expert_id=proposal.expert_id,
                        reason=log_entry["block_reason"],
                    ))

            elif proposal.edit_type == "split":
                result = self.split_expert(
                    parent_id=proposal.expert_id,
                    optimizer=optimizer,
                    check_calm_gate=False,  # Already checked above
                )
                if result is not None:
                    child_a_id, child_b_id = result
                    log_entry["child_a_id"] = child_a_id
                    log_entry["child_b_id"] = child_b_id
                    results["executed"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL EXECUTED: SPLIT expert {proposal.expert_id} → [{child_a_id}, {child_b_id}]")

                    # Report success to controller
                    self.controller.apply(EditResult(
                        edit_type="split",
                        success=True,
                        expert_id=proposal.expert_id,
                        new_expert_id=child_a_id,  # Primary child for compatibility
                        reason=f"Split into experts {child_a_id} and {child_b_id}",
                    ))
                else:
                    log_entry["action"] = "REJECTED"
                    log_entry["block_reason"] = "Split failed (capacity/gates)"
                    results["rejected"] += 1
                    print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL FAILED: SPLIT - {log_entry['block_reason']}")

                    # Report failure to controller
                    self.controller.apply(EditResult(
                        edit_type="split",
                        success=False,
                        expert_id=proposal.expert_id,
                        reason=log_entry["block_reason"],
                    ))

            else:
                # MERGE not implemented yet (Milestone E deferred)
                log_entry["action"] = "REJECTED"
                log_entry["block_reason"] = f"Unsupported edit type: {proposal.edit_type}"
                results["rejected"] += 1
                print(f"  [ChronoMoE Layer {self.layer_id}] PROPOSAL REJECTED: {log_entry['block_reason']}")

                # Report to controller
                self.controller.apply(EditResult(
                    edit_type=proposal.edit_type,
                    success=False,
                    expert_id=proposal.expert_id,
                    reason=log_entry["block_reason"],
                ))

            results["log"].append(log_entry)

        return results
