"""
Coherence tracking demonstration.

Shows how to:
1. Create MoETrace from a forward pass
2. Compute coherence phi_e
3. Track coherence over time with three-clock EMA
4. Detect expert degradation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from chronomoe_v3 import (
    MoETrace,
    CoherenceState,
    ChronoConfig,
    batch_update_coherence,
)


def simulate_forward_pass(
    num_experts: int = 8,
    num_tokens: int = 32,
    d_model: int = 128,
    expert_health: dict = None,
) -> MoETrace:
    """
    Simulate one forward pass through an MoE layer.

    Args:
        num_experts: Number of experts in layer
        num_tokens: Number of tokens (batch×seq)
        d_model: Hidden dimension
        expert_health: Dict of expert_id -> health (1.0=perfect, 0.0=broken)

    Returns:
        MoETrace capturing the forward pass state
    """
    if expert_health is None:
        expert_health = {i: 1.0 for i in range(num_experts)}

    # Simulate mixture direction
    mixture_direction = torch.randn(d_model)
    mixture_direction = mixture_direction / mixture_direction.norm()

    # Each expert processes some tokens
    tokens_per_expert = num_tokens // num_experts

    active_expert_ids = []
    expert_mean_outputs = []
    token_row_indices = []
    gate_weights = []

    for expert_idx in range(num_experts):
        if expert_idx not in expert_health:
            continue  # Expert not active

        health = expert_health[expert_idx]

        # Expert output = mixture direction + noise (noise increases with poor health)
        noise_scale = 1.0 - health
        expert_output = mixture_direction + noise_scale * torch.randn(d_model)
        expert_output = expert_output / expert_output.norm()

        active_expert_ids.append(expert_idx)
        expert_mean_outputs.append(expert_output)

        # Token assignment
        start_idx = expert_idx * tokens_per_expert
        end_idx = start_idx + tokens_per_expert
        token_row_indices.append(torch.arange(start_idx, end_idx))
        gate_weights.append(torch.ones(tokens_per_expert) / num_experts)

    # Mixture is weighted sum of expert outputs
    mixture = mixture_direction.unsqueeze(0).repeat(num_tokens, 1)

    return MoETrace(
        mixture=mixture,
        active_expert_ids=active_expert_ids,
        expert_mean_outputs=expert_mean_outputs,
        token_row_indices=token_row_indices,
        gate_weights=gate_weights,
    )


def main():
    """Demonstrate coherence tracking over time."""
    print("=" * 80)
    print("ChronoMoEv3 Coherence Tracking Demo")
    print("=" * 80)

    # Configuration
    config = ChronoConfig()
    layer_id = 0
    num_experts = 8
    num_tokens = 32
    d_model = 128

    # Expert coherence states
    states = {}

    # Scenario: All experts healthy initially
    print("\n--- Phase 1: All experts healthy (steps 0-100) ---")
    expert_health = {i: 1.0 for i in range(num_experts)}

    for step in range(100):
        trace = simulate_forward_pass(
            num_experts=num_experts,
            num_tokens=num_tokens,
            d_model=d_model,
            expert_health=expert_health,
        )

        batch_update_coherence(
            states=states,
            trace=trace,
            alpha_fast=config.alpha_fast,
            alpha_mid=config.alpha_mid,
            alpha_slow=config.alpha_slow,
            step=step,
            layer_id=layer_id,
        )

        if step in [0, 10, 50, 99]:
            phi = trace.compute_coherence()
            print(f"  Step {step:3d}: phi_mean={phi.mean():.3f}")

    # Check steady state
    print("\n  Steady state coherence:")
    for expert_id, state in sorted(states.items()):
        print(
            f"    {expert_id}: "
            f"fast={state.phi_fast:.3f}, "
            f"mid={state.phi_mid:.3f}, "
            f"slow={state.phi_slow:.3f}"
        )

    # Scenario: Expert 3 starts failing
    print("\n--- Phase 2: Expert 3 degrades (steps 100-200) ---")
    expert_health[3] = 0.1  # Expert 3 nearly broken

    for step in range(100, 200):
        trace = simulate_forward_pass(
            num_experts=num_experts,
            num_tokens=num_tokens,
            d_model=d_model,
            expert_health=expert_health,
        )

        batch_update_coherence(
            states=states,
            trace=trace,
            alpha_fast=config.alpha_fast,
            alpha_mid=config.alpha_mid,
            alpha_slow=config.alpha_slow,
            step=step,
            layer_id=layer_id,
        )

        if step in [100, 110, 150, 199]:
            phi = trace.compute_coherence()
            expert_3_phi = phi[3] if len(trace.active_expert_ids) > 3 else 0.0
            print(f"  Step {step:3d}: expert_3 phi={expert_3_phi:.3f}")

    # Check degradation detection
    print("\n  Expert 3 coherence state:")
    expert_3_state = states["L0_E3"]
    print(
        f"    fast={expert_3_state.phi_fast:.3f}, "
        f"mid={expert_3_state.phi_mid:.3f}, "
        f"slow={expert_3_state.phi_slow:.3f}"
    )
    print(f"    delta (fast-slow)={expert_3_state.phi_delta:.3f}")
    print(f"    is_degrading={expert_3_state.is_degrading}")

    # Scenario: Expert 3 removed, Expert 5 starts failing
    print("\n--- Phase 3: Expert 3 pruned, Expert 5 degrades (steps 200-300) ---")
    del expert_health[3]  # Prune expert 3
    expert_health[5] = 0.2  # Expert 5 degrading

    for step in range(200, 300):
        trace = simulate_forward_pass(
            num_experts=num_experts,
            num_tokens=num_tokens,
            d_model=d_model,
            expert_health=expert_health,
        )

        # Expert 3 not in trace anymore (pruned)
        batch_update_coherence(
            states=states,
            trace=trace,
            alpha_fast=config.alpha_fast,
            alpha_mid=config.alpha_mid,
            alpha_slow=config.alpha_slow,
            step=step,
            layer_id=layer_id,
        )

    # Final state summary
    print("\n--- Final coherence summary ---")
    print("\nHealthy experts:")
    for expert_id, state in sorted(states.items()):
        if state.last_update_step >= 290:  # Recently active
            if state.phi_slow > 0.7:
                print(
                    f"  {expert_id}: "
                    f"slow={state.phi_slow:.3f}, "
                    f"tokens={state.total_tokens_seen}"
                )

    print("\nDegraded experts:")
    for expert_id, state in sorted(states.items()):
        if state.last_update_step >= 290:  # Recently active
            if state.phi_slow <= 0.7:
                print(
                    f"  {expert_id}: "
                    f"slow={state.phi_slow:.3f}, "
                    f"delta={state.phi_delta:.3f}, "
                    f"degrading={state.is_degrading}"
                )

    print("\nPruned experts:")
    for expert_id, state in sorted(states.items()):
        if state.last_update_step < 200:  # Not updated recently
            print(
                f"  {expert_id}: "
                f"last_seen_step={state.last_update_step}, "
                f"final_slow={state.phi_slow:.3f}"
            )

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
