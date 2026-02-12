#!/usr/bin/env python3
"""
Test suppression trials (Milestone F Phase 1.5).

Validates penalty system for testing redundancy before MERGE execution.

Tests:
- Test 20: Penalty shifts router selection
- Test 21: Cooldown blocks selection entirely
- Test 22: Penalty decays and selection returns
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from chronomoe_integration.controller import (
    ChronoController,
    create_controller,
)


def test_penalty_shifts_selection():
    """
    Test 20: Penalty Shifts Router Selection

    When expert has penalty, router prefers other experts (all else equal).

    Deterministic construction:
    - Two experts with identical logits (equal preference)
    - Apply penalty to expert 0
    - Verify router selects expert 1 instead
    """
    print("\n" + "=" * 70)
    print("TEST 20: Penalty Shifts Router Selection")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=False,  # Not testing proposals, just penalty mechanism
    )

    # Apply penalty to expert 0
    controller.suppress_expert(expert_id=0, duration_steps=0)  # Penalty only, no cooldown

    # Get routing adjustments (two-channel)
    current_step = 0
    hard_blocks, soft_penalties = controller.get_routing_adjustments(current_step)

    print(f"✓ Hard blocks: {hard_blocks}")
    print(f"✓ Soft penalties: {soft_penalties}")
    assert 0 not in hard_blocks, "Expected no hard block (no cooldown)"
    assert 0 in soft_penalties, "Expected soft penalty for expert 0"
    assert soft_penalties[0] > 0, "Expected positive penalty (subtracted from logits)"

    # Simulate router decision with equal logits
    num_experts = 4
    base_logits = torch.ones(num_experts) * 5.0  # All equal

    # Apply penalties (as router would)
    adjusted_logits = base_logits.clone()
    for expert_id in hard_blocks:
        adjusted_logits[expert_id] = float('-inf')
    for expert_id, penalty in soft_penalties.items():
        adjusted_logits[expert_id] -= penalty

    print(f"✓ Base logits: {base_logits.tolist()}")
    print(f"✓ Adjusted logits: {adjusted_logits.tolist()}")

    # Verify expert 0 has lower adjusted logit
    assert adjusted_logits[0] < base_logits[0], "Expected expert 0 logit reduced"
    assert adjusted_logits[1] == base_logits[1], "Expected expert 1 logit unchanged"

    # Router should prefer expert 1 (argmax)
    selected = adjusted_logits.argmax().item()
    print(f"✓ Router selects expert: {selected}")
    assert selected != 0, "Expected router to avoid penalized expert 0"

    print("\n✓ TEST 20 PASSED: Penalty shifts selection correctly\n")


def test_cooldown_blocks_selection():
    """
    Test 21: Cooldown Blocks Selection Entirely

    During cooldown, expert is unavailable (logit → -inf equivalent).

    Deterministic construction:
    - Apply penalty + cooldown to expert 0
    - Verify routing adjustment is very large (blocks selection)
    - Verify expert cannot be selected even with high base logit
    """
    print("\n" + "=" * 70)
    print("TEST 21: Cooldown Blocks Selection Entirely")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=False,
    )

    # Apply penalty + cooldown to expert 0
    cooldown_duration = 100
    controller.suppress_expert(expert_id=0, duration_steps=cooldown_duration)

    # Simulate observation to set step
    controller.observation_history.append(None)  # Dummy observation
    current_step = 1

    # Get routing adjustments during cooldown (two-channel)
    hard_blocks, soft_penalties = controller.get_routing_adjustments(current_step)

    print(f"✓ Hard blocks during cooldown: {hard_blocks}")
    print(f"✓ Soft penalties during cooldown: {soft_penalties}")
    assert 0 in hard_blocks, "Expected expert 0 in hard blocks during cooldown"
    assert 0 not in soft_penalties, "Expected no soft penalty during cooldown (hard block takes precedence)"

    # Simulate router with expert 0 having highest base logit
    base_logits = torch.tensor([10.0, 5.0, 5.0, 5.0])  # Expert 0 preferred

    # Apply adjustments (as router would)
    adjusted_logits = base_logits.clone()
    for expert_id in hard_blocks:
        adjusted_logits[expert_id] = float('-inf')
    for expert_id, penalty in soft_penalties.items():
        adjusted_logits[expert_id] -= penalty

    print(f"✓ Base logits: {base_logits.tolist()}")
    print(f"✓ Adjusted logits: {adjusted_logits[:4].tolist()}")

    # Verify expert 0 is blocked (hard mask applied)
    assert adjusted_logits[0] == float('-inf'), "Expected expert 0 hard blocked (-inf)"

    # Router should select different expert
    selected = adjusted_logits.argmax().item()
    print(f"✓ Router selects expert: {selected}")
    assert selected != 0, "Expected router to avoid cooled-down expert 0"
    assert selected in [1, 2, 3], f"Expected selection from available experts, got {selected}"

    print("\n✓ TEST 21 PASSED: Cooldown blocks selection correctly\n")


def test_penalty_decays():
    """
    Test 22: Penalty Decays and Selection Returns

    After decay, penalty diminishes and router returns to normal.

    Deterministic construction:
    - Apply penalty to expert 0 (no cooldown)
    - Call update_penalties() multiple times
    - Verify penalty decays
    - Verify selection returns to normal
    """
    print("\n" + "=" * 70)
    print("TEST 22: Penalty Decays and Selection Returns")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=False,
    )

    # Set decay rate to 0.2 (20% decay per step)
    controller.config["suppression"]["decay_rate"] = 0.2

    # Apply penalty to expert 0 (no cooldown)
    initial_penalty = 10.0
    controller.config["suppression"]["penalty_magnitude"] = initial_penalty
    controller.suppress_expert(expert_id=0, duration_steps=0)

    # Verify initial penalty (two-channel)
    hard_blocks, soft_penalties = controller.get_routing_adjustments(current_step=0)
    assert 0 not in hard_blocks, "Expected no hard block (no cooldown)"
    assert 0 in soft_penalties, "Expected initial soft penalty"
    assert abs(soft_penalties[0] - initial_penalty) < 0.01, f"Expected penalty ~{initial_penalty}"
    print(f"✓ Initial penalty: {soft_penalties[0]:.4f}")

    # Decay penalty over 10 steps
    penalties_over_time = [soft_penalties[0]]
    for step in range(1, 11):
        controller.update_penalties()
        hard_blocks, soft_penalties = controller.get_routing_adjustments(current_step=step)
        if 0 in soft_penalties:
            penalties_over_time.append(soft_penalties[0])
        else:
            penalties_over_time.append(0.0)  # Penalty cleaned up

    print(f"✓ Penalty decay over 10 steps: {[f'{p:.4f}' for p in penalties_over_time]}")

    # Verify decay pattern
    # With decay_rate=0.2: penalty *= 0.8 each step
    # After 10 steps: penalty ≈ 10.0 * (0.8^10) ≈ 1.07
    expected_final = initial_penalty * (0.8 ** 10)
    actual_final = penalties_over_time[-1]
    print(f"✓ Expected final penalty: {expected_final:.4f}")
    print(f"✓ Actual final penalty: {actual_final:.4f}")
    assert actual_final < initial_penalty * 0.2, "Expected penalty to decay significantly"

    # Verify penalties are strictly decreasing (until cleanup)
    for i in range(len(penalties_over_time) - 1):
        if penalties_over_time[i] > 0.01:  # Before cleanup threshold
            assert penalties_over_time[i] >= penalties_over_time[i+1], \
                f"Expected monotonic decay at step {i}"

    # Simulate router selection with equal logits
    base_logits = torch.ones(4) * 5.0

    # With decayed penalty
    adjusted_logits = base_logits.clone()
    if actual_final > 0.01:
        adjusted_logits[0] -= actual_final

    # Selection should be less biased (penalty weakened)
    logit_diff = base_logits[0] - adjusted_logits[0]
    print(f"✓ Logit difference after decay: {logit_diff:.4f}")
    assert logit_diff < initial_penalty * 0.2, "Expected weakened penalty effect"

    print("\n✓ TEST 22 PASSED: Penalty decays correctly\n")


if __name__ == "__main__":
    test_penalty_shifts_selection()
    test_cooldown_blocks_selection()
    test_penalty_decays()

    print("=" * 70)
    print("✓ ALL SUPPRESSION TESTS PASSED (3/3)")
    print("=" * 70)
