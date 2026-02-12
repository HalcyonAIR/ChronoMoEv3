#!/usr/bin/env python3
"""
Test router integration with suppression adjustments (Milestone F Phase 1.5).

Validates:
- Test 23: Hard block prevents expert from receiving tokens
- Test 24: Soft penalty reduces expert utilization
- Test 25: Active mask is absolute (penalties can't revive inactive experts)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from chronomoe_integration.chronomoe_layer import ChronoMoE


class DummyMLP(nn.Module):
    """Minimal MLP for testing."""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc(x), {}


class Config:
    """Minimal config for testing."""
    def __init__(self):
        self.n_embd = 64
        self.max_experts = 8
        self.moe_num_experts = 4  # Initial active experts
        self.moe_num_experts_per_tok = 2  # Top-k
        self.moe_softmax_order = "softmax_topk"


def test_hard_block_prevents_routing():
    """
    Test 23: Hard Block Prevents Expert from Receiving Tokens

    When expert is hard blocked (cooldown), it should receive zero tokens
    even if router would normally select it.
    """
    print("\n" + "=" * 70)
    print("TEST 23: Hard Block Prevents Expert from Receiving Tokens")
    print("=" * 70)

    config = Config()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Get baseline utilization (no suppression)
    x = torch.randn(2, 10, config.n_embd)  # [batch=2, seq=10, d_model=64]

    baseline_utils = []
    for _ in range(20):
        output, metadata = layer(x)
        baseline_utils.append(metadata['expert_utilization'].clone())

    baseline_util = torch.stack(baseline_utils).float().mean(dim=0)
    print(f"✓ Baseline utilization (20 passes): {baseline_util.tolist()}")

    # Find expert with highest baseline utilization
    target_expert = baseline_util.argmax().item()
    print(f"✓ Target expert (highest baseline): {target_expert}")
    assert baseline_util[target_expert] > 0, "Target expert should have baseline usage"

    # Apply hard block (cooldown) to target expert
    layer.controller.suppress_expert(expert_id=target_expert, duration_steps=100)
    print(f"✓ Hard block applied to expert {target_expert} (cooldown=100 steps)")

    # Run forward passes with hard block
    blocked_utils = []
    for _ in range(20):
        layer.current_step += 1
        output, metadata = layer(x)
        blocked_utils.append(metadata['expert_utilization'].clone())

    blocked_util = torch.stack(blocked_utils).float().mean(dim=0)
    print(f"✓ Blocked utilization (20 passes): {blocked_util.tolist()}")

    # Verify target expert received ZERO tokens
    print(f"✓ Target expert {target_expert}: baseline={baseline_util[target_expert]:.1f}, blocked={blocked_util[target_expert]:.1f}")
    assert blocked_util[target_expert] == 0, \
        f"Hard blocked expert should receive ZERO tokens, got {blocked_util[target_expert]}"

    print("\n✓ TEST 23 PASSED: Hard block prevents routing correctly\n")


def test_soft_penalty_reduces_utilization():
    """
    Test 24: Soft Penalty Reduces Expert Utilization

    When expert has soft penalty (no cooldown), it should receive fewer tokens
    but not zero (unless penalty is very large).
    """
    print("\n" + "=" * 70)
    print("TEST 24: Soft Penalty Reduces Expert Utilization")
    print("=" * 70)

    config = Config()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Get baseline utilization
    x = torch.randn(2, 10, config.n_embd)

    baseline_utils = []
    for _ in range(20):
        output, metadata = layer(x)
        baseline_utils.append(metadata['expert_utilization'].clone())

    baseline_util = torch.stack(baseline_utils).float().mean(dim=0)
    print(f"✓ Baseline utilization (20 passes): {baseline_util.tolist()}")

    # Find expert with highest baseline utilization
    target_expert = baseline_util.argmax().item()
    print(f"✓ Target expert (highest baseline): {target_expert}")
    assert baseline_util[target_expert] > 0, "Target expert should have baseline usage"

    # Apply soft penalty (no cooldown)
    layer.controller.suppress_expert(expert_id=target_expert, duration_steps=0)
    print(f"✓ Soft penalty applied to expert {target_expert} (no cooldown)")

    # Run forward passes with soft penalty
    penalized_utils = []
    for _ in range(20):
        layer.current_step += 1
        output, metadata = layer(x)
        penalized_utils.append(metadata['expert_utilization'].clone())

    penalized_util = torch.stack(penalized_utils).float().mean(dim=0)
    print(f"✓ Penalized utilization (20 passes): {penalized_util.tolist()}")

    # Verify target expert utilization decreased but not zero
    print(f"✓ Target expert {target_expert}: baseline={baseline_util[target_expert]:.1f}, penalized={penalized_util[target_expert]:.1f}")
    assert penalized_util[target_expert] < baseline_util[target_expert], \
        "Soft penalty should reduce utilization"

    # Note: We don't assert > 0 because with a large enough penalty (10.0),
    # the expert might receive zero tokens. The key is it's not HARD blocked
    # (no -inf mask), just heavily discouraged.

    print("\n✓ TEST 24 PASSED: Soft penalty reduces utilization correctly\n")


def test_active_mask_absolute():
    """
    Test 25: Active Mask is Absolute (Suppression Can't Revive Inactive Experts)

    CRITICAL: Suppression adjustments applied AFTER active_mask.
    Penalties cannot revive inactive experts.
    """
    print("\n" + "=" * 70)
    print("TEST 25: Active Mask is Absolute (No Inactive Expert Revival)")
    print("=" * 70)

    config = Config()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Initial state: 4 active experts (0-3), 4 inactive (4-7)
    active_mask = layer.registry.active_mask
    print(f"✓ Active experts: {[i for i in range(8) if active_mask[i]]}")
    print(f"✓ Inactive experts: {[i for i in range(8) if not active_mask[i]]}")

    # Try to suppress inactive expert (should fail at forward assertion)
    inactive_expert = 4
    assert not active_mask[inactive_expert], "Expert 4 should be inactive"

    # Apply suppression to inactive expert
    layer.controller.suppress_expert(expert_id=inactive_expert, duration_steps=100)

    x = torch.randn(2, 10, config.n_embd)

    try:
        output, metadata = layer(x)
        # If we get here, assertion didn't fire (BUG)
        assert False, "Expected assertion error for suppressing inactive expert"
    except AssertionError as e:
        if "cannot block inactive expert" in str(e):
            print(f"✓ Assertion fired correctly: {e}")
            print("✓ Active mask is absolute - suppression rejected")
        else:
            # Different assertion, re-raise
            raise

    # Clean up and verify with normal forward pass (no suppression)
    del layer.controller.expert_penalties[inactive_expert]
    del layer.controller.expert_cooldowns[inactive_expert]

    output, metadata = layer(x)
    utilization = metadata['expert_utilization']
    assert utilization[inactive_expert] == 0, \
        f"Inactive expert {inactive_expert} should never receive tokens"
    print(f"✓ Inactive expert {inactive_expert} utilization: {utilization[inactive_expert]} (expected 0)")

    print("\n✓ TEST 25 PASSED: Active mask is absolute, penalties cannot revive inactive experts\n")


if __name__ == "__main__":
    test_hard_block_prevents_routing()
    test_soft_penalty_reduces_utilization()
    test_active_mask_absolute()

    print("=" * 70)
    print("✓ ALL ROUTER INTEGRATION TESTS PASSED (3/3)")
    print("=" * 70)
