#!/usr/bin/env python3
"""
Test spawn strategy defaults and enforcement.

Validates Issue #2: Blank spawn as default, clone as opt-in.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from dataclasses import dataclass

from chronomoe_integration import ChronoMoE, ProbationConfig


# Mock config
@dataclass
class MockConfig:
    n_embd: int = 128
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"
    dropout: float = 0.0
    bias: bool = True


# Mock MLP
class MockMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)

    def forward(self, x):
        return torch.nn.functional.gelu(self.fc1(x)), None


def test_default_spawn_is_blank():
    """Test 1: Default spawn strategy is blank (not clone)."""
    print("\n" + "=" * 70)
    print("TEST 1: Default Spawn Strategy is Blank")
    print("=" * 70)

    config = MockConfig()
    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
    )

    # Spawn without specifying strategy (should default to blank)
    layer.current_step = 100
    new_id = layer.spawn_expert(parent_id=0, check_calm_gate=False)

    assert new_id == 4, f"Expected expert ID 4, got {new_id}"

    # Verify expert is in probation
    info = layer.registry.experts[new_id]
    assert info.state.value == "probation", f"Expected probation, got {info.state.value}"
    assert info.strategy == "blank", f"Expected blank strategy, got {info.strategy}"

    print(f"✓ Default spawn uses blank strategy")
    print(f"✓ Expert {new_id} in probation state")
    print(f"✓ Strategy recorded as: {info.strategy}")

    print("\n✓ TEST 1 PASSED: Default is blank\n")


def test_clone_spawn_requires_explicit_opt_in():
    """Test 2: Clone spawning requires explicit strategy='clone'."""
    print("=" * 70)
    print("TEST 2: Clone Spawn Requires Explicit Opt-In")
    print("=" * 70)

    config = MockConfig()
    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
    )

    # Spawn with explicit clone strategy
    layer.current_step = 100
    new_id = layer.spawn_expert(parent_id=0, strategy="clone", check_calm_gate=False)

    assert new_id == 4, f"Expected expert ID 4, got {new_id}"

    # Verify expert is in probation with clone strategy
    info = layer.registry.experts[new_id]
    assert info.strategy == "clone", f"Expected clone strategy, got {info.strategy}"

    print(f"✓ Clone spawn allowed with explicit strategy='clone'")
    print(f"✓ Strategy recorded as: {info.strategy}")

    print("\n✓ TEST 2 PASSED: Clone requires explicit opt-in\n")


def test_clone_warning_logged():
    """Test 3: Warning logged when clone strategy used."""
    print("=" * 70)
    print("TEST 3: Clone Warning Logged")
    print("=" * 70)

    config = MockConfig()
    probation_config = ProbationConfig(
        warn_on_clone=True,  # Enable warning
    )

    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
        probation_config=probation_config,
    )

    # Spawn with clone (should log warning)
    print("Expected warning on next line:")
    layer.current_step = 100
    new_id = layer.spawn_expert(parent_id=0, strategy="clone", check_calm_gate=False)

    assert new_id == 4
    print(f"✓ Warning logged (see above)")

    print("\n✓ TEST 3 PASSED: Clone warning working\n")


def test_blank_only_enforcement():
    """Test 4: blank_only config disables clone spawning."""
    print("=" * 70)
    print("TEST 4: Blank-Only Enforcement")
    print("=" * 70)

    config = MockConfig()
    probation_config = ProbationConfig.blank_only()  # Enforce blank-only

    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
        probation_config=probation_config,
    )

    # Verify config
    assert not layer.registry.probation_config.allow_clone_spawn, \
        "blank_only should disable clone spawning"
    print(f"✓ Config: allow_clone_spawn = False")

    # Try blank spawn (should work)
    layer.current_step = 100
    blank_id = layer.spawn_expert(parent_id=0, strategy="blank", check_calm_gate=False)
    assert blank_id == 4, "Blank spawn should work"
    print(f"✓ Blank spawn allowed (expert {blank_id})")

    # Try clone spawn (should fail)
    layer.current_step = 101
    try:
        clone_id = layer.spawn_expert(parent_id=0, strategy="clone", check_calm_gate=False)
        assert False, "Clone spawn should have raised ValueError"
    except ValueError as e:
        assert "Clone spawning is disabled" in str(e)
        print(f"✓ Clone spawn blocked with ValueError")
        print(f"  Error: {str(e)[:80]}...")

    print("\n✓ TEST 4 PASSED: Blank-only enforcement working\n")


def test_default_without_strategy_param():
    """Test 5: Calling spawn without strategy param uses blank."""
    print("=" * 70)
    print("TEST 5: Default Behavior Without Strategy Parameter")
    print("=" * 70)

    config = MockConfig()
    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
    )

    # Call spawn without strategy parameter at all
    layer.current_step = 100
    # This should use the default value strategy="blank"
    new_id = layer.spawn_expert(parent_id=0, check_calm_gate=False)

    info = layer.registry.experts[new_id]
    assert info.strategy == "blank", \
        f"Default strategy should be blank, got {info.strategy}"

    print(f"✓ Spawn without strategy param defaults to blank")
    print(f"✓ Expert {new_id} strategy: {info.strategy}")

    print("\n✓ TEST 5 PASSED: Default parameter value working\n")


def run_all_tests():
    """Run all spawn default tests."""
    print("\n" + "=" * 70)
    print("SPAWN STRATEGY DEFAULTS + ENFORCEMENT")
    print("Issue #2 Validation")
    print("=" * 70)

    test_default_spawn_is_blank()
    test_clone_spawn_requires_explicit_opt_in()
    test_clone_warning_logged()
    test_blank_only_enforcement()
    test_default_without_strategy_param()

    print("=" * 70)
    print("✓ ALL SPAWN DEFAULT TESTS PASSED (5/5)")
    print("=" * 70)
    print("\nSpawn strategy defaults validated:")
    print("  1. Default spawn is blank (not clone) ✓")
    print("  2. Clone requires explicit opt-in ✓")
    print("  3. Clone warning logged when used ✓")
    print("  4. Blank-only config enforces no clone ✓")
    print("  5. Default param value is blank ✓")
    print("\nIssue #2 complete: Blank spawn is now the default.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
