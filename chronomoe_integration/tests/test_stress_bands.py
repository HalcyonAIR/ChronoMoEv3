#!/usr/bin/env python3
"""
Test stress bands and calm gate enforcement.

Validates that lifecycle operations respect stress bands and calm credit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from dataclasses import dataclass

from chronomoe_integration import (
    ChronoMoE,
    ProbationConfig,
    StressBandsConfig,
    Band,
    step_stress_bands,
    init_stress_bands,
)


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


def test_stress_band_classification():
    """Test 1: Stress band classification with hysteresis."""
    print("\n" + "=" * 70)
    print("TEST 1: Stress Band Classification")
    print("=" * 70)

    config = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
        hysteresis_margin=0.05,
    )

    state = init_stress_bands(config)

    # Test comfort → strain transition
    result = step_stress_bands(state, config, stress=1.2)
    assert result.band_now == Band.STRAIN, f"Expected STRAIN at stress=1.2, got {result.band_now}"
    print(f"✓ Comfort → Strain at stress=1.2")

    # Test strain → panic transition (need multiple steps for EMA to reach panic threshold)
    for _ in range(20):
        result = step_stress_bands(state, config, stress=2.5)
    assert result.band_now == Band.PANIC, f"Expected PANIC after sustained stress=2.5, got {result.band_now}"
    print(f"✓ Strain → Panic at sustained stress=2.5")

    # Test hysteresis (panic → strain requires stress < 2.0 * 0.95 = 1.9)
    # EMA needs time to drop, so apply sustained lower stress
    for _ in range(10):
        result = step_stress_bands(state, config, stress=1.95)
    # EMA is now ~1.95, still in panic due to hysteresis
    assert result.band_now == Band.PANIC, f"Expected PANIC (hysteresis) at sustained stress=1.95, got {result.band_now}"
    print(f"✓ Hysteresis: Still panic at sustained stress=1.95")

    # Lower stress below hysteresis threshold
    for _ in range(10):
        result = step_stress_bands(state, config, stress=1.7)
    assert result.band_now == Band.STRAIN, f"Expected STRAIN at sustained stress=1.7, got {result.band_now}"
    print(f"✓ Panic → Strain at sustained stress=1.7 (below hysteresis threshold)")

    print("\n✓ TEST 1 PASSED: Stress band classification working\n")


def test_calm_credit_accumulation():
    """Test 2: Calm credit accumulates in comfort band."""
    print("=" * 70)
    print("TEST 2: Calm Credit Accumulation")
    print("=" * 70)

    config = StressBandsConfig(
        comfort_ceiling_init=1.0,
        spawn_calm_steps=10,
    )

    state = init_stress_bands(config)

    # Stay in comfort for 5 steps
    for _ in range(5):
        step_stress_bands(state, config, stress=0.5)

    assert state.time_in_comfort == 5, f"Expected 5 steps in comfort, got {state.time_in_comfort}"
    print(f"✓ Calm credit: {state.time_in_comfort} steps after 5 comfort steps")

    # Enter strain (calm credit resets) - need sustained stress for EMA
    for _ in range(15):
        step_stress_bands(state, config, stress=1.5)
    assert state.current_band == Band.STRAIN, f"Expected STRAIN, got {state.current_band}"
    assert state.time_in_comfort == 0, f"Expected calm credit reset to 0, got {state.time_in_comfort}"
    print(f"✓ Calm credit reset to 0 when entering strain")

    # Return to comfort (EMA needs time to drop, so actual comfort steps will be less than 12)
    for i in range(12):
        step_stress_bands(state, config, stress=0.5)

    assert state.time_in_comfort > 0, f"Expected some calm credit after low stress, got {state.time_in_comfort}"
    assert state.current_band == Band.COMFORT, f"Expected COMFORT band, got {state.current_band}"
    print(f"✓ Calm credit: {state.time_in_comfort} steps after returning to comfort (EMA lag accounted for)")

    print("\n✓ TEST 2 PASSED: Calm credit accumulation working\n")


def test_spawn_calm_gate():
    """Test 3: Spawn blocked until calm credit sufficient."""
    print("=" * 70)
    print("TEST 3: Spawn Calm Gate")
    print("=" * 70)

    config = MockConfig()
    stress_config = StressBandsConfig(
        comfort_ceiling_init=1.0,
        spawn_calm_steps=10,  # Need 10 steps in comfort
    )

    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
        stress_bands_config=stress_config,
    )

    # Try spawn with 0 calm credit (should be blocked)
    layer.current_step = 100
    result = layer.spawn_expert(parent_id=0, strategy="blank", check_calm_gate=True)

    assert result is None, "Spawn should be blocked with 0 calm credit"
    print(f"✓ Spawn blocked with 0 calm credit")

    # Build calm credit by staying in comfort
    for i in range(12):
        layer.update_stress_bands(stress=0.5)

    # Try spawn again (should succeed)
    result = layer.spawn_expert(parent_id=0, strategy="blank", check_calm_gate=True)

    assert result is not None, "Spawn should succeed with sufficient calm credit"
    assert result == 4, f"Expected expert ID 4, got {result}"
    print(f"✓ Spawn allowed after {layer.stress_bands.time_in_comfort} steps in comfort")

    print("\n✓ TEST 3 PASSED: Spawn calm gate working\n")


def test_prune_calm_gate():
    """Test 4: Prune blocked in strain band."""
    print("=" * 70)
    print("TEST 4: Prune Calm Gate")
    print("=" * 70)

    config = MockConfig()
    stress_config = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
        prune_calm_steps=10,
    )

    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
        stress_bands_config=stress_config,
    )

    # Enter strain band
    layer.update_stress_bands(stress=1.5)
    assert layer.stress_bands.current_band == Band.STRAIN

    # Try prune (should be blocked in strain)
    pruned = layer.prune_expert(expert_id=0, check_calm_gate=True)

    assert not pruned, "Prune should be blocked in strain band"
    print(f"✓ Prune blocked in strain band")

    # Return to comfort and build calm credit (need more steps for EMA to drop)
    for _ in range(25):
        layer.update_stress_bands(stress=0.5)

    assert layer.stress_bands.current_band == Band.COMFORT, \
        f"Expected COMFORT, got {layer.stress_bands.current_band}"

    # Try prune again (should succeed)
    pruned = layer.prune_expert(expert_id=0, check_calm_gate=True)

    assert pruned, "Prune should succeed in comfort with calm credit"
    assert not layer.registry.active_mask[0], "Expert 0 should be inactive"
    print(f"✓ Prune allowed in comfort after {layer.stress_bands.time_in_comfort} steps")

    print("\n✓ TEST 4 PASSED: Prune calm gate working\n")


def test_panic_freezes_all():
    """Test 5: Panic band freezes all lifecycle operations."""
    print("=" * 70)
    print("TEST 5: Panic Freezes All Operations")
    print("=" * 70)

    config = MockConfig()
    stress_config = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
    )

    layer = ChronoMoE(
        config=config,
        mlp=MockMLP,
        layer_id=0,
        max_experts=8,
        stress_bands_config=stress_config,
    )

    # Build calm credit first
    for _ in range(20):
        layer.update_stress_bands(stress=0.5)

    # Enter panic (need sustained high stress for EMA to reach panic threshold)
    for _ in range(25):
        layer.update_stress_bands(stress=3.0)
    assert layer.stress_bands.current_band == Band.PANIC, \
        f"Expected PANIC, got {layer.stress_bands.current_band}"
    print(f"✓ Entered panic band (sustained stress=3.0)")

    # Try spawn (should be blocked)
    spawn_result = layer.spawn_expert(parent_id=0, strategy="blank", check_calm_gate=True)
    assert spawn_result is None, "Spawn should be blocked in panic"
    print(f"✓ Spawn blocked in panic")

    # Try prune (should be blocked)
    prune_result = layer.prune_expert(expert_id=0, check_calm_gate=True)
    assert not prune_result, "Prune should be blocked in panic"
    print(f"✓ Prune blocked in panic")

    print("\n✓ TEST 5 PASSED: Panic freezes all lifecycle operations\n")


def run_all_tests():
    """Run all stress bands tests."""
    print("\n" + "=" * 70)
    print("STRESS BANDS + CALM GATE VALIDATION")
    print("=" * 70)

    test_stress_band_classification()
    test_calm_credit_accumulation()
    test_spawn_calm_gate()
    test_prune_calm_gate()
    test_panic_freezes_all()

    print("=" * 70)
    print("✓ ALL STRESS BANDS TESTS PASSED (5/5)")
    print("=" * 70)
    print("\nStress bands validated:")
    print("  1. Band classification with hysteresis ✓")
    print("  2. Calm credit accumulation ✓")
    print("  3. Spawn calm gate enforcement ✓")
    print("  4. Prune calm gate enforcement ✓")
    print("  5. Panic freezes all operations ✓")
    print("\nCalm gating ready for integration.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
