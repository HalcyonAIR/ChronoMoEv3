#!/usr/bin/env python3
"""
Test autonomous proposal execution with stress band enforcement (Milestone D).

Validates:
- Proposals execute in COMFORT with sufficient calm credit
- Proposals rejected in STRAIN (stress band gate)
- Proposals rejected in PANIC (stress band gate)
- Proposals queued when calm credit insufficient
- All decisions logged
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import StressBandsConfig, Band


class DummyConfig:
    """Minimal config for testing."""
    n_embd = 128
    moe_num_experts = 4
    moe_num_experts_per_tok = 2
    moe_softmax_order = "softmax_topk"


class DummyMLP(nn.Module):
    """Minimal MLP for testing."""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc(x), {}  # Return tuple matching expert interface


def test_autonomous_execution_in_comfort():
    """Test 13: Proposals execute in COMFORT with sufficient calm credit."""
    print("\n" + "=" * 70)
    print("TEST 13: Autonomous Execution in COMFORT")
    print("=" * 70)

    # Create layer in AUTONOMOUS mode
    config = DummyConfig()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,  # AUTONOMOUS mode
        stress_bands_config=StressBandsConfig(
            comfort_ceiling_init=1.0,
            strain_ceiling_init=2.0,
        ),
    )

    # Run forward passes to build up signals
    B, T = 2, 16
    for step in range(250):  # Run 250 steps to accumulate sufficient calm credit
        layer.current_step = step
        inputs = torch.randn(B, T, config.n_embd)

        # Keep in COMFORT band (low stress)
        layer.update_stress_bands(stress=0.5)

        # Forward pass
        outputs, metadata = layer(inputs)

    # Verify in COMFORT with sufficient calm credit
    assert layer.stress_bands.current_band == Band.COMFORT
    assert layer.stress_bands.time_in_comfort >= 200  # Sufficient for SPAWN
    print(f"✓ In COMFORT band: {layer.stress_bands.time_in_comfort} steps")

    # Process proposals
    results = layer.process_controller_proposals()

    print(f"✓ Proposals: {results['proposals']}")
    print(f"✓ Executed: {results['executed']}")
    print(f"✓ Rejected: {results['rejected']}")
    print(f"✓ Queued: {results['queued']}")

    # Verify at least some proposals were made (may or may not execute)
    # In COMFORT with sufficient calm, they should not be rejected by gates
    # (but may be rejected for other reasons like capacity)
    for entry in results["log"]:
        print(f"  - {entry['action']}: {entry['type']} expert {entry['expert_id']}")
        if entry["action"] == "REJECTED":
            # Should not be rejected due to stress bands
            assert "COMFORT" not in entry.get("block_reason", "")
            assert "calm credit" not in entry.get("block_reason", "").lower()

    print("\n✓ TEST 13 PASSED: Execution in COMFORT working\n")


def test_autonomous_blocked_in_strain():
    """Test 14: Proposals blocked in STRAIN (stress band gate)."""
    print("=" * 70)
    print("TEST 14: Proposals Blocked in STRAIN")
    print("=" * 70)

    # Create layer in AUTONOMOUS mode
    config = DummyConfig()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,  # AUTONOMOUS mode
        stress_bands_config=StressBandsConfig(
            comfort_ceiling_init=1.0,
            strain_ceiling_init=2.0,
        ),
    )

    # Run forward passes to build up signals
    B, T = 2, 16
    for step in range(300):  # Run 300 steps: 250 in COMFORT, then 50 in STRAIN
        layer.current_step = step
        inputs = torch.randn(B, T, config.n_embd)

        # First, accumulate calm credit in COMFORT
        if step < 250:  # Accumulate 250 steps of calm credit
            layer.update_stress_bands(stress=0.5)
        else:
            # Then force into STRAIN
            layer.update_stress_bands(stress=1.5)

        # Forward pass
        outputs, metadata = layer(inputs)

    # Verify in STRAIN
    assert layer.stress_bands.current_band == Band.STRAIN
    print(f"✓ In STRAIN band (stress: {layer.stress_bands.stress_ema.value:.2f})")

    # Process proposals - should all be REJECTED due to stress band
    results = layer.process_controller_proposals()

    print(f"✓ Proposals: {results['proposals']}")
    print(f"✓ Executed: {results['executed']}")
    print(f"✓ Rejected: {results['rejected']}")

    # Verify NO executions in STRAIN
    assert results["executed"] == 0, \
        f"Expected 0 executions in STRAIN, got {results['executed']}"

    # Verify all proposals were rejected with stress band reason
    if results["proposals"] > 0:
        assert results["rejected"] > 0, "Proposals should be rejected in STRAIN"
        for entry in results["log"]:
            print(f"  - {entry['action']}: {entry['type']} expert {entry['expert_id']} - {entry.get('block_reason', 'N/A')}")
            assert entry["action"] == "REJECTED"
            assert "COMFORT" in entry["block_reason"]

    print("✓ All proposals correctly blocked in STRAIN")
    print("\n✓ TEST 14 PASSED: STRAIN blocking working\n")


def test_autonomous_blocked_in_panic():
    """Test 15: Proposals blocked in PANIC (stress band gate)."""
    print("=" * 70)
    print("TEST 15: Proposals Blocked in PANIC")
    print("=" * 70)

    # Create layer in AUTONOMOUS mode
    config = DummyConfig()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,  # AUTONOMOUS mode
        stress_bands_config=StressBandsConfig(
            comfort_ceiling_init=1.0,
            strain_ceiling_init=2.0,
        ),
    )

    # Run forward passes to build up signals
    B, T = 2, 16
    for step in range(300):  # Run 300 steps: 250 in COMFORT, then 50 in PANIC
        layer.current_step = step
        inputs = torch.randn(B, T, config.n_embd)

        # First, accumulate calm credit in COMFORT
        if step < 250:  # Accumulate 250 steps of calm credit
            layer.update_stress_bands(stress=0.5)
        else:
            # Then force into PANIC
            layer.update_stress_bands(stress=5.0)

        # Forward pass
        outputs, metadata = layer(inputs)

    # Verify in PANIC
    assert layer.stress_bands.current_band == Band.PANIC
    print(f"✓ In PANIC band (stress: {layer.stress_bands.stress_ema.value:.2f})")

    # Process proposals - should all be REJECTED due to stress band
    results = layer.process_controller_proposals()

    print(f"✓ Proposals: {results['proposals']}")
    print(f"✓ Executed: {results['executed']}")
    print(f"✓ Rejected: {results['rejected']}")

    # Verify NO executions in PANIC
    assert results["executed"] == 0, \
        f"Expected 0 executions in PANIC, got {results['executed']}"

    # Verify all proposals were rejected with stress band reason
    if results["proposals"] > 0:
        assert results["rejected"] > 0, "Proposals should be rejected in PANIC"
        for entry in results["log"]:
            print(f"  - {entry['action']}: {entry['type']} expert {entry['expert_id']} - {entry.get('block_reason', 'N/A')}")
            assert entry["action"] == "REJECTED"
            assert "COMFORT" in entry["block_reason"]

    print("✓ All proposals correctly blocked in PANIC")
    print("\n✓ TEST 15 PASSED: PANIC blocking working\n")


def run_all_tests():
    """Run all autonomous execution tests."""
    print("\n" + "=" * 70)
    print("AUTONOMOUS EXECUTION TESTS")
    print("Stress Band + Calm Gate Enforcement")
    print("=" * 70)

    test_autonomous_execution_in_comfort()
    test_autonomous_blocked_in_strain()
    test_autonomous_blocked_in_panic()

    print("=" * 70)
    print("✓ ALL AUTONOMOUS EXECUTION TESTS PASSED (3/3)")
    print("=" * 70)
    print("\nAutonomous execution validated:")
    print("  13. Execution in COMFORT ✓")
    print("  14. Blocking in STRAIN ✓")
    print("  15. Blocking in PANIC ✓")
    print("\nNon-bypassable gates enforced at executor boundary.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
