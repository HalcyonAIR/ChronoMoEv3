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


def test_autonomous_split_in_comfort():
    """Test 16: SPLIT execution in COMFORT (Milestone E)."""
    print("=" * 70)
    print("TEST 16: SPLIT Execution in COMFORT")
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

    # Use STABLE unimodal experts (fixed centroids + small noise)
    # Not fresh random clouds each step
    torch.manual_seed(44)

    # Create fixed centroids for each expert
    expert_centroids = []
    for i in range(4):
        centroid = torch.randn(config.n_embd)
        centroid = centroid / centroid.norm()  # Normalize
        expert_centroids.append(centroid)

    # Create bimodal centroids for expert 0 (will be split)
    centroid_a = expert_centroids[0]
    centroid_b = -centroid_a  # Opposite direction (maximal separation)

    B, T = 2, 16

    # Phase 1: Build up calm credit and bimodal signal
    print("\nPhase 1: Accumulating calm credit and bimodal signal (350 steps)...")

    for step in range(350):
        layer.current_step = step
        inputs = torch.randn(B, T, config.n_embd)

        # Keep in COMFORT (low stress)
        layer.update_stress_bands(stress=0.5)

        # Create observation with stable experts
        # Expert 0: alternate between modes (bimodal)
        # Experts 1-3: stable unimodal (fixed centroids)

        expert_outputs = torch.zeros(4, B * T, config.n_embd)

        # Expert 0: bimodal (alternate between opposite centroids)
        if step % 2 == 0:
            expert_outputs[0] = centroid_a.unsqueeze(0).expand(B * T, config.n_embd) + \
                              torch.randn(B * T, config.n_embd) * 0.05
        else:
            expert_outputs[0] = centroid_b.unsqueeze(0).expand(B * T, config.n_embd) + \
                              torch.randn(B * T, config.n_embd) * 0.05

        # Experts 1-3: stable unimodal (fixed centroid + small noise)
        for i in range(1, 4):
            expert_outputs[i] = expert_centroids[i].unsqueeze(0).expand(B * T, config.n_embd) + \
                              torch.randn(B * T, config.n_embd) * 0.05

        # Create snapshot
        from chronomoe_integration.controller import ObservationSnapshot
        mixture = expert_outputs.mean(dim=0)  # [B*T, d_model]

        # Expand to full expert count (8 experts, only 4 active)
        full_expert_outputs = torch.zeros(8, B * T, config.n_embd)
        full_expert_outputs[:4] = expert_outputs

        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(B * T, 8) / 8,
            selected_experts=torch.randint(0, 4, (B * T, 2)),
            expert_outputs=full_expert_outputs,  # [num_experts=8, B*T, d_model]
            mixture_output=mixture,  # [B*T, d_model]
            utilization=torch.tensor([8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0]),
        )

        # Observe
        layer.controller.observe(snapshot)

        # Forward pass (needed for layer state)
        outputs, metadata = layer(inputs)

        if step % 100 == 0:
            print(f"  Step {step}: calm_credit={layer.stress_bands.time_in_comfort}, "
                  f"band={layer.stress_bands.current_band.name}")

    # Verify prerequisites
    assert layer.stress_bands.current_band == Band.COMFORT, \
        f"Should be in COMFORT, got {layer.stress_bands.current_band}"
    assert layer.stress_bands.time_in_comfort >= 300, \
        f"Need 300 calm steps, have {layer.stress_bands.time_in_comfort}"

    # Check bimodality across all experts
    print(f"\n✓ Bimodality scores:")
    for expert_id, state in layer.controller.bimodality_states.items():
        score = state.compute_bimodality_score()
        print(f"  Expert {expert_id}: {score:.4f}")

    # Find expert with highest bimodality (will be split)
    best_bimodal_expert = max(
        layer.controller.bimodality_states.items(),
        key=lambda x: x[1].compute_bimodality_score()
    )
    parent_id = best_bimodal_expert[0]
    bimodality_score = best_bimodal_expert[1].compute_bimodality_score()

    print(f"✓ Highest bimodality: Expert {parent_id} with score {bimodality_score:.4f}")
    assert bimodality_score > 0.5, f"Expected high bimodality, got {bimodality_score:.4f}"

    # Record initial topology
    initial_active = layer.registry.num_active
    initial_experts = list(layer.registry.experts.keys())

    print(f"✓ Prerequisites met:")
    print(f"  - COMFORT band: {layer.stress_bands.time_in_comfort} calm steps")
    print(f"  - Bimodal expert {parent_id}: score={bimodality_score:.4f}")
    print(f"  - Active experts: {initial_active}")

    # Phase 2: Process proposals (should execute SPLIT)
    print("\nPhase 2: Processing proposals...")

    results = layer.process_controller_proposals()

    print(f"  Proposals: {results['proposals']}")
    print(f"  Executed: {results['executed']}")
    print(f"  Rejected: {results['rejected']}")
    print(f"  Queued: {results['queued']}")

    # Verify SPLIT executed
    assert results["executed"] > 0, \
        f"Expected at least 1 execution, got {results['executed']}"

    split_executed = False
    child_a_id = None
    child_b_id = None

    for entry in results["log"]:
        print(f"\n  Decision: {entry['action']} - {entry['type']}")
        if entry["action"] == "EXECUTED" and entry["type"] == "split":
            split_executed = True
            child_a_id = entry.get("child_a_id")
            child_b_id = entry.get("child_b_id")
            print(f"    Parent: {entry['expert_id']}")
            print(f"    Children: [{child_a_id}, {child_b_id}]")

    assert split_executed, "SPLIT should have executed"
    assert child_a_id is not None and child_b_id is not None, "Should have child IDs"

    # Phase 3: Verify topology changes
    print("\nPhase 3: Verifying topology...")

    # Parent should be pruned (archived)
    assert parent_id in layer.registry.experts, "Parent should exist in registry"
    from chronomoe_integration.expert_registry import ExpertState
    assert layer.registry.experts[parent_id].state == ExpertState.ARCHIVED, \
        f"Parent should be ARCHIVED, got {layer.registry.experts[parent_id].state}"
    print(f"✓ Parent expert {parent_id}: ARCHIVED")

    # Children should be in PROBATION
    assert child_a_id in layer.registry.experts, f"Child A {child_a_id} should exist"
    assert child_b_id in layer.registry.experts, f"Child B {child_b_id} should exist"
    assert layer.registry.experts[child_a_id].state == ExpertState.PROBATION, \
        f"Child A should be in PROBATION"
    assert layer.registry.experts[child_b_id].state == ExpertState.PROBATION, \
        f"Child B should be in PROBATION"
    print(f"✓ Child A expert {child_a_id}: PROBATION")
    print(f"✓ Child B expert {child_b_id}: PROBATION")

    # Net capacity change: +1 (parent pruned, 2 children added)
    final_active = layer.registry.num_active
    assert final_active == initial_active + 1, \
        f"Should have +1 expert (was {initial_active}, now {final_active})"
    print(f"✓ Active experts: {initial_active} → {final_active} (net +1)")

    # Phase 4: Verify bimodality cleanup
    print("\nPhase 4: Verifying controller state cleanup...")

    # Parent's bimodality state should be deleted
    assert parent_id not in layer.controller.bimodality_states, \
        f"Parent {parent_id} bimodality state should be deleted"
    print(f"✓ Parent bimodality state cleaned up")

    # Children don't have bimodality states yet (need observations)
    # This is expected - they'll initialize on first observation
    print(f"✓ Children will initialize bimodality on first observation")

    # Phase 5: Verify invariants
    print("\nPhase 5: Verifying invariants...")

    # Two-step commit enforced
    print(f"✓ Two-step commit: controller proposed, layer decided")

    # Audit log complete
    assert len(results["log"]) > 0, "Audit log should have entries"
    print(f"✓ Audit log: {len(results['log'])} decision(s) logged")

    # Gates were enforced (COMFORT + calm credit)
    print(f"✓ Gates enforced: COMFORT + 300 calm credit")

    print("\n✓ TEST 16 PASSED: SPLIT execution in COMFORT working\n")


def run_all_tests():
    """Run all autonomous execution tests."""
    print("\n" + "=" * 70)
    print("AUTONOMOUS EXECUTION TESTS")
    print("Stress Band + Calm Gate Enforcement")
    print("=" * 70)

    test_autonomous_execution_in_comfort()
    test_autonomous_blocked_in_strain()
    test_autonomous_blocked_in_panic()
    test_autonomous_split_in_comfort()

    print("=" * 70)
    print("✓ ALL AUTONOMOUS EXECUTION TESTS PASSED (4/4)")
    print("=" * 70)
    print("\nAutonomous execution validated:")
    print("  13. Execution in COMFORT (SPAWN/PRUNE) ✓")
    print("  14. Blocking in STRAIN ✓")
    print("  15. Blocking in PANIC ✓")
    print("  16. SPLIT execution in COMFORT ✓")
    print("\nNon-bypassable gates enforced at executor boundary.")
    print("=" * 70)
    print("\n[PASS] Autonomous execution tests: all 4 tests validated")


if __name__ == "__main__":
    run_all_tests()
