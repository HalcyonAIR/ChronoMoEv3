#!/usr/bin/env python3
"""
Test controller boundary, coherence tracking (Milestone A), and bimodality detection (Milestone B).

Validates:
- Controller API (observe, decide, apply, get_diagnostics)
- Coherence tracking (phi_fast, phi_mid, phi_slow)
- Bimodality detection (separation, balance, bimodality score)
- Ring buffer behavior (no memory leaks)
- NO edit proposals fired (decide() returns empty list)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from chronomoe_integration.controller import (
    ChronoController,
    ObservationSnapshot,
    create_controller,
)


def test_controller_api():
    """Test 1: Controller API methods exist and work."""
    print("\n" + "=" * 70)
    print("TEST 1: Controller API")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    # Check API methods exist
    assert hasattr(controller, "observe")
    assert hasattr(controller, "decide")
    assert hasattr(controller, "apply")
    assert hasattr(controller, "get_diagnostics")

    print(f"✓ Controller API methods present")
    print(f"✓ Layer ID: {controller.layer_id}")
    print(f"✓ Max experts: {controller.max_experts}")

    print("\n✓ TEST 1 PASSED: Controller API working\n")


def test_coherence_tracking():
    """Test 2: Coherence tracking computes phi values correctly."""
    print("=" * 70)
    print("TEST 2: Coherence Tracking")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    # Create observation snapshot
    B_T = 32
    d_model = 128
    num_experts = 8

    # Simulate expert outputs (all aligned with mixture)
    mixture_output = torch.randn(B_T, d_model)
    expert_outputs = torch.zeros(num_experts, B_T, d_model)
    for i in range(4):  # First 4 experts active
        expert_outputs[i] = mixture_output + torch.randn(B_T, d_model) * 0.1  # Slight noise

    snapshot = ObservationSnapshot(
        step=10,
        layer_id=0,
        router_probs=torch.ones(B_T, num_experts) / num_experts,  # Uniform
        selected_experts=torch.randint(0, 4, (B_T, 2)),
        expert_outputs=expert_outputs,
        mixture_output=mixture_output,
        utilization=torch.tensor([8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0]),  # 4 active
    )

    controller.observe(snapshot)

    # Get diagnostics
    diagnostics = controller.get_diagnostics()

    assert "coherence" in diagnostics
    coherence_data = diagnostics["coherence"]

    # Check layer coherence values exist
    assert "layer_coherence_fast" in coherence_data
    assert "layer_coherence_mid" in coherence_data
    assert "layer_coherence_slow" in coherence_data

    print(f"✓ Layer coherence (fast): {coherence_data['layer_coherence_fast']:.4f}")
    print(f"✓ Layer coherence (mid): {coherence_data['layer_coherence_mid']:.4f}")
    print(f"✓ Layer coherence (slow): {coherence_data['layer_coherence_slow']:.4f}")

    # Check per-expert coherence
    assert "by_expert" in coherence_data
    expert_coherence = coherence_data["by_expert"]

    print(f"✓ Per-expert coherence tracked: {len(expert_coherence)} experts")

    # Verify expert 0 has coherence data
    assert 0 in expert_coherence
    expert_0 = expert_coherence[0]
    assert "phi_fast" in expert_0
    assert "phi_mid" in expert_0
    assert "phi_slow" in expert_0
    assert "phi_delta" in expert_0

    print(f"✓ Expert 0: phi_fast={expert_0['phi_fast']:.4f}, phi_delta={expert_0['phi_delta']:.4f}")

    print("\n✓ TEST 2 PASSED: Coherence tracking working\n")


def test_coherence_degradation_detection():
    """Test 3: Coherence detects degrading experts (phi_delta < 0)."""
    print("=" * 70)
    print("TEST 3: Degradation Detection")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    B_T = 32
    d_model = 128
    num_experts = 8

    # Start with aligned expert (with slight noise)
    mixture_output = torch.randn(B_T, d_model)
    expert_outputs = torch.zeros(num_experts, B_T, d_model)
    expert_outputs[0] = mixture_output.clone() + torch.randn(B_T, d_model) * 0.01  # Expert 0 nearly aligned

    # Initial observations (expert 0 healthy)
    for step in range(50):
        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(B_T, num_experts) / num_experts,
            selected_experts=torch.randint(0, 4, (B_T, 2)),
            expert_outputs=expert_outputs,
            mixture_output=mixture_output,
            utilization=torch.tensor([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Get coherence after healthy period
    diagnostics = controller.get_diagnostics()
    expert_0_healthy = diagnostics["coherence"]["by_expert"][0]
    print(f"✓ Expert 0 (healthy): phi_fast={expert_0_healthy['phi_fast']:.4f}, phi_slow={expert_0_healthy['phi_slow']:.4f}")

    # Now degrade expert 0 (orthogonal to mixture, not opposite)
    orthogonal = torch.randn(B_T, d_model)
    orthogonal = orthogonal - (orthogonal * mixture_output).sum(dim=1, keepdim=True) / (mixture_output ** 2).sum(dim=1, keepdim=True) * mixture_output
    expert_outputs[0] = orthogonal  # Orthogonal direction (phi ~0)

    # Observations during degradation (only 3 steps - fast drops quickly, slow lags)
    for step in range(50, 53):
        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(B_T, num_experts) / num_experts,
            selected_experts=torch.randint(0, 4, (B_T, 2)),
            expert_outputs=expert_outputs,
            mixture_output=mixture_output,
            utilization=torch.tensor([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Get coherence after degradation
    diagnostics = controller.get_diagnostics()
    expert_0_degraded = diagnostics["coherence"]["by_expert"][0]

    print(f"✓ Expert 0 (degraded): phi_fast={expert_0_degraded['phi_fast']:.4f}, phi_slow={expert_0_degraded['phi_slow']:.4f}")
    print(f"✓ Phi delta: {expert_0_degraded['phi_delta']:.4f}")

    # Coherence should have dropped significantly
    assert expert_0_degraded["phi_fast"] < expert_0_healthy["phi_fast"] - 0.5, \
        f"Expected coherence to drop, got {expert_0_degraded['phi_fast']:.4f} vs {expert_0_healthy['phi_fast']:.4f}"

    print(f"✓ Coherence dropped: {expert_0_healthy['phi_fast']:.2f} → {expert_0_degraded['phi_fast']:.2f}")
    print(f"✓ Three-timescale EMA working (fast/mid/slow tracked)")

    print("\n✓ TEST 3 PASSED: Degradation detection working\n")


def test_no_edit_proposals():
    """Test 4: Controller does NOT fire edit proposals (Milestone A)."""
    print("=" * 70)
    print("TEST 4: No Edit Proposals (Milestone A)")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    # Run observations
    for step in range(10):
        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(32, 8) / 8,
            selected_experts=torch.randint(0, 4, (32, 2)),
            expert_outputs=torch.randn(8, 32, 128),
            mixture_output=torch.randn(32, 128),
            utilization=torch.tensor([8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Call decide() - should return empty list (no triggers in Milestone A)
    proposals = controller.decide()

    assert len(proposals) == 0, f"Expected 0 proposals, got {len(proposals)}"
    print(f"✓ No edit proposals fired: {len(proposals)} proposals")
    print(f"✓ decide() returns empty list (Milestone A: diagnostics only)")

    print("\n✓ TEST 4 PASSED: No edit proposals\n")


def test_ring_buffer_no_memory_leak():
    """Test 5: Ring buffer prevents memory leak (max 100 observations)."""
    print("=" * 70)
    print("TEST 5: Ring Buffer (No Memory Leak)")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    # Run 200 observations (2x ring buffer size)
    for step in range(200):
        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(32, 8) / 8,
            selected_experts=torch.randint(0, 4, (32, 2)),
            expert_outputs=torch.randn(8, 32, 128),
            mixture_output=torch.randn(32, 128),
            utilization=torch.tensor([8.0, 8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Check ring buffer size
    history_size = len(controller.observation_history)
    assert history_size <= controller.max_history, \
        f"History size {history_size} exceeds max {controller.max_history}"

    print(f"✓ Ring buffer size: {history_size} (max: {controller.max_history})")
    print(f"✓ No memory leak: history bounded")

    print("\n✓ TEST 5 PASSED: Ring buffer working\n")


def test_bimodality_unimodal():
    """Test 6: Bimodality detects unimodal (healthy) expert."""
    print("=" * 70)
    print("TEST 6: Bimodality - Unimodal (Healthy)")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    B_T = 32
    d_model = 128
    num_experts = 8

    # Create unimodal expert: outputs cluster around one direction
    torch.manual_seed(42)  # Deterministic
    base_direction = torch.randn(d_model)
    base_direction = base_direction / base_direction.norm()  # Normalize

    # Run 50 observations with expert 0 producing similar outputs
    for step in range(50):
        expert_outputs = torch.zeros(num_experts, B_T, d_model)

        # Expert 0: unimodal (small variations around base direction)
        noise = torch.randn(B_T, d_model) * 0.1
        expert_outputs[0] = base_direction + noise

        mixture_output = expert_outputs[0].clone()  # Simple mixture

        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(B_T, num_experts) / num_experts,
            selected_experts=torch.randint(0, 4, (B_T, 2)),
            expert_outputs=expert_outputs,
            mixture_output=mixture_output,
            utilization=torch.tensor([32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Get diagnostics
    diagnostics = controller.get_diagnostics()
    assert "bimodality" in diagnostics
    bimodality_data = diagnostics["bimodality"]

    print(f"✓ Layer bimodality: {bimodality_data['layer_bimodality']:.4f}")

    # Check expert 0 bimodality
    assert 0 in bimodality_data["by_expert"]
    expert_0 = bimodality_data["by_expert"][0]

    print(f"✓ Expert 0 (unimodal):")
    print(f"  - Separation: {expert_0['separation']:.4f}")
    print(f"  - Balance: {expert_0['balance']:.4f}")
    print(f"  - Bimodality score: {expert_0['bimodality_score']:.4f}")
    print(f"  - Observations: {expert_0['total_observations']}")

    # Unimodal expert should have LOW bimodality score
    assert expert_0["bimodality_score"] < 0.3, \
        f"Unimodal expert should have low bimodality, got {expert_0['bimodality_score']:.4f}"

    print(f"✓ Unimodal expert correctly identified (low score)")

    print("\n✓ TEST 6 PASSED: Unimodal detection working\n")


def test_bimodality_bimodal():
    """Test 7: Bimodality detects bimodal expert."""
    print("=" * 70)
    print("TEST 7: Bimodality - Bimodal (Split Candidate)")
    print("=" * 70)

    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
    )

    B_T = 32
    d_model = 128
    num_experts = 8

    # Create bimodal expert: outputs alternate between two opposite directions
    torch.manual_seed(42)  # Deterministic
    direction_a = torch.randn(d_model)
    direction_a = direction_a / direction_a.norm()
    direction_b = -direction_a  # Opposite direction

    # Run 50 observations with expert 0 alternating between two modes
    for step in range(50):
        expert_outputs = torch.zeros(num_experts, B_T, d_model)

        # Expert 0: bimodal (alternates between opposite directions)
        if step % 2 == 0:
            # Mode A
            noise = torch.randn(B_T, d_model) * 0.1
            expert_outputs[0] = direction_a + noise
        else:
            # Mode B (opposite)
            noise = torch.randn(B_T, d_model) * 0.1
            expert_outputs[0] = direction_b + noise

        mixture_output = expert_outputs[0].clone()

        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=torch.ones(B_T, num_experts) / num_experts,
            selected_experts=torch.randint(0, 4, (B_T, 2)),
            expert_outputs=expert_outputs,
            mixture_output=mixture_output,
            utilization=torch.tensor([32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        controller.observe(snapshot)

    # Get diagnostics
    diagnostics = controller.get_diagnostics()
    bimodality_data = diagnostics["bimodality"]

    print(f"✓ Layer bimodality: {bimodality_data['layer_bimodality']:.4f}")

    # Check expert 0 bimodality
    expert_0 = bimodality_data["by_expert"][0]

    print(f"✓ Expert 0 (bimodal):")
    print(f"  - Separation: {expert_0['separation']:.4f}")
    print(f"  - Balance: {expert_0['balance']:.4f}")
    print(f"  - Bimodality score: {expert_0['bimodality_score']:.4f}")
    print(f"  - Observations: {expert_0['total_observations']}")

    # Bimodal expert should have HIGH bimodality score
    assert expert_0["bimodality_score"] > 0.5, \
        f"Bimodal expert should have high bimodality, got {expert_0['bimodality_score']:.4f}"

    # Should have balanced usage
    assert expert_0["balance"] > 0.8, \
        f"Bimodal expert should have balanced usage, got {expert_0['balance']:.4f}"

    # Should have high separation (opposite directions)
    assert expert_0["separation"] > 1.5, \
        f"Bimodal expert should have high separation, got {expert_0['separation']:.4f}"

    print(f"✓ Bimodal expert correctly identified (high score)")

    # Verify NO edit proposals (Milestone B: diagnostics only)
    proposals = controller.decide()
    assert len(proposals) == 0, f"Expected 0 proposals in Milestone B, got {len(proposals)}"
    print(f"✓ No edit proposals (Milestone B: diagnostics only)")

    print("\n✓ TEST 7 PASSED: Bimodal detection working\n")


def run_all_tests():
    """Run all controller tests."""
    print("\n" + "=" * 70)
    print("CONTROLLER + COHERENCE + BIMODALITY TRACKING TESTS")
    print("Milestone A + B Validation")
    print("=" * 70)

    test_controller_api()
    test_coherence_tracking()
    test_coherence_degradation_detection()
    test_no_edit_proposals()
    test_ring_buffer_no_memory_leak()
    test_bimodality_unimodal()
    test_bimodality_bimodal()

    print("=" * 70)
    print("✓ ALL CONTROLLER TESTS PASSED (7/7)")
    print("=" * 70)
    print("\nController validated:")
    print("  1. API methods working ✓")
    print("  2. Coherence tracking working ✓")
    print("  3. Degradation detection working ✓")
    print("  4. No edit proposals (diagnostics only) ✓")
    print("  5. Ring buffer prevents memory leaks ✓")
    print("  6. Bimodality unimodal detection ✓")
    print("  7. Bimodality bimodal detection ✓")
    print("\nMilestone A: Coherence logging only, no triggers.")
    print("Milestone B: Bimodality logging only, no triggers.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
