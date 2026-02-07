"""
Tests for lifecycle coordinator (Step 5).
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    LifecycleCoordinator,
    PruneDecision,
    compute_neff,
    compute_saturation,
)
from chronomoe_v3.coherence import CoherenceState


def test_lifecycle_coordinator_init():
    """Test LifecycleCoordinator initialization."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        min_tokens_threshold=1000,
        starvation_threshold=0.5,
    )

    assert coordinator.prune_threshold == 0.3
    assert coordinator.min_tokens_threshold == 1000
    assert coordinator.starvation_threshold == 0.5
    assert len(coordinator.decisions) == 0

    print("✓ LifecycleCoordinator initialization")


def test_prune_detection_low_coherence():
    """Test detection of low coherence experts."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        starvation_threshold=0.4,  # Lower so layer isn't starving
    )

    # Create mock coherence snapshot
    # Expert 0: Low coherence, observed enough
    # Expert 1: High coherence (dominates, keeps layer above starvation)
    # Expert 2: Low coherence, not observed enough
    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.2,  # Below threshold
            phi_fast=0.15,
            total_tokens_seen=1000,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,  # Above threshold (keeps layer healthy)
            phi_fast=0.82,
            total_tokens_seen=5000,  # Much more observed
        ),
        2: CoherenceState(
            expert_id="L0_E2",
            layer_id=0,
            d_model=64,
            phi_slow=0.1,  # Below threshold
            phi_fast=0.08,
            total_tokens_seen=100,  # Not enough observations
        ),
    }

    # Evaluate
    decisions = coordinator.evaluate_layer(0, snapshot, step=1000)

    # Should detect expert 0 only (expert 2 not observed enough)
    assert len(decisions) == 1
    assert decisions[0].expert_id == "L0_E0"
    assert decisions[0].reason == "low_coherence"
    assert decisions[0].phi_slow == 0.2

    print(f"✓ Prune detection: {len(decisions)} candidate detected")


def test_starvation_prevention():
    """Test that pruning is prevented when layer is starving."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        starvation_threshold=0.5,
    )

    # All experts have low coherence (layer starving)
    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.2,
            phi_fast=0.2,
            total_tokens_seen=2000,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.3,
            phi_fast=0.3,
            total_tokens_seen=2000,
        ),
    }

    # Evaluate
    decisions = coordinator.evaluate_layer(0, snapshot, step=1000)

    # No pruning should occur (layer starving)
    assert len(decisions) == 0

    print("✓ Starvation prevention: no pruning when layer struggling")


def test_min_tokens_filter():
    """Test that experts with insufficient observations are skipped."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        min_tokens_threshold=1000,
        starvation_threshold=0.2,  # Very low to allow pruning
    )

    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.1,  # Very low
            phi_fast=0.1,
            total_tokens_seen=500,  # Below min threshold
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.1,  # Very low
            phi_fast=0.1,
            total_tokens_seen=2000,  # Above min threshold
        ),
        2: CoherenceState(
            expert_id="L0_E2",
            layer_id=0,
            d_model=64,
            phi_slow=0.6,  # High (keeps layer above starvation)
            phi_fast=0.6,
            total_tokens_seen=3000,
        ),
    }

    decisions = coordinator.evaluate_layer(0, snapshot, step=1000)

    # Only expert 1 should be detected (expert 0 not observed enough)
    assert len(decisions) == 1
    assert decisions[0].expert_id == "L0_E1"

    print("✓ Min tokens filter: insufficient observations skipped")


def test_layer_coherence_computation():
    """Test weighted layer coherence computation."""
    coordinator = LifecycleCoordinator()

    # Expert 0: High coherence, many observations
    # Expert 1: Low coherence, few observations
    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,
            total_tokens_seen=1000,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.2,
            total_tokens_seen=100,
        ),
    }

    layer_coh = coordinator._compute_layer_coherence(snapshot)

    # Should be weighted toward expert 0 (10× more observations)
    # Expected: (0.8 * 1000 + 0.2 * 100) / 1100 ≈ 0.75
    expected = (0.8 * 1000 + 0.2 * 100) / 1100
    assert abs(layer_coh - expected) < 0.01

    print(f"✓ Layer coherence computation: {layer_coh:.3f}")


def test_decision_logging():
    """Test that decisions are logged correctly."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        starvation_threshold=0.4,  # Lower so layer isn't starving
    )

    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.2,
            phi_fast=0.15,
            total_tokens_seen=2000,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,  # High coherence (keeps layer healthy)
            phi_fast=0.82,
            total_tokens_seen=3000,
        ),
    }

    # Evaluate multiple times
    coordinator.evaluate_layer(0, snapshot, step=1000)
    coordinator.evaluate_layer(0, snapshot, step=2000)

    # Check decision log
    assert len(coordinator.decisions) == 2
    assert coordinator.decisions[0].step == 1000
    assert coordinator.decisions[1].step == 2000

    # Test statistics
    stats = coordinator.get_statistics()
    assert stats["total_decisions"] == 2
    assert stats["by_reason"]["low_coherence"] == 2

    print(f"✓ Decision logging: {stats['total_decisions']} decisions logged")


def test_compute_neff():
    """Test Neff (effective number of experts) computation."""
    # Uniform distribution → Neff ≈ num_experts
    p_uniform = torch.ones(100, 8) / 8
    neff_uniform = compute_neff(p_uniform)
    assert abs(neff_uniform - 8.0) < 0.1  # Should be close to 8

    # Concentrated distribution → Neff ≈ 1
    p_concentrated = torch.zeros(100, 8)
    p_concentrated[:, 0] = 1.0
    neff_concentrated = compute_neff(p_concentrated)
    assert abs(neff_concentrated - 1.0) < 0.1  # Should be close to 1

    # Half-and-half → Neff ≈ 2
    p_half = torch.zeros(100, 8)
    p_half[:, 0] = 0.5
    p_half[:, 1] = 0.5
    neff_half = compute_neff(p_half)
    assert abs(neff_half - 2.0) < 0.1  # Should be close to 2

    print(
        f"✓ Neff computation: uniform={neff_uniform:.1f}, "
        f"concentrated={neff_concentrated:.1f}, half={neff_half:.1f}"
    )


def test_compute_saturation():
    """Test routing saturation computation."""
    # Uniform distribution → low saturation
    p_uniform = torch.ones(100, 8) / 8
    sat_uniform = compute_saturation(p_uniform)
    assert abs(sat_uniform - 0.125) < 0.01  # 1/8

    # One expert dominates → high saturation
    p_dominated = torch.zeros(100, 8)
    p_dominated[:, 0] = 0.9
    p_dominated[:, 1:] = 0.1 / 7
    sat_dominated = compute_saturation(p_dominated)
    assert abs(sat_dominated - 0.9) < 0.01

    print(
        f"✓ Saturation computation: uniform={sat_uniform:.3f}, "
        f"dominated={sat_dominated:.2f}"
    )


def test_prune_decision_serialization():
    """Test PruneDecision to_dict serialization."""
    decision = PruneDecision(
        expert_id="L0_E3",
        reason="low_coherence",
        phi_slow=0.2,
        phi_fast=0.15,
        phi_delta=-0.05,
        total_tokens_seen=2000,
        step=1000,
    )

    d = decision.to_dict()

    assert d["expert_id"] == "L0_E3"
    assert d["reason"] == "low_coherence"
    assert d["phi_slow"] == 0.2
    assert d["step"] == 1000

    print("✓ PruneDecision serialization")


def test_dry_run_no_execution():
    """Test that lifecycle is dry-run only (no execution)."""
    coordinator = LifecycleCoordinator(
        prune_threshold=0.3,
        starvation_threshold=0.4,  # Lower so layer isn't starving
    )

    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.1,
            phi_fast=0.1,
            total_tokens_seen=2000,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,  # High coherence (keeps layer healthy)
            phi_fast=0.8,
            total_tokens_seen=3000,
        ),
    }

    # Evaluate
    decisions = coordinator.evaluate_layer(0, snapshot, step=1000)

    # Decisions detected
    assert len(decisions) == 1

    # But snapshot unchanged (dry-run)
    assert snapshot[0].phi_slow == 0.1  # Still there
    assert len(snapshot) == 2  # Not removed (both experts still present)

    print("✓ Dry-run verified: decisions logged but not executed")


if __name__ == "__main__":
    print("\nTesting Lifecycle Coordinator (Step 5)\n")

    test_lifecycle_coordinator_init()
    test_prune_detection_low_coherence()
    test_starvation_prevention()
    test_min_tokens_filter()
    test_layer_coherence_computation()
    test_decision_logging()
    test_compute_neff()
    test_compute_saturation()
    test_prune_decision_serialization()
    test_dry_run_no_execution()

    print("\n✅ All Step 5 tests passed!")
