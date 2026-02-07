"""
Tests for bimodality detection (Phase 3).
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.bimodality import BimodalityState, BimodalityDetector


def test_bimodality_state_init():
    """Test BimodalityState initialization."""
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    assert state.expert_id == "L0_E0"
    assert state.layer_id == 0
    assert state.d_model == 64
    assert state.count_a == 0
    assert state.count_b == 0
    assert state.alpha == 0.95

    print("✓ BimodalityState initialization")


def test_single_mode_no_bimodality():
    """Test that single-mode expert shows no bimodality."""
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    # Feed same mode repeatedly (with small noise)
    torch.manual_seed(42)
    base_direction = torch.randn(64)
    base_direction = base_direction / base_direction.norm()

    for _ in range(50):
        # Same direction with small noise
        y = base_direction + torch.randn(64) * 0.1
        state.update(y)

    separation = state.compute_separation()
    balance = state.compute_balance()
    score = state.compute_bimodality_score()

    # Should have low separation (centroids similar)
    assert separation < 0.3, f"Expected low separation, got {separation}"
    # Score should be low (either low separation or low balance)
    assert score < 0.2, f"Expected low score, got {score}"

    print(
        f"✓ Single mode: separation={separation:.3f}, balance={balance:.3f}, score={score:.3f}"
    )


def test_two_modes_high_bimodality():
    """Test that two-mode expert shows high bimodality."""
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    # Define two distinct modes
    torch.manual_seed(42)
    mode_a = torch.randn(64)
    mode_a = mode_a / mode_a.norm()

    mode_b = torch.randn(64)
    mode_b = mode_b / mode_b.norm()

    # Ensure modes are different
    cos_sim = torch.nn.functional.cosine_similarity(
        mode_a.unsqueeze(0), mode_b.unsqueeze(0)
    )
    print(f"  Mode similarity: {cos_sim.item():.3f}")

    # Alternate between modes (balanced)
    for i in range(100):
        if i % 2 == 0:
            y = mode_a + torch.randn(64) * 0.1  # Mode A with noise
        else:
            y = mode_b + torch.randn(64) * 0.1  # Mode B with noise
        state.update(y)

    separation = state.compute_separation()
    balance = state.compute_balance()
    score = state.compute_bimodality_score()

    # Should have high separation (centroids different)
    assert separation > 0.3, f"Expected high separation, got {separation}"
    # Balance should be high (roughly equal assignments)
    assert balance > 0.8, f"Expected high balance, got {balance}"
    # Score should be high
    assert score > 0.3, f"Expected high score, got {score}"

    print(
        f"✓ Two modes: separation={separation:.3f}, balance={balance:.3f}, score={score:.3f}"
    )


def test_skewed_modes_lower_bimodality():
    """Test that skewed mode usage lowers bimodality score."""
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    # Two distinct modes
    torch.manual_seed(42)
    mode_a = torch.randn(64)
    mode_a = mode_a / mode_a.norm()

    mode_b = torch.randn(64)
    mode_b = mode_b / mode_b.norm()

    # Feed mostly mode A, occasionally mode B (skewed 90/10)
    for i in range(100):
        if i % 10 == 0:
            y = mode_b + torch.randn(64) * 0.1  # Mode B (10%)
        else:
            y = mode_a + torch.randn(64) * 0.1  # Mode A (90%)
        state.update(y)

    separation = state.compute_separation()
    balance = state.compute_balance()
    score = state.compute_bimodality_score()

    # Separation should be high (modes different)
    assert separation > 0.3, f"Expected high separation, got {separation}"
    # Balance should be low (skewed toward mode A)
    assert balance < 0.3, f"Expected low balance (skewed), got {balance}"
    # Score should be moderate (high sep × low balance)
    assert 0.05 < score < 0.3, f"Expected moderate score, got {score}"

    print(
        f"✓ Skewed modes: separation={separation:.3f}, balance={balance:.3f}, score={score:.3f}"
    )


def test_bimodality_detector_init():
    """Test BimodalityDetector initialization."""
    detector = BimodalityDetector(
        layer_id=0,
        num_experts=8,
        d_model=64,
        device="cpu",
        split_threshold=0.3,
        min_observations=100,
    )

    assert detector.layer_id == 0
    assert detector.num_experts == 8
    assert len(detector.states) == 8
    assert detector.split_threshold == 0.3
    assert detector.min_observations == 100

    print("✓ BimodalityDetector initialization")


def test_detector_update_and_snapshot():
    """Test updating detector and taking snapshots."""
    detector = BimodalityDetector(
        layer_id=0, num_experts=4, d_model=64, device="cpu"
    )

    # Update expert 0 a few times
    torch.manual_seed(42)
    for _ in range(10):
        y = torch.randn(64)
        detector.update(expert_id=0, y_expert_mean=y)

    # Check state updated
    state = detector.states[0]
    assert state.count_a + state.count_b == 10

    # Take snapshot
    snapshot = detector.snapshot()
    assert len(snapshot) == 4
    assert 0 in snapshot

    print("✓ Detector update and snapshot")


def test_split_candidate_detection():
    """Test detection of split candidates."""
    detector = BimodalityDetector(
        layer_id=0,
        num_experts=4,
        d_model=64,
        device="cpu",
        split_threshold=0.3,
        min_observations=50,
    )

    # Expert 0: Bimodal (should be candidate)
    torch.manual_seed(42)
    mode_a = torch.randn(64)
    mode_a = mode_a / mode_a.norm()
    mode_b = -mode_a  # Opposite direction

    for i in range(100):
        if i % 2 == 0:
            y = mode_a + torch.randn(64) * 0.1
        else:
            y = mode_b + torch.randn(64) * 0.1
        detector.update(expert_id=0, y_expert_mean=y)

    # Expert 1: Unimodal (should not be candidate)
    for _ in range(100):
        y = torch.randn(64)
        detector.update(expert_id=1, y_expert_mean=y)

    # Expert 2: Not enough observations (should not be candidate)
    for _ in range(20):
        y = mode_a + torch.randn(64) * 0.1
        detector.update(expert_id=2, y_expert_mean=y)

    # Detect candidates
    candidates = detector.detect_split_candidates()

    # Should detect expert 0 only
    assert 0 in candidates, "Expert 0 should be split candidate"
    assert 1 not in candidates, "Expert 1 should not be split candidate (unimodal)"
    assert 2 not in candidates, "Expert 2 should not be split candidate (not enough obs)"

    print(f"✓ Split candidate detection: {len(candidates)} candidate(s) detected")


def test_statistics_reporting():
    """Test statistics reporting."""
    detector = BimodalityDetector(
        layer_id=0, num_experts=2, d_model=64, device="cpu"
    )

    # Update expert 0
    torch.manual_seed(42)
    for _ in range(50):
        y = torch.randn(64)
        detector.update(expert_id=0, y_expert_mean=y)

    # Get statistics
    stats = detector.get_statistics(expert_id=0)

    assert "expert_id" in stats
    assert "separation" in stats
    assert "balance" in stats
    assert "bimodality_score" in stats
    assert "count_a" in stats
    assert "count_b" in stats

    print(f"✓ Statistics reporting: {stats['expert_id']}, score={stats['bimodality_score']:.3f}")


def test_reset_counts():
    """Test resetting assignment counts."""
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    # Update a few times
    torch.manual_seed(42)
    for _ in range(10):
        y = torch.randn(64)
        state.update(y)

    # Counts should be non-zero
    assert state.count_a + state.count_b > 0

    # Reset
    state.reset_counts()

    # Counts should be zero
    assert state.count_a == 0
    assert state.count_b == 0

    print("✓ Reset counts")


def test_false_coherence_scenario():
    """
    Test the key scenario: expert with high average coherence
    but serving two incompatible basins.
    """
    state = BimodalityState(
        expert_id="L0_E0",
        layer_id=0,
        d_model=64,
        centroid_a=torch.zeros(64),
        centroid_b=torch.zeros(64),
    )

    # Two modes with opposite directions (maximum incompatibility)
    torch.manual_seed(42)
    mode_a = torch.randn(64)
    mode_a = mode_a / mode_a.norm()
    mode_b = -mode_a  # Opposite

    # Simulate expert alternating between basins
    coherence_values = []
    mixture_direction = torch.randn(64)
    mixture_direction = mixture_direction / mixture_direction.norm()

    for i in range(100):
        if i % 2 == 0:
            y = mode_a
        else:
            y = mode_b

        # Compute instantaneous coherence (cosine similarity to mixture)
        coherence = torch.nn.functional.cosine_similarity(
            y.unsqueeze(0), mixture_direction.unsqueeze(0)
        ).item()
        coherence_values.append(coherence)

        # Update bimodality state
        state.update(y)

    # Average coherence might be moderate (averaging over opposites)
    avg_coherence = sum(coherence_values) / len(coherence_values)

    # But bimodality score should be high
    score = state.compute_bimodality_score()

    print(f"\n  Average coherence: {avg_coherence:.3f} (could look healthy)")
    print(f"  Bimodality score: {score:.3f} (reveals pathology)")

    # Key insight: bimodality detector catches what coherence average misses
    assert score > 0.5, "High bimodality should be detected"

    print("✓ False coherence scenario: bimodality detector reveals pathology")


if __name__ == "__main__":
    print("\nTesting Bimodality Detection (Phase 3)\n")

    test_bimodality_state_init()
    test_single_mode_no_bimodality()
    test_two_modes_high_bimodality()
    test_skewed_modes_lower_bimodality()
    test_bimodality_detector_init()
    test_detector_update_and_snapshot()
    test_split_candidate_detection()
    test_statistics_reporting()
    test_reset_counts()
    test_false_coherence_scenario()

    print("\n✅ All Phase 3 tests passed!")
    print("\nKey validation:")
    print("  ✓ Unimodal experts: low bimodality score")
    print("  ✓ Bimodal experts: high bimodality score")
    print("  ✓ Skewed modes: reduced by balance term")
    print("  ✓ False coherence: detector reveals pathology")
    print()
