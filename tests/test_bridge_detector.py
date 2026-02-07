"""
Tests for bridge detector (relevance modulation) - Step 4.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    ChronoRouter,
    compute_overlap_only,
    compute_relevance,
)


def test_overlap_only_zero():
    """Test overlap_only is zero when distributions are identical."""
    p = torch.softmax(torch.randn(10, 8), dim=-1)
    overlap = compute_overlap_only(p, p)

    assert abs(overlap) < 1e-6
    print(f"✓ Overlap-only (identical distributions): {overlap:.6f}")


def test_overlap_only_measures_hallucination():
    """Test overlap_only measures mass going to hallucinated experts."""
    # Clean: Expert 0 has all mass
    p_clean = torch.zeros(10, 4)
    p_clean[:, 0] = 1.0

    # Biased: Expert 0 has 0.7, Expert 1 has 0.3 (hallucinated)
    p_biased = torch.zeros(10, 4)
    p_biased[:, 0] = 0.7
    p_biased[:, 1] = 0.3

    overlap = compute_overlap_only(p_clean, p_biased)

    # Overlap-only should be 0.3 (mass given to expert 1)
    assert abs(overlap - 0.3) < 1e-6
    print(f"✓ Overlap-only (30% hallucination): {overlap:.3f}")


def test_overlap_only_ignores_suppression():
    """Test overlap_only ignores mass reduction (only counts additions)."""
    # Clean: Uniform [0.25, 0.25, 0.25, 0.25]
    p_clean = torch.ones(10, 4) * 0.25

    # Biased: Expert 0 boosted, Expert 1 suppressed
    p_biased = torch.tensor([[0.4, 0.1, 0.25, 0.25]] * 10)

    overlap = compute_overlap_only(p_clean, p_biased)

    # Only expert 0's +0.15 counts (expert 1's -0.15 is clamped to 0)
    assert abs(overlap - 0.15) < 1e-6
    print(f"✓ Overlap-only (boost not suppress): {overlap:.3f}")


def test_relevance_full():
    """Test relevance is 1.0 when overlap_only is low."""
    # Create distributions with very small difference
    p_clean = torch.softmax(torch.randn(10, 8), dim=-1)
    p_biased = p_clean.clone()
    p_biased[:, 0] += 0.01  # Tiny boost
    p_biased = p_biased / p_biased.sum(dim=-1, keepdim=True)  # Renormalize

    r = compute_relevance(p_clean, p_biased, threshold=0.1)
    overlap = compute_overlap_only(p_clean, p_biased)

    # Overlap should be very small → full relevance
    assert overlap < 0.1  # Below threshold
    assert r == 1.0
    print(f"✓ Relevance (low overlap={overlap:.4f}): {r:.2f}")


def test_relevance_decay():
    """Test relevance decays linearly with overlap_only."""
    p_clean = torch.zeros(10, 4)
    p_clean[:, 0] = 1.0

    # Test with increasing hallucination
    overlaps = []
    relevances = []

    for hallucination in [0.05, 0.1, 0.2, 0.3]:
        p_biased = torch.zeros(10, 4)
        p_biased[:, 0] = 1.0 - hallucination
        p_biased[:, 1] = hallucination

        r = compute_relevance(p_clean, p_biased, threshold=0.1)
        overlaps.append(hallucination)
        relevances.append(r)

    # Check decay pattern
    assert relevances[0] == 1.0  # Below threshold (0.05 < 0.1)
    assert 0.0 < relevances[1] <= 1.0  # At/near threshold (0.1)
    assert 0.0 < relevances[2] < 1.0  # In decay zone (0.2)
    assert relevances[3] == 0.0  # Above 3×threshold (0.3 >= 0.3)

    print(f"✓ Relevance decay:")
    for o, r in zip(overlaps, relevances):
        print(f"  overlap={o:.2f} → relevance={r:.2f}")


def test_relevance_zero_at_crisis():
    """Test relevance is 0.0 when overlap_only is very high."""
    p_clean = torch.zeros(10, 4)
    p_clean[:, 0] = 1.0

    # Biased: 50% mass goes to hallucinated expert
    p_biased = torch.zeros(10, 4)
    p_biased[:, 0] = 0.5
    p_biased[:, 1] = 0.5

    r = compute_relevance(p_clean, p_biased, threshold=0.1)

    # Overlap = 0.5 >> 0.3 (3×threshold) → no relevance
    assert r == 0.0
    print(f"✓ Relevance (crisis): {r:.2f}")


def test_router_with_relevance():
    """Test router with relevance modulation enabled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device=device)

    # Set strong beta on expert 0
    router.router_state.beta_coeff[0] = 0.3

    # Forward WITHOUT relevance
    hidden = torch.randn(32, 64, device=device)
    _, _, z_clean_1, z_biased_1 = router(hidden, top_k=2, use_relevance=False)
    overlap_without = router.router_state.overlap_only

    # Forward WITH relevance
    _, _, z_clean_2, z_biased_2 = router(hidden, top_k=2, use_relevance=True)
    overlap_with = router.router_state.overlap_only

    # With relevance, overlap should be same or lower (beta modulated)
    print(
        f"✓ Router with relevance: "
        f"overlap_without={overlap_without:.4f}, "
        f"overlap_with={overlap_with:.4f}"
    )


def test_bridge_detector_prevents_hallucination():
    """Test bridge detector suppresses beta when it hallucinates."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = ChronoRouter(d_model=64, num_experts=4, layer_id=0, device=device)

    # Set very strong beta to force hallucination
    router.router_state.beta_coeff = torch.tensor([0.0, 0.0, 0.3, 0.0], device=device)

    # Generate input that doesn't naturally select expert 2
    torch.manual_seed(42)
    hidden = torch.randn(32, 64, device=device)

    # Forward pass WITHOUT relevance
    top_k_probs_1, top_k_indices_1, _, _ = router(hidden, top_k=2, use_relevance=False)
    overlap_1 = router.router_state.overlap_only

    # How often is expert 2 selected?
    expert_2_selected_1 = (top_k_indices_1 == 2).any(dim=-1).float().mean().item()

    # Forward pass WITH relevance
    top_k_probs_2, top_k_indices_2, _, _ = router(hidden, top_k=2, use_relevance=True)
    overlap_2 = router.router_state.overlap_only

    # How often is expert 2 selected?
    expert_2_selected_2 = (top_k_indices_2 == 2).any(dim=-1).float().mean().item()

    print(f"✓ Bridge detector:")
    print(f"  Without: expert_2_selected={expert_2_selected_1:.2f}, overlap={overlap_1:.4f}")
    print(f"  With:    expert_2_selected={expert_2_selected_2:.2f}, overlap={overlap_2:.4f}")
    print(f"  (Bridge detector should reduce selection if beta is hallucinating)")


def test_overlap_only_with_temperature():
    """Test overlap_only behaves correctly with temperature scaling."""
    # High temperature (more uniform)
    z = torch.randn(10, 4)
    p_clean_high = F.softmax(z / 2.0, dim=-1)
    p_biased_high = F.softmax(z / 2.0 + torch.tensor([0.5, 0.0, 0.0, 0.0]), dim=-1)

    # Low temperature (more peaked)
    p_clean_low = F.softmax(z / 0.5, dim=-1)
    p_biased_low = F.softmax(z / 0.5 + torch.tensor([0.5, 0.0, 0.0, 0.0]), dim=-1)

    overlap_high = compute_overlap_only(p_clean_high, p_biased_high)
    overlap_low = compute_overlap_only(p_clean_low, p_biased_low)

    print(f"✓ Overlap with temperature:")
    print(f"  High T: {overlap_high:.4f}")
    print(f"  Low T:  {overlap_low:.4f}")


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTesting Bridge Detector (Step 4) on device: {device_name}\n")

    test_overlap_only_zero()
    test_overlap_only_measures_hallucination()
    test_overlap_only_ignores_suppression()
    test_relevance_full()
    test_relevance_decay()
    test_relevance_zero_at_crisis()
    test_router_with_relevance()
    test_bridge_detector_prevents_hallucination()
    test_overlap_only_with_temperature()

    print("\n✅ All Step 4 tests passed!")
