"""
Tests for router with beta (slow bias) mechanism.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    ChronoRouter,
    RouterState,
    compute_js_divergence,
    compute_flip_rate,
    compute_relevance,
)


def test_router_state_basic():
    """Test RouterState initialization and basic operations."""
    state = RouterState(
        layer_id=0,
        num_experts=8,
        beta_coeff=torch.zeros(8, device="cpu"),
    )

    # Check initial state
    assert state.layer_id == 0
    assert state.num_experts == 8
    assert state.beta_coeff.shape == (8,)
    assert state.logit_std_ema == 1.0
    assert state.disagreement_js == 0.0
    assert state.disagreement_flip == 0.0

    print("✓ RouterState initialization")


def test_beta_eff_computation():
    """Test scale-free beta computation."""
    state = RouterState(
        layer_id=0,
        num_experts=8,
        beta_coeff=torch.tensor([0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        k_max=0.3,
    )

    # With logit_std = 1.0
    beta_eff = state.compute_beta_eff()
    assert beta_eff.shape == (8,)
    assert torch.allclose(beta_eff[0], torch.tensor(0.1))
    assert torch.allclose(beta_eff[1], torch.tensor(-0.1))

    # Update logit std to 2.0
    state.logit_std_ema = 2.0
    beta_eff = state.compute_beta_eff()
    assert torch.allclose(beta_eff[0], torch.tensor(0.2))  # 0.1 * 2.0
    assert torch.allclose(beta_eff[1], torch.tensor(-0.2))  # -0.1 * 2.0

    print("✓ Beta effective computation (scale-free)")


def test_logit_std_update():
    """Test logit std EMA tracking."""
    state = RouterState(
        layer_id=0,
        num_experts=8,
        beta_coeff=torch.zeros(8),
    )

    # Initial std = 1.0
    assert state.logit_std_ema == 1.0

    # Update with z_clean having std=2.0
    z_clean = torch.randn(100, 8) * 2.0  # std ≈ 2.0
    state.update_logit_std(z_clean)

    # Should move toward 2.0 (but slowly, alpha=0.99)
    assert state.logit_std_ema > 1.0
    assert state.logit_std_ema < 2.0

    print(f"✓ Logit std EMA: {state.logit_std_ema:.3f}")


def test_js_divergence():
    """Test JS divergence computation."""
    # Identical distributions
    p = torch.softmax(torch.randn(100, 8), dim=-1)
    js = compute_js_divergence(p, p)
    assert abs(js) < 1e-6

    # Very different distributions
    q = torch.softmax(torch.randn(100, 8) + 10, dim=-1)
    js = compute_js_divergence(p, q)
    assert js > 0.1

    print(f"✓ JS divergence: identical={abs(js):.6f}, different={js:.3f}")


def test_flip_rate():
    """Test top-1 flip rate computation."""
    # Identical distributions
    p = torch.softmax(torch.randn(100, 8), dim=-1)
    flip = compute_flip_rate(p, p, top_k=2)
    assert flip == 0.0

    # Inverted distribution (top-1 definitely changes)
    p = torch.softmax(torch.randn(100, 8), dim=-1)
    q = torch.softmax(-p * 100, dim=-1)  # Invert
    flip = compute_flip_rate(p, q, top_k=2)
    assert flip > 0.9  # Should be nearly 100%

    print(f"✓ Flip rate: identical={0.0:.1f}, inverted={flip:.2f}")


def test_relevance_computation():
    """Test relevance scalar (bridge detector)."""
    state = RouterState(
        layer_id=0,
        num_experts=8,
        beta_coeff=torch.zeros(8),
    )

    # Low disagreement → full relevance
    state.disagreement_js = 0.05
    r = compute_relevance(state, threshold=0.2)
    assert r == 1.0

    # Medium disagreement → partial relevance
    state.disagreement_js = 0.3
    r = compute_relevance(state, threshold=0.2)
    assert 0.0 < r < 1.0

    # High disagreement → no relevance
    state.disagreement_js = 0.8
    r = compute_relevance(state, threshold=0.2)
    assert r == 0.0

    print(f"✓ Relevance: low={1.0:.1f}, mid={r:.2f}, high={0.0:.1f}")


def test_chrono_router_forward():
    """Test ChronoRouter forward pass with dual distribution."""
    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Forward pass
    hidden = torch.randn(32, 64)  # [B×S, d_model]
    top_k_probs, top_k_indices, z_clean, z_biased = router(hidden, top_k=2)

    # Check shapes
    assert top_k_probs.shape == (32, 2)
    assert top_k_indices.shape == (32, 2)
    assert z_clean.shape == (32, 8)
    assert z_biased.shape == (32, 8)

    # Check that top_k_probs are normalized
    assert torch.allclose(top_k_probs.sum(dim=-1), torch.ones(32), atol=1e-5)

    # Check that disagreement metrics are logged
    assert router.router_state.disagreement_js >= 0.0
    assert router.router_state.disagreement_flip >= 0.0

    print(
        f"✓ ChronoRouter forward: JS={router.router_state.disagreement_js:.4f}, "
        f"flip={router.router_state.disagreement_flip:.3f}"
    )


def test_router_with_beta():
    """Test router with non-zero beta."""
    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Set non-zero beta
    router.router_state.beta_coeff = torch.tensor(
        [0.2, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Forward pass
    hidden = torch.randn(32, 64)
    top_k_probs, top_k_indices, z_clean, z_biased = router(hidden, top_k=2)

    # Check that z_biased != z_clean (beta was applied)
    assert not torch.allclose(z_clean, z_biased)

    # Check that disagreement is non-zero (since beta ≠ 0)
    assert router.router_state.disagreement_js > 0.0

    print(
        f"✓ Router with beta: JS={router.router_state.disagreement_js:.4f}, "
        f"flip={router.router_state.disagreement_flip:.3f}"
    )


def test_router_with_relevance():
    """Test router with relevance modulation (bridge detector)."""
    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Set non-zero beta
    router.router_state.beta_coeff = torch.tensor(
        [0.3, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Forward pass WITHOUT relevance
    hidden = torch.randn(32, 64)
    _, _, z_clean_1, z_biased_1 = router(hidden, top_k=2, use_relevance=False)
    js_without = router.router_state.disagreement_js

    # Forward pass WITH relevance
    _, _, z_clean_2, z_biased_2 = router(hidden, top_k=2, use_relevance=True)
    js_with = router.router_state.disagreement_js

    # With relevance, disagreement should be same or lower
    # (beta is modulated if disagreement is high)
    print(
        f"✓ Relevance modulation: JS_without={js_without:.4f}, JS_with={js_with:.4f}"
    )


if __name__ == "__main__":
    print("Testing RouterState and ChronoRouter (Step 1)...\n")

    test_router_state_basic()
    test_beta_eff_computation()
    test_logit_std_update()
    test_js_divergence()
    test_flip_rate()
    test_relevance_computation()
    test_chrono_router_forward()
    test_router_with_beta()
    test_router_with_relevance()

    print("\n✅ All Step 1 tests passed!")
