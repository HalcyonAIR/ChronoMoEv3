"""
Tests for beta update function (Step 3).
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    RouterState,
    CoherenceBuffer,
    update_beta,
    update_beta_from_buffer,
)
from chronomoe_v3.coherence import CoherenceState


def test_update_beta_promotion_prior():
    """Test PROMOTION prior: high phi_slow → increase beta."""
    router_state = RouterState(
        layer_id=0,
        num_experts=4,
        beta_coeff=torch.zeros(4),
    )

    # Create mock coherence snapshot
    # Expert 0: phi_slow=0.8 (high, above tau=0.5)
    # Expert 1: phi_slow=0.3 (low, below tau=0.5)
    # Expert 2: phi_slow=0.5 (at tau)
    # Expert 3: phi_slow=0.1 (very low)

    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,
            total_tokens_seen=100,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.3,
            total_tokens_seen=100,
        ),
        2: CoherenceState(
            expert_id="L0_E2",
            layer_id=0,
            d_model=64,
            phi_slow=0.5,
            total_tokens_seen=100,
        ),
        3: CoherenceState(
            expert_id="L0_E3",
            layer_id=0,
            d_model=64,
            phi_slow=0.1,
            total_tokens_seen=100,
        ),
    }

    # Update beta
    update_beta(router_state, snapshot, eta=0.1, tau=0.5)

    # Check results
    # Expert 0: delta = 0.1 * (0.8 - 0.5) = 0.03
    assert router_state.beta_coeff[0] > 0  # Promoted
    assert torch.isclose(router_state.beta_coeff[0], torch.tensor(0.03))

    # Expert 1: delta = 0.1 * (0.3 - 0.5) = -0.02
    assert router_state.beta_coeff[1] < 0  # Demoted
    assert torch.isclose(router_state.beta_coeff[1], torch.tensor(-0.02))

    # Expert 2: delta = 0.1 * (0.5 - 0.5) = 0
    assert torch.isclose(router_state.beta_coeff[2], torch.tensor(0.0))

    # Expert 3: delta = 0.1 * (0.1 - 0.5) = -0.04
    assert router_state.beta_coeff[3] < 0
    assert torch.isclose(router_state.beta_coeff[3], torch.tensor(-0.04))

    print("✓ PROMOTION prior: high phi_slow → positive beta")


def test_update_beta_clamping():
    """Test beta clamping to [-k_max, k_max]."""
    router_state = RouterState(
        layer_id=0,
        num_experts=2,
        beta_coeff=torch.tensor([0.29, -0.29]),  # Near limits
        k_max=0.3,
    )

    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.9,  # Very high, would push beta > 0.3
            total_tokens_seen=100,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.0,  # Very low, would push beta < -0.3
            total_tokens_seen=100,
        ),
    }

    # Update beta with large eta
    update_beta(router_state, snapshot, eta=0.5, tau=0.5)

    # Check clamping
    assert router_state.beta_coeff[0] <= 0.3  # Clamped at k_max
    assert router_state.beta_coeff[1] >= -0.3  # Clamped at -k_max

    print(f"✓ Beta clamping: [{router_state.beta_coeff[0]:.3f}, {router_state.beta_coeff[1]:.3f}]")


def test_update_beta_observation_filter():
    """Test that unobserved experts are skipped."""
    router_state = RouterState(
        layer_id=0,
        num_experts=2,
        beta_coeff=torch.zeros(2),
    )

    # Expert 0: observed (tokens > 0)
    # Expert 1: not observed (tokens = 0)
    snapshot = {
        0: CoherenceState(
            expert_id="L0_E0",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,
            total_tokens_seen=100,
        ),
        1: CoherenceState(
            expert_id="L0_E1",
            layer_id=0,
            d_model=64,
            phi_slow=0.8,
            total_tokens_seen=0,  # Not observed
        ),
    }

    update_beta(router_state, snapshot, eta=0.1, tau=0.5)

    # Expert 0 should be updated
    assert router_state.beta_coeff[0] != 0.0

    # Expert 1 should NOT be updated (not observed)
    assert router_state.beta_coeff[1] == 0.0

    print("✓ Observation filter: unobserved experts skipped")


def test_update_beta_from_buffer():
    """Test GPU-optimized beta update from coherence buffer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_state = RouterState(
        layer_id=0,
        num_experts=4,
        beta_coeff=torch.zeros(4, device=device),
    )

    # Mock coherence buffer state
    phi_slow = torch.tensor([0.8, 0.3, 0.5, 0.1], device=device)
    total_tokens_seen = torch.tensor([100, 100, 100, 5], device=device)

    # Update beta from buffer (stays on GPU)
    update_beta_from_buffer(
        router_state,
        phi_slow,
        total_tokens_seen,
        eta=0.1,
        tau=0.5,
        min_tokens=10,
    )

    # Check results
    assert router_state.beta_coeff[0] > 0  # High phi_slow
    assert router_state.beta_coeff[1] < 0  # Low phi_slow
    assert torch.isclose(router_state.beta_coeff[2], torch.tensor(0.0, device=device))  # At tau
    assert router_state.beta_coeff[3] == 0.0  # Not observed (tokens < 10)

    print(f"✓ GPU beta update (device={device})")


def test_beta_convergence():
    """Test beta converges to stable value over multiple updates."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_state = RouterState(
        layer_id=0,
        num_experts=1,
        beta_coeff=torch.zeros(1, device=device),
    )

    # Simulate 100 updates with consistent high coherence
    phi_slow = torch.tensor([0.8], device=device)
    total_tokens_seen = torch.tensor([100], device=device)

    beta_history = []
    for _ in range(100):
        update_beta_from_buffer(
            router_state,
            phi_slow,
            total_tokens_seen,
            eta=0.01,
            tau=0.5,
        )
        beta_history.append(router_state.beta_coeff[0].item())

    # Beta should converge (changes become smaller)
    early_change = abs(beta_history[10] - beta_history[0])
    late_change = abs(beta_history[99] - beta_history[90])

    assert late_change < early_change  # Convergence
    assert router_state.beta_coeff[0] > 0  # Stable positive value
    assert router_state.beta_coeff[0] <= 0.3  # At or below k_max

    print(
        f"✓ Beta convergence: early_Δ={early_change:.4f}, "
        f"late_Δ={late_change:.4f}, final={router_state.beta_coeff[0].item():.3f}"
    )


def test_beta_response_to_coherence_drop():
    """Test beta responds to sudden coherence drop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_state = RouterState(
        layer_id=0,
        num_experts=1,
        beta_coeff=torch.zeros(1, device=device),
    )

    # Phase 1: High coherence (50 steps)
    phi_slow = torch.tensor([0.8], device=device)
    total_tokens_seen = torch.tensor([100], device=device)

    for _ in range(50):
        update_beta_from_buffer(
            router_state, phi_slow, total_tokens_seen, eta=0.01, tau=0.5
        )

    beta_after_high = router_state.beta_coeff[0].item()
    assert beta_after_high > 0

    # Phase 2: Coherence drops (50 steps)
    phi_slow = torch.tensor([0.2], device=device)

    for _ in range(50):
        update_beta_from_buffer(
            router_state, phi_slow, total_tokens_seen, eta=0.01, tau=0.5
        )

    beta_after_low = router_state.beta_coeff[0].item()

    # Beta should have decreased (possibly negative)
    assert beta_after_low < beta_after_high

    print(
        f"✓ Beta responds to coherence drop: "
        f"high={beta_after_high:.3f} → low={beta_after_low:.3f}"
    )


def test_integration_with_coherence_buffer():
    """Test full integration: CoherenceBuffer → beta update."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create coherence buffer with faster slow clock for testing
    coherence_buffer = CoherenceBuffer(
        layer_id=0,
        num_experts=4,
        d_model=64,
        device=device,
        alpha_slow=0.95,  # Faster convergence for test
    )

    # Create router state
    router_state = RouterState(
        layer_id=0,
        num_experts=4,
        beta_coeff=torch.zeros(4, device=device),
    )

    # Simulate forward passes with different coherence patterns
    # Expert 0: consistently high
    # Expert 1: consistently low
    # Expert 2: starts high, drops
    # Expert 3: starts low, improves

    for step in range(200):  # More steps to let slow clock converge
        if step < 100:
            phi_raw = torch.tensor([0.8, 0.3, 0.8, 0.3], device=device)
        else:
            phi_raw = torch.tensor([0.8, 0.3, 0.3, 0.8], device=device)

        active_ids = torch.tensor([0, 1, 2, 3], device=device)
        num_tokens = torch.tensor([10, 10, 10, 10], device=device)

        # Update coherence
        coherence_buffer.update(phi_raw, active_ids, step, num_tokens)

        # Update beta every 10 steps (after phi_slow has converged a bit)
        if step >= 50 and step % 10 == 0:
            update_beta_from_buffer(
                router_state,
                coherence_buffer.get_phi_slow(),
                coherence_buffer.total_tokens_seen,
                eta=0.02,
                tau=0.5,
            )

    # Check final beta values
    assert router_state.beta_coeff[0] > 0  # Always high → promoted
    assert router_state.beta_coeff[1] < 0  # Always low → demoted

    # Expert 2 and 3 should reflect recent trajectory
    print(
        f"✓ Integration test: beta=[{router_state.beta_coeff[0].item():.3f}, "
        f"{router_state.beta_coeff[1].item():.3f}, "
        f"{router_state.beta_coeff[2].item():.3f}, "
        f"{router_state.beta_coeff[3].item():.3f}]"
    )


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTesting Beta Update (Step 3) on device: {device_name}\n")

    test_update_beta_promotion_prior()
    test_update_beta_clamping()
    test_update_beta_observation_filter()
    test_update_beta_from_buffer()
    test_beta_convergence()
    test_beta_response_to_coherence_drop()
    test_integration_with_coherence_buffer()

    print("\n✅ All Step 3 tests passed!")
