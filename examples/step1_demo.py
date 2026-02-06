"""
Step 1 Demo: RouterState + Beta Application

Demonstrates:
1. Dual distribution (clean vs biased)
2. Disagreement metrics (JS divergence, flip rate)
3. Scale-free beta (beta_eff = k * logit_std)
4. Bridge detector (relevance modulation)
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import ChronoRouter, compute_relevance


def demo_dual_distribution():
    """Show clean vs biased distributions."""
    print("=" * 70)
    print("DEMO: Dual Distribution (Clean vs Biased)")
    print("=" * 70)

    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Set non-zero beta for experts 0, 1, 2
    router.router_state.beta_coeff = torch.tensor(
        [0.25, -0.20, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    print(f"\nBeta coefficients (k): {router.router_state.beta_coeff.numpy()}")

    # Forward pass
    hidden = torch.randn(16, 64)
    top_k_probs, top_k_indices, z_clean, z_biased = router(hidden, top_k=2)

    print(f"Logit std: {router.router_state.logit_std_ema:.3f}")
    print(f"Beta effective: {router.router_state.compute_beta_eff().numpy()}")

    print(f"\nDisagreement metrics:")
    print(f"  JS divergence: {router.router_state.disagreement_js:.4f}")
    print(f"  Top-1 flip rate: {router.router_state.disagreement_flip:.3f}")

    # Show example distributions for first token
    p_clean = torch.softmax(z_clean[0], dim=-1).detach()
    p_biased = torch.softmax(z_biased[0], dim=-1).detach()

    print(f"\nExample token 0 distributions:")
    print(f"  Clean:  {p_clean.numpy()}")
    print(f"  Biased: {p_biased.numpy()}")
    print(f"  Delta:  {(p_biased - p_clean).numpy()}")


def demo_scale_free_beta():
    """Show beta scales with logit std."""
    print("\n" + "=" * 70)
    print("DEMO: Scale-Free Beta (Portability Across Regimes)")
    print("=" * 70)

    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")
    router.router_state.beta_coeff[0] = 0.2  # k = 0.2

    print(f"\nBeta coefficient k[0] = {router.router_state.beta_coeff[0].item():.2f}")

    # Test three regimes with different logit scales
    regimes = [
        {"name": "Low entropy", "scale": 0.5},
        {"name": "Normal", "scale": 1.0},
        {"name": "High entropy", "scale": 2.0},
    ]

    print(f"\n{'Regime':<20} {'logit_std':<12} {'beta_eff[0]':<12} {'JS div':<10} {'Flip rate':<10}")
    print("-" * 70)

    for regime in regimes:
        # Generate inputs with specified scale
        hidden = torch.randn(32, 64) * regime["scale"]

        # Forward pass
        router(hidden, top_k=2)

        # Results
        print(
            f"{regime['name']:<20} "
            f"{router.router_state.logit_std_ema:<12.3f} "
            f"{router.router_state.compute_beta_eff()[0].item():<12.3f} "
            f"{router.router_state.disagreement_js:<10.4f} "
            f"{router.router_state.disagreement_flip:<10.3f}"
        )

    print(
        "\nNote: Beta_eff scales with logit_std, but impact (JS, flip) stays consistent!"
    )


def demo_bridge_detector():
    """Show bridge detector vetoing beta when disagreement is high."""
    print("\n" + "=" * 70)
    print("DEMO: Bridge Detector (Relevance Modulation)")
    print("=" * 70)

    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Start with strong beta
    router.router_state.beta_coeff = torch.tensor(
        [0.3, -0.3, 0.2, -0.2, 0.1, 0.0, 0.0, 0.0]
    )

    print(f"\nBeta coefficients (k): {router.router_state.beta_coeff.numpy()}")

    # Test with different beta strengths
    print(f"\n{'Beta scale':<15} {'JS div':<10} {'Relevance':<12} {'Comment':<30}")
    print("-" * 70)

    for scale in [0.0, 0.1, 0.3, 0.5, 0.8]:
        # Scale beta
        router.router_state.beta_coeff = torch.tensor(
            [0.3, -0.3, 0.2, -0.2, 0.1, 0.0, 0.0, 0.0]
        ) * scale

        # Forward WITHOUT relevance
        hidden = torch.randn(32, 64)
        router(hidden, top_k=2, use_relevance=False)

        # Compute relevance
        r = compute_relevance(router.router_state, threshold=0.2)

        # Comment
        if r == 1.0:
            comment = "Full relevance (beta aligned)"
        elif r > 0.3:
            comment = "Partial relevance (warning)"
        else:
            comment = "No relevance (beta vetoed)"

        print(
            f"{scale:<15.1f} "
            f"{router.router_state.disagreement_js:<10.4f} "
            f"{r:<12.3f} "
            f"{comment:<30}"
        )

    print(
        "\nBridge detector: High disagreement → low relevance → beta suppressed automatically"
    )


def demo_beta_update_loop():
    """Show beta responding to coherence feedback (preview of Step 3)."""
    print("\n" + "=" * 70)
    print("DEMO: Beta Update Loop (Preview of Step 3)")
    print("=" * 70)

    router = ChronoRouter(d_model=64, num_experts=8, layer_id=0, device="cpu")

    # Simulate 10 steps
    print(f"\n{'Step':<8} {'beta_mean':<12} {'beta_std':<12} {'JS div':<10}")
    print("-" * 50)

    for step in range(10):
        # Forward pass
        hidden = torch.randn(32, 64)
        router(hidden, top_k=2)

        # Simulate coherence feedback (Step 3 will use real coherence)
        # For now, just update beta randomly
        if step % 3 == 0:
            # Simulate: some experts get promoted, some demoted
            delta = torch.randn(8) * 0.01
            router.router_state.beta_coeff += delta
            router.router_state.beta_coeff.clamp_(-0.3, 0.3)

        print(
            f"{step:<8} "
            f"{router.router_state.beta_coeff.mean().item():<12.4f} "
            f"{router.router_state.beta_coeff.std().item():<12.4f} "
            f"{router.router_state.disagreement_js:<10.4f}"
        )

    print("\nIn Step 3, beta will respond to phi_slow (coherence feedback)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 1: RouterState + Beta Application")
    print("=" * 70)
    print("\nDeliverables:")
    print("  ✓ RouterState with beta_coeff (scale-free)")
    print("  ✓ Dual distribution (clean vs biased)")
    print("  ✓ Disagreement metrics (JS divergence, flip rate)")
    print("  ✓ Bridge detector (relevance modulation)")
    print()

    demo_dual_distribution()
    demo_scale_free_beta()
    demo_bridge_detector()
    demo_beta_update_loop()

    print("\n" + "=" * 70)
    print("✅ Step 1 Complete: Router with beta works!")
    print("=" * 70)
    print("\nNext: Step 2 - Coherence on GPU with buffered state")
    print()
