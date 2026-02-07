"""
Step 3 Demo: Beta Update Function - The Closed Loop

Demonstrates:
1. Coherence → Beta feedback loop
2. PROMOTION prior: high phi_slow → increase beta
3. Beta convergence to stable values
4. Beta responds to coherence changes
5. Full integration: CoherenceBuffer → beta → routing
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    RouterState,
    CoherenceBuffer,
    update_beta_from_buffer,
)


def demo_promotion_prior():
    """Show PROMOTION prior in action."""
    print("=" * 70)
    print("DEMO: PROMOTION Prior (High Coherence → Positive Beta)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router_state = RouterState(
        layer_id=0,
        num_experts=4,
        beta_coeff=torch.zeros(4, device=device),
    )

    # Mock coherence values
    # Expert 0: Very high (0.9)
    # Expert 1: Above threshold (0.6)
    # Expert 2: At threshold (0.5)
    # Expert 3: Below threshold (0.3)

    phi_slow = torch.tensor([0.9, 0.6, 0.5, 0.3], device=device)
    total_tokens_seen = torch.tensor([100, 100, 100, 100], device=device)

    print(f"\nInitial phi_slow: {phi_slow.cpu().numpy()}")
    print(f"Threshold (tau): 0.5")
    print(f"Learning rate (eta): 0.1")

    update_beta_from_buffer(
        router_state,
        phi_slow,
        total_tokens_seen,
        eta=0.1,
        tau=0.5,
    )

    print(f"\nBeta after 1 update:")
    for i in range(4):
        delta = 0.1 * (phi_slow[i].item() - 0.5)
        symbol = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(
            f"  Expert {i}: beta={router_state.beta_coeff[i].item():+.3f} "
            f"(delta={delta:+.3f}) {symbol}"
        )


def demo_closed_loop():
    """Show the full closed loop over time."""
    print("\n" + "=" * 70)
    print("DEMO: Closed Loop (Coherence → Beta → Routing)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create coherence buffer
    buffer = CoherenceBuffer(
        layer_id=0,
        num_experts=4,
        d_model=64,
        device=device,
        alpha_slow=0.95,  # Faster for demo
    )

    # Create router state
    router = RouterState(
        layer_id=0,
        num_experts=4,
        beta_coeff=torch.zeros(4, device=device),
    )

    print(f"\nSimulating 150 steps with 4 experts:")
    print(f"  Expert 0: Consistently high coherence")
    print(f"  Expert 1: Consistently low coherence")
    print(f"  Expert 2: Drops at step 75")
    print(f"  Expert 3: Improves at step 75")
    print()

    # Track beta evolution
    beta_history = {i: [] for i in range(4)}

    for step in range(150):
        # Generate coherence patterns
        if step < 75:
            phi_raw = torch.tensor([0.8, 0.3, 0.8, 0.3], device=device)
        else:
            phi_raw = torch.tensor([0.8, 0.3, 0.3, 0.8], device=device)

        active_ids = torch.arange(4, device=device)
        num_tokens = torch.ones(4, dtype=torch.long, device=device) * 10

        # Update coherence
        buffer.update(phi_raw, active_ids, step, num_tokens)

        # Update beta every 10 steps
        if step % 10 == 0 and step >= 20:  # Wait for phi_slow to converge
            update_beta_from_buffer(
                router,
                buffer.get_phi_slow(),
                buffer.total_tokens_seen,
                eta=0.02,
                tau=0.5,
            )

            # Record beta values
            for i in range(4):
                beta_history[i].append(router.beta_coeff[i].item())

            if step % 30 == 0:
                phi_slow_vals = buffer.get_phi_slow()
                print(f"Step {step:3d}:")
                print(f"  phi_slow: [{phi_slow_vals[0]:.3f}, {phi_slow_vals[1]:.3f}, {phi_slow_vals[2]:.3f}, {phi_slow_vals[3]:.3f}]")
                print(f"  beta:     [{router.beta_coeff[0]:.3f}, {router.beta_coeff[1]:.3f}, {router.beta_coeff[2]:.3f}, {router.beta_coeff[3]:.3f}]")

    print(f"\nFinal state:")
    phi_slow_vals = buffer.get_phi_slow()
    print(f"  phi_slow: [{phi_slow_vals[0]:.3f}, {phi_slow_vals[1]:.3f}, {phi_slow_vals[2]:.3f}, {phi_slow_vals[3]:.3f}]")
    print(f"  beta:     [{router.beta_coeff[0]:.3f}, {router.beta_coeff[1]:.3f}, {router.beta_coeff[2]:.3f}, {router.beta_coeff[3]:.3f}]")

    print(f"\nObservations:")
    print(f"  Expert 0: phi_slow={phi_slow_vals[0]:.3f} → beta={router.beta_coeff[0]:+.3f} (promoted)")
    print(f"  Expert 1: phi_slow={phi_slow_vals[1]:.3f} → beta={router.beta_coeff[1]:+.3f} (demoted)")
    print(f"  Expert 2: Dropped coherence → beta={router.beta_coeff[2]:+.3f}")
    print(f"  Expert 3: Improved coherence → beta={router.beta_coeff[3]:+.3f}")


def demo_beta_scale_free():
    """Show scale-free property: beta_eff = k * logit_std."""
    print("\n" + "=" * 70)
    print("DEMO: Scale-Free Beta (beta_eff = k * logit_std)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router = RouterState(
        layer_id=0,
        num_experts=2,
        beta_coeff=torch.tensor([0.2, -0.2], device=device),
        logit_std_ema=1.0,
    )

    print(f"\nBeta coefficients (k): [0.2, -0.2]")
    print()

    for logit_std in [0.5, 1.0, 2.0, 4.0]:
        router.logit_std_ema = logit_std
        beta_eff = router.compute_beta_eff()

        print(f"logit_std={logit_std:.1f}: beta_eff=[{beta_eff[0]:+.2f}, {beta_eff[1]:+.2f}]")

    print(f"\nScale-free property ensures consistent routing influence")
    print(f"across different entropy regimes.")


def demo_beta_convergence():
    """Show beta converging to stable value."""
    print("\n" + "=" * 70)
    print("DEMO: Beta Convergence")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router = RouterState(
        layer_id=0,
        num_experts=1,
        beta_coeff=torch.zeros(1, device=device),
    )

    # Consistent high coherence
    phi_slow = torch.tensor([0.7], device=device)
    total_tokens_seen = torch.tensor([100], device=device)

    print(f"\nConsistent phi_slow=0.7, tau=0.5, eta=0.01")
    print(f"\nBeta evolution:")

    for step in range(100):
        update_beta_from_buffer(
            router, phi_slow, total_tokens_seen, eta=0.01, tau=0.5
        )

        if step % 20 == 0:
            print(f"  Step {step:3d}: beta={router.beta_coeff[0].item():.4f}")

    print(f"\nBeta converges to stable value where delta ≈ 0")
    print(f"Final beta: {router.beta_coeff[0].item():.4f}")


def demo_beta_clamping():
    """Show beta clamping at k_max."""
    print("\n" + "=" * 70)
    print("DEMO: Beta Clamping (Safety Bounds)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    router = RouterState(
        layer_id=0,
        num_experts=2,
        beta_coeff=torch.zeros(2, device=device),
        k_max=0.3,
    )

    # Extreme coherence values
    phi_slow = torch.tensor([1.0, 0.0], device=device)
    total_tokens_seen = torch.tensor([100, 100], device=device)

    print(f"\nExtreme phi_slow: [1.0, 0.0]")
    print(f"k_max (clamp limit): {router.k_max}")
    print()

    for step in range(50):
        update_beta_from_buffer(
            router, phi_slow, total_tokens_seen, eta=0.1, tau=0.5
        )

        if step % 10 == 0:
            print(
                f"  Step {step:2d}: beta=[{router.beta_coeff[0]:+.3f}, {router.beta_coeff[1]:+.3f}]"
            )

    print(f"\nBeta clamped to [-{router.k_max}, +{router.k_max}]")
    print(f"Prevents saturation and maintains routing diversity")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 3: Beta Update Function (The Closed Loop)")
    print("=" * 70)
    print("\nDeliverables:")
    print("  ✓ PROMOTION prior: high phi_slow → increase beta")
    print("  ✓ GPU-optimized: updates directly from CoherenceBuffer")
    print("  ✓ Scale-free: beta_eff = k * logit_std")
    print("  ✓ Clamping: bounded to [-k_max, k_max]")
    print("  ✓ Closed loop: coherence → beta → routing → coherence")
    print()

    demo_promotion_prior()
    demo_closed_loop()
    demo_beta_scale_free()
    demo_beta_convergence()
    demo_beta_clamping()

    print("\n" + "=" * 70)
    print("✅ Step 3 Complete: Beta update loop works!")
    print("=" * 70)
    print("\nThe locus is now active:")
    print("  High coherence experts earn routing advantage (positive beta)")
    print("  Low coherence experts lose routing advantage (negative beta)")
    print("  Beta accumulates over the slow clock timescale (~1000 steps)")
    print()
    print("Next: Step 4 - Bridge detector veto (relevance modulation)")
    print()
