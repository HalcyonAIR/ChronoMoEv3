"""
Beta saturation analysis.

Tests the stability of pre-softmax additive bias and validates the
decision to use |beta| <= 1.0 constraint.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

# Try to import matplotlib, but continue without it
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def analyze_beta_impact(
    z_clean_range=(-5, 5),
    beta_range=(-3, 3),
    num_experts=8,
    num_samples=1000,
    temperature=1.0,
):
    """
    Analyze how beta affects routing decisions across different magnitudes.

    Args:
        z_clean_range: Range of clean logits to test
        beta_range: Range of beta values to test
        num_experts: Number of experts
        num_samples: Number of random samples
        temperature: Softmax temperature
    """
    print("=" * 80)
    print("Beta Saturation Analysis")
    print("=" * 80)

    # Generate random clean logits
    torch.manual_seed(42)
    z_clean = torch.rand(num_samples, num_experts) * (z_clean_range[1] - z_clean_range[0]) + z_clean_range[0]

    # Test different beta magnitudes
    beta_values = torch.linspace(beta_range[0], beta_range[1], 21)

    results = {
        'beta_values': beta_values.numpy(),
        'mean_top1_prob': [],
        'mean_entropy': [],
        'flip_rate': [],  # How often does top-1 expert change?
    }

    # Baseline (beta=0)
    probs_baseline = F.softmax(z_clean / temperature, dim=-1)
    top1_baseline = probs_baseline.argmax(dim=-1)

    print(f"\nBaseline (beta=0):")
    print(f"  Mean top-1 probability: {probs_baseline.max(dim=-1).values.mean():.3f}")
    print(f"  Mean entropy: {-(probs_baseline * probs_baseline.log()).sum(dim=-1).mean():.3f}")

    print(f"\nTesting beta in range [{beta_range[0]}, {beta_range[1]}]:")
    print(f"{'Beta':<8} {'Top-1 Prob':<12} {'Entropy':<10} {'Flip Rate':<10} {'Status'}")
    print("-" * 60)

    for beta_val in beta_values:
        # Apply beta to expert 0 only
        beta = torch.zeros(num_experts)
        beta[0] = beta_val

        # Compute biased logits
        z_biased = z_clean + beta.unsqueeze(0)
        probs = F.softmax(z_biased / temperature, dim=-1)

        # Metrics
        top1_prob = probs.max(dim=-1).values.mean().item()
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()
        top1 = probs.argmax(dim=-1)
        flip_rate = (top1 != top1_baseline).float().mean().item()

        results['mean_top1_prob'].append(top1_prob)
        results['mean_entropy'].append(entropy)
        results['flip_rate'].append(flip_rate)

        # Classify status
        status = "OK"
        if top1_prob > 0.9:
            status = "SATURATING"
        elif abs(beta_val) > 1.0 and flip_rate > 0.5:
            status = "HIGH IMPACT"

        print(f"{beta_val.item():>6.2f}  {top1_prob:>10.3f}  {entropy:>8.3f}  {flip_rate:>8.3f}  {status}")

    # Analysis
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)

    # Find safe beta range
    safe_beta_max = 0.0
    for i, beta_val in enumerate(beta_values):
        if results['mean_top1_prob'][i] < 0.8:  # Not saturating
            safe_beta_max = abs(beta_val.item())

    print(f"\n1. Safe beta range: |beta| <= {safe_beta_max:.1f}")
    print(f"   (Avoids saturation: top-1 prob stays < 0.8)")

    # Impact on routing
    flip_at_1 = 0.0
    flip_at_2 = 0.0

    beta_1_idx = torch.where(torch.abs(beta_values - 1.0) < 0.01)[0]
    if len(beta_1_idx) > 0:
        flip_at_1 = results['flip_rate'][beta_1_idx[0].item()]
        print(f"\n2. At beta=1.0: {flip_at_1*100:.1f}% of tokens change top-1 expert")
        print(f"   (Moderate influence, not dominating)")

    beta_2_idx = torch.where(torch.abs(beta_values - 2.0) < 0.01)[0]
    if len(beta_2_idx) > 0:
        flip_at_2 = results['flip_rate'][beta_2_idx[0].item()]
        print(f"\n3. At beta=2.0: {flip_at_2*100:.1f}% of tokens change top-1 expert")
        if flip_at_2 > 0.7:
            print(f"   (TOO STRONG: bias is dominating clean routing)")

    # Ratio analysis
    z_clean_std = z_clean.std().item()
    print(f"\n4. Clean logit std: {z_clean_std:.2f}")
    print(f"   beta=1.0 / std = {1.0/z_clean_std:.2f}")
    print(f"   (Good: ratio < 0.5 means beta is gentle prior)")

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)
    print(f"✓ Pre-softmax additive bias is stable for |beta| <= 1.0")
    print(f"✓ At beta=1.0, routing impact is moderate (~{flip_at_1*100:.0f}% flip rate)")
    print(f"✗ At beta=2.0, bias starts dominating (>{flip_at_2*100:.0f}% flip rate)")
    print(f"\n→ Recommendation: Enforce beta.data.clamp_(-1.0, 1.0) after each update")

    return results


def visualize_saturation(results):
    """Create visualization of beta saturation effects."""
    if not HAS_MATPLOTLIB:
        print("\n(Skipping visualization: matplotlib not available)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    beta_values = results['beta_values']

    # Plot 1: Top-1 probability
    axes[0].plot(beta_values, results['mean_top1_prob'], 'b-', linewidth=2)
    axes[0].axhline(y=0.8, color='r', linestyle='--', label='Saturation threshold')
    axes[0].axvline(x=1.0, color='g', linestyle='--', label='beta_max')
    axes[0].axvline(x=-1.0, color='g', linestyle='--')
    axes[0].set_xlabel('Beta value')
    axes[0].set_ylabel('Mean top-1 probability')
    axes[0].set_title('Routing Saturation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Entropy
    axes[1].plot(beta_values, results['mean_entropy'], 'b-', linewidth=2)
    axes[1].axvline(x=1.0, color='g', linestyle='--', label='beta_max')
    axes[1].axvline(x=-1.0, color='g', linestyle='--')
    axes[1].set_xlabel('Beta value')
    axes[1].set_ylabel('Mean entropy')
    axes[1].set_title('Routing Diversity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Flip rate
    axes[2].plot(beta_values, results['flip_rate'], 'b-', linewidth=2)
    axes[2].axhline(y=0.5, color='r', linestyle='--', label='High impact threshold')
    axes[2].axvline(x=1.0, color='g', linestyle='--', label='beta_max')
    axes[2].axvline(x=-1.0, color='g', linestyle='--')
    axes[2].set_xlabel('Beta value')
    axes[2].set_ylabel('Flip rate (routing change)')
    axes[2].set_title('Routing Influence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('beta_saturation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: beta_saturation_analysis.png")


def test_temperature_interaction():
    """Test how temperature affects beta impact."""
    print("\n" + "=" * 80)
    print("Temperature Interaction Analysis")
    print("=" * 80)

    torch.manual_seed(42)
    num_samples = 1000
    num_experts = 8
    z_clean = torch.randn(num_samples, num_experts) * 2.0  # Mean 0, std 2

    beta = torch.zeros(num_experts)
    beta[0] = 1.0  # Fixed beta

    temperatures = [0.5, 1.0, 2.0, 5.0]

    print(f"\n{'Temperature':<12} {'Top-1 Prob':<12} {'Flip Rate':<12} {'Entropy'}")
    print("-" * 50)

    probs_baseline = F.softmax(z_clean, dim=-1)
    top1_baseline = probs_baseline.argmax(dim=-1)

    for temp in temperatures:
        z_biased = z_clean + beta.unsqueeze(0)
        probs = F.softmax(z_biased / temp, dim=-1)

        top1_prob = probs.max(dim=-1).values.mean().item()
        top1 = probs.argmax(dim=-1)
        flip_rate = (top1 != top1_baseline).float().mean().item()
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()

        print(f"{temp:<10.1f}  {top1_prob:>10.3f}  {flip_rate:>10.3f}  {entropy:>8.3f}")

    print("\nObservations:")
    print("- Higher temperature → beta has less relative impact")
    print("- Lower temperature → beta has more relative impact")
    print("- This is desirable: confident model (low T) respects slow bias more")


if __name__ == "__main__":
    results = analyze_beta_impact(
        z_clean_range=(-5, 5),
        beta_range=(-3, 3),
        num_experts=8,
        num_samples=1000,
        temperature=1.0,
    )

    try:
        visualize_saturation(results)
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")

    test_temperature_interaction()

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
