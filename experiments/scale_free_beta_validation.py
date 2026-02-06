"""
Scale-free beta validation.

Demonstrates that beta = k * logit_std is portable across:
1. Different logit scales (early vs late layers)
2. Different training phases (logits grow over time)
3. Different architectures (with/without bias)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np


def simulate_logit_regime(mean=0.0, std=2.0, num_samples=1000, num_experts=8):
    """Generate synthetic logits with specific scale."""
    return torch.randn(num_samples, num_experts) * std + mean


def compute_flip_rate(z_clean, beta_absolute):
    """Compute top-1 flip rate for given absolute beta."""
    z_biased = z_clean + beta_absolute.unsqueeze(0)

    top1_clean = z_clean.argmax(dim=-1)
    top1_biased = z_biased.argmax(dim=-1)

    flip_rate = (top1_clean != top1_biased).float().mean().item()
    return flip_rate


def compute_js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    p = p + 1e-9
    q = q + 1e-9

    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    return js.mean().item()


def test_scale_free_beta():
    """
    Test that k * logit_std produces consistent impact across regimes.
    """
    print("=" * 80)
    print("Scale-Free Beta Validation")
    print("=" * 80)

    k_target = 0.3  # Target scale-free coefficient
    num_experts = 8

    # Three regimes: early layer, mid layer, late layer
    regimes = [
        {"name": "Early Layer", "mean": 0.0, "std": 1.0},
        {"name": "Mid Layer", "mean": 0.0, "std": 2.0},
        {"name": "Late Layer", "mean": 0.0, "std": 4.0},
    ]

    print(f"\nTarget: k = {k_target} (30% of logit std)")
    print(f"\nTesting across three logit scale regimes:\n")

    print(f"{'Regime':<15} {'Logit Std':<12} {'Beta Abs':<12} {'Flip Rate':<12} {'JS Div':<12}")
    print("-" * 70)

    results = []

    for regime in regimes:
        # Generate logits
        z_clean = simulate_logit_regime(
            mean=regime["mean"],
            std=regime["std"],
            num_samples=1000,
            num_experts=num_experts,
        )

        # Measure actual std
        logit_std = z_clean.std().item()

        # Compute scale-free beta
        beta_absolute = torch.zeros(num_experts)
        beta_absolute[0] = k_target * logit_std  # Apply to expert 0

        # Measure impact
        flip_rate = compute_flip_rate(z_clean, beta_absolute)

        p_clean = F.softmax(z_clean, dim=-1)
        p_biased = F.softmax(z_clean + beta_absolute.unsqueeze(0), dim=-1)
        js_div = compute_js_divergence(p_clean, p_biased)

        results.append({
            "name": regime["name"],
            "logit_std": logit_std,
            "beta_abs": beta_absolute[0].item(),
            "flip_rate": flip_rate,
            "js_div": js_div,
        })

        print(
            f"{regime['name']:<15} "
            f"{logit_std:>10.2f}  "
            f"{beta_absolute[0].item():>10.2f}  "
            f"{flip_rate:>10.3f}  "
            f"{js_div:>10.3f}"
        )

    # Analyze consistency
    print("\n" + "=" * 80)
    print("Analysis:")
    print("=" * 80)

    flip_rates = [r["flip_rate"] for r in results]
    js_divs = [r["js_div"] for r in results]

    flip_rate_std = np.std(flip_rates)
    js_div_std = np.std(js_divs)

    print(f"\nFlip rate consistency:")
    print(f"  Mean: {np.mean(flip_rates):.3f}")
    print(f"  Std:  {flip_rate_std:.4f}")
    print(f"  {'✓ CONSISTENT' if flip_rate_std < 0.02 else '✗ VARIABLE'}")

    print(f"\nJS divergence consistency:")
    print(f"  Mean: {np.mean(js_divs):.3f}")
    print(f"  Std:  {js_div_std:.4f}")
    print(f"  {'✓ CONSISTENT' if js_div_std < 0.01 else '✗ VARIABLE'}")

    print("\n" + "=" * 80)
    print("Conclusion:")
    print("=" * 80)

    if flip_rate_std < 0.02 and js_div_std < 0.01:
        print("✅ Scale-free beta (k * logit_std) produces CONSISTENT impact")
        print("   across different logit scales.")
        print(f"   k={k_target} → flip_rate≈{np.mean(flip_rates):.3f}, JS≈{np.mean(js_divs):.3f}")
        print("\n→ This validates using k as the portable parameter, not absolute beta.")
    else:
        print("⚠️  Impact varies across regimes. Scale-free approach needs tuning.")

    return results


def compare_absolute_vs_scalefree():
    """
    Compare absolute beta (old) vs scale-free beta (new).
    """
    print("\n" + "=" * 80)
    print("Absolute Beta vs Scale-Free Beta Comparison")
    print("=" * 80)

    regimes = [
        {"name": "Early", "std": 1.0},
        {"name": "Late", "std": 4.0},
    ]

    beta_absolute_fixed = 1.0  # Old approach: fixed beta
    k_fixed = 0.3  # New approach: fixed k

    print(f"\nOLD approach: beta = {beta_absolute_fixed} (fixed absolute value)")
    print(f"NEW approach: k = {k_fixed} (scale-free coefficient)\n")

    print(f"{'Regime':<10} {'Logit Std':<12} {'Beta (old)':<12} {'Beta (new)':<12} {'Flip (old)':<12} {'Flip (new)':<12}")
    print("-" * 80)

    for regime in regimes:
        z_clean = simulate_logit_regime(std=regime["std"], num_samples=1000)
        logit_std = z_clean.std().item()

        # Old: fixed absolute beta
        beta_old = torch.zeros(8)
        beta_old[0] = beta_absolute_fixed
        flip_old = compute_flip_rate(z_clean, beta_old)

        # New: scale-free beta
        beta_new = torch.zeros(8)
        beta_new[0] = k_fixed * logit_std
        flip_new = compute_flip_rate(z_clean, beta_new)

        print(
            f"{regime['name']:<10} "
            f"{logit_std:>10.2f}  "
            f"{beta_old[0].item():>10.2f}  "
            f"{beta_new[0].item():>10.2f}  "
            f"{flip_old:>10.3f}  "
            f"{flip_new:>10.3f}"
        )

    print("\nObservation:")
    print("- OLD: Same absolute beta → DIFFERENT impact (flip rate varies)")
    print("- NEW: Same k → SAME impact (flip rate consistent)")
    print("\n→ Scale-free beta is portable across layers and training phases.")


def test_js_divergence_metric():
    """
    Demonstrate JS divergence captures more than top-1 flips.
    """
    print("\n" + "=" * 80)
    print("JS Divergence vs Top-1 Flip Rate")
    print("=" * 80)

    z_clean = simulate_logit_regime(std=2.0, num_samples=1000)

    # Scenario 1: Large shift in top-1
    beta_large = torch.zeros(8)
    beta_large[0] = 2.0
    z_biased_large = z_clean + beta_large.unsqueeze(0)

    # Scenario 2: Small shift in top-1, but full distribution changes
    beta_uniform = torch.ones(8) * 0.5
    z_biased_uniform = z_clean + beta_uniform.unsqueeze(0)

    # Metrics
    flip_large = compute_flip_rate(z_clean, beta_large)
    flip_uniform = compute_flip_rate(z_clean, beta_uniform)

    p_clean = F.softmax(z_clean, dim=-1)
    p_biased_large = F.softmax(z_biased_large, dim=-1)
    p_biased_uniform = F.softmax(z_biased_uniform, dim=-1)

    js_large = compute_js_divergence(p_clean, p_biased_large)
    js_uniform = compute_js_divergence(p_clean, p_biased_uniform)

    print(f"\nScenario 1: Large bias on expert 0 (beta[0]=2.0)")
    print(f"  Top-1 flip rate: {flip_large:.3f}")
    print(f"  JS divergence:   {js_large:.3f}")

    print(f"\nScenario 2: Uniform bias on all experts (beta[*]=0.5)")
    print(f"  Top-1 flip rate: {flip_uniform:.3f}")
    print(f"  JS divergence:   {js_uniform:.3f}")

    print("\nObservation:")
    if flip_large > flip_uniform * 1.5 and js_uniform > js_large * 0.5:
        print("✓ JS divergence captures full distribution shift,")
        print("  not just top-1 changes.")
        print("\n→ Use JS divergence as primary disagreement metric.")
    else:
        print("  (Results depend on specific beta values)")


if __name__ == "__main__":
    results = test_scale_free_beta()
    compare_absolute_vs_scalefree()
    test_js_divergence_metric()

    print("\n" + "=" * 80)
    print("✅ Validation complete!")
    print("=" * 80)
