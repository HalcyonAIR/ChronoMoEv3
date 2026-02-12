#!/usr/bin/env python3
"""
Phase 3: Attack Surface Exhaustion → Convergence Correlation

Test hypothesis: Method-space clustering precedes convergence.

Experiment design:
1. Measure baseline method diversity distributions (natural behavior)
2. Run both detectors simultaneously (exhaustion + convergence)
3. Test correlation: Does exhaustion precede convergence?
4. Build ROC curve: Precision/recall of exhaustion as convergence predictor

Expected result:
- If exhaustion is predictive: Exhaustion events occur 50-200 steps before convergence
- If not predictive: Random timing or happens simultaneously
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our modules
from chronomoe_integration.convergence import (
    ConvergenceDetector,
    ConvergenceThresholds,
    derive_thresholds_from_baseline as derive_convergence_thresholds,
    compute_deformation_regime,
)
from chronomoe_integration.challenge_receipts import (
    ExhaustionDetector,
    ExhaustionThresholds,
    ChallengeReceipt,
    derive_exhaustion_thresholds_from_baseline,
    compute_method_diversity,
)

# Model setup (simplified for testing)
# We don't need the full ChronoMoE layer - just need to simulate behavior


class MockLayer:
    """
    Mock layer for testing behavioral fingerprinting.

    We don't need a real MoE layer - just need to simulate routing patterns.
    """
    def __init__(self, d_model: int, max_experts: int):
        self.d_model = d_model
        self.max_experts = max_experts
        self.last_routing_weights = None

    def forward(self, x: torch.Tensor, scar_strength: float) -> torch.Tensor:
        """
        Simulate forward pass with routing pattern that depends on input and scar.

        As scar increases, routing becomes more concentrated (less diverse).
        """
        batch_size = x.shape[0]

        # Generate routing pattern based on input hash and scar strength
        # As scar increases, routing concentrates on fewer experts
        input_hash = float(torch.sum(x).item()) % 1.0

        # Base routing (uniform with noise)
        routing = torch.ones(self.max_experts) / self.max_experts

        # Add input-dependent bias
        for i in range(self.max_experts):
            routing[i] += 0.3 * np.sin(2 * np.pi * i / self.max_experts + input_hash * 10)

        # Apply scar-dependent concentration
        # As scar increases, concentrate on expert 0
        concentration = np.clip(scar_strength / 5.0, 0, 1)  # 0.0 at scar=0, 1.0 at scar=5
        routing[0] += concentration * 2.0

        # Normalize
        routing = routing / routing.sum()

        self.last_routing_weights = routing

        # Simulate output (weighted sum based on routing)
        output = x * (1 + concentration)  # Output magnitude increases with scar

        return output


def create_test_model(
    d_model: int = 128,
    max_experts: int = 8,
) -> MockLayer:
    """Create mock layer for testing."""
    return MockLayer(d_model, max_experts)


def create_synthetic_challenge(
    step: int,
    d_model: int,
    challenge_type: str = "normal",
) -> torch.Tensor:
    """
    Create synthetic challenge for testing.

    Types:
    - normal: Random input
    - mode_a: Clustered input (mode A)
    - mode_b: Clustered input (mode B)
    """
    if challenge_type == "mode_a":
        # Mode A: bias toward certain dimensions
        base = torch.randn(1, d_model) * 0.5
        base[:, :d_model//2] += 2.0  # Strong signal in first half
        return base

    elif challenge_type == "mode_b":
        # Mode B: bias toward different dimensions
        base = torch.randn(1, d_model) * 0.5
        base[:, d_model//2:] += 2.0  # Strong signal in second half
        return base

    else:  # normal
        return torch.randn(1, d_model)


def extract_behavioral_fingerprint(
    layer: MockLayer,
    input_tensor: torch.Tensor,
    step: int,
    scar_strength: float,
) -> ChallengeReceipt:
    """
    Extract behavioral fingerprint from layer's response to input.

    This is BEHAVIORAL EVIDENCE, not self-report.
    We observe what the layer does, not what it says.
    """
    with torch.no_grad():
        # Forward pass
        output = layer.forward(input_tensor, scar_strength)

        # Extract routing pattern (which experts fired)
        routing_pattern = layer.last_routing_weights.cpu().numpy() if isinstance(layer.last_routing_weights, torch.Tensor) else layer.last_routing_weights.numpy()

        # Activation magnitude
        activation_magnitude = float(torch.norm(output).item())

        # Gradient direction (use output as proxy)
        gradient_direction = output.cpu().numpy().flatten()
        gradient_direction = gradient_direction / (np.linalg.norm(gradient_direction) + 1e-8)

        # Determine regime
        regime = compute_deformation_regime(scar_strength).value

        receipt = ChallengeReceipt(
            step=step,
            challenge_id=step,  # Simple: one challenge per step
            routing_pattern=routing_pattern,
            activation_magnitude=activation_magnitude,
            gradient_direction=gradient_direction,
            scar_strength=scar_strength,
            regime=regime,
        )

        return receipt


def simulate_training_trajectory(
    num_steps: int,
    d_model: int,
    scar_schedule: str = "gradual",
) -> Tuple[List[ChallengeReceipt], List[Dict]]:
    """
    Simulate training trajectory with increasing scar strength.

    Returns:
        (receipts, convergence_observations)
    """
    layer = create_test_model(d_model=d_model)

    receipts: List[ChallengeReceipt] = []
    convergence_obs: List[Dict] = []

    # Scar schedule
    if scar_schedule == "gradual":
        # Gradual increase: 0.0 → 5.0 over training
        scar_strengths = np.linspace(0.0, 5.0, num_steps)
    elif scar_schedule == "sudden":
        # Sudden jump: 0.0 for half, then jump to 4.0
        scar_strengths = np.concatenate([
            np.zeros(num_steps // 2),
            np.full(num_steps // 2, 4.0),
        ])
    else:  # "cyclic"
        # Cyclic: oscillate between 0.0 and 5.0
        scar_strengths = 2.5 * (1 + np.sin(np.linspace(0, 4 * np.pi, num_steps)))

    for step in range(num_steps):
        scar_strength = float(scar_strengths[step])

        # Determine challenge type based on scar strength
        # As scar increases, challenges become more clustered (less diverse)
        if scar_strength < 2.0:
            # Exploration: diverse challenges
            challenge_type = np.random.choice(["normal", "mode_a", "mode_b"], p=[0.7, 0.15, 0.15])
        elif scar_strength < 4.0:
            # Transition: starting to cluster
            challenge_type = np.random.choice(["normal", "mode_a", "mode_b"], p=[0.3, 0.35, 0.35])
        else:
            # Saturation: heavily clustered (mode A dominant)
            challenge_type = np.random.choice(["normal", "mode_a", "mode_b"], p=[0.1, 0.7, 0.2])

        # Create challenge
        input_tensor = create_synthetic_challenge(step, d_model, challenge_type)

        # Extract behavioral fingerprint
        receipt = extract_behavioral_fingerprint(layer, input_tensor, step, scar_strength)
        receipts.append(receipt)

        # Convergence observation (simplified)
        # In saturation, Bob and maniac start agreeing more
        regime = compute_deformation_regime(scar_strength)

        if regime.value == "saturation":
            bob_agreed = np.random.random() < 0.8  # High agreement in saturation
            maniac_downgraded = np.random.random() < 0.1  # Few downgrades
        elif regime.value == "transition":
            bob_agreed = np.random.random() < 0.5
            maniac_downgraded = np.random.random() < 0.3
        else:  # exploration
            bob_agreed = np.random.random() < 0.2
            maniac_downgraded = np.random.random() < 0.5

        # Diversity (routing entropy as proxy)
        diversity_current = -np.sum(receipt.routing_pattern * np.log(receipt.routing_pattern + 1e-8))

        convergence_obs.append({
            "step": step,
            "bob_agreed_with_maniac": bob_agreed,
            "claim_impact": 0.8,  # Fixed high impact
            "maniac_downgraded": maniac_downgraded,
            "diversity_current": float(diversity_current),
            "scar_strength_total": scar_strength,
        })

    return receipts, convergence_obs


def measure_baseline_distributions(
    num_steps: int = 5000,
    d_model: int = 128,
) -> Dict:
    """
    Phase 1: Measure baseline distributions for both exhaustion and convergence.

    Returns:
        Baseline data with both exhaustion and convergence distributions
    """
    print("=" * 70)
    print("PHASE 3 BASELINE MEASUREMENT: Method Diversity + Convergence")
    print("=" * 70)
    print()
    print(f"Measuring {num_steps} steps to derive thresholds from percentiles")
    print()

    # Simulate training
    receipts, convergence_obs = simulate_training_trajectory(
        num_steps=num_steps,
        d_model=d_model,
        scar_schedule="gradual",
    )

    # Measure method diversity over time (sliding window)
    print("Phase 1: Measuring Method Diversity Distributions")
    print("-" * 70)

    window_size = 100
    diversity_samples = []
    cluster_fractions = []

    for i in range(window_size, len(receipts), 10):  # Sample every 10 steps
        window_receipts = receipts[i-window_size:i]

        # Compute diversity
        diversity = compute_method_diversity(window_receipts, sample_size=50)
        diversity_samples.append(diversity)

        # Compute clustering (simplified: use routing pattern variance)
        routing_patterns = np.array([r.routing_pattern for r in window_receipts])
        pattern_variance = np.std(routing_patterns, axis=0)

        # Cluster fraction: what fraction of variance is in dominant expert
        if np.sum(pattern_variance) > 0:
            largest_fraction = np.max(pattern_variance) / np.sum(pattern_variance)
        else:
            largest_fraction = 1.0

        cluster_fractions.append(largest_fraction)

    print(f"Method diversity samples: n={len(diversity_samples)}")
    print(f"  Mean: {np.mean(diversity_samples):.3f}")
    print(f"  Median: {np.median(diversity_samples):.3f}")
    print(f"  10th percentile: {np.percentile(diversity_samples, 10):.3f}")
    print(f"  90th percentile: {np.percentile(diversity_samples, 90):.3f}")
    print()

    print(f"Cluster fractions: n={len(cluster_fractions)}")
    print(f"  Mean: {np.mean(cluster_fractions):.3f}")
    print(f"  Median: {np.median(cluster_fractions):.3f}")
    print(f"  10th percentile: {np.percentile(cluster_fractions, 10):.3f}")
    print(f"  90th percentile: {np.percentile(cluster_fractions, 90):.3f}")
    print()

    # Measure convergence distributions (reuse Phase 2 logic)
    agreement_lengths = []
    silence_periods = []
    diversity_by_regime = {"exploration": [], "transition": [], "saturation": []}

    current_agreement = 0
    current_silence = 0
    last_downgrade = -1

    for obs in convergence_obs:
        # Agreement tracking
        if obs["bob_agreed_with_maniac"] and obs["claim_impact"] >= 0.7:
            current_agreement += 1
        else:
            if current_agreement > 0:
                agreement_lengths.append(current_agreement)
            current_agreement = 0

        # Silence tracking
        if obs["maniac_downgraded"]:
            if last_downgrade >= 0:
                silence = obs["step"] - last_downgrade
                silence_periods.append(silence)
            last_downgrade = obs["step"]

        # Diversity by regime
        regime = compute_deformation_regime(obs["scar_strength_total"]).value
        diversity_by_regime[regime].append(obs["diversity_current"])

    # Final agreement if any
    if current_agreement > 0:
        agreement_lengths.append(current_agreement)

    print("Convergence distributions:")
    print(f"  Agreement lengths: n={len(agreement_lengths)}, "
          f"mean={np.mean(agreement_lengths) if agreement_lengths else 0:.1f}, "
          f"90th={np.percentile(agreement_lengths, 90) if agreement_lengths else 0:.0f}")
    print(f"  Silence periods: n={len(silence_periods)}, "
          f"mean={np.mean(silence_periods) if silence_periods else 0:.1f}, "
          f"90th={np.percentile(silence_periods, 90) if silence_periods else 0:.0f}")
    print()

    baseline_data = {
        # Exhaustion baseline
        "method_diversity_samples": diversity_samples,
        "cluster_fractions": cluster_fractions,

        # Convergence baseline
        "agreement_lengths": agreement_lengths,
        "silence_periods": silence_periods,
        "diversity_by_regime": diversity_by_regime,
    }

    # Save baseline
    baseline_path = Path("exhaustion_baseline.json")
    with open(baseline_path, "w") as f:
        json.dump({k: v if isinstance(v, dict) else [float(x) for x in v]
                   for k, v in baseline_data.items()}, f, indent=2)

    print(f"Baseline saved: {baseline_path}")
    print()

    return baseline_data


def test_exhaustion_convergence_correlation(
    baseline_data: Dict,
    num_steps: int = 5000,
    d_model: int = 128,
) -> Dict:
    """
    Phase 2: Test correlation between exhaustion and convergence.

    Run both detectors simultaneously and measure:
    1. Do exhaustion events precede convergence events?
    2. By how many steps?
    3. Precision/recall of exhaustion as predictor

    Returns:
        Correlation statistics
    """
    print("=" * 70)
    print("PHASE 3 CORRELATION TEST: Exhaustion → Convergence")
    print("=" * 70)
    print()

    # Derive thresholds
    exhaustion_thresholds = derive_exhaustion_thresholds_from_baseline(baseline_data)
    convergence_thresholds = derive_convergence_thresholds(baseline_data)

    print("Derived Thresholds:")
    print("-" * 70)
    print(f"Exhaustion:")
    print(f"  diversity_min: {exhaustion_thresholds.diversity_min:.4f}")
    print(f"  max_cluster_fraction: {exhaustion_thresholds.max_cluster_fraction:.4f}")
    print(f"  K_persistence: {exhaustion_thresholds.K_persistence}")
    print()
    print(f"Convergence:")
    print(f"  W_min: {convergence_thresholds.W_min}")
    print(f"  S_min: {convergence_thresholds.S_min}")
    print(f"  D_exploration: {convergence_thresholds.D_exploration:.4f}")
    print(f"  D_transition: {convergence_thresholds.D_transition:.4f}")
    print(f"  D_saturation: {convergence_thresholds.D_saturation:.4f}")
    print()

    # Create detectors
    exhaustion_detector = ExhaustionDetector(exhaustion_thresholds, window_size=100)

    # Convergence detector needs baseline diversity
    baseline_diversity = {
        compute_deformation_regime(0.5).value: np.mean(baseline_data["diversity_by_regime"]["exploration"]) if baseline_data["diversity_by_regime"]["exploration"] else 1.0,
        compute_deformation_regime(3.0).value: np.mean(baseline_data["diversity_by_regime"]["transition"]) if baseline_data["diversity_by_regime"]["transition"] else 1.0,
        compute_deformation_regime(5.0).value: np.mean(baseline_data["diversity_by_regime"]["saturation"]) if baseline_data["diversity_by_regime"]["saturation"] else 1.0,
    }
    convergence_detector = ConvergenceDetector(convergence_thresholds, baseline_diversity)

    # Simulate training (same trajectory type as baseline)
    print("Running Correlation Test (5000 steps)")
    print("-" * 70)

    receipts, convergence_obs = simulate_training_trajectory(
        num_steps=num_steps,
        d_model=d_model,
        scar_schedule="gradual",
    )

    # Run both detectors
    for i, (receipt, obs) in enumerate(zip(receipts, convergence_obs)):
        # Update exhaustion detector
        exhaustion_state = exhaustion_detector.update(receipt)

        # Update convergence detector
        convergence_state = convergence_detector.update(
            step=obs["step"],
            bob_agreed_with_maniac=obs["bob_agreed_with_maniac"],
            claim_impact=obs["claim_impact"],
            maniac_downgraded=obs["maniac_downgraded"],
            diversity_current=obs["diversity_current"],
            scar_strength_total=obs["scar_strength_total"],
        )

        # Log transitions
        if exhaustion_state.is_exhausted and exhaustion_state.exhausted_since == obs["step"]:
            print(f"  Step {obs['step']:4d}: Exhaustion detected (regime={exhaustion_state.regime})")

        if convergence_state.is_converged and convergence_state.converged_since == obs["step"]:
            print(f"  Step {obs['step']:4d}: Convergence detected (regime={convergence_state.regime})")

    print()

    # Analyze correlation
    print("=" * 70)
    print("CORRELATION STATISTICS")
    print("=" * 70)
    print()

    exhaustion_events = exhaustion_detector.exhaustion_events
    convergence_events = convergence_detector.convergence_events

    print(f"Exhaustion events: {len(exhaustion_events)}")
    print(f"Convergence events: {len(convergence_events)}")
    print()

    # Measure lead time: exhaustion → convergence
    lead_times = []

    for conv_event in convergence_events:
        # Find most recent exhaustion event before this convergence
        preceding_exhaustion = None
        for exh_event in exhaustion_events:
            if exh_event.step < conv_event.step:
                if preceding_exhaustion is None or exh_event.step > preceding_exhaustion.step:
                    preceding_exhaustion = exh_event

        if preceding_exhaustion is not None:
            lead_time = conv_event.step - preceding_exhaustion.step
            lead_times.append(lead_time)

            # Update convergence event with lead time
            conv_event.steps_before_convergence = lead_time

            print(f"Convergence at step {conv_event.step}: exhaustion preceded by {lead_time} steps")

    if lead_times:
        print()
        print(f"Lead time statistics (n={len(lead_times)}):")
        print(f"  Mean: {np.mean(lead_times):.1f} steps")
        print(f"  Median: {np.median(lead_times):.1f} steps")
        print(f"  Range: [{min(lead_times)}, {max(lead_times)}] steps")
    else:
        print()
        print("No convergence events preceded by exhaustion")

    print()

    # Predictive power: Did exhaustion predict convergence?
    exhaustion_predicted_convergence = len(lead_times) > 0

    if exhaustion_predicted_convergence:
        precision = len(lead_times) / len(exhaustion_events) if exhaustion_events else 0.0
        recall = len(lead_times) / len(convergence_events) if convergence_events else 0.0

        print(f"Predictive Power:")
        print(f"  Precision: {precision:.2%} (fraction of exhaustion events that preceded convergence)")
        print(f"  Recall: {recall:.2%} (fraction of convergence events preceded by exhaustion)")
    else:
        print("Exhaustion did NOT predict convergence (no lead times observed)")

    print()

    # Save results
    results = {
        "num_exhaustion_events": len(exhaustion_events),
        "num_convergence_events": len(convergence_events),
        "lead_times": lead_times,
        "lead_time_stats": {
            "mean": float(np.mean(lead_times)) if lead_times else 0.0,
            "median": float(np.median(lead_times)) if lead_times else 0.0,
            "min": int(min(lead_times)) if lead_times else 0,
            "max": int(max(lead_times)) if lead_times else 0,
        },
        "predictive_power": {
            "precision": float(len(lead_times) / len(exhaustion_events)) if exhaustion_events else 0.0,
            "recall": float(len(lead_times) / len(convergence_events)) if convergence_events else 0.0,
        },
        "exhaustion_events": [
            {
                "step": e.step,
                "regime": e.regime,
                "scar_strength": float(e.scar_strength),
                "method_diversity": float(e.method_diversity),
                "num_clusters": e.num_clusters,
                "largest_cluster_fraction": float(e.largest_cluster_fraction),
            }
            for e in exhaustion_events
        ],
        "convergence_events": [
            {
                "step": e.step,
                "regime": e.regime.value,
                "scar_strength": float(e.scar_strength),
                "steps_before_convergence": e.steps_before_convergence,
            }
            for e in convergence_events
        ],
    }

    results_path = Path("exhaustion_correlation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return results


def main():
    """Run Phase 3: Exhaustion → Convergence correlation test."""

    # Phase 1: Measure baseline distributions
    baseline_data = measure_baseline_distributions(num_steps=5000, d_model=128)

    # Phase 2: Test correlation
    results = test_exhaustion_convergence_correlation(
        baseline_data,
        num_steps=5000,
        d_model=128,
    )

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Baseline distributions measured for method diversity and convergence")
    print("✓ Thresholds derived from percentiles (not intuition)")
    print("✓ Both detectors instrumented (exhaustion + convergence)")
    print("✓ Correlation tested: exhaustion → convergence lead time")
    print()

    if results["num_exhaustion_events"] > 0 and results["num_convergence_events"] > 0:
        if results["lead_time_stats"]["mean"] > 0:
            print(f"✓ PREDICTIVE: Exhaustion precedes convergence by ~{results['lead_time_stats']['mean']:.0f} steps")
            print(f"  Precision: {results['predictive_power']['precision']:.1%}, "
                  f"Recall: {results['predictive_power']['recall']:.1%}")
        else:
            print("✗ NOT PREDICTIVE: Exhaustion does not precede convergence")
    else:
        print("⚠ Insufficient events for correlation analysis")
    print()
    print("Ready for Phase 4: Policy integration (intervention on exhaustion signal)")
    print()


if __name__ == "__main__":
    main()
