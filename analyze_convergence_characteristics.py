#!/usr/bin/env python3
"""
Convergence Characteristics Analysis

Measure, don't tune. Sharpen hypotheses before real training.

Questions:
1. Base rate: How often does convergence occur? (K=5 vs K=3)
2. Lead time by regime: Does early warning collapse in saturation?
3. Convergence proximity: What conditions trigger convergence?
4. Performance proxy: Does convergence correlate with adaptation failure?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from experiment_exhaustion_correlation import (
    simulate_training_trajectory,
    create_synthetic_challenge,
)
from chronomoe_integration.convergence import (
    ConvergenceDetector,
    ConvergenceThresholds,
    derive_thresholds_from_baseline as derive_convergence_thresholds,
    compute_deformation_regime,
)
from chronomoe_integration.challenge_receipts import (
    ExhaustionDetector,
    ExhaustionThresholds,
    derive_exhaustion_thresholds_from_baseline,
)


def measure_base_rate(
    convergence_detector: ConvergenceDetector,
    convergence_obs: List[Dict],
    K: int,
) -> Dict:
    """
    Measure convergence base rate.

    Returns:
        - pct_time_converged: % of steps in converged state
        - mean_duration: Average duration of convergence episodes
        - persistence_distribution: Histogram of persistence lengths
    """
    # Track convergence episodes
    episodes = []
    current_episode_start = None
    persistence_lengths = []

    for obs in convergence_obs:
        state = convergence_detector.update(
            step=obs["step"],
            bob_agreed_with_maniac=obs["bob_agreed_with_maniac"],
            claim_impact=obs["claim_impact"],
            maniac_downgraded=obs["maniac_downgraded"],
            diversity_current=obs["diversity_current"],
            scar_strength_total=obs["scar_strength_total"],
        )

        # Track persistence lengths
        if state.persistence_count > 0:
            persistence_lengths.append(state.persistence_count)

        # Track episodes
        if state.is_converged and current_episode_start is None:
            current_episode_start = state.converged_since

        elif not state.is_converged and current_episode_start is not None:
            # Episode ended
            duration = obs["step"] - current_episode_start
            episodes.append({
                "start": current_episode_start,
                "end": obs["step"],
                "duration": duration,
            })
            current_episode_start = None

    # Final episode if still converged
    if current_episode_start is not None:
        duration = convergence_obs[-1]["step"] - current_episode_start
        episodes.append({
            "start": current_episode_start,
            "end": convergence_obs[-1]["step"],
            "duration": duration,
        })

    # Compute metrics
    num_steps = len(convergence_obs)
    steps_converged = sum(ep["duration"] for ep in episodes)
    pct_time_converged = (steps_converged / num_steps * 100) if num_steps > 0 else 0.0

    mean_duration = np.mean([ep["duration"] for ep in episodes]) if episodes else 0.0

    # Persistence distribution
    persistence_hist = defaultdict(int)
    for p in persistence_lengths:
        persistence_hist[p] += 1

    return {
        "K": K,
        "num_episodes": len(episodes),
        "pct_time_converged": pct_time_converged,
        "mean_duration": float(mean_duration),
        "std_duration": float(np.std([ep["duration"] for ep in episodes])) if episodes else 0.0,
        "persistence_distribution": dict(persistence_hist),
        "max_persistence": max(persistence_lengths) if persistence_lengths else 0,
    }


def stratify_lead_time_by_regime(
    exhaustion_events: List,
    convergence_events: List,
) -> Dict:
    """
    Stratify exhaustion→convergence lead time by scar regime.

    Returns:
        Mean ± std lead time per regime (exploration/transition/saturation)
    """
    lead_times_by_regime = {
        "exploration": [],
        "transition": [],
        "saturation": [],
    }

    for conv_event in convergence_events:
        # Find most recent exhaustion before this convergence
        preceding_exhaustion = None
        for exh_event in exhaustion_events:
            if exh_event["step"] < conv_event["step"]:
                if preceding_exhaustion is None or exh_event["step"] > preceding_exhaustion["step"]:
                    preceding_exhaustion = exh_event

        if preceding_exhaustion:
            lead_time = conv_event["step"] - preceding_exhaustion["step"]
            regime = conv_event["regime"]
            lead_times_by_regime[regime].append(lead_time)

    # Compute stats per regime
    stats = {}
    for regime, lead_times in lead_times_by_regime.items():
        if lead_times:
            stats[regime] = {
                "n": len(lead_times),
                "mean": float(np.mean(lead_times)),
                "std": float(np.std(lead_times)),
                "min": int(min(lead_times)),
                "max": int(max(lead_times)),
            }
        else:
            stats[regime] = {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
            }

    return stats


def measure_convergence_proximity(
    convergence_events: List,
) -> Dict:
    """
    Measure conditions at convergence.

    Returns:
        - average_diversity: Mean routing diversity at convergence
        - average_scar: Mean scar strength at convergence
        - volatility: Variance in routing patterns at convergence
    """
    if not convergence_events:
        return {
            "n": 0,
            "diversity": {"mean": 0.0, "std": 0.0},
            "scar_strength": {"mean": 0.0, "std": 0.0},
            "regime_distribution": {"exploration": 0, "transition": 0, "saturation": 0},
        }

    diversities = []
    scars = []
    regime_counts = defaultdict(int)

    for event in convergence_events:
        # Get diversity and scar from event (if available)
        # For now, use what's logged in the event
        if "diversity" in event:
            diversities.append(event["diversity"])
        if "scar_strength" in event:
            scars.append(event["scar_strength"])
        if "regime" in event:
            regime_counts[event["regime"]] += 1

    return {
        "n": len(convergence_events),
        "diversity": {
            "mean": float(np.mean(diversities)) if diversities else 0.0,
            "std": float(np.std(diversities)) if diversities else 0.0,
        },
        "scar_strength": {
            "mean": float(np.mean(scars)) if scars else 0.0,
            "std": float(np.std(scars)) if scars else 0.0,
        },
        "regime_distribution": {
            "exploration": regime_counts.get("exploration", 0),
            "transition": regime_counts.get("transition", 0),
            "saturation": regime_counts.get("saturation", 0),
        },
    }


def measure_performance_proxy(
    convergence_obs: List[Dict],
    convergence_events: List,
    window_size: int = 50,
) -> Dict:
    """
    Lightweight performance proxy.

    Measure routing volatility before/after convergence.

    Args:
        convergence_obs: Full observation sequence
        convergence_events: List of convergence events
        window_size: Steps before/after convergence to measure

    Returns:
        - volatility_before: Mean routing diversity before convergence
        - volatility_after: Mean routing diversity after convergence
        - volatility_ratio: after/before (>1 = increased volatility)
    """
    if not convergence_events:
        return {
            "n": 0,
            "volatility_before": 0.0,
            "volatility_after": 0.0,
            "volatility_ratio": 0.0,
        }

    volatility_before_list = []
    volatility_after_list = []

    for event in convergence_events:
        conv_step = event["step"]

        # Get window before convergence
        before_window = [
            obs for obs in convergence_obs
            if conv_step - window_size <= obs["step"] < conv_step
        ]

        # Get window after convergence
        after_window = [
            obs for obs in convergence_obs
            if conv_step <= obs["step"] < conv_step + window_size
        ]

        # Compute volatility (routing diversity variance)
        if before_window:
            before_div = [obs["diversity_current"] for obs in before_window]
            volatility_before_list.append(np.std(before_div))

        if after_window:
            after_div = [obs["diversity_current"] for obs in after_window]
            volatility_after_list.append(np.std(after_div))

    volatility_before = float(np.mean(volatility_before_list)) if volatility_before_list else 0.0
    volatility_after = float(np.mean(volatility_after_list)) if volatility_after_list else 0.0
    volatility_ratio = (volatility_after / volatility_before) if volatility_before > 0 else 0.0

    return {
        "n": len(convergence_events),
        "volatility_before": volatility_before,
        "volatility_after": volatility_after,
        "volatility_ratio": volatility_ratio,
    }


def run_analysis(seed: int = 42) -> Dict:
    """
    Run full convergence characteristics analysis.

    Returns:
        Comprehensive report (K=5 and K=3 comparison)
    """
    print("=" * 70)
    print(f"CONVERGENCE CHARACTERISTICS ANALYSIS (seed={seed})")
    print("=" * 70)
    print()

    # Set seed
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load baseline from existing file
    with open("exhaustion_baseline.json", "r") as f:
        baseline_data = json.load(f)

    # Derive thresholds
    convergence_baseline = {
        "agreement_lengths": baseline_data["agreement_lengths"],
        "silence_periods": baseline_data["silence_periods"],
        "diversity_by_regime": baseline_data["convergence_diversity_by_regime"],
    }

    exhaustion_thresholds = derive_exhaustion_thresholds_from_baseline(baseline_data)

    # Baseline diversity for convergence detector
    baseline_diversity = {
        "exploration": np.mean(baseline_data["convergence_diversity_by_regime"]["exploration"]) if baseline_data["convergence_diversity_by_regime"]["exploration"] else 1.0,
        "transition": np.mean(baseline_data["convergence_diversity_by_regime"]["transition"]) if baseline_data["convergence_diversity_by_regime"]["transition"] else 1.0,
        "saturation": np.mean(baseline_data["convergence_diversity_by_regime"]["saturation"]) if baseline_data["convergence_diversity_by_regime"]["saturation"] else 1.0,
    }

    # Simulate training
    print("Simulating training trajectory (5000 steps, extreme saturation mode)...")
    receipts, convergence_obs = simulate_training_trajectory(
        num_steps=5000,
        d_model=128,
        scar_schedule="gradual",
        saturation_mode="extreme",
    )
    print()

    # Run analysis for K=5 and K=3
    results = {}

    for K in [5, 3]:
        print(f"Analyzing with K_persistence={K}")
        print("-" * 70)

        # Derive thresholds with specific K
        thresholds_K = derive_convergence_thresholds(convergence_baseline)
        thresholds_K.K_persistence = K

        # Create detectors
        convergence_detector = ConvergenceDetector(thresholds_K, baseline_diversity)
        exhaustion_detector = ExhaustionDetector(exhaustion_thresholds, window_size=100)

        # Run both detectors
        for receipt, obs in zip(receipts, convergence_obs):
            exhaustion_detector.update(receipt)
            convergence_detector.update(
                step=obs["step"],
                bob_agreed_with_maniac=obs["bob_agreed_with_maniac"],
                claim_impact=obs["claim_impact"],
                maniac_downgraded=obs["maniac_downgraded"],
                diversity_current=obs["diversity_current"],
                scar_strength_total=obs["scar_strength_total"],
            )

        # Get events
        exhaustion_events = [
            {
                "step": e.step,
                "regime": e.regime,
                "scar_strength": float(e.scar_strength),
                "method_diversity": float(e.method_diversity),
            }
            for e in exhaustion_detector.exhaustion_events
        ]

        convergence_events = [
            {
                "step": e.step,
                "regime": e.regime.value,
                "scar_strength": float(e.scar_strength),
                "diversity": float(e.diversity),
            }
            for e in convergence_detector.convergence_events
        ]

        # 1. Base rate (use a fresh detector to re-measure)
        print(f"  1. Base rate analysis...")
        base_rate_detector = ConvergenceDetector(thresholds_K, baseline_diversity)
        base_rate = measure_base_rate(
            base_rate_detector,
            convergence_obs,
            K,
        )

        # 2. Lead time by regime
        print(f"  2. Lead time stratification...")
        lead_time_by_regime = stratify_lead_time_by_regime(
            exhaustion_events,
            convergence_events,
        )

        # 3. Convergence proximity
        print(f"  3. Convergence proximity...")
        proximity = measure_convergence_proximity(convergence_events)

        # 4. Performance proxy
        print(f"  4. Performance proxy...")
        performance = measure_performance_proxy(
            convergence_obs,
            convergence_events,
        )

        results[f"K={K}"] = {
            "base_rate": base_rate,
            "lead_time_by_regime": lead_time_by_regime,
            "proximity": proximity,
            "performance_proxy": performance,
            "num_exhaustion_events": len(exhaustion_events),
            "num_convergence_events": len(convergence_events),
        }

        print()

    return results


def print_report(results: Dict):
    """Print comprehensive analysis report."""
    print("=" * 70)
    print("ANALYSIS REPORT")
    print("=" * 70)
    print()

    for K_label in ["K=5", "K=3"]:
        r = results[K_label]
        K = r["base_rate"]["K"]

        print(f"{K_label}: Production threshold" if K == 5 else f"{K_label}: Synthetic validation")
        print("=" * 70)
        print()

        # Base rate
        print("1. BASE RATE")
        print("-" * 70)
        br = r["base_rate"]
        print(f"  Convergence episodes: {br['num_episodes']}")
        print(f"  % time converged: {br['pct_time_converged']:.2f}%")
        print(f"  Mean duration: {br['mean_duration']:.1f} ± {br['std_duration']:.1f} steps")
        print(f"  Max persistence reached: {br['max_persistence']}")
        print(f"  Persistence distribution:")
        for p in sorted(br["persistence_distribution"].keys()):
            count = br["persistence_distribution"][p]
            print(f"    {p}: {count} occurrences")
        print()

        # Lead time by regime
        print("2. LEAD TIME BY REGIME")
        print("-" * 70)
        for regime in ["exploration", "transition", "saturation"]:
            lt = r["lead_time_by_regime"][regime]
            if lt["n"] > 0:
                print(f"  {regime.capitalize()}:")
                print(f"    n = {lt['n']}")
                print(f"    mean = {lt['mean']:.1f} ± {lt['std']:.1f} steps")
                print(f"    range = [{lt['min']}, {lt['max']}] steps")
            else:
                print(f"  {regime.capitalize()}: No convergence events")
        print()

        # Proximity
        print("3. CONVERGENCE PROXIMITY")
        print("-" * 70)
        prox = r["proximity"]
        print(f"  Convergence events: {prox['n']}")
        if prox["n"] > 0:
            print(f"  Average diversity: {prox['diversity']['mean']:.4f} ± {prox['diversity']['std']:.4f}")
            print(f"  Average scar strength: {prox['scar_strength']['mean']:.2f} ± {prox['scar_strength']['std']:.2f}")
            print(f"  Regime distribution:")
            print(f"    Exploration: {prox['regime_distribution']['exploration']}")
            print(f"    Transition: {prox['regime_distribution']['transition']}")
            print(f"    Saturation: {prox['regime_distribution']['saturation']}")
        print()

        # Performance proxy
        print("4. PERFORMANCE PROXY (Routing Volatility)")
        print("-" * 70)
        perf = r["performance_proxy"]
        if perf["n"] > 0:
            print(f"  Convergence events analyzed: {perf['n']}")
            print(f"  Volatility before convergence: {perf['volatility_before']:.4f}")
            print(f"  Volatility after convergence: {perf['volatility_after']:.4f}")
            print(f"  Volatility ratio (after/before): {perf['volatility_ratio']:.2f}x")
            if perf["volatility_ratio"] > 1.1:
                print(f"  → Increased volatility after convergence (unstable)")
            elif perf["volatility_ratio"] < 0.9:
                print(f"  → Decreased volatility after convergence (locked in)")
            else:
                print(f"  → Volatility unchanged")
        else:
            print(f"  No convergence events to analyze")
        print()

        print()

    # Comparison
    print("=" * 70)
    print("K=5 vs K=3 COMPARISON")
    print("=" * 70)
    print()

    k5 = results["K=5"]
    k3 = results["K=3"]

    print(f"Convergence episodes:")
    print(f"  K=5: {k5['base_rate']['num_episodes']}")
    print(f"  K=3: {k3['base_rate']['num_episodes']}")
    print()

    print(f"% time converged:")
    print(f"  K=5: {k5['base_rate']['pct_time_converged']:.2f}%")
    print(f"  K=3: {k3['base_rate']['pct_time_converged']:.2f}%")
    print()

    print(f"Mean lead time (all regimes):")
    k5_lead = [lt["mean"] for lt in k5["lead_time_by_regime"].values() if lt["n"] > 0]
    k3_lead = [lt["mean"] for lt in k3["lead_time_by_regime"].values() if lt["n"] > 0]
    if k5_lead:
        print(f"  K=5: {np.mean(k5_lead):.1f} steps")
    else:
        print(f"  K=5: N/A (no convergence)")
    if k3_lead:
        print(f"  K=3: {np.mean(k3_lead):.1f} steps")
    else:
        print(f"  K=3: N/A (no convergence)")
    print()


def main():
    """Run analysis and save report."""
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    results = run_analysis(seed=seed)

    # Print report
    print_report(results)

    # Save results
    results_path = Path(f"convergence_characteristics_analysis_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()


if __name__ == "__main__":
    main()
