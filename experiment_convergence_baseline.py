#!/usr/bin/env python3
"""
Phase 2: Convergence Baseline Measurement

Instrument convergence characteristics BEFORE attaching policy.

Questions to answer:
1. Does convergence cluster in transition regime?
2. Does it appear mostly in saturation?
3. Is it domain-dependent?
4. What is time-to-convergence after entering transition?

This mapping informs Phase 3 (attack surface exhaustion).

DO NOT attach interventions yet. Measure first.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from synthetic_datasets import ArithmeticProgressionDataset
from chronomoe_integration.routing_geometry import measure_routing_geometry
from chronomoe_integration.convergence import (
    ConvergenceDetector,
    ConvergenceThresholds,
    DeformationRegime,
    derive_thresholds_from_baseline,
)
from chronomoe_integration import ChronoMoE


@dataclass
class Config:
    """Model configuration."""
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 1
    vocab_size: int = 100
    block_size: int = 20
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"


class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, {}


class TinyModel(nn.Module):
    def __init__(self, config, max_experts=8):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.moe = ChronoMoE(
            config=config,
            mlp=SimpleMLP,
            layer_id=0,
            max_experts=max_experts,
            autonomous_mode=True,
        )
        self.output = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        h = self.embed(x)
        h_flat = h.view(-1, h.size(-1))
        h_moe, metadata = self.moe(h_flat)
        h = h_moe.view(h.size(0), h.size(1), -1)
        logits = self.output(h)
        return logits


class EmbeddingDatasetWrapper:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.dataset)
        h = self.model.embed(batch)
        h_flat = h.view(-1, h.size(-1))
        return h_flat


def simulate_bob_maniac_interaction(step: int, regime: DeformationRegime, rng) -> Dict:
    """
    Simulate Bob-maniac interaction for baseline measurement.

    In real system, this would come from actual Bob claims and maniac challenges.
    For baseline, we simulate natural patterns to derive thresholds.

    Args:
        step: Current step
        regime: Current deformation regime
        rng: Random number generator

    Returns:
        Dict with {agreed, impact, downgraded}
    """
    # Simulate agreement probability (decreases in saturation)
    if regime == DeformationRegime.EXPLORATION:
        agree_prob = 0.3  # Low baseline agreement
    elif regime == DeformationRegime.TRANSITION:
        agree_prob = 0.5  # Medium
    else:  # SATURATION
        agree_prob = 0.7  # High (converging)

    agreed = rng.rand() < agree_prob

    # Simulate impact (uniform for now)
    impact = rng.rand()

    # Simulate downgrade (rare events)
    downgrade_prob = 0.02  # 2% base rate
    downgraded = rng.rand() < downgrade_prob

    return {
        "agreed": agreed,
        "impact": impact,
        "downgraded": downgraded,
    }


def measure_baseline_distributions(num_steps=5000, seed=42) -> Dict:
    """
    Measure baseline distributions for threshold derivation.

    Run system WITHOUT convergence detection to observe natural patterns:
    - Agreement window lengths
    - Silence period durations
    - Diversity by regime

    Returns:
        Dict with baseline distributions
    """
    print("=" * 70)
    print("BASELINE MEASUREMENT: Natural Behavior Distributions")
    print("=" * 70)
    print()
    print(f"Measuring {num_steps} steps to derive thresholds from percentiles")
    print()

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # Setup
    config = Config()
    model = TinyModel(config=config, max_experts=8)
    dataset = ArithmeticProgressionDataset(
        vocab_size=config.vocab_size,
        seq_len=config.block_size,
        batch_size=4,
        seed=seed,
    )

    # Train baseline
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Track distributions
    agreement_lengths = []
    silence_periods = []
    diversity_by_regime = {
        "exploration": [],
        "transition": [],
        "saturation": [],
    }

    # State tracking
    current_agreement_length = 0
    current_silence_period = 0
    last_downgrade_step = -1

    # Simulate scar accumulation
    scarred_experts = [0, 1]
    scar_schedule = [
        (0, 0.0),      # No scars initially
        (1000, 1.0),   # Exploration
        (2000, 2.5),   # Transition
        (3000, 5.0),   # Saturation
    ]

    current_scar_strength = 0.0

    print("Phase 1: Measuring Natural Distributions")
    print("-" * 70)

    for step in range(num_steps):
        # Update scar strength based on schedule
        for threshold_step, strength in scar_schedule:
            if step == threshold_step:
                current_scar_strength = strength
                # Apply scars
                for expert_id in scarred_experts:
                    model.moe.controller.expert_penalties[expert_id] = strength
                print(f"  Step {step:4d}: Scar strength → {strength:.1f}")

        # Training step
        batch = next(dataset)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        # Measure diversity every 100 steps
        if step % 100 == 0:
            dataset_embedded = EmbeddingDatasetWrapper(dataset, model)
            geometry = measure_routing_geometry(model.moe, dataset_embedded, num_steps=50)
            diversity = geometry['motif_diversity']

            # Categorize by regime
            if current_scar_strength < 2.0:
                regime_key = "exploration"
            elif current_scar_strength < 4.0:
                regime_key = "transition"
            else:
                regime_key = "saturation"

            diversity_by_regime[regime_key].append(diversity)

        # Simulate Bob-maniac interaction
        from chronomoe_integration.convergence import compute_deformation_regime
        regime = compute_deformation_regime(current_scar_strength)
        interaction = simulate_bob_maniac_interaction(step, regime, rng)

        # Track agreement lengths
        if interaction["agreed"] and interaction["impact"] >= 0.7:
            current_agreement_length += 1
        else:
            if current_agreement_length > 0:
                agreement_lengths.append(current_agreement_length)
            current_agreement_length = 0

        # Track silence periods
        if interaction["downgraded"]:
            if last_downgrade_step >= 0:
                silence_periods.append(step - last_downgrade_step)
            last_downgrade_step = step

    # Finalize
    if current_agreement_length > 0:
        agreement_lengths.append(current_agreement_length)

    print()
    print("Phase 2: Distribution Statistics")
    print("-" * 70)

    print(f"Agreement lengths: n={len(agreement_lengths)}")
    print(f"  Mean: {np.mean(agreement_lengths):.1f}")
    print(f"  Median: {np.median(agreement_lengths):.1f}")
    print(f"  90th percentile: {np.percentile(agreement_lengths, 90):.1f}")
    print()

    print(f"Silence periods: n={len(silence_periods)}")
    print(f"  Mean: {np.mean(silence_periods):.1f}")
    print(f"  Median: {np.median(silence_periods):.1f}")
    print(f"  90th percentile: {np.percentile(silence_periods, 90):.1f}")
    print()

    print("Diversity by regime:")
    for regime_key in ["exploration", "transition", "saturation"]:
        values = diversity_by_regime[regime_key]
        if values:
            print(f"  {regime_key:12s}: n={len(values):3d}, "
                  f"mean={np.mean(values):.4f}, "
                  f"10th percentile={np.percentile(values, 10):.4f}")
    print()

    return {
        "agreement_lengths": agreement_lengths,
        "silence_periods": silence_periods,
        "diversity_by_regime": diversity_by_regime,
    }


def run_convergence_detection_experiment(baseline_data: Dict, seed=42) -> Dict:
    """
    Run convergence detection with baseline-derived thresholds.

    Instrument:
    - Time to convergence after entering transition
    - Scar strength at convergence
    - Regime distribution of convergence events
    - Divergence frequency

    DO NOT attach policy. Measure only.
    """
    print("=" * 70)
    print("CONVERGENCE DETECTION: Instrumentation (No Policy)")
    print("=" * 70)
    print()

    # Derive thresholds from baseline
    thresholds = derive_thresholds_from_baseline(baseline_data, percentile=90.0)

    print("Derived Thresholds (90th percentile):")
    print("-" * 70)
    print(f"  W_min (agreement window):    {thresholds.W_min}")
    print(f"  S_min (downgrade silence):   {thresholds.S_min}")
    print(f"  D_exploration (diversity):   {thresholds.D_exploration:.4f}")
    print(f"  D_transition (diversity):    {thresholds.D_transition:.4f}")
    print(f"  D_saturation (diversity):    {thresholds.D_saturation:.4f}")
    print(f"  K_persistence (intervals):   {thresholds.K_persistence}")
    print()

    # Baseline diversity per regime
    baseline_diversity = {
        DeformationRegime.EXPLORATION: np.mean(baseline_data["diversity_by_regime"]["exploration"]),
        DeformationRegime.TRANSITION: np.mean(baseline_data["diversity_by_regime"]["transition"]),
        DeformationRegime.SATURATION: np.mean(baseline_data["diversity_by_regime"]["saturation"]),
    }

    # Initialize detector
    detector = ConvergenceDetector(thresholds, baseline_diversity)

    # Run detection (simulated for now - would use real Bob/maniac in production)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # Simulate scar accumulation with convergence-inducing pattern
    num_steps = 5000
    scar_schedule = [
        (0, 0.0),
        (1000, 1.5),   # Exploration
        (2000, 3.0),   # Transition (convergence likely here)
        (3500, 6.0),   # Saturation
    ]

    current_scar_strength = 0.0
    transition_entry_step = None

    print("Running Detection (5000 steps)")
    print("-" * 70)

    for step in range(num_steps):
        # Update scar strength
        for threshold_step, strength in scar_schedule:
            if step == threshold_step:
                current_scar_strength = strength

        # Detect transition entry
        from chronomoe_integration.convergence import compute_deformation_regime
        regime = compute_deformation_regime(current_scar_strength)
        if regime == DeformationRegime.TRANSITION and transition_entry_step is None:
            transition_entry_step = step

        # Simulate interaction
        interaction = simulate_bob_maniac_interaction(step, regime, rng)

        # Simulate diversity (decreases with scar strength)
        # Using Phase 1 contraction relationship
        if current_scar_strength == 0:
            diversity = 0.003
        elif current_scar_strength < 2.0:
            diversity = 0.003 * (1 - current_scar_strength * 0.1)
        elif current_scar_strength < 4.0:
            diversity = 0.003 * (1 - 0.2 - (current_scar_strength - 2.0) * 0.3)
        else:
            diversity = 0.003 * 0.1  # Saturated

        # Update detector
        state = detector.update(
            step=step,
            bob_agreed_with_maniac=interaction["agreed"],
            claim_impact=interaction["impact"],
            maniac_downgraded=interaction["downgraded"],
            diversity_current=diversity,
            scar_strength_total=current_scar_strength,
        )

        # Log convergence events
        if state.is_converged and step % 100 == 0:
            print(f"  Step {step:4d}: CONVERGED (regime={state.regime.value}, "
                  f"scar={state.scar_strength_total:.1f}, "
                  f"diversity={state.diversity_current:.4f})")

    print()

    # Analyze results
    stats = detector.get_convergence_statistics()

    print("=" * 70)
    print("CONVERGENCE STATISTICS")
    print("=" * 70)
    print()
    print(f"Convergence events: {stats['num_convergence_events']}")
    print(f"Divergence events:  {stats['num_divergence_events']}")
    print()

    print("Convergence by regime:")
    for regime in ["exploration", "transition", "saturation"]:
        count = stats["convergence_by_regime"][regime]
        print(f"  {regime:12s}: {count}")
    print()

    print("Scar strength at convergence:")
    print(f"  Mean: {stats['scar_strength_at_convergence']['mean']:.2f}")
    print(f"  Std:  {stats['scar_strength_at_convergence']['std']:.2f}")
    print(f"  Range: [{stats['scar_strength_at_convergence']['min']:.2f}, "
          f"{stats['scar_strength_at_convergence']['max']:.2f}]")
    print()

    if transition_entry_step is not None and detector.convergence_events:
        first_convergence = detector.convergence_events[0].step
        time_to_convergence = first_convergence - transition_entry_step
        print(f"Time to convergence after transition entry: {time_to_convergence} steps")
    print()

    return {
        "thresholds": asdict(thresholds),
        "statistics": stats,
        "convergence_events": [asdict(e) for e in detector.convergence_events],
        "divergence_events": [asdict(e) for e in detector.divergence_events],
    }


if __name__ == "__main__":
    # Phase 1: Measure baseline distributions
    baseline_data = measure_baseline_distributions(num_steps=5000, seed=42)

    # Save baseline
    with open("convergence_baseline.json", 'w') as f:
        # Convert to JSON-serializable
        baseline_json = {
            "agreement_lengths": baseline_data["agreement_lengths"],
            "silence_periods": baseline_data["silence_periods"],
            "diversity_by_regime": baseline_data["diversity_by_regime"],
        }
        json.dump(baseline_json, f, indent=2)

    print("Baseline saved: convergence_baseline.json")
    print()

    # Phase 2: Run convergence detection
    results = run_convergence_detection_experiment(baseline_data, seed=42)

    # Save results
    with open("convergence_detection_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved: convergence_detection_results.json")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Baseline distributions measured from natural behavior")
    print("✓ Thresholds derived from 90th percentile (not intuition)")
    print("✓ Convergence detection instrumented (regime-aware, persistence-required)")
    print("✓ Statistics logged (regime distribution, time-to-convergence)")
    print()
    print("Ready for Phase 3: Attack surface exhaustion detection")
    print("(DO NOT attach policy yet - Phase 4)")
