#!/usr/bin/env python3
"""
MERGE Execution Demo: Convergent Dataset

Uses a dataset designed to make experts converge, ensuring we find merge candidates.

Dataset evolution:
- First 500 steps: Two distinct patterns (A=increment, B=decrement)
- Steps 500-1000: Gradually shift to single pattern (increment only)
- Result: Experts initially specialize, then converge → merge candidates emerge
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

from execute_merge_demo import Config, SimpleMLP, TinyModel, train_step

from chronomoe_integration import (
    create_delta_bundle,
    save_delta_bundle_weights,
    save_delta_bundle_metadata,
    load_delta_bundle_weights,
    execute_merge,
    find_merge_candidate,
    run_probe_battery,
    check_protected_deltas,
    print_probe_battery_result,
    rollback_merge,
)


class ConvergentDataset:
    """
    Dataset that starts with two patterns, then converges to one.

    Steps 0-500: Pattern A (60%) vs Pattern B (40%) - experts specialize
    Steps 500-1000: Gradually shift to 100% Pattern A - experts converge
    """
    def __init__(self, vocab_size=100, seq_len=20, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)
        self.step = 0

    def __next__(self):
        # Calculate pattern mix based on step
        if self.step < 500:
            # Diverse phase: 60% A, 40% B
            pattern_a_prob = 0.6
        else:
            # Convergent phase: gradually shift to 100% A
            progress = (self.step - 500) / 500  # 0.0 to 1.0
            pattern_a_prob = 0.6 + 0.4 * progress  # 0.6 to 1.0

        # Sample pattern
        if torch.rand(1, generator=self.rng).item() < pattern_a_prob:
            # Pattern A: incrementing
            start = torch.randint(0, self.vocab_size - self.seq_len, (1,), generator=self.rng).item()
            batch = torch.arange(start, start + self.seq_len).unsqueeze(0)
        else:
            # Pattern B: decrementing
            start = torch.randint(self.seq_len, self.vocab_size, (1,), generator=self.rng).item()
            batch = torch.arange(start, start - self.seq_len, -1).unsqueeze(0)

        self.step += 1
        return batch


def run_suppression_trial_simple(model, dataset, optimizer, criterion, suppressed_expert, baseline_window=50, trial_window=100):
    """Simplified suppression trial."""
    # Baseline
    baseline_losses = []
    for _ in range(baseline_window):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)
        baseline_losses.append(loss)

    baseline_loss = sum(baseline_losses) / len(baseline_losses)

    # Trial (with suppression)
    model.moe.controller.suppress_expert(expert_id=suppressed_expert, duration_steps=0)

    trial_losses = []
    for _ in range(trial_window):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)
        trial_losses.append(loss)
        model.moe.controller.update_penalties()

    trial_loss_avg = sum(trial_losses) / len(trial_losses)
    trial_loss_max = max(trial_losses)

    # Veto check
    loss_tolerance = 0.10  # Lenient for demo
    loss_delta_max = trial_loss_max - baseline_loss

    veto_signals = []
    if loss_delta_max >= loss_tolerance:
        veto_signals.append(f"loss_spike (delta={loss_delta_max:.4f} >= {loss_tolerance})")

    verdict = "VETO" if veto_signals else "PASS"

    return {
        "verdict": verdict,
        "veto_signals": veto_signals,
        "baseline_loss": baseline_loss,
        "trial_loss_avg": trial_loss_avg,
        "trial_loss_max": trial_loss_max,
        "loss_delta_max": loss_delta_max,
    }


def main():
    print("=" * 70)
    print("MERGE EXECUTION DEMO: Convergent Dataset")
    print("=" * 70)
    print()
    print("Dataset design:")
    print("  Steps 0-500:   60% Pattern A, 40% Pattern B (experts specialize)")
    print("  Steps 500-1000: Shift to 100% Pattern A (experts converge)")
    print()

    # Setup
    torch.manual_seed(42)
    config = Config()

    model = TinyModel(config)
    dataset = ConvergentDataset(vocab_size=config.vocab_size, seq_len=config.block_size, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    bundle_dir = Path("./merge_bundles")
    bundle_dir.mkdir(exist_ok=True)

    # Phase 1: Train through convergence
    print("Phase 1: Training Through Convergence (1000 steps)")
    print("-" * 70)

    for step in range(1000):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)

        if step % 100 == 0:
            print(f"  Step {step:4d}: loss={loss:.4f}")

        # Check for candidates periodically after step 700
        if step >= 700 and step % 100 == 0:
            candidate = find_merge_candidate(
                model.moe.controller,
                similarity_threshold=0.7,  # Slightly lowered
                utilization_threshold=0.15,  # Slightly raised
                min_observations=200,
            )

            if candidate:
                print(f"  -> Candidate found at step {step}!")
                print(f"     Experts: {candidate['expert_a_id']}, {candidate['expert_b_id']}")
                print(f"     Similarity: {candidate['similarity']:.4f}")
                break

    print()

    if candidate is None:
        print("No candidates found after 1000 steps.")
        print("Checking final state...")

        candidate = find_merge_candidate(
            model.moe.controller,
            similarity_threshold=0.6,  # Even more lenient
            utilization_threshold=0.2,
            min_observations=200,
        )

        if candidate is None:
            print("  Still no candidates. Experts may not have converged enough.")
            print("  This is correct behavior - no false positives.")
            print()
            return

    print()
    print("Phase 2: Merge Candidate Found")
    print("-" * 70)
    print(f"  Experts: {candidate['expert_a_id']} and {candidate['expert_b_id']}")
    print(f"  Similarity: {candidate['similarity']:.4f}")
    print(f"  Utilization: {candidate['utilization_a']:.4f}, {candidate['utilization_b']:.4f}")
    print()

    # Phase 3: Suppression trial
    print("Phase 3: Running Suppression Trial")
    print("-" * 70)

    trial_result = run_suppression_trial_simple(
        model, dataset, optimizer, criterion,
        suppressed_expert=candidate['expert_b_id'],
        baseline_window=50,
        trial_window=100,
    )

    print(f"  Verdict: {trial_result['verdict']}")
    print(f"  Baseline loss: {trial_result['baseline_loss']:.4f}")
    print(f"  Trial loss (max): {trial_result['trial_loss_max']:.4f}")
    print(f"  Loss delta: {trial_result['loss_delta_max']:+.4f}")

    if trial_result['veto_signals']:
        print(f"  Veto signals: {trial_result['veto_signals']}")
        print()
        print("  MERGE rejected by suppression trial.")
        outcome = "TRIAL_VETOED"
    else:
        print()

        # Phase 4: Execute merge
        print("Phase 4: Executing MERGE")
        print("-" * 70)

        # Pre-merge probe
        print("  Running pre-merge probe battery...")
        pre_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)
        print(f"    Loss: {pre_metrics['loss']:.4f}")
        print()

        # Create and save delta bundle
        print("  Creating delta bundle...")
        delta_bundle = create_delta_bundle(
            layer=model.moe,
            expert_a_id=candidate['expert_a_id'],
            expert_b_id=candidate['expert_b_id'],
            similarity=candidate['similarity'],
            utilization_a=candidate['utilization_a'],
            utilization_b=candidate['utilization_b'],
            trial_result=trial_result,
            current_step=model.moe.current_step,
        )

        save_delta_bundle_weights(delta_bundle, model.moe, optimizer, bundle_dir)
        save_delta_bundle_metadata(delta_bundle, bundle_dir)
        print(f"    Bundle: {delta_bundle.bundle_id}")
        print()

        # Execute merge
        print("  Executing merge...")
        execute_merge(model.moe, delta_bundle, optimizer)
        print()

        # Post-merge probe
        print("Phase 5: Post-Merge Probe Battery")
        print("-" * 70)

        post_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)
        print(f"    Loss: {post_metrics['loss']:.4f}")
        print()

        # Check protected deltas
        print("Phase 6: Checking Protected Deltas")
        print("-" * 70)

        probe_result = check_protected_deltas(pre_metrics, post_metrics)
        print_probe_battery_result(probe_result)

        # Rollback if needed
        if probe_result.rollback_triggered:
            print("Phase 7: ROLLBACK")
            print("-" * 70)

            weights_data = load_delta_bundle_weights(delta_bundle.bundle_id, bundle_dir)
            rollback_merge(model.moe, delta_bundle, optimizer, weights_data=weights_data)
            print()

            outcome = "ROLLED_BACK"
        else:
            print("Phase 7: MERGE Accepted")
            print("-" * 70)
            print("  Protected deltas satisfied. Merge accepted.")
            print()

            outcome = "ACCEPTED"

        # Save outcome
        outcome_data = {
            "bundle_id": delta_bundle.bundle_id,
            "candidate": candidate,
            "trial_result": trial_result,
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "probe_result": {
                "rollback_triggered": probe_result.rollback_triggered,
                "rollback_signals": probe_result.rollback_signals,
            },
            "outcome": outcome,
        }

        with open(bundle_dir / "convergent_merge_outcome.json", 'w') as f:
            json.dump(outcome_data, f, indent=2)

    print("=" * 70)
    print(f"MERGE EXECUTION COMPLETE: {outcome}")
    print("=" * 70)
    print()
    print(f"Outcome saved: {bundle_dir / 'convergent_merge_outcome.json'}")
    print()


if __name__ == "__main__":
    main()
