#!/usr/bin/env python3
"""
Adversarial MERGE Execution Demo: Force merge conditions to test rollback.

This demo intentionally creates conditions where:
1. MERGE candidate exists (artificially lowered thresholds)
2. Suppression trial passes (lenient criteria)
3. Post-merge regression is injected to test rollback

Purpose: Validate that rollback works when protected deltas regress.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

# Import from regular demo
from execute_merge_demo import (
    Config, SimpleMLP, TinyModel, SimpleDataset, train_step,
)

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
    verify_rollback,
)


def main():
    print("=" * 70)
    print("ADVERSARIAL MERGE EXECUTION DEMO (Rollback Test)")
    print("=" * 70)
    print()
    print("This demo forces merge conditions to validate rollback.")
    print()

    # Setup
    torch.manual_seed(42)
    config = Config()

    model = TinyModel(config)
    dataset = SimpleDataset(vocab_size=config.vocab_size, seq_len=config.block_size, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    bundle_dir = Path("./merge_bundles")
    bundle_dir.mkdir(exist_ok=True)

    # Phase 1: Train baseline (shorter for speed)
    print("Phase 1: Training Baseline (100 steps)")
    print("-" * 70)

    for step in range(100):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)

        if step % 25 == 0:
            print(f"  Step {step:3d}: loss={loss:.4f}")

    print()

    # Phase 2: Find MERGE candidate (with LOWERED thresholds to force finding one)
    print("Phase 2: Identifying MERGE Candidate (Lowered Thresholds)")
    print("-" * 70)

    candidate = find_merge_candidate(
        model.moe.controller,
        similarity_threshold=0.5,  # Lowered from 0.8
        utilization_threshold=0.3,  # Raised from 0.1
        min_observations=50,  # Lowered from 100
    )

    if candidate is None:
        print("  No candidates found even with lowered thresholds.")
        print("  This is expected if experts are still very dissimilar.")
        print()
        return

    print(f"  Candidate found:")
    print(f"    Experts: {candidate['expert_a_id']} and {candidate['expert_b_id']}")
    print(f"    Similarity: {candidate['similarity']:.4f}")
    print(f"    Utilization: {candidate['utilization_a']:.4f}, {candidate['utilization_b']:.4f}")
    print()

    # Phase 3: Skip suppression trial (assume it passed)
    print("Phase 3: Skipping Suppression Trial (Assumed PASS)")
    print("-" * 70)
    print("  For adversarial testing, we assume trial passed.")
    print()

    trial_result = {
        "verdict": "PASS",
        "veto_signals": [],
        "baseline_loss": 2.5,
        "trial_loss_avg": 2.48,
        "trial_loss_max": 2.52,
        "loss_delta_max": 0.02,
    }

    # Phase 4: Execute merge
    print("Phase 4: Executing MERGE")
    print("-" * 70)

    # 4a. Run probe battery before merge
    print("  Running pre-merge probe battery...")
    pre_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)
    print(f"    Loss: {pre_metrics['loss']:.4f}")
    print(f"    Perplexity: {pre_metrics['perplexity']:.2f}")
    print(f"    Neff: {pre_metrics['neff']:.2f}")
    print()

    # 4b. Create delta bundle
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
        merge_strategy="average",
        merge_alpha=0.5,
    )

    # Save delta bundle
    save_delta_bundle_weights(delta_bundle, model.moe, optimizer, bundle_dir)
    save_delta_bundle_metadata(delta_bundle, bundle_dir)
    print(f"    Bundle saved: {delta_bundle.bundle_id}")
    print()

    # 4c. Execute merge
    print("  Executing merge...")
    execute_merge(model.moe, delta_bundle, optimizer)
    print()

    # Phase 5: Probe battery after merge (with INJECTED REGRESSION)
    print("Phase 5: Running Post-Merge Probe Battery")
    print("-" * 70)
    print("  INJECTING REGRESSION: Adding noise to simulate capability loss")

    post_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)

    # INJECT REGRESSION (simulate merge harming model)
    post_metrics_injected = {
        "loss": post_metrics["loss"] + 0.15,  # Inject loss spike (above 0.05 tolerance)
        "perplexity": post_metrics["perplexity"] * 1.2,  # 20% increase
        "f_l": post_metrics["f_l"],
        "coherence": post_metrics["coherence"],
        "neff": post_metrics["neff"],
    }

    print(f"    Original loss: {post_metrics['loss']:.4f}")
    print(f"    Injected loss: {post_metrics_injected['loss']:.4f} (regression: +0.15)")
    print()

    # Phase 6: Check protected deltas (should trigger rollback)
    print("Phase 6: Checking Protected Deltas")
    print("-" * 70)

    probe_result = check_protected_deltas(pre_metrics, post_metrics_injected)
    print_probe_battery_result(probe_result)

    # Phase 7: Rollback (should be triggered)
    if not probe_result.rollback_triggered:
        print("  ERROR: Rollback was NOT triggered despite injected regression!")
        print("  This means veto criteria are too lenient.")
        return

    print("Phase 7: ROLLBACK TRIGGERED")
    print("-" * 70)

    # Load delta bundle weights
    weights_data = load_delta_bundle_weights(delta_bundle.bundle_id, bundle_dir)

    # Rollback
    rollback_merge(model.moe, delta_bundle, optimizer, weights_data=weights_data)

    # Verify rollback
    print()
    print("  Verifying rollback...")
    post_rollback_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)

    rollback_verified = verify_rollback(
        model.moe,
        delta_bundle,
        pre_metrics,
        post_rollback_metrics,
        tolerance=0.05,  # Slightly higher tolerance for verification
    )

    if rollback_verified:
        print()
        print("  ✓ ROLLBACK VERIFIED: Metrics returned to baseline")
    else:
        print()
        print("  ✗ ROLLBACK VERIFICATION FAILED: Metrics did not return to baseline")

    outcome = "ROLLED_BACK"
    print()

    # Save outcome
    outcome_data = {
        "bundle_id": delta_bundle.bundle_id,
        "candidate": candidate,
        "trial_result": trial_result,
        "pre_metrics": pre_metrics,
        "post_metrics_original": post_metrics,
        "post_metrics_injected": post_metrics_injected,
        "post_rollback_metrics": post_rollback_metrics,
        "probe_result": {
            "rollback_triggered": probe_result.rollback_triggered,
            "rollback_signals": probe_result.rollback_signals,
        },
        "rollback_verified": rollback_verified,
        "outcome": outcome,
    }

    with open(bundle_dir / "adversarial_rollback_outcome.json", 'w') as f:
        json.dump(outcome_data, f, indent=2)

    print("=" * 70)
    print(f"ADVERSARIAL DEMO COMPLETE: {outcome}")
    print("=" * 70)
    print()
    print("Key validations:")
    print(f"  ✓ MERGE executed (expert {candidate['expert_b_id']} merged into {candidate['expert_a_id']})")
    print(f"  ✓ Regression detected (loss delta {probe_result.loss_delta:+.4f} > {probe_result.loss_tolerance})")
    print(f"  ✓ Rollback triggered ({len(probe_result.rollback_signals)} veto signals)")
    print(f"  {'✓' if rollback_verified else '✗'} Rollback verified (metrics returned to baseline)")
    print()
    print(f"Outcome saved: {bundle_dir / 'adversarial_rollback_outcome.json'}")
    print()


if __name__ == "__main__":
    main()
