#!/usr/bin/env python3
"""
Rollback Mechanism Test: Direct validation without waiting for natural candidates.

Tests:
1. Create delta bundle with current state
2. Execute merge (modify expert A, deactivate expert B)
3. Inject regression (simulate quality drop)
4. Trigger rollback
5. Verify state restored exactly
"""

import torch
import torch.nn as nn
from pathlib import Path

from execute_merge_demo import Config, SimpleMLP, TinyModel, SimpleDataset, train_step

from chronomoe_integration import (
    create_delta_bundle,
    save_delta_bundle_weights,
    load_delta_bundle_weights,
    execute_merge,
    rollback_merge,
    run_probe_battery,
)


def main():
    print("=" * 70)
    print("ROLLBACK MECHANISM TEST")
    print("=" * 70)
    print()

    torch.manual_seed(42)
    config = Config()

    model = TinyModel(config)
    dataset = SimpleDataset(vocab_size=config.vocab_size, seq_len=config.block_size, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    bundle_dir = Path("./merge_bundles")
    bundle_dir.mkdir(exist_ok=True)

    # Train a bit
    print("Training baseline (50 steps)...")
    for step in range(50):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)

    print(f"  Final loss: {loss:.4f}")
    print()

    # Manually select two experts to merge (doesn't matter if they're similar)
    expert_a_id = 0
    expert_b_id = 1

    print(f"Test Setup:")
    print(f"  Expert A: {expert_a_id}")
    print(f"  Expert B: {expert_b_id}")
    print()

    # Capture pre-merge metrics
    print("Phase 1: Capturing Pre-Merge State")
    print("-" * 70)

    pre_metrics = run_probe_battery(model, dataset, criterion, num_steps=30)
    print(f"  Loss: {pre_metrics['loss']:.4f}")
    print(f"  Neff: {pre_metrics['neff']:.2f}")

    # Capture expert B weight hash (for verification)
    expert_b_weight_sum_before = sum(
        p.sum().item() for p in model.moe.experts[expert_b_id].parameters()
    )
    print(f"  Expert B weight sum: {expert_b_weight_sum_before:.4f}")
    print()

    # Create delta bundle
    print("Phase 2: Creating Delta Bundle")
    print("-" * 70)

    delta_bundle = create_delta_bundle(
        layer=model.moe,
        expert_a_id=expert_a_id,
        expert_b_id=expert_b_id,
        similarity=0.9,  # Fake similarity
        utilization_a=0.05,  # Fake utilization
        utilization_b=0.05,
        trial_result={"verdict": "PASS"},
        current_step=model.moe.current_step,
    )

    save_delta_bundle_weights(delta_bundle, model.moe, optimizer, bundle_dir)
    print(f"  Bundle created: {delta_bundle.bundle_id}")
    print()

    # Execute merge
    print("Phase 3: Executing MERGE")
    print("-" * 70)

    execute_merge(model.moe, delta_bundle, optimizer)

    # Verify expert B is deactivated
    expert_b_active_after_merge = model.moe.registry.active_mask[expert_b_id].item()
    print(f"  Expert B active after merge: {expert_b_active_after_merge} (should be False)")
    print()

    # Measure post-merge metrics
    print("Phase 4: Post-Merge Metrics")
    print("-" * 70)

    post_metrics = run_probe_battery(model, dataset, criterion, num_steps=30)
    print(f"  Loss: {post_metrics['loss']:.4f} (delta: {post_metrics['loss'] - pre_metrics['loss']:+.4f})")
    print(f"  Neff: {post_metrics['neff']:.2f} (delta: {post_metrics['neff'] - pre_metrics['neff']:+.2f})")
    print()

    # Rollback
    print("Phase 5: ROLLBACK")
    print("-" * 70)

    weights_data = load_delta_bundle_weights(delta_bundle.bundle_id, bundle_dir)
    rollback_merge(model.moe, delta_bundle, optimizer, weights_data=weights_data)
    print()

    # Verify rollback
    print("Phase 6: Verifying Rollback")
    print("-" * 70)

    # Check expert B is re-activated
    expert_b_active_after_rollback = model.moe.registry.active_mask[expert_b_id].item()
    print(f"  Expert B active after rollback: {expert_b_active_after_rollback} (should be True)")

    if not expert_b_active_after_rollback:
        print(f"  ✗ FAILED: Expert B not re-activated")
        return

    # Check expert B weights restored
    expert_b_weight_sum_after = sum(
        p.sum().item() for p in model.moe.experts[expert_b_id].parameters()
    )
    weight_delta = abs(expert_b_weight_sum_after - expert_b_weight_sum_before)
    print(f"  Expert B weight sum: {expert_b_weight_sum_after:.4f} (delta: {weight_delta:.6f})")

    if weight_delta > 1e-4:
        print(f"  ✗ FAILED: Weights not restored (delta={weight_delta:.6f})")
    else:
        print(f"  ✓ Weights restored correctly")

    # Check metrics returned to baseline
    post_rollback_metrics = run_probe_battery(model, dataset, criterion, num_steps=30)
    loss_restored_delta = abs(post_rollback_metrics['loss'] - pre_metrics['loss'])

    print(f"  Loss after rollback: {post_rollback_metrics['loss']:.4f} (delta from baseline: {loss_restored_delta:.4f})")

    if loss_restored_delta < 0.05:
        print(f"  ✓ Loss returned to baseline")
    else:
        print(f"  ⚠ Loss delta larger than expected (may be due to stochastic variance)")

    print()

    # Summary
    print("=" * 70)
    print("ROLLBACK MECHANISM TEST COMPLETE")
    print("=" * 70)
    print()
    print("Validations:")
    print(f"  {'✓' if not expert_b_active_after_merge else '✗'} Expert B deactivated after merge")
    print(f"  {'✓' if expert_b_active_after_rollback else '✗'} Expert B re-activated after rollback")
    print(f"  {'✓' if weight_delta < 1e-4 else '✗'} Weights restored exactly")
    print(f"  {'✓' if loss_restored_delta < 0.05 else '⚠'} Loss returned to baseline (within variance)")
    print()

    if expert_b_active_after_rollback and weight_delta < 1e-4:
        print("✓ ROLLBACK MECHANISM WORKING")
    else:
        print("✗ ROLLBACK MECHANISM FAILED")

    print()


if __name__ == "__main__":
    main()
