#!/usr/bin/env python3
"""
Rollback: Restore exact pre-merge state.

If a merge causes protected deltas to regress, rollback restores:
- Expert weights
- Optimizer state (Adam momentum/variance)
- Registry state (active mask, expert info)
"""

import torch
from pathlib import Path
from typing import Optional

from chronomoe_integration.delta_bundle import (
    DeltaBundleMerge,
    load_delta_bundle_weights,
)
from chronomoe_integration.expert_registry import ExpertState


def rollback_merge(
    layer,
    bundle: DeltaBundleMerge,
    optimizer: torch.optim.Optimizer,
    weights_data: Optional[dict] = None,
    bundle_dir: Optional[Path] = None,
) -> None:
    """
    Rollback a merge to exact pre-merge state.

    Restores:
    1. Expert A weights (before merge)
    2. Expert B weights (before merge)
    3. Active mask (re-activate expert B)
    4. Expert registry state (state, probation tokens)
    5. Optimizer state (Adam momentum/variance)

    Args:
        layer: ChronoMoE layer
        bundle: DeltaBundleMerge metadata
        optimizer: Optimizer (Adam/AdamW)
        weights_data: Pre-loaded weights dict (if already in memory)
        bundle_dir: Directory containing bundle files (if loading from disk)
    """
    # Load weights if not provided
    if weights_data is None:
        if bundle_dir is None:
            raise ValueError("Must provide either weights_data or bundle_dir")
        weights_data = load_delta_bundle_weights(bundle.bundle_id, bundle_dir)

    print(f"  [Rollback] Reverting merge {bundle.bundle_id}...")

    # 1. Restore expert A weights
    layer.experts[bundle.expert_a_id].load_state_dict(weights_data["expert_a_weights"])
    print(f"    Restored expert {bundle.expert_a_id} weights")

    # 2. Restore expert B weights
    layer.experts[bundle.expert_b_id].load_state_dict(weights_data["expert_b_weights"])
    print(f"    Restored expert {bundle.expert_b_id} weights")

    # 3. Restore active mask (re-activate expert B)
    layer.registry.active_mask = weights_data["active_mask"].clone()
    print(f"    Restored active mask (expert {bundle.expert_b_id} re-activated)")

    # 4. Restore expert registry state
    # Expert A
    layer.registry.experts[bundle.expert_a_id].state = ExpertState[bundle.expert_a_state_before]
    layer.registry.experts[bundle.expert_a_id].probation_tokens_accumulated = bundle.expert_a_probation_tokens
    print(f"    Restored expert {bundle.expert_a_id} registry state: {bundle.expert_a_state_before}")

    # Expert B
    layer.registry.experts[bundle.expert_b_id].state = ExpertState[bundle.expert_b_state_before]
    layer.registry.experts[bundle.expert_b_id].probation_tokens_accumulated = bundle.expert_b_probation_tokens
    print(f"    Restored expert {bundle.expert_b_id} registry state: {bundle.expert_b_state_before}")

    # 5. Restore optimizer state (CRITICAL for Adam/AdamW)
    # This restores momentum (first moment) and variance (second moment) for each parameter
    _restore_optimizer_state(
        optimizer,
        layer.experts[bundle.expert_a_id],
        weights_data["expert_a_optim"],
    )
    print(f"    Restored expert {bundle.expert_a_id} optimizer state")

    _restore_optimizer_state(
        optimizer,
        layer.experts[bundle.expert_b_id],
        weights_data["expert_b_optim"],
    )
    print(f"    Restored expert {bundle.expert_b_id} optimizer state")

    print(f"  [Rollback] Complete. Merge reverted to step {bundle.created_at_step} state.")


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    expert_module: torch.nn.Module,
    saved_state: dict,
) -> None:
    """
    Restore optimizer state for expert parameters.

    Args:
        optimizer: Optimizer
        expert_module: Expert module
        saved_state: Saved optimizer state (param_id -> state dict)
    """
    # Get current parameters
    current_params = list(expert_module.parameters())

    # Clear current optimizer state for these parameters
    for param in current_params:
        if param in optimizer.state:
            del optimizer.state[param]

    # Restore saved state
    # Assumption: parameter order is stable (same expert structure)
    saved_states_list = list(saved_state.values())
    for i, param in enumerate(current_params):
        if i < len(saved_states_list):
            optimizer.state[param] = saved_states_list[i]


def verify_rollback(
    layer,
    bundle: DeltaBundleMerge,
    pre_merge_metrics: dict,
    post_rollback_metrics: dict,
    tolerance: float = 0.001,
) -> bool:
    """
    Verify rollback restored state correctly.

    Checks:
    - Expert B is active again
    - Metrics returned to pre-merge values (within tolerance)

    Args:
        layer: ChronoMoE layer
        bundle: DeltaBundleMerge metadata
        pre_merge_metrics: Metrics before merge
        post_rollback_metrics: Metrics after rollback
        tolerance: Tolerance for metric comparison

    Returns:
        True if rollback successful
    """
    print(f"  [Verification] Checking rollback correctness...")

    # Check 1: Expert B is active
    expert_b_active = layer.registry.active_mask[bundle.expert_b_id].item()
    if not expert_b_active:
        print(f"    ✗ Expert {bundle.expert_b_id} not re-activated")
        return False
    print(f"    ✓ Expert {bundle.expert_b_id} re-activated")

    # Check 2: Metrics returned to baseline
    metric_checks = []

    for metric_name in ["loss", "perplexity", "f_l", "coherence", "neff"]:
        pre = pre_merge_metrics.get(metric_name)
        post = post_rollback_metrics.get(metric_name)

        if pre is None or post is None:
            continue

        delta = abs(post - pre)
        if delta > tolerance:
            print(f"    ✗ {metric_name}: {post:.4f} vs {pre:.4f} (delta={delta:.4f} > {tolerance})")
            metric_checks.append(False)
        else:
            print(f"    ✓ {metric_name}: {post:.4f} ≈ {pre:.4f} (delta={delta:.4f})")
            metric_checks.append(True)

    if all(metric_checks):
        print(f"  [Verification] Rollback verified ✓")
        return True
    else:
        print(f"  [Verification] Rollback verification failed ✗")
        return False
