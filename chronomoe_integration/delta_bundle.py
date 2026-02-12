#!/usr/bin/env python3
"""
Delta Bundle: State snapshot for MERGE rollback.

A delta bundle captures the exact state before a merge, enabling complete rollback
if protected deltas regress after merge execution.
"""

import json
import torch
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
from pathlib import Path


@dataclass
class DeltaBundleMerge:
    """
    Delta bundle for MERGE operation.

    Contains everything needed to rollback a merge to exact pre-merge state.
    """
    # Identification
    bundle_id: str  # Unique ID for this merge (e.g., "merge_layer0_step500_exp2_exp3")
    created_at_step: int

    # Target experts
    expert_a_id: int  # Expert to keep (receives merged weights)
    expert_b_id: int  # Expert to prune (will be deactivated)

    # Evidence for merge decision
    similarity: float
    utilization_a: float
    utilization_b: float
    suppression_trial_result: Dict[str, Any]  # Full trial verdict

    # Registry state (for rollback) - stored as dicts not ExpertInfo objects
    expert_a_state_before: str  # "ACTIVE" or "PROBATION"
    expert_b_state_before: str
    expert_a_probation_tokens: int
    expert_b_probation_tokens: int

    # Merge strategy
    merge_strategy: str = "average"  # "average", "weighted", "keep_a", "keep_b"
    merge_alpha: float = 0.5  # Weight for expert A (if weighted average)

    # Note: Weight deltas and optimizer state stored separately as .pt files
    # (too large for JSON, stored as tensors)


def create_delta_bundle(
    layer,
    expert_a_id: int,
    expert_b_id: int,
    similarity: float,
    utilization_a: float,
    utilization_b: float,
    trial_result: Dict[str, Any],
    current_step: int,
    merge_strategy: str = "average",
    merge_alpha: float = 0.5,
) -> DeltaBundleMerge:
    """
    Create delta bundle capturing state before merge.

    Args:
        layer: ChronoMoE layer
        expert_a_id: Expert to keep
        expert_b_id: Expert to prune
        similarity: Cosine similarity between experts
        utilization_a: Utilization of expert A
        utilization_b: Utilization of expert B
        trial_result: Suppression trial verdict
        current_step: Current training step
        merge_strategy: How to merge weights
        merge_alpha: Weight for expert A (if weighted)

    Returns:
        DeltaBundleMerge with full state snapshot
    """
    bundle_id = f"merge_layer{layer.layer_id}_step{current_step}_exp{expert_a_id}_exp{expert_b_id}"

    # Capture registry state
    expert_a_info = layer.registry.experts[expert_a_id]
    expert_b_info = layer.registry.experts[expert_b_id]

    bundle = DeltaBundleMerge(
        bundle_id=bundle_id,
        created_at_step=current_step,
        expert_a_id=expert_a_id,
        expert_b_id=expert_b_id,
        similarity=similarity,
        utilization_a=utilization_a,
        utilization_b=utilization_b,
        suppression_trial_result=trial_result,
        merge_strategy=merge_strategy,
        merge_alpha=merge_alpha,
        expert_a_state_before=expert_a_info.state.name,
        expert_b_state_before=expert_b_info.state.name,
        expert_a_probation_tokens=expert_a_info.probation_tokens_accumulated,
        expert_b_probation_tokens=expert_b_info.probation_tokens_accumulated,
    )

    return bundle


def save_delta_bundle_weights(
    bundle: DeltaBundleMerge,
    layer,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
) -> None:
    """
    Save weight deltas and optimizer state to disk.

    Args:
        bundle: DeltaBundleMerge metadata
        layer: ChronoMoE layer
        optimizer: Optimizer
        output_dir: Directory to save bundle files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save expert A weights
    expert_a_weights = layer.experts[bundle.expert_a_id].state_dict()
    torch.save(expert_a_weights, output_dir / f"{bundle.bundle_id}_expert_a_weights.pt")

    # Save expert B weights
    expert_b_weights = layer.experts[bundle.expert_b_id].state_dict()
    torch.save(expert_b_weights, output_dir / f"{bundle.bundle_id}_expert_b_weights.pt")

    # Save optimizer state for expert A
    expert_a_params = list(layer.experts[bundle.expert_a_id].parameters())
    expert_a_optimizer_state = {
        id(p): optimizer.state[p] for p in expert_a_params if p in optimizer.state
    }
    torch.save(expert_a_optimizer_state, output_dir / f"{bundle.bundle_id}_expert_a_optim.pt")

    # Save optimizer state for expert B
    expert_b_params = list(layer.experts[bundle.expert_b_id].parameters())
    expert_b_optimizer_state = {
        id(p): optimizer.state[p] for p in expert_b_params if p in optimizer.state
    }
    torch.save(expert_b_optimizer_state, output_dir / f"{bundle.bundle_id}_expert_b_optim.pt")

    # Save active mask
    torch.save(layer.registry.active_mask.clone(), output_dir / f"{bundle.bundle_id}_active_mask.pt")


def save_delta_bundle_metadata(
    bundle: DeltaBundleMerge,
    output_dir: Path,
) -> None:
    """
    Save delta bundle metadata to JSON.

    Args:
        bundle: DeltaBundleMerge metadata
        output_dir: Directory to save bundle files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = asdict(bundle)
    with open(output_dir / f"{bundle.bundle_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def load_delta_bundle_weights(
    bundle_id: str,
    input_dir: Path,
) -> Dict[str, Any]:
    """
    Load weight deltas and optimizer state from disk.

    Args:
        bundle_id: Bundle ID
        input_dir: Directory containing bundle files

    Returns:
        Dict with expert weights, optimizer state, and active mask
    """
    input_dir = Path(input_dir)

    return {
        "expert_a_weights": torch.load(input_dir / f"{bundle_id}_expert_a_weights.pt"),
        "expert_b_weights": torch.load(input_dir / f"{bundle_id}_expert_b_weights.pt"),
        "expert_a_optim": torch.load(input_dir / f"{bundle_id}_expert_a_optim.pt"),
        "expert_b_optim": torch.load(input_dir / f"{bundle_id}_expert_b_optim.pt"),
        "active_mask": torch.load(input_dir / f"{bundle_id}_active_mask.pt"),
    }


def load_delta_bundle_metadata(
    bundle_id: str,
    input_dir: Path,
) -> DeltaBundleMerge:
    """
    Load delta bundle metadata from JSON.

    Args:
        bundle_id: Bundle ID
        input_dir: Directory containing bundle files

    Returns:
        DeltaBundleMerge
    """
    input_dir = Path(input_dir)

    with open(input_dir / f"{bundle_id}_metadata.json", 'r') as f:
        metadata = json.load(f)

    return DeltaBundleMerge(**metadata)
