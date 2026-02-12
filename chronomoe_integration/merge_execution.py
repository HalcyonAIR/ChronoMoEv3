#!/usr/bin/env python3
"""
Merge Execution: Execute MERGE operation.

Merges two experts' weights and prunes one expert.
Phase 2 MVP uses simple average strategy.
"""

import torch
from collections import OrderedDict

from chronomoe_integration.delta_bundle import DeltaBundleMerge
from chronomoe_integration.expert_registry import ExpertState


def execute_merge(
    layer,
    bundle: DeltaBundleMerge,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Execute MERGE operation.

    Steps:
    1. Merge weights (expert A ← merge(A, B))
    2. Prune expert B (deactivate in registry)
    3. Update optimizer state (remove expert B params)

    Args:
        layer: ChronoMoE layer
        bundle: DeltaBundleMerge with merge strategy
        optimizer: Optimizer
    """
    expert_a_id = bundle.expert_a_id
    expert_b_id = bundle.expert_b_id

    print(f"  [Merge] Executing MERGE: expert {expert_b_id} → expert {expert_a_id}")
    print(f"    Strategy: {bundle.merge_strategy}, alpha={bundle.merge_alpha}")

    # Step 1: Merge weights
    _merge_weights(
        layer.experts[expert_a_id],
        layer.experts[expert_b_id],
        strategy=bundle.merge_strategy,
        alpha=bundle.merge_alpha,
    )
    print(f"    ✓ Weights merged")

    # Step 2: Prune expert B
    layer.registry.prune_expert(expert_b_id)
    print(f"    ✓ Expert {expert_b_id} pruned (deactivated)")

    # Step 3: Clean up optimizer state for expert B
    # Note: Expert B parameters still exist in memory, but optimizer won't update them
    # This is acceptable - they're just dead weight
    # For production, could explicitly remove from optimizer.state to save memory
    print(f"    ✓ Merge complete")


def _merge_weights(
    expert_a: torch.nn.Module,
    expert_b: torch.nn.Module,
    strategy: str = "average",
    alpha: float = 0.5,
) -> None:
    """
    Merge expert B weights into expert A.

    Modifies expert A weights in-place.

    Args:
        expert_a: Expert to keep (will receive merged weights)
        expert_b: Expert to prune (weights will be averaged in)
        strategy: Merge strategy ("average", "weighted", "keep_a", "keep_b")
        alpha: Weight for expert A (if weighted average)
    """
    with torch.no_grad():
        state_a = expert_a.state_dict()
        state_b = expert_b.state_dict()

        merged_state = OrderedDict()

        for key in state_a.keys():
            if key not in state_b:
                # Expert A has param that expert B doesn't (shouldn't happen)
                merged_state[key] = state_a[key]
                continue

            param_a = state_a[key]
            param_b = state_b[key]

            if strategy == "average":
                # Simple average
                merged_state[key] = (param_a + param_b) / 2.0

            elif strategy == "weighted":
                # Weighted average (alpha * A + (1-alpha) * B)
                merged_state[key] = alpha * param_a + (1.0 - alpha) * param_b

            elif strategy == "keep_a":
                # Keep expert A weights (no merge)
                merged_state[key] = param_a

            elif strategy == "keep_b":
                # Replace with expert B weights
                merged_state[key] = param_b

            else:
                raise ValueError(f"Unknown merge strategy: {strategy}")

        # Load merged weights into expert A
        expert_a.load_state_dict(merged_state)


def find_merge_candidate(
    controller,
    similarity_threshold: float = 0.8,
    utilization_threshold: float = 0.1,
    min_observations: int = 100,
):
    """
    Find MERGE candidate from controller state.

    Args:
        controller: ChronoController
        similarity_threshold: Minimum cosine similarity
        utilization_threshold: Maximum utilization (both experts)
        min_observations: Minimum observations required

    Returns:
        Dict with candidate info, or None if no candidate found
    """
    # Check if we have sufficient observations
    if len(controller.observation_history) < min_observations:
        return None

    # Get last observation
    if not controller.observation_history:
        return None

    last_obs = controller.observation_history[-1]
    total_tokens = last_obs.utilization.sum().item()

    if total_tokens == 0:
        return None

    # Find pairs of experts with high similarity and low utilization
    best_candidate = None
    best_score = 0.0

    for expert_a_id, state_a in controller.bimodality_states.items():
        for expert_b_id, state_b in controller.bimodality_states.items():
            if expert_a_id >= expert_b_id:
                continue  # Skip self and duplicates

            # Check observations
            if state_a.count_a < min_observations or state_b.count_a < min_observations:
                continue

            # Compute similarity
            centroid_a = state_a.centroid_a
            centroid_b = state_b.centroid_a

            if centroid_a.norm() == 0 or centroid_b.norm() == 0:
                continue

            similarity = torch.cosine_similarity(
                centroid_a.unsqueeze(0),
                centroid_b.unsqueeze(0),
                dim=1
            ).item()

            # Check similarity threshold
            if similarity < similarity_threshold:
                continue

            # Compute utilization
            util_a = last_obs.utilization[expert_a_id].item() / total_tokens
            util_b = last_obs.utilization[expert_b_id].item() / total_tokens

            # Check utilization threshold (both must be low)
            if util_a > utilization_threshold or util_b > utilization_threshold:
                continue

            # Score candidate (higher similarity + lower utilization = better)
            score = similarity * (1.0 - max(util_a, util_b))

            if score > best_score:
                best_score = score
                best_candidate = {
                    "expert_a_id": expert_a_id,
                    "expert_b_id": expert_b_id,
                    "similarity": similarity,
                    "utilization_a": util_a,
                    "utilization_b": util_b,
                    "score": score,
                }

    return best_candidate
