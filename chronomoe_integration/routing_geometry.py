#!/usr/bin/env python3
"""
Routing Geometry Measurement Harness

Measures reachable solution manifold geometry to detect:
1. Scar-induced contraction (scars shrink reachability)
2. Epoch review-induced expansion (removing obsolete scars restores reachability)

Core metrics:
- Motif diversity (unique routing patterns over time)
- Effective rank (participation ratio over routing logits)
- Router entropy (baseline diversity measure)
- Time-to-recover (adaptation speed after domain shift)

CRITICAL: This is not bookkeeping. This measures geometric deformation.
If scars don't contract motif diversity, they're not architecture.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import time


def extract_motifs(routing_sequences: List[List[int]], window_size: int = 4) -> List[Tuple[int, ...]]:
    """
    Extract motifs from routing sequences.

    A motif is a sliding window pattern of expert selections.

    Args:
        routing_sequences: List of routing decisions (expert IDs selected per step)
        window_size: Size of sliding window

    Returns:
        List of motif tuples
    """
    motifs = []
    for seq in routing_sequences:
        if len(seq) < window_size:
            continue
        for i in range(len(seq) - window_size + 1):
            motif = tuple(seq[i:i + window_size])
            motifs.append(motif)

    return motifs


def compute_participation_ratio(singular_values: torch.Tensor) -> float:
    """
    Compute participation ratio from singular values.

    PR = (sum(s_i))^2 / sum(s_i^2)

    Measures effective dimensionality. If all singular values equal,
    PR = n (full rank). If only one singular value, PR = 1.

    Args:
        singular_values: Tensor of singular values from SVD

    Returns:
        Participation ratio (effective rank)
    """
    s_sum = singular_values.sum()
    s_squared_sum = (singular_values ** 2).sum()

    if s_squared_sum == 0:
        return 0.0

    pr = (s_sum ** 2) / s_squared_sum
    return pr.item()


def compute_routing_entropy(routing_weights: torch.Tensor) -> float:
    """
    Compute Shannon entropy of routing distribution.

    Args:
        routing_weights: [batch, num_experts] routing probabilities

    Returns:
        Average entropy across batch
    """
    # Avoid log(0)
    weights = routing_weights.clamp(min=1e-10)

    # Shannon entropy: -sum(p * log(p))
    entropy = -(weights * torch.log(weights)).sum(dim=-1)

    return entropy.mean().item()


def get_expert_sequence(router_weights: torch.Tensor, k: int = 2) -> List[int]:
    """
    Extract top-k expert sequence from routing weights.

    Args:
        router_weights: [batch, num_experts] routing probabilities
        k: Number of top experts to track

    Returns:
        List of expert IDs in top-k order
    """
    # Get top-k experts per token
    top_k_experts = torch.topk(router_weights, k, dim=-1).indices  # [batch, k]

    # Flatten to sequence
    sequence = top_k_experts.flatten().tolist()

    return sequence


def compute_effective_rank(routing_logits_history: List[torch.Tensor]) -> float:
    """
    Compute effective rank (participation ratio) over routing logits.

    CRITICAL: Normalizes logits to avoid measuring amplitude drift
    instead of structural diversity.

    Args:
        routing_logits_history: List of [batch, num_experts] logit tensors

    Returns:
        Effective rank (participation ratio)
    """
    if len(routing_logits_history) == 0:
        return 0.0

    # Stack into [time, batch, num_experts]
    logits_matrix = torch.stack(routing_logits_history)

    # Center and normalize across time to remove scale effects
    logits_centered = logits_matrix - logits_matrix.mean(dim=0, keepdim=True)
    logits_std = logits_centered.std(dim=0, keepdim=True)

    # Check for zero std (constant logits)
    if (logits_std < 1e-8).all():
        # All logits are constant - rank is effectively 1
        return 1.0

    logits_normalized = logits_centered / (logits_std + 1e-8)

    # Reshape to [time * batch, num_experts] for SVD
    logits_flat = logits_normalized.reshape(-1, logits_normalized.shape[-1])

    # Check for NaN or Inf
    if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
        return 0.0

    # Compute singular values
    try:
        singular_values = torch.linalg.svdvals(logits_flat)

        # Check for valid singular values
        if torch.isnan(singular_values).any() or torch.isinf(singular_values).any():
            return 0.0

        effective_rank = compute_participation_ratio(singular_values)

        # Check result
        if np.isnan(effective_rank) or np.isinf(effective_rank):
            return 0.0

        return effective_rank
    except (RuntimeError, ValueError) as e:
        # SVD failed (degenerate matrix)
        return 0.0


def measure_routing_geometry(
    layer,
    dataset,
    num_steps: int = 1000,
) -> Dict[str, float]:
    """
    Measure routing behavior distribution.

    Core metrics:
    - Motif diversity: Ratio of unique routing patterns
    - Effective rank: Participation ratio over routing logits
    - Router entropy: Shannon entropy of routing distribution

    Args:
        layer: ChronoMoE layer
        dataset: Iterator yielding batches
        num_steps: Number of steps to sample

    Returns:
        Dict with geometry metrics
    """
    routing_sequences = []
    routing_logits_history = []
    router_entropies = []

    layer.eval()  # Evaluation mode (no gradient tracking)

    with torch.no_grad():
        for step in range(num_steps):
            try:
                batch = next(dataset)
            except StopIteration:
                # Dataset exhausted, use what we have
                break

            # Forward pass - returns (outputs, metadata)
            outputs, metadata = layer(batch)

            # Extract routing information from metadata
            router_probs = metadata["router_probs"]  # [B*T, num_experts]
            router_logits = metadata["router_logits"]  # [B*T, num_experts]

            # Capture routing decisions (top-2 experts per token)
            routing_seq = get_expert_sequence(router_probs, k=2)
            routing_sequences.append(routing_seq)

            # Capture routing logits (before softmax)
            routing_logits_history.append(router_logits.detach().cpu())

            # Measure routing entropy
            router_entropy = compute_routing_entropy(router_probs)
            router_entropies.append(router_entropy)

    # Compute motif diversity
    motifs = extract_motifs(routing_sequences, window_size=4)
    unique_motifs = set(motifs)
    motif_diversity = len(unique_motifs) / max(len(motifs), 1)

    # Compute effective rank
    effective_rank = compute_effective_rank(routing_logits_history)

    # Average router entropy
    avg_router_entropy = np.mean(router_entropies)

    return {
        "motif_diversity": motif_diversity,
        "effective_rank": effective_rank,
        "router_entropy": avg_router_entropy,
        "num_unique_motifs": len(unique_motifs),
        "num_steps": len(routing_sequences),
    }


def measure_adaptation_speed(
    layer,
    dataset,
    num_steps: int = 500,
) -> Dict[str, float]:
    """
    Measure how quickly routing entropy recovers after domain shift.

    Metrics:
    - Entropy trajectory over time
    - T_90: Time to reach 90% of asymptotic entropy
    - Adaptation rate: Average slope of recovery curve

    Args:
        layer: ChronoMoE layer
        dataset: Iterator yielding batches
        num_steps: Number of steps to measure

    Returns:
        Dict with adaptation metrics
    """
    entropy_trajectory = []

    layer.eval()

    with torch.no_grad():
        for step in range(num_steps):
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Forward pass - returns (outputs, metadata)
            outputs, metadata = layer(batch)

            # Extract routing information
            router_probs = metadata["router_probs"]

            # Measure routing entropy
            entropy = compute_routing_entropy(router_probs)
            entropy_trajectory.append(entropy)

    if len(entropy_trajectory) < 50:
        # Not enough data
        return {
            "T_90": 0,
            "adaptation_rate": 0.0,
            "asymptotic_entropy": 0.0,
        }

    # Estimate asymptotic entropy (average of last 50 steps)
    asymptotic_entropy = np.mean(entropy_trajectory[-50:])

    # Find T_90: time to reach 90% of asymptote
    target = 0.9 * asymptotic_entropy
    T_90 = next((i for i, e in enumerate(entropy_trajectory) if e >= target), num_steps)

    # Compute adaptation rate (average slope in recovery region)
    if T_90 > 10:
        recovery_region = entropy_trajectory[10:T_90]
        if len(recovery_region) > 1:
            adaptation_rate = np.gradient(recovery_region).mean()
        else:
            adaptation_rate = 0.0
    else:
        adaptation_rate = 0.0

    return {
        "T_90": T_90,
        "adaptation_rate": float(adaptation_rate),
        "asymptotic_entropy": float(asymptotic_entropy),
        "entropy_trajectory": entropy_trajectory,
    }


def compute_gradient_variance(layer, batch1, batch2):
    """
    Compute gradient variance between two similar inputs.

    Measures sensitivity to perturbation.

    Args:
        layer: ChronoMoE layer
        batch1: Original batch
        batch2: Perturbed batch

    Returns:
        Gradient variance (scalar)
    """
    layer.train()

    # Forward + backward on batch1
    outputs1 = layer(batch1)
    loss1 = outputs1.mean()
    grads1 = torch.autograd.grad(loss1, layer.parameters(), create_graph=False, allow_unused=True)

    # Forward + backward on batch2
    outputs2 = layer(batch2)
    loss2 = outputs2.mean()
    grads2 = torch.autograd.grad(loss2, layer.parameters(), create_graph=False, allow_unused=True)

    # Compute variance across gradients
    grad_variance = 0.0
    num_grads = 0
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            grad_variance += ((g1 - g2) ** 2).mean().item()
            num_grads += 1

    layer.eval()

    if num_grads > 0:
        return grad_variance / num_grads
    else:
        return 0.0


def run_orthogonal_probe(
    layer,
    candidate,
    dataset,
    optimizer,
    num_samples: int = 50,
) -> Dict[str, float]:
    """
    Probe that measures metrics orthogonal to correctness.

    Neither Bob nor maniac optimize for these:
    - Latency stability (variance in forward pass time)
    - Gradient variance under perturbation

    Args:
        layer: ChronoMoE layer
        candidate: Merge candidate dict
        dataset: Iterator yielding batches
        optimizer: Optimizer
        num_samples: Number of samples to collect

    Returns:
        Dict with orthogonal metrics and regression status
    """
    from chronomoe_integration.merge_execution import execute_merge
    from chronomoe_integration.delta_bundle import create_delta_bundle
    from chronomoe_integration.rollback import rollback_merge, load_delta_bundle_weights
    from pathlib import Path

    # Pre-merge baseline
    pre_latencies = []
    pre_grad_variance = []

    for _ in range(num_samples):
        try:
            batch = next(dataset)
        except StopIteration:
            break

        # Measure latency
        start = time.time()
        with torch.no_grad():
            outputs = layer(batch)
        latency = time.time() - start
        pre_latencies.append(latency)

        # Measure gradient variance under small perturbation
        perturbed_batch = batch + torch.randn_like(batch) * 0.01
        grad_var = compute_gradient_variance(layer, batch, perturbed_batch)
        pre_grad_variance.append(grad_var)

    # Create delta bundle
    bundle_dir = Path("./merge_bundles")
    bundle_dir.mkdir(exist_ok=True)

    bundle = create_delta_bundle(
        layer=layer,
        expert_a_id=candidate['expert_a_id'],
        expert_b_id=candidate['expert_b_id'],
        similarity=candidate['similarity'],
        utilization_a=candidate.get('utilization_a', 0.0),
        utilization_b=candidate.get('utilization_b', 0.0),
        trial_result={},
        current_step=layer.current_step,
    )

    # Execute merge
    execute_merge(layer, bundle, optimizer)

    # Post-merge measurement
    post_latencies = []
    post_grad_variance = []

    for _ in range(num_samples):
        try:
            batch = next(dataset)
        except StopIteration:
            break

        start = time.time()
        with torch.no_grad():
            outputs = layer(batch)
        latency = time.time() - start
        post_latencies.append(latency)

        perturbed_batch = batch + torch.randn_like(batch) * 0.01
        grad_var = compute_gradient_variance(layer, batch, perturbed_batch)
        post_grad_variance.append(grad_var)

    # Rollback merge (restore state)
    weights_data = load_delta_bundle_weights(bundle.bundle_id, bundle_dir)
    rollback_merge(layer, bundle, optimizer, weights_data=weights_data)

    # Check regression
    if len(pre_latencies) > 0 and len(post_latencies) > 0:
        latency_var_increase = np.var(post_latencies) / max(np.var(pre_latencies), 1e-8)
    else:
        latency_var_increase = 1.0

    if len(pre_grad_variance) > 0 and len(post_grad_variance) > 0:
        grad_var_increase = np.mean(post_grad_variance) / max(np.mean(pre_grad_variance), 1e-8)
    else:
        grad_var_increase = 1.0

    regression_detected = latency_var_increase > 1.2 or grad_var_increase > 1.2

    return {
        "latency_stability_delta": latency_var_increase - 1.0,
        "gradient_variance_delta": grad_var_increase - 1.0,
        "regression_detected": regression_detected,
    }
