#!/usr/bin/env python3
"""
MERGE Diagnostic: Collect Distribution Data

Samples expert pair statistics to inform threshold calibration.
Does NOT propose merges - just collects data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np

# Reuse model from main diagnostic
from validation_merge_diagnostic import TinyModel, TinyDataset


def collect_distributions(seed=42, max_steps=1000, sample_interval=100):
    """
    Collect similarity and utilization distributions without threshold filtering.

    Samples ALL expert pairs at regular intervals to understand actual distributions.
    """
    print("=" * 70)
    print("MERGE DISTRIBUTION COLLECTION")
    print("=" * 70)
    print(f"Sampling every {sample_interval} steps")
    print()

    torch.manual_seed(seed)

    # Create model
    model = TinyModel(vocab_size=5000, n_embd=128, max_experts=16)
    dataset = TinyDataset(vocab_size=5000, seq_len=16, n_samples=1000, seed=seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Collect samples
    similarity_samples = []
    utilization_samples = []
    step_samples = []

    step = 0
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Forward/backward
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.moe.update_stress_bands(loss.item())
        model.moe.current_step = step

        # Sample distributions at intervals
        if step % sample_interval == 0 and step > 0:
            controller = model.moe.controller

            # Only sample if we have sufficient data
            if controller.free_energy_state and controller.observation_history:
                print(f"Step {step}: Sampling expert pair distributions...")

                last_obs = controller.observation_history[-1]
                total_tokens = last_obs.utilization.sum().item()

                if total_tokens > 0:
                    expert_ids = list(controller.bimodality_states.keys())

                    for i in range(len(expert_ids)):
                        for j in range(i + 1, len(expert_ids)):
                            expert_a = expert_ids[i]
                            expert_b = expert_ids[j]

                            # Get bimodality states
                            state_a = controller.bimodality_states.get(expert_a)
                            state_b = controller.bimodality_states.get(expert_b)

                            if not state_a or not state_b:
                                continue

                            # Check observations
                            obs_a = state_a.count_a + state_a.count_b
                            obs_b = state_b.count_a + state_b.count_b

                            if obs_a < 100 or obs_b < 100:
                                continue  # Not enough data

                            # Compute similarity
                            centroid_a = state_a.centroid_a
                            centroid_b = state_b.centroid_a

                            if centroid_a is None or centroid_b is None:
                                continue

                            sim = torch.nn.functional.cosine_similarity(
                                centroid_a.unsqueeze(0),
                                centroid_b.unsqueeze(0),
                                dim=1
                            ).item()

                            # Compute utilization
                            util_a = last_obs.utilization[expert_a].item() / total_tokens
                            util_b = last_obs.utilization[expert_b].item() / total_tokens

                            # Record
                            similarity_samples.append(sim)
                            utilization_samples.append((util_a, util_b))
                            step_samples.append(step)

        step += 1

    print()
    print("=" * 70)
    print("DISTRIBUTION REPORT")
    print("=" * 70)
    print(f"Total samples: {len(similarity_samples)}")

    if similarity_samples:
        print()
        print("SIMILARITY DISTRIBUTION:")
        print(f"  Min:        {min(similarity_samples):.4f}")
        print(f"  25th %ile:  {np.percentile(similarity_samples, 25):.4f}")
        print(f"  50th %ile:  {np.percentile(similarity_samples, 50):.4f}")
        print(f"  75th %ile:  {np.percentile(similarity_samples, 75):.4f}")
        print(f"  90th %ile:  {np.percentile(similarity_samples, 90):.4f}")
        print(f"  95th %ile:  {np.percentile(similarity_samples, 95):.4f}")
        print(f"  Max:        {max(similarity_samples):.4f}")

        # Count how many exceed threshold
        above_threshold = sum(1 for s in similarity_samples if s > 0.8)
        print(f"  Above 0.8:  {above_threshold} ({above_threshold / len(similarity_samples) * 100:.1f}%)")

    if utilization_samples:
        utils_flat = [u for pair in utilization_samples for u in pair]
        print()
        print("UTILIZATION DISTRIBUTION:")
        print(f"  Min:        {min(utils_flat):.4f}")
        print(f"  25th %ile:  {np.percentile(utils_flat, 25):.4f}")
        print(f"  50th %ile:  {np.percentile(utils_flat, 50):.4f}")
        print(f"  75th %ile:  {np.percentile(utils_flat, 75):.4f}")
        print(f"  90th %ile:  {np.percentile(utils_flat, 90):.4f}")
        print(f"  95th %ile:  {np.percentile(utils_flat, 95):.4f}")
        print(f"  Max:        {max(utils_flat):.4f}")

        # Count how many below threshold
        below_threshold = sum(1 for u in utils_flat if u < 0.1)
        print(f"  Below 0.1:  {below_threshold} ({below_threshold / len(utils_flat) * 100:.1f}%)")

        # Check joint condition: both below 0.1 AND similarity > 0.8
        joint_candidates = 0
        for (util_a, util_b), sim in zip(utilization_samples, similarity_samples):
            if util_a < 0.1 and util_b < 0.1 and sim > 0.8:
                joint_candidates += 1

        print()
        print(f"JOINT CONDITION (util_a < 0.1 AND util_b < 0.1 AND sim > 0.8): {joint_candidates}")
        if len(utilization_samples) > 0:
            print(f"  ({joint_candidates / len(utilization_samples) * 100:.1f}% of pairs)")

    # Save raw data
    with open("merge_distributions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "similarity", "util_a", "util_b"])
        for step, sim, (util_a, util_b) in zip(step_samples, similarity_samples, utilization_samples):
            writer.writerow([step, f"{sim:.4f}", f"{util_a:.4f}", f"{util_b:.4f}"])

    print()
    print("Raw data saved to: merge_distributions.csv")
    print("=" * 70)

    return similarity_samples, utilization_samples


if __name__ == "__main__":
    collect_distributions(seed=42, max_steps=1000, sample_interval=100)
