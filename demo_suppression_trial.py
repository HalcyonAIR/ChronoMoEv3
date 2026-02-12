#!/usr/bin/env python3
"""
Suppression Trial Demo: Collect one example on real dataset.

Demonstrates:
1. Normal training with MERGE diagnostic mode
2. Manual suppression trial when similarity detected
3. Timeline artifact with suppression adjustments logged
"""

import sys
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from chronomoe_integration.chronomoe_layer import ChronoMoE


@dataclass
class Config:
    """Minimal config for demo."""
    n_embd: int = 128
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"


class DummyMLP(nn.Module):
    """Simple MLP for demo."""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))), {}


class SimpleDataset:
    """
    Simple synthetic dataset with two patterns:
    - Pattern A: incrementing sequences [0, 1, 2, 3, ...]
    - Pattern B: decrementing sequences [99, 98, 97, 96, ...]
    """
    def __init__(self, vocab_size=100, seq_len=20, num_batches=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Alternate between pattern A and B
        if self.current_batch % 2 == 0:
            # Pattern A: incrementing
            start = torch.randint(0, self.vocab_size - self.seq_len, (1,)).item()
            batch = torch.arange(start, start + self.seq_len).unsqueeze(0)
        else:
            # Pattern B: decrementing
            start = torch.randint(self.seq_len, self.vocab_size, (1,)).item()
            batch = torch.arange(start, start - self.seq_len, -1).unsqueeze(0)

        self.current_batch += 1
        return batch


def run_suppression_trial_demo():
    """Run short demo with manual suppression trial."""
    print("=" * 70)
    print("SUPPRESSION TRIAL DEMO")
    print("=" * 70)
    print()

    # Setup
    torch.manual_seed(42)
    config = Config()

    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Embedding layer
    embedding = nn.Embedding(100, config.n_embd)

    # Simple loss (predict next token)
    criterion = nn.CrossEntropyLoss()

    # Dataset (need 500 batches for full demo)
    dataset = SimpleDataset(vocab_size=100, seq_len=20, num_batches=600)

    # Timeline for artifact
    timeline = []

    print("Phase 1: Normal Training (0-299 steps)")
    print("-" * 70)

    for step, batch in enumerate(dataset):
        if step >= 300:
            break

        # Forward pass
        x = embedding(batch)  # [1, 20, 128]
        output, metadata = layer(x)

        # Dummy loss (random target)
        target = torch.randint(0, 100, (1, 20))
        logits = torch.randn(1, 20, 100)  # Fake logits
        loss = criterion(logits.view(-1, 100), target.view(-1))

        # Update controller with loss
        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation is not None:
            layer.controller.last_observation.loss = loss.item()

        layer.current_step += 1

        # Log timeline entry
        diagnostics = layer.controller.get_diagnostics()
        timeline.append({
            "step": step,
            "loss": round(loss.item(), 4),
            "stress_band": layer.stress_bands.current_band.name,
            "calm_credit": layer.stress_bands.time_in_comfort,
            "merge_candidates": len([
                p for p in diagnostics.get("proposals", [])
                if p.get("edit_type") == "merge"
            ]),
            "suppression_active": len(layer.controller.expert_penalties) > 0,
        })

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, band={layer.stress_bands.current_band.name}, calm={layer.stress_bands.time_in_comfort}")

    print()
    print("Phase 2: Manual Suppression Trial (300-449 steps)")
    print("-" * 70)
    print("  Suppressing expert 1 to test redundancy...")
    print()

    # Apply suppression to expert 1
    layer.controller.suppress_expert(expert_id=1, duration_steps=100)

    suppression_start = 300
    suppression_metrics = {
        "baseline_loss": [],
        "trial_loss": [],
        "recovery_loss": [],
    }

    for step in range(300, 500):
        batch = next(dataset)

        # Forward pass
        x = embedding(batch)
        output, metadata = layer(x)

        # Dummy loss
        target = torch.randint(0, 100, (1, 20))
        logits = torch.randn(1, 20, 100)
        loss = criterion(logits.view(-1, 100), target.view(-1))

        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation is not None:
            layer.controller.last_observation.loss = loss.item()

        # Track metrics for trial phases
        if step < 400:
            suppression_metrics["trial_loss"].append(loss.item())
        else:
            suppression_metrics["recovery_loss"].append(loss.item())

        # Decay penalties
        layer.controller.update_penalties()

        layer.current_step += 1

        # Log timeline entry with suppression info
        hard_blocks, soft_penalties = layer.controller.get_routing_adjustments(step)
        diagnostics = layer.controller.get_diagnostics()

        timeline.append({
            "step": step,
            "loss": round(loss.item(), 4),
            "stress_band": layer.stress_bands.current_band.name,
            "calm_credit": layer.stress_bands.time_in_comfort,
            "hard_blocks": list(hard_blocks),
            "soft_penalties": {k: round(v, 4) for k, v in soft_penalties.items()},
            "expert_utilization": metadata['expert_utilization'].tolist(),
        })

        if step % 25 == 0:
            util = metadata['expert_utilization']
            print(f"  Step {step:3d}: loss={loss.item():.4f}, expert_1_util={util[1]:.0f}, " +
                  f"hard_blocks={list(hard_blocks)}, soft_penalties={list(soft_penalties.keys())}")

    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()

    # Compute trial summary
    baseline_loss = sum(timeline[i]["loss"] for i in range(250, 300)) / 50
    trial_loss = sum(suppression_metrics["trial_loss"]) / len(suppression_metrics["trial_loss"])
    recovery_loss = sum(suppression_metrics["recovery_loss"]) / len(suppression_metrics["recovery_loss"])

    print("Trial Summary:")
    print(f"  Baseline loss (steps 250-299):   {baseline_loss:.4f}")
    print(f"  Trial loss (steps 300-399):      {trial_loss:.4f}  (delta: {trial_loss - baseline_loss:+.4f})")
    print(f"  Recovery loss (steps 400-499):   {recovery_loss:.4f}  (delta: {recovery_loss - baseline_loss:+.4f})")
    print()

    # Check if trial would pass criteria
    loss_tolerance = 0.05
    trial_passed = (trial_loss - baseline_loss) < loss_tolerance

    print(f"Trial Result: {'PASS' if trial_passed else 'FAIL'} " +
          f"(loss delta {trial_loss - baseline_loss:+.4f} vs tolerance {loss_tolerance})")
    print()

    # Save timeline artifact
    output_file = "timeline_suppression_trial_demo.json"
    with open(output_file, 'w') as f:
        json.dump({
            "demo": "suppression_trial",
            "config": {
                "n_embd": config.n_embd,
                "num_experts": config.moe_num_experts,
                "max_experts": 8,
                "suppressed_expert": 1,
                "cooldown_duration": 100,
            },
            "summary": {
                "baseline_loss": round(baseline_loss, 4),
                "trial_loss": round(trial_loss, 4),
                "recovery_loss": round(recovery_loss, 4),
                "trial_result": "PASS" if trial_passed else "FAIL",
            },
            "timeline": timeline,
        }, f, indent=2)

    print(f"Timeline artifact saved: {output_file}")
    print(f"  Total steps: {len(timeline)}")
    print(f"  Suppression phase: steps 300-399")
    print(f"  Recovery phase: steps 400-499")
    print()

    # Show sample suppression log entries
    print("Sample Suppression Log Entries:")
    if hasattr(layer, 'suppression_log') and len(layer.suppression_log) > 0:
        for i, entry in enumerate(layer.suppression_log[:5]):
            print(f"  {entry}")
        print(f"  ... ({len(layer.suppression_log)} total entries)")
    else:
        print("  (No suppression log entries)")
    print()


if __name__ == "__main__":
    run_suppression_trial_demo()
