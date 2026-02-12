#!/usr/bin/env python3
"""
Autonomous MERGE Execution Demo: swiss-ai/MoE Testbed

Fixed script demonstrating:
1. Train baseline (200 steps)
2. Find MERGE candidate (high similarity + low utilization)
3. Run suppression trial (if candidate found)
4. If PASS → Execute merge (delta bundle)
5. Run probe battery immediately after merge
6. If protected delta regresses → Auto-rollback
7. Record outcome
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass

from chronomoe_integration import (
    ChronoMoE,
    create_delta_bundle,
    save_delta_bundle_weights,
    save_delta_bundle_metadata,
    execute_merge,
    find_merge_candidate,
    run_probe_battery,
    check_protected_deltas,
    print_probe_battery_result,
    rollback_merge,
    verify_rollback,
)


@dataclass
class Config:
    """Model configuration."""
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 1  # Single layer for testing
    vocab_size: int = 100
    block_size: int = 20
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"


class SimpleMLP(nn.Module):
    """Simple MLP for expert."""
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
    """Minimal transformer with ChronoMoE layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # ChronoMoE layer (replaces standard FFN)
        self.moe = ChronoMoE(
            config=config,
            mlp=SimpleMLP,
            layer_id=0,
            max_experts=8,
            autonomous_mode=True,
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # [B, T, n_embd]

        # MoE layer
        x, metadata = self.moe(x)  # [B, T, n_embd]

        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]

        return logits, metadata


class SimpleDataset:
    """
    Simple dataset with two patterns to encourage expert specialization.
    Pattern A: incrementing sequences
    Pattern B: decrementing sequences
    """
    def __init__(self, vocab_size=100, seq_len=20, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Alternate between patterns
        if self.step % 2 == 0:
            # Pattern A: incrementing
            start = torch.randint(0, self.vocab_size - self.seq_len, (1,), generator=self.rng).item()
            batch = torch.arange(start, start + self.seq_len).unsqueeze(0)
        else:
            # Pattern B: decrementing
            start = torch.randint(self.seq_len, self.vocab_size, (1,), generator=self.rng).item()
            batch = torch.arange(start, start - self.seq_len, -1).unsqueeze(0)

        self.step += 1
        return batch


def train_step(model, batch, optimizer, criterion):
    """Single training step."""
    model.train()

    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Forward
    logits, metadata = model(inputs)

    # Loss
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update stress bands
    model.moe.update_stress_bands(loss.item())
    model.moe.current_step += 1

    # Update controller with loss
    if hasattr(model.moe.controller, 'last_observation') and model.moe.controller.last_observation:
        model.moe.controller.last_observation.loss = loss.item()

    return loss.item()


def run_suppression_trial_simple(model, dataset, optimizer, criterion, suppressed_expert, baseline_window=50, trial_window=100):
    """Simplified suppression trial for demo."""
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
    loss_tolerance = 0.05
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
    print("AUTONOMOUS MERGE EXECUTION DEMO")
    print("=" * 70)
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

    # Phase 1: Train baseline
    print("Phase 1: Training Baseline")
    print("-" * 70)

    for step in range(200):
        batch = next(dataset)
        loss = train_step(model, batch, optimizer, criterion)

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss={loss:.4f}")

    print()

    # Phase 2: Find MERGE candidate
    print("Phase 2: Identifying MERGE Candidate")
    print("-" * 70)

    candidate = find_merge_candidate(
        model.moe.controller,
        similarity_threshold=0.8,
        utilization_threshold=0.1,
        min_observations=100,
    )

    if candidate is None:
        print("  No candidates found. Exiting.")
        print()
        return

    print(f"  Candidate found:")
    print(f"    Experts: {candidate['expert_a_id']} and {candidate['expert_b_id']}")
    print(f"    Similarity: {candidate['similarity']:.4f}")
    print(f"    Utilization: {candidate['utilization_a']:.4f}, {candidate['utilization_b']:.4f}")
    print()

    # Phase 3: Run suppression trial
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
        print("  MERGE rejected by suppression trial. Exiting.")
        print()
        return

    print()

    # Phase 4: Execute merge
    print("Phase 4: Executing MERGE")
    print("-" * 70)

    # 4a. Run probe battery before merge
    print("  Running pre-merge probe battery...")
    pre_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)
    print(f"    Loss: {pre_metrics['loss']:.4f}")
    print(f"    Perplexity: {pre_metrics['perplexity']:.2f}")
    print(f"    F_l: {pre_metrics['f_l']:.4f}")
    print(f"    Coherence: {pre_metrics['coherence']:.4f}")
    print(f"    Neff: {pre_metrics['neff']:.2f}")
    print()

    # 4b. Create delta bundle (for rollback)
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

    # Phase 5: Probe battery after merge
    print("Phase 5: Running Post-Merge Probe Battery")
    print("-" * 70)

    post_metrics = run_probe_battery(model, dataset, criterion, num_steps=50)
    print(f"    Loss: {post_metrics['loss']:.4f}")
    print(f"    Perplexity: {post_metrics['perplexity']:.2f}")
    print(f"    F_l: {post_metrics['f_l']:.4f}")
    print(f"    Coherence: {post_metrics['coherence']:.4f}")
    print(f"    Neff: {post_metrics['neff']:.2f}")

    # Phase 6: Check protected deltas
    print()
    print("Phase 6: Checking Protected Deltas")
    print("-" * 70)

    probe_result = check_protected_deltas(pre_metrics, post_metrics)
    print_probe_battery_result(probe_result)

    # Phase 7: Rollback if needed
    if probe_result.rollback_triggered:
        print("Phase 7: ROLLBACK TRIGGERED")
        print("-" * 70)

        # Load delta bundle weights
        from chronomoe_integration.delta_bundle import load_delta_bundle_weights
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
            tolerance=0.01,
        )

        outcome = "ROLLED_BACK"
        print()
    else:
        print("Phase 7: Protected Deltas Satisfied")
        print("-" * 70)
        print("  MERGE accepted. No rollback needed.")
        outcome = "ACCEPTED"
        print()

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

    with open(bundle_dir / "merge_execution_outcome.json", 'w') as f:
        json.dump(outcome_data, f, indent=2)

    print("=" * 70)
    print(f"MERGE EXECUTION COMPLETE: {outcome}")
    print("=" * 70)
    print()
    print(f"Outcome saved: {bundle_dir / 'merge_execution_outcome.json'}")
    print()


if __name__ == "__main__":
    main()
