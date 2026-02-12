#!/usr/bin/env python3
"""
Domain Shift Experiment: Arithmetic → Geometric

Measures geometric deformation to validate core hypothesis:
1. Scars (penalties) contract reachable manifold
2. Epoch review (penalty removal) expands reachability

Falsifiable criterion:
- If motif diversity does NOT drop by ≥5% under shift with scars → scars are bookkeeping
- If epoch review does NOT expand motif diversity by ≥5% → review doesn't work

One-sentence summary: "Scars accumulated under arithmetic patterns contract routing
diversity, epoch review detects obsolescence under geometric patterns and restores
reachability."
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import json
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_datasets import create_domain_shift_datasets
from chronomoe_integration.routing_geometry import (
    measure_routing_geometry,
    measure_adaptation_speed,
)
from chronomoe_integration import (
    ChronoMoE,
    create_controller,
)


@dataclass
class Config:
    """Model configuration."""
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 1
    vocab_size: int = 100
    block_size: int = 20
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"


# Simple MLP for experts
class SimpleMLP(nn.Module):
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
    """Minimal model with one ChronoMoE layer."""

    def __init__(self, config, max_experts=8):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)

        self.moe = ChronoMoE(
            config=config,
            mlp=SimpleMLP,
            layer_id=0,
            max_experts=max_experts,
            autonomous_mode=True,
        )

        self.output = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        h = self.embed(x)  # [batch, seq_len, d_model]
        h_flat = h.view(-1, h.size(-1))  # [batch*seq_len, d_model]
        h_moe, metadata = self.moe(h_flat)  # Returns (outputs, metadata)
        h = h_moe.view(h.size(0), h.size(1), -1)  # [batch, seq_len, d_model]
        logits = self.output(h)  # [batch, seq_len, vocab_size]
        return logits


def train_step(model, batch, optimizer, criterion):
    """Single training step."""
    # batch: [batch_size, seq_len]
    inputs = batch[:, :-1]  # All but last token
    targets = batch[:, 1:]  # All but first token

    optimizer.zero_grad()
    logits = model(inputs)  # [batch, seq_len-1, vocab_size]

    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss.item()


def simulate_scar_accumulation(model, old_dataset, optimizer, criterion, num_steps=5000):
    """
    Train on old dataset and simulate scar accumulation.

    Returns:
        List of expert IDs that would have scars (high penalties)
    """
    print(f"Training {num_steps} steps on old dataset (arithmetic)...")

    model.train()
    losses = []

    for step in range(num_steps):
        batch = next(old_dataset)
        loss = train_step(model, batch, optimizer, criterion)
        losses.append(loss)

        if step % 500 == 0:
            print(f"  Step {step:5d}: loss={loss:.4f}")

    # Identify "scarred" experts (simulate: experts with low utilization get scars)
    # In real system, scars would form from harm events
    # Here, we'll artificially mark half the experts as "scarred"
    num_experts = model.moe.registry.max_experts
    active_experts = [i for i in range(num_experts) if model.moe.registry.active_mask[i]]

    # Mark half as scarred (simulate experts that caused harm)
    scarred_experts = active_experts[: len(active_experts) // 2]

    print(f"\n  Simulated scars on experts: {scarred_experts}")
    print(f"  Average loss: {sum(losses[-100:]) / 100:.4f}")

    return scarred_experts


def apply_scars(model, scarred_experts, penalty_magnitude=5.0):
    """
    Apply scars (penalties) to specified experts.

    This simulates the effect of scars on routing.
    """
    print(f"\nApplying scars (penalties) to {len(scarred_experts)} experts...")

    for expert_id in scarred_experts:
        model.moe.controller.expert_penalties[expert_id] = penalty_magnitude

    print(f"  Penalty magnitude: {penalty_magnitude}")
    print(f"  Scarred experts: {scarred_experts}")


def remove_scars(model, scarred_experts):
    """
    Remove scars (penalties) from specified experts.

    This simulates epoch review removing obsolete scars.
    """
    print(f"\nRemoving scars from {len(scarred_experts)} experts...")

    for expert_id in scarred_experts:
        if expert_id in model.moe.controller.expert_penalties:
            del model.moe.controller.expert_penalties[expert_id]

    print(f"  Scars removed: {scarred_experts}")


def embed_batch(model, batch):
    """Embed token batch for MoE layer."""
    # batch: [batch_size, seq_len] token IDs
    # returns: [batch_size * seq_len, n_embd] embeddings
    h = model.embed(batch)  # [batch, seq_len, n_embd]
    h_flat = h.view(-1, h.size(-1))  # [batch*seq_len, n_embd]
    return h_flat


class EmbeddingDatasetWrapper:
    """Wraps dataset to embed batches."""

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.dataset)
        return embed_batch(self.model, batch)


def run_domain_shift_experiment():
    """
    Complete domain shift experiment.

    Phases:
    1. Baseline reachability (old dataset, no scars)
    2. Train with scar accumulation (old dataset)
    3. Apply scars and measure on new dataset (contraction expected)
    4. Remove scars and measure on new dataset (expansion expected)
    """
    print("=" * 70)
    print("DOMAIN SHIFT EXPERIMENT: Arithmetic → Geometric")
    print("=" * 70)
    print()
    print("Hypothesis: Scars contract reachable manifold")
    print("Falsifiable: Motif diversity must drop by ≥5% with scars")
    print()

    # Setup
    torch.manual_seed(42)

    config = Config(
        n_embd=64,
        vocab_size=100,
        block_size=20,
        moe_num_experts=4,
        moe_num_experts_per_tok=2,
    )

    old_dataset, new_dataset = create_domain_shift_datasets(
        vocab_size=config.vocab_size,
        seq_len=config.block_size,
        batch_size=4,
        seed=42,
    )

    model = TinyModel(config=config, max_experts=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    results = {}

    # Phase 1: Baseline Reachability (old dataset, no scars)
    print("Phase 1: Baseline Reachability (Arithmetic, No Scars)")
    print("-" * 70)

    # Wrap datasets to embed batches
    old_dataset_embedded = EmbeddingDatasetWrapper(old_dataset, model)
    new_dataset_embedded = EmbeddingDatasetWrapper(new_dataset, model)

    baseline_geometry = measure_routing_geometry(model.moe, old_dataset_embedded, num_steps=500)
    baseline_adaptation = measure_adaptation_speed(model.moe, old_dataset_embedded, num_steps=300)

    print(f"  Motif diversity:  {baseline_geometry['motif_diversity']:.4f}")
    print(f"  Effective rank:   {baseline_geometry['effective_rank']:.2f}")
    print(f"  Router entropy:   {baseline_geometry['router_entropy']:.4f}")
    print(f"  Unique motifs:    {baseline_geometry['num_unique_motifs']}")
    print(f"  T_90 recovery:    {baseline_adaptation['T_90']} steps")
    print()

    results["baseline"] = baseline_geometry

    # Phase 2: Train with scar accumulation
    print("Phase 2: Training with Scar Accumulation (Arithmetic)")
    print("-" * 70)

    scarred_experts = simulate_scar_accumulation(
        model, old_dataset, optimizer, criterion, num_steps=5000
    )
    print()

    # Phase 3: Domain shift with scars active
    print("Phase 3: Domain Shift with Scars Active (Geometric)")
    print("-" * 70)

    # Apply scars
    apply_scars(model, scarred_experts, penalty_magnitude=5.0)

    # Measure reachability on new dataset
    scarred_geometry_new = measure_routing_geometry(model.moe, new_dataset_embedded, num_steps=500)
    scarred_adaptation_new = measure_adaptation_speed(model.moe, new_dataset_embedded, num_steps=300)

    print(f"  Motif diversity:  {scarred_geometry_new['motif_diversity']:.4f}")
    print(f"  Effective rank:   {scarred_geometry_new['effective_rank']:.2f}")
    print(f"  Router entropy:   {scarred_geometry_new['router_entropy']:.4f}")
    print(f"  Unique motifs:    {scarred_geometry_new['num_unique_motifs']}")
    print(f"  T_90 recovery:    {scarred_adaptation_new['T_90']} steps")
    print()

    # CRITICAL MEASUREMENT: Contraction
    contraction = baseline_geometry['motif_diversity'] - scarred_geometry_new['motif_diversity']
    contraction_pct = (contraction / baseline_geometry['motif_diversity']) * 100

    print(f"  Reachability contraction: {contraction:.4f} ({contraction_pct:.1f}%)")

    if contraction_pct < 5.0:
        print(f"  ⚠️  WARNING: Contraction < 5% threshold")
        print(f"  ⚠️  Scars may be bookkeeping, not geometric deformation")
    else:
        print(f"  ✓ Significant contraction detected")

    print()

    results["scarred_new"] = scarred_geometry_new
    results["contraction"] = contraction
    results["contraction_pct"] = contraction_pct

    # Phase 4: Epoch review (remove obsolete scars)
    print("Phase 4: Epoch Review (Remove Obsolete Scars)")
    print("-" * 70)

    remove_scars(model, scarred_experts)
    print()

    # Phase 5: Measure recovery
    print("Phase 5: Post-Review Reachability (Geometric)")
    print("-" * 70)

    recovered_geometry = measure_routing_geometry(model.moe, new_dataset_embedded, num_steps=500)
    recovered_adaptation = measure_adaptation_speed(model.moe, new_dataset_embedded, num_steps=300)

    print(f"  Motif diversity:  {recovered_geometry['motif_diversity']:.4f}")
    print(f"  Effective rank:   {recovered_geometry['effective_rank']:.2f}")
    print(f"  Router entropy:   {recovered_geometry['router_entropy']:.4f}")
    print(f"  Unique motifs:    {recovered_geometry['num_unique_motifs']}")
    print(f"  T_90 recovery:    {recovered_adaptation['T_90']} steps")
    print()

    # CRITICAL MEASUREMENT: Expansion
    expansion = recovered_geometry['motif_diversity'] - scarred_geometry_new['motif_diversity']
    expansion_pct = (expansion / scarred_geometry_new['motif_diversity']) * 100 if scarred_geometry_new['motif_diversity'] > 0 else 0

    print(f"  Reachability expansion: {expansion:.4f} ({expansion_pct:.1f}%)")

    if expansion_pct < 5.0:
        print(f"  ⚠️  WARNING: Expansion < 5% threshold")
        print(f"  ⚠️  Epoch review may not restore reachability")
    else:
        print(f"  ✓ Significant expansion detected")

    print()

    results["recovered"] = recovered_geometry
    results["expansion"] = expansion
    results["expansion_pct"] = expansion_pct

    # Summary Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    hypothesis_confirmed = contraction_pct >= 5.0 and expansion_pct >= 5.0

    if hypothesis_confirmed:
        print("✓ HYPOTHESIS CONFIRMED:")
        print("  - Scars contract reachable geometry (≥5% motif diversity drop)")
        print("  - Epoch review expands reachability (≥5% motif diversity recovery)")
        print("  - Architecture validated, not just bookkeeping")
        print()
        print("  This is measurable geometric deformation.")
    else:
        print("✗ HYPOTHESIS FALSIFIED:")
        print()
        if contraction_pct < 5.0:
            print("  - Scars do NOT contract manifold (< 5% drop)")
            print("  - Mechanism is bookkeeping, not architectural")
        if expansion_pct < 5.0:
            print("  - Epoch review does NOT restore reachability (< 5% recovery)")
            print("  - Review mechanism ineffective")
        print()
        print("  Return to design phase. Governance is metaphysics without deformation.")

    print()
    print("=" * 70)

    results["hypothesis_confirmed"] = hypothesis_confirmed

    # Save results
    output_path = Path("./domain_shift_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {output_path}")
    print()

    return results


if __name__ == "__main__":
    results = run_domain_shift_experiment()

    # Exit with status code based on hypothesis
    if results["hypothesis_confirmed"]:
        print("✓ Experiment passed: Geometric deformation validated")
        sys.exit(0)
    else:
        print("✗ Experiment failed: No geometric deformation detected")
        sys.exit(1)
