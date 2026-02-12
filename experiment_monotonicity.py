#!/usr/bin/env python3
"""
Phase 1: Monotonicity Validation

Maps scar strength → geometric deformation to validate:
1. Monotonicity (contraction increases with scar strength)
2. Smoothness (no discontinuous jumps)
3. Saturation (diminishing returns)
4. Absence of fragility (mild shift + mild scar = small contraction)

Grid: Scar strength × Shift magnitude
Outputs: Deformation surface plot, monotonicity report
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from synthetic_datasets import ArithmeticProgressionDataset
from chronomoe_integration.routing_geometry import (
    measure_routing_geometry,
    measure_adaptation_speed,
)
from chronomoe_integration import ChronoMoE


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
        h = self.embed(x)
        h_flat = h.view(-1, h.size(-1))
        h_moe, metadata = self.moe(h_flat)
        h = h_moe.view(h.size(0), h.size(1), -1)
        logits = self.output(h)
        return logits


class EmbeddingDatasetWrapper:
    """Wraps dataset to embed batches."""

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.dataset)
        h = self.model.embed(batch)
        h_flat = h.view(-1, h.size(-1))
        return h_flat


def create_shift_dataset(shift_type: str, vocab_size: int, seq_len: int, batch_size: int, seed: int):
    """
    Create dataset with specified shift magnitude.

    Shift types:
    - "none": Arithmetic → Arithmetic (control)
    - "mild": Arithmetic + 10% multiplicative noise
    - "moderate": Arithmetic × 1.5 (scaled pattern)
    - "severe": Geometric (exponential, original experiment)
    """
    if shift_type == "none":
        # Control: Same as baseline
        return ArithmeticProgressionDataset(vocab_size, seq_len, batch_size, seed)

    elif shift_type == "mild":
        # Mild: Arithmetic with noise
        class MildNoiseDataset:
            def __init__(self, vocab_size, seq_len, batch_size, seed):
                self.base = ArithmeticProgressionDataset(vocab_size, seq_len, batch_size, seed)
                self.rng = np.random.RandomState(seed + 100)

            def __iter__(self):
                return self

            def __next__(self):
                batch = next(self.base)
                # Add 10% multiplicative noise
                noise = self.rng.randn(*batch.shape) * 0.1
                noisy = (batch.float() * (1 + noise)).long()
                return noisy.clamp(0, self.base.vocab_size - 1)

        return MildNoiseDataset(vocab_size, seq_len, batch_size, seed)

    elif shift_type == "moderate":
        # Moderate: Scaled arithmetic
        class ModerateScaleDataset:
            def __init__(self, vocab_size, seq_len, batch_size, seed):
                self.base = ArithmeticProgressionDataset(vocab_size, seq_len, batch_size, seed)

            def __iter__(self):
                return self

            def __next__(self):
                batch = next(self.base)
                # Scale by 1.5
                scaled = (batch.float() * 1.5).long()
                return scaled % self.base.vocab_size

        return ModerateScaleDataset(vocab_size, seq_len, batch_size, seed)

    elif shift_type == "severe":
        # Severe: Geometric (original experiment)
        from synthetic_datasets import GeometricProgressionDataset
        return GeometricProgressionDataset(vocab_size, seq_len, batch_size, seed)

    else:
        raise ValueError(f"Unknown shift type: {shift_type}")


def train_baseline(model, dataset, num_steps=1000):
    """Train baseline model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(num_steps):
        batch = next(dataset)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"    Step {step:4d}: loss={loss.item():.4f}")


def apply_scars(model, scarred_experts: List[int], penalty_magnitude: float):
    """Apply scars (penalties) to specified experts."""
    for expert_id in scarred_experts:
        model.moe.controller.expert_penalties[expert_id] = penalty_magnitude


def remove_scars(model, scarred_experts: List[int]):
    """Remove scars from specified experts."""
    for expert_id in scarred_experts:
        if expert_id in model.moe.controller.expert_penalties:
            del model.moe.controller.expert_penalties[expert_id]


def measure_contraction(
    model,
    baseline_diversity: float,
    shift_dataset,
    scar_strength: float,
    scarred_experts: List[int],
) -> dict:
    """
    Measure contraction at specific scar strength.

    Returns:
        Dict with contraction metrics
    """
    # Apply scars
    apply_scars(model, scarred_experts, penalty_magnitude=scar_strength)

    # Wrap dataset
    shift_embedded = EmbeddingDatasetWrapper(shift_dataset, model)

    # Measure geometry
    geometry = measure_routing_geometry(model.moe, shift_embedded, num_steps=500)

    # Remove scars
    remove_scars(model, scarred_experts)

    # Compute contraction
    contraction = baseline_diversity - geometry['motif_diversity']
    contraction_pct = (contraction / baseline_diversity) * 100 if baseline_diversity > 0 else 0

    return {
        "motif_diversity": geometry['motif_diversity'],
        "effective_rank": geometry['effective_rank'],
        "router_entropy": geometry['router_entropy'],
        "contraction": contraction,
        "contraction_pct": contraction_pct,
    }


def run_monotonicity_experiment():
    """
    Run full monotonicity experiment.

    Grid: Scar strength × Shift magnitude
    Outputs: Deformation surface, monotonicity analysis
    """
    print("=" * 70)
    print("PHASE 1: MONOTONICITY VALIDATION")
    print("=" * 70)
    print()
    print("Hypothesis: Scar strength produces graded, monotonic deformation")
    print("Grid: Scar strength × Shift magnitude")
    print()

    # Setup
    torch.manual_seed(42)
    config = Config()

    # Scar strength grid
    scar_strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

    # Shift magnitude grid
    shift_types = ["none", "mild", "moderate", "severe"]

    # Results storage
    results = {
        "scar_strengths": scar_strengths,
        "shift_types": shift_types,
        "grid": {},  # shift_type -> scar_strength -> metrics
    }

    # Create baseline dataset (arithmetic)
    baseline_dataset = ArithmeticProgressionDataset(
        vocab_size=config.vocab_size,
        seq_len=config.block_size,
        batch_size=4,
        seed=42,
    )

    # Train baseline model
    print("Training baseline model (1000 steps)...")
    print("-" * 70)
    model = TinyModel(config=config, max_experts=8)
    train_baseline(model, baseline_dataset, num_steps=1000)
    print()

    # Measure baseline geometry
    print("Measuring baseline geometry...")
    print("-" * 70)
    baseline_dataset_embedded = EmbeddingDatasetWrapper(baseline_dataset, model)
    baseline_geometry = measure_routing_geometry(model.moe, baseline_dataset_embedded, num_steps=500)
    baseline_diversity = baseline_geometry['motif_diversity']

    print(f"  Baseline motif diversity: {baseline_diversity:.6f}")
    print(f"  Baseline router entropy:  {baseline_geometry['router_entropy']:.4f}")
    print()

    # Experts to scar (half of active experts)
    num_experts = model.moe.registry.max_experts
    active_experts = [i for i in range(num_experts) if model.moe.registry.active_mask[i]]
    scarred_experts = active_experts[: len(active_experts) // 2]

    print(f"Scarring experts: {scarred_experts}")
    print()

    # Run grid
    for shift_type in shift_types:
        print(f"Shift Type: {shift_type.upper()}")
        print("-" * 70)

        # Create shift dataset
        shift_dataset = create_shift_dataset(
            shift_type,
            vocab_size=config.vocab_size,
            seq_len=config.block_size,
            batch_size=4,
            seed=42 + 1000,
        )

        results["grid"][shift_type] = {}

        for scar_strength in scar_strengths:
            print(f"  Scar strength: {scar_strength:4.1f} ", end="")

            metrics = measure_contraction(
                model,
                baseline_diversity,
                shift_dataset,
                scar_strength,
                scarred_experts,
            )

            results["grid"][shift_type][scar_strength] = metrics

            print(f"→ Contraction: {metrics['contraction_pct']:6.1f}% "
                  f"(diversity: {metrics['motif_diversity']:.6f})")

        print()

    # Analyze monotonicity
    print("=" * 70)
    print("MONOTONICITY ANALYSIS")
    print("=" * 70)
    print()

    for shift_type in shift_types:
        print(f"Shift: {shift_type}")

        contractions = [
            results["grid"][shift_type][strength]["contraction_pct"]
            for strength in scar_strengths
        ]

        # Check monotonicity
        is_monotonic = all(contractions[i] <= contractions[i+1]
                          for i in range(len(contractions)-1))

        # Check smoothness (no large jumps)
        diffs = [contractions[i+1] - contractions[i] for i in range(len(contractions)-1)]
        max_jump = max(diffs) if diffs else 0

        print(f"  Monotonic: {'✓' if is_monotonic else '✗'}")
        print(f"  Max jump:  {max_jump:.1f}%")
        print(f"  Contractions: {[f'{c:.1f}' for c in contractions]}")
        print()

    # Save results
    output_path = Path("./monotonicity_results.json")
    with open(output_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"Results saved: {output_path}")
    print()

    return results


if __name__ == "__main__":
    results = run_monotonicity_experiment()

    # Check if monotonicity holds across all shifts
    all_monotonic = True
    for shift_type in results["shift_types"]:
        contractions = [
            results["grid"][shift_type][strength]["contraction_pct"]
            for strength in results["scar_strengths"]
        ]
        is_monotonic = all(contractions[i] <= contractions[i+1]
                          for i in range(len(contractions)-1))
        if not is_monotonic:
            all_monotonic = False

    if all_monotonic:
        print("✓ MONOTONICITY VALIDATED: Scar strength produces graded deformation")
        sys.exit(0)
    else:
        print("✗ MONOTONICITY FAILED: Deformation is not monotonic")
        print("  → Identity may be phase-based, not scalar")
        sys.exit(1)
