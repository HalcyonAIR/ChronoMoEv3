#!/usr/bin/env python3
"""
Validation: MERGE Diagnostic-Only Mode (Milestone F Phase 1)

Purpose: Collect merge candidate data for threshold calibration.

Constraints:
- Controller mode: AUTONOMOUS (proposals allowed)
- merge.enabled: True (proposals generated)
- merge execution: HARD DISABLED (no apply path, even if proposed)

Output: CSV timeline with merge candidate data for analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
import csv
import json

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import Band, StressBandsConfig


# Use same dataset as SPLIT validation for comparability
class TinyDataset(torch.utils.data.Dataset):
    """
    Learnable synthetic dataset (increment/decrement sequences).
    Same as used for SPLIT validation.
    """
    def __init__(self, vocab_size=5000, seq_len=16, n_samples=1000, seed=42):
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        # Generate learnable patterns
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

        # Create patterns: even samples increment, odd samples decrement
        for i in range(n_samples):
            if i % 2 == 0:
                # Increment sequence
                for j in range(1, seq_len):
                    self.data[i, j] = (self.data[i, j-1] + 1) % vocab_size
            else:
                # Decrement sequence
                for j in range(1, seq_len):
                    self.data[i, j] = (self.data[i, j-1] - 1) % vocab_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


class TinyMLP(nn.Module):
    """MLP expert."""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x))), {}


class TinyModel(nn.Module):
    """Minimal transformer-like model with ChronoMoE integration."""
    def __init__(self, vocab_size=5000, n_embd=128, max_experts=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        class Config:
            pass

        config = Config()
        config.n_embd = n_embd
        config.moe_num_experts = 4
        config.moe_num_experts_per_tok = 2
        config.moe_softmax_order = "softmax_topk"

        self.embedding = nn.Embedding(vocab_size, n_embd)

        # ChronoMoE layer - MERGE diagnostic config applied after init
        self.moe = ChronoMoE(
            config=config,
            mlp=TinyMLP,
            layer_id=0,
            max_experts=max_experts,
            autonomous_mode=True,  # AUTONOMOUS (proposals generated)
            stress_bands_config=StressBandsConfig(
                comfort_ceiling_init=5.0,
                strain_ceiling_init=8.0,
            ),
        )

        # Enable MERGE diagnostic mode
        self.moe.controller.config["merge"]["enabled"] = True

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)  # [B, T] -> [B, T, D]
        x, metadata = self.moe(x)  # ChronoMoE layer
        logits = self.lm_head(x)  # [B, T, D] -> [B, T, vocab]
        return logits


@dataclass
class MergeDiagnosticCriteria:
    """Track merge candidate appearances."""
    candidates_found: int = 0
    total_steps: int = 0
    first_candidate_step: Optional[int] = None

    # Distributions for calibration
    similarity_scores: list = None
    utilization_pairs: list = None
    delta_f_values: list = None

    def __post_init__(self):
        self.similarity_scores = []
        self.utilization_pairs = []
        self.delta_f_values = []


def run_merge_diagnostic(seed=42, max_steps=2000, csv_path="merge_candidates.csv"):
    """
    Run merge diagnostic collection.

    CRITICAL: Merge proposals are generated but NEVER executed.
    This is enforced by not calling any merge execution code.
    """
    print("=" * 70)
    print("MERGE DIAGNOSTIC VALIDATION (Milestone F Phase 1)")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Controller mode: AUTONOMOUS (proposals allowed)")
    print("  - merge.enabled: True (diagnostic proposals)")
    print("  - Merge execution: HARD DISABLED (no apply path)")
    print()

    # Set seed
    torch.manual_seed(seed)

    # Create model and dataset
    model = TinyModel(vocab_size=5000, n_embd=128, max_experts=16)
    dataset = TinyDataset(vocab_size=5000, seq_len=16, n_samples=1000, seed=seed)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    criteria = MergeDiagnosticCriteria()
    timeline = []

    # CSV output
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'step', 'band', 'calm', 'loss',
        'candidate_found', 'expert_a', 'expert_b',
        'similarity', 'util_a', 'util_b', 'delta_f_pred'
    ])

    print(f"Training for {max_steps} steps")
    print(f"Dataset: TinyStories-like (synthetic)")
    print(f"Starting experts: 4")
    print(f"CSV output: {csv_path}")
    print()

    step = 0
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Forward pass
        inputs = batch[:, :-1]  # [B, T-1]
        targets = batch[:, 1:]  # [B, T-1]

        logits = model(inputs)  # [B, T-1, vocab]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stress bands
        model.moe.update_stress_bands(loss.item())
        model.moe.current_step = step

        # Get current state
        current_band = model.moe.stress_bands.current_band
        calm = model.moe.stress_bands.time_in_comfort

        # Process controller proposals
        # NOTE: This generates proposals but merge execution is DISABLED
        # The layer has no merge_experts() call path in process_controller_proposals()
        results = model.moe.process_controller_proposals(optimizer)

        # Check for MERGE proposals (diagnostic only)
        merge_logs = [e for e in results.get("log", []) if e["type"] == "merge"]

        candidate_found = False
        expert_a = None
        expert_b = None
        similarity = None
        util_a = None
        util_b = None
        delta_f_pred = None

        if merge_logs:
            entry = merge_logs[0]
            candidate_found = True

            # Extract from proposal evidence
            evidence = entry.get("evidence", {})
            expert_a = entry.get("expert_id")
            expert_b = evidence.get("expert_b_id")
            similarity = evidence.get("similarity_score")
            util_a = evidence.get("utilization_a")
            util_b = evidence.get("utilization_b")
            delta_f_pred = evidence.get("predicted_delta_f")

            # Track for distributions
            if similarity is not None:
                criteria.similarity_scores.append(similarity)
            if util_a is not None and util_b is not None:
                criteria.utilization_pairs.append((util_a, util_b))
            if delta_f_pred is not None:
                criteria.delta_f_values.append(delta_f_pred)

            criteria.candidates_found += 1
            if criteria.first_candidate_step is None:
                criteria.first_candidate_step = step
                print(f"  [FIRST CANDIDATE] Step {step}, experts ({expert_a}, {expert_b}), similarity={similarity:.4f}")

        # Write CSV row
        csv_writer.writerow([
            step,
            current_band.name,
            calm,
            f"{loss.item():.4f}",
            1 if candidate_found else 0,
            expert_a or '',
            expert_b or '',
            f"{similarity:.4f}" if similarity else '',
            f"{util_a:.4f}" if util_a else '',
            f"{util_b:.4f}" if util_b else '',
            f"{delta_f_pred:.4f}" if delta_f_pred else '',
        ])

        # Timeline for JSON
        timeline.append({
            "step": step,
            "loss": loss.item(),
            "band": current_band.name,
            "calm": calm,
            "candidate_found": candidate_found,
        })

        # Log progress
        if step % 200 == 0:
            diagnostics = model.moe.controller.get_diagnostics()
            merge_diag = diagnostics.get("merge")
            cand_status = "YES" if merge_diag and merge_diag.get("candidate_found") else "NO"

            print(f"Step {step:4d}: loss={loss.item():.4f}, band={current_band.name:7s}, "
                  f"calm={calm:3d}, candidate={cand_status}, "
                  f"total_candidates={criteria.candidates_found}")

        step += 1
        criteria.total_steps = step

    csv_file.close()

    # Save timeline JSON
    timeline_file = f"timeline_merge_diagnostic_seed{seed}.json"
    with open(timeline_file, "w") as f:
        json.dump({
            "seed": seed,
            "timeline": timeline,
            "criteria": {
                "candidates_found": criteria.candidates_found,
                "total_steps": criteria.total_steps,
                "first_candidate_step": criteria.first_candidate_step,
            }
        }, f, indent=2)

    print()
    print("=" * 70)
    print("MERGE DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Total steps: {criteria.total_steps}")
    print(f"Candidates found: {criteria.candidates_found}")
    print(f"Candidate rate: {criteria.candidates_found / criteria.total_steps * 100:.2f}%")

    if criteria.first_candidate_step is not None:
        print(f"First candidate at step: {criteria.first_candidate_step}")
    else:
        print("No candidates found (check thresholds)")

    # Distribution stats
    if criteria.similarity_scores:
        print()
        print("Similarity distribution:")
        print(f"  Min: {min(criteria.similarity_scores):.4f}")
        print(f"  Max: {max(criteria.similarity_scores):.4f}")
        print(f"  Mean: {sum(criteria.similarity_scores) / len(criteria.similarity_scores):.4f}")

    if criteria.utilization_pairs:
        print()
        print("Utilization distribution:")
        utils_flat = [u for pair in criteria.utilization_pairs for u in pair]
        print(f"  Min: {min(utils_flat):.4f}")
        print(f"  Max: {max(utils_flat):.4f}")
        print(f"  Mean: {sum(utils_flat) / len(utils_flat):.4f}")

    if criteria.delta_f_values:
        print()
        print("Predicted ΔF_l distribution:")
        print(f"  Min: {min(criteria.delta_f_values):.4f}")
        print(f"  Max: {max(criteria.delta_f_values):.4f}")
        print(f"  Mean: {sum(criteria.delta_f_values) / len(criteria.delta_f_values):.4f}")

    print("=" * 70)
    print()
    print(f"CSV saved to: {csv_path}")
    print(f"Timeline saved to: {timeline_file}")

    return criteria


if __name__ == "__main__":
    criteria = run_merge_diagnostic(seed=42, max_steps=2000, csv_path="merge_candidates.csv")
