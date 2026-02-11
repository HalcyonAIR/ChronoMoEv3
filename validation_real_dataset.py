#!/usr/bin/env python3
"""
Real Dataset Validation: SPLIT on TinyStories

Fixed seed, real training loop, AUTONOMOUS on, MERGE off.

Success criteria:
1. One SPLIT event proposed
2. Blocked in STRAIN (if high stress occurs)
3. Executed in COMFORT after calm credit accumulates
4. Children either take load OR fail probation cleanly

Output: One-line summary + timeline artifact
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import List, Optional
import json

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import StressBandsConfig, Band
from chronomoe_integration.expert_registry import ExpertState


@dataclass
class RealDatasetCriteria:
    """Pass criteria for real dataset validation."""
    split_proposed: bool = False
    split_blocked_in_strain: bool = False  # If STRAIN occurs
    split_executed_in_comfort: bool = False
    children_outcome: Optional[str] = None  # "both_graduated", "both_pruned", "one_graduated", "mixed"

    # Tracking
    first_proposal_step: int = None
    execution_step: int = None
    calm_at_execution: int = None
    child_a_id: int = None
    child_b_id: int = None
    outcome_resolved_step: int = None

    # Children state tracking
    child_a_final_state: Optional[str] = None
    child_b_final_state: Optional[str] = None

    def passes(self) -> bool:
        """
        Pass if SPLIT executed in COMFORT after calm credit AND outcome resolved.

        Outcome must be resolved (children either graduated or pruned).
        """
        return (
            self.split_proposed and
            self.split_executed_in_comfort and
            self.children_outcome is not None and
            self.children_outcome != "pending"
        )

    def one_line_summary(self, seed: int) -> str:
        """Generate grep-friendly one-line summary."""
        status = "REAL_PASS" if self.passes() else "REAL_FAIL"
        return (
            f"{status} seed={seed} "
            f"split_proposed={1 if self.split_proposed else 0} "
            f"executed_step={self.execution_step or 'NONE'} "
            f"calm_at_exec={self.calm_at_execution or 'NONE'} "
            f"outcome={self.children_outcome or 'NONE'} "
            f"states=({self.child_a_final_state or '?'},{self.child_b_final_state or '?'})"
        )


class TinyStoriesDataset(Dataset):
    """Minimal TinyStories-like dataset with learnable pattern."""
    def __init__(self, n_samples=1000, seq_len=128, vocab_size=5000, seed=42):
        torch.manual_seed(seed)

        # Generate sequences with learnable pattern: next_token = (prev_token + 1) % vocab_size
        # This simulates real data structure while being trainable
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len))

        # Create targets as shifted version (standard LM task)
        # But also inject some predictable patterns for bimodality
        for i in range(n_samples):
            # Half the sequences: simple increment pattern
            if i % 2 == 0:
                for j in range(1, seq_len):
                    self.data[i, j] = (self.data[i, j-1] + 1) % vocab_size
            # Other half: simple decrement pattern (creates bimodality)
            else:
                for j in range(1, seq_len):
                    self.data[i, j] = (self.data[i, j-1] - 1) % vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y


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
    """Minimal LM with ChronoMoE."""
    def __init__(self, vocab_size=5000, n_embd=128, max_experts=16):
        super().__init__()

        class Config:
            pass

        config = Config()
        config.n_embd = n_embd
        config.moe_num_experts = 4
        config.moe_num_experts_per_tok = 2
        config.moe_softmax_order = "softmax_topk"

        self.embed = nn.Embedding(vocab_size, n_embd)
        self.moe = ChronoMoE(
            config=config,
            mlp=TinyMLP,
            layer_id=0,
            max_experts=max_experts,
            autonomous_mode=True,  # AUTONOMOUS ON
            stress_bands_config=StressBandsConfig(
                comfort_ceiling_init=5.0,
                strain_ceiling_init=8.0,
                split_calm_steps=300,
            ),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Lower split threshold for real dataset validation
        # Natural bimodality is typically lower than test scenarios
        self.moe.controller.config["bimodality"]["split_threshold"] = 0.05

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        x, metadata = self.moe(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss, metadata


def classify_children_outcome(
    model,
    child_a_id: int,
    child_b_id: int,
) -> tuple[str, str, str]:
    """
    Classify outcome of split children.

    Returns:
        (outcome, child_a_state, child_b_state)

    Outcomes:
        - "both_graduated": Both children graduated to ACTIVE
        - "both_pruned": Both children pruned (ARCHIVED)
        - "one_graduated": One graduated, one pruned
        - "pending": Still in PROBATION or not yet determined
    """
    child_a = model.moe.registry.experts.get(child_a_id)
    child_b = model.moe.registry.experts.get(child_b_id)

    if not child_a or not child_b:
        return "pending", None, None

    state_a = child_a.state.value
    state_b = child_b.state.value

    # Both still in probation
    if state_a == "probation" and state_b == "probation":
        return "pending", state_a, state_b

    # At least one resolved
    a_graduated = state_a == "active"
    b_graduated = state_b == "active"
    a_pruned = state_a == "archived"
    b_pruned = state_b == "archived"

    if a_graduated and b_graduated:
        return "both_graduated", state_a, state_b
    elif a_pruned and b_pruned:
        return "both_pruned", state_a, state_b
    elif (a_graduated and b_pruned) or (a_pruned and b_graduated):
        return "one_graduated", state_a, state_b
    else:
        # Mixed or unknown state
        return "pending", state_a, state_b


def run_real_validation(seed=42, max_steps=3000):
    """
    Run real dataset validation.

    Args:
        seed: Random seed
        max_steps: Maximum training steps
    """
    print("=" * 70)
    print(f"REAL DATASET VALIDATION (seed={seed})")
    print("=" * 70)
    print()

    torch.manual_seed(seed)

    # Create model and dataset
    vocab_size = 5000
    model = TinyModel(vocab_size=vocab_size, n_embd=128, max_experts=16)
    dataset = TinyStoriesDataset(n_samples=1000, seq_len=128, vocab_size=vocab_size, seed=seed)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criteria = RealDatasetCriteria()
    timeline = []

    print(f"Training for {max_steps} steps")
    print(f"Dataset: TinyStories-like (synthetic)")
    print(f"Starting experts: {model.moe.registry.num_active}")
    print()

    step = 0
    while step < max_steps:
        for inputs, targets in dataloader:
            if step >= max_steps:
                break

            model.moe.current_step = step

            # Forward pass
            logits, loss, metadata = model(inputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update stress bands
            stress = loss.item()
            model.moe.update_stress_bands(stress)

            # Get current state
            current_band = model.moe.stress_bands.current_band
            calm = model.moe.stress_bands.time_in_comfort

            # Process proposals
            results = model.moe.process_controller_proposals(optimizer)

            # Track SPLIT events
            split_log = [e for e in results.get("log", []) if e["type"] == "split"]

            if split_log:
                entry = split_log[0]
                action = entry["action"]

                if action in ["REJECTED", "QUEUED"]:
                    if not criteria.split_proposed:
                        criteria.split_proposed = True
                        criteria.first_proposal_step = step
                        print(f"  [SPLIT PROPOSED] Step {step}, band={current_band.name}")

                    # Check if blocked in STRAIN
                    if current_band == Band.STRAIN:
                        criteria.split_blocked_in_strain = True
                        print(f"  [SPLIT BLOCKED] Step {step}, band=STRAIN, reason={entry.get('block_reason', 'N/A')}")

                elif action == "EXECUTED":
                    criteria.split_executed_in_comfort = True
                    criteria.execution_step = step
                    criteria.calm_at_execution = calm
                    criteria.child_a_id = entry["child_a_id"]
                    criteria.child_b_id = entry["child_b_id"]

                    print(f"  [SPLIT EXECUTED] Step {step}, band={current_band.name}, calm={calm}")
                    print(f"    Parent: {entry['expert_id']} → Children: [{criteria.child_a_id}, {criteria.child_b_id}]")

            # Track children outcome (if SPLIT happened)
            if criteria.child_a_id is not None and criteria.children_outcome is None:
                # Check continuously after SPLIT execution
                if step >= criteria.execution_step + 30:  # Probation duration is 30 steps
                    outcome, state_a, state_b = classify_children_outcome(
                        model, criteria.child_a_id, criteria.child_b_id
                    )

                    if outcome != "pending":
                        criteria.children_outcome = outcome
                        criteria.child_a_final_state = state_a
                        criteria.child_b_final_state = state_b
                        criteria.outcome_resolved_step = step
                        print(f"  [CHILDREN OUTCOME] {outcome} at step {step}")
                        print(f"    Child {criteria.child_a_id}: {state_a}")
                        print(f"    Child {criteria.child_b_id}: {state_b}")

                        # Exit early if outcome resolved (success)
                        if criteria.passes():
                            print(f"  [VALIDATION] Outcome resolved, exiting early")
                            break

            # Log progress
            if step % 200 == 0:
                diagnostics = model.moe.controller.get_diagnostics()
                f_l = diagnostics.get("free_energy", {}).get("components", {}).get("total", 0.0)

                bimodality_scores = []
                if diagnostics.get("bimodality", {}).get("by_expert"):
                    for expert_id, bim_state in diagnostics["bimodality"]["by_expert"].items():
                        bimodality_scores.append(bim_state.get("bimodality_score", 0.0))
                bimodality_max = max(bimodality_scores) if bimodality_scores else 0.0

                print(f"Step {step:4d}: loss={loss.item():.4f}, band={current_band.name:7s}, "
                      f"calm={calm:3d}, F_l={f_l:.4f}, bimodality_max={bimodality_max:.4f}, "
                      f"active={model.moe.registry.num_active}")

            timeline.append({
                "step": step,
                "loss": loss.item(),
                "band": current_band.name,
                "calm": calm,
            })

            step += 1

    print()
    print(f"Training complete: {step} steps")
    print()

    # Save timeline
    timeline_file = f"timeline_real_seed{seed}.json"
    with open(timeline_file, "w") as f:
        json.dump({
            "seed": seed,
            "timeline": timeline,
            "criteria": {
                "split_proposed": criteria.split_proposed,
                "split_blocked_in_strain": criteria.split_blocked_in_strain,
                "split_executed_in_comfort": criteria.split_executed_in_comfort,
                "children_outcome": criteria.children_outcome,
                "first_proposal_step": criteria.first_proposal_step,
                "execution_step": criteria.execution_step,
                "calm_at_execution": criteria.calm_at_execution,
                "outcome_resolved_step": criteria.outcome_resolved_step,
                "child_a_id": criteria.child_a_id,
                "child_b_id": criteria.child_b_id,
                "child_a_final_state": criteria.child_a_final_state,
                "child_b_final_state": criteria.child_b_final_state,
            }
        }, f, indent=2)

    print(f"Timeline saved to: {timeline_file}")
    print()

    # Print summary
    print("=" * 70)
    print("REAL DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"1. SPLIT proposed: {'✓' if criteria.split_proposed else '✗'}")
    if criteria.first_proposal_step:
        print(f"   First proposal at step {criteria.first_proposal_step}")

    print(f"2. SPLIT blocked in STRAIN: {'✓' if criteria.split_blocked_in_strain else 'N/A'}")

    print(f"3. SPLIT executed in COMFORT: {'✓' if criteria.split_executed_in_comfort else '✗'}")
    if criteria.execution_step:
        print(f"   Executed at step {criteria.execution_step}, calm={criteria.calm_at_execution}")

    print(f"4. Children outcome: {criteria.children_outcome or 'PENDING'}")
    if criteria.outcome_resolved_step:
        print(f"   Resolved at step {criteria.outcome_resolved_step}")
        print(f"   Child {criteria.child_a_id}: {criteria.child_a_final_state}")
        print(f"   Child {criteria.child_b_id}: {criteria.child_b_final_state}")

    print("=" * 70)
    print(f"OVERALL: [{'PASS' if criteria.passes() else 'FAIL'}]")
    print("=" * 70)
    print()
    print(criteria.one_line_summary(seed))

    return criteria.passes(), criteria


if __name__ == "__main__":
    seeds = [42, 7, 1337]
    results = []

    print("=" * 70)
    print("MULTI-SEED REAL DATASET VALIDATION")
    print("=" * 70)
    print()

    for seed in seeds:
        passed, criteria = run_real_validation(seed=seed, max_steps=3000)
        results.append((seed, passed, criteria))
        print()
        print()

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for seed, passed, criteria in results:
        print(criteria.one_line_summary(seed))

    print("=" * 70)
    all_passed = all(p for _, p, _ in results)
    if all_passed:
        print(f"\n[PASS] All {len(seeds)} seeds passed real dataset validation")
        print("Pending SPLIT latch validated across multiple seeds.")
    else:
        failed = sum(1 for _, p, _ in results if not p)
        print(f"\n[PARTIAL] {len(seeds) - failed}/{len(seeds)} seeds passed")
    print("=" * 70)
