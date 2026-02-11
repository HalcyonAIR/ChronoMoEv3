#!/usr/bin/env python3
"""
Validation Training: SPLIT under real gradient noise

Pass criteria (defined in code):
1. At least one SPLIT proposed
2. At least one SPLIT rejected/queued for valid constitutional reason
3. At least one SPLIT executes in COMFORT after two-step commit
4. Both children take non-trivial load within probation window
5. At least one "bad outcome" handled correctly:
   - Child fails probation and gets pruned, OR
   - System refrains from further splits (calm/MIN_DELTA_F blocks)
6. Forced STRAIN segment: proposals appear but executions blocked

Output: Timeline artifact showing event and child utilization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import StressBandsConfig, Band


@dataclass
class ValidationCriteria:
    """Pass criteria for validation run (defined in code, not assumptions)."""

    # Criteria 1-3: Proposal lifecycle
    split_proposed: bool = False
    split_rejected_or_queued: bool = False
    split_executed_in_comfort: bool = False

    # Criteria 4: Children take load
    children_took_load: bool = False
    child_a_id: Optional[int] = None
    child_b_id: Optional[int] = None
    split_step: Optional[int] = None

    # Criteria 5: Bad outcome handled
    bad_outcome_handled: bool = False
    bad_outcome_type: Optional[str] = None  # "child_pruned" or "further_splits_blocked"

    # Criteria 6: STRAIN blocking
    strain_blocked_split: bool = False

    def passes(self) -> bool:
        """Check if all criteria met."""
        return (
            self.split_proposed and
            self.split_rejected_or_queued and
            self.split_executed_in_comfort and
            self.children_took_load and
            self.bad_outcome_handled and
            self.strain_blocked_split
        )

    def report(self) -> str:
        """Generate pass/fail report."""
        lines = ["=" * 70]
        lines.append("VALIDATION CRITERIA REPORT")
        lines.append("=" * 70)

        def status(passed: bool) -> str:
            return "✓ PASS" if passed else "✗ FAIL"

        lines.append(f"1. SPLIT proposed: {status(self.split_proposed)}")
        lines.append(f"2. SPLIT rejected/queued: {status(self.split_rejected_or_queued)}")
        lines.append(f"3. SPLIT executed in COMFORT: {status(self.split_executed_in_comfort)}")
        lines.append(f"4. Children took load: {status(self.children_took_load)}")
        if self.children_took_load:
            lines.append(f"   - Child A: {self.child_a_id}, Child B: {self.child_b_id}")
        lines.append(f"5. Bad outcome handled: {status(self.bad_outcome_handled)}")
        if self.bad_outcome_type:
            lines.append(f"   - Type: {self.bad_outcome_type}")
        lines.append(f"6. STRAIN blocked SPLIT: {status(self.strain_blocked_split)}")

        lines.append("=" * 70)
        overall = "PASS" if self.passes() else "FAIL"
        lines.append(f"OVERALL: [{overall}]")
        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class TimelineEvent:
    """Single event in timeline."""
    step: int
    band: str
    f_l: float
    bimodality_max: float
    proposal_type: Optional[str] = None
    proposal_action: Optional[str] = None
    expert_id: Optional[int] = None
    child_a_util: float = 0.0
    child_b_util: float = 0.0


class TinyModel(nn.Module):
    """Minimal model for validation."""
    def __init__(self, vocab_size=1000, n_embd=64, max_experts=12):
        super().__init__()

        class Config:
            pass

        config = Config()
        config.n_embd = n_embd
        config.moe_num_experts = 4  # Start with 4
        config.moe_num_experts_per_tok = 2
        config.moe_softmax_order = "softmax_topk"

        self.embed = nn.Embedding(vocab_size, n_embd)
        self.moe = ChronoMoE(
            config=config,
            mlp=TinyMLP,
            layer_id=0,
            max_experts=max_experts,
            autonomous_mode=True,  # AUTONOMOUS mode
            stress_bands_config=StressBandsConfig(
                comfort_ceiling_init=5.0,  # Higher for untrained model
                strain_ceiling_init=8.0,
                split_calm_steps=300,
            ),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, top_k_override=None, inject_bimodal=False):
        x = self.embed(idx)

        # Override top_k for forced stress
        if top_k_override is not None:
            old_top_k = self.moe.top_k
            self.moe.top_k = top_k_override

        x, metadata = self.moe(x)

        if top_k_override is not None:
            self.moe.top_k = old_top_k

        # Inject bimodal pattern for expert 0 (for validation)
        if inject_bimodal and hasattr(self.moe, 'controller') and self.moe.controller is not None:
            # Alternate between two opposite centroids for expert 0
            step = self.moe.current_step
            if step % 2 == 0:
                centroid = torch.randn(self.config.n_embd) * 2.0
            else:
                centroid = -centroid if 'centroid' in locals() else torch.randn(self.config.n_embd) * 2.0

            # Store the centroid to make it persistent
            if not hasattr(self, '_bimodal_centroid'):
                self._bimodal_centroid = torch.randn(self.config.n_embd) * 2.0

            # Alternate between centroid and -centroid
            if step % 2 == 0:
                fake_output = self._bimodal_centroid
            else:
                fake_output = -self._bimodal_centroid

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss, metadata


class TinyMLP(nn.Module):
    """Minimal MLP expert."""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.fc(x)), {}


def create_synthetic_dataset(vocab_size=1000, n_samples=1000, seq_len=32, seed=42):
    """Create fixed synthetic dataset with learnable pattern."""
    torch.manual_seed(seed)

    # Generate sequences with learnable pattern: targets = (inputs + 1) % vocab_size
    # This is simple enough to learn, creating decreasing loss over time
    inputs = torch.randint(0, vocab_size, (n_samples, seq_len))
    targets = (inputs + 1) % vocab_size  # Simple shift pattern

    return TensorDataset(inputs, targets)


def run_validation(seed: int, max_steps: int = 1000, strain_window: tuple = (400, 500)):
    """
    Run validation training with fixed seed.

    Args:
        seed: Random seed for reproducibility
        max_steps: Maximum training steps
        strain_window: (start, end) steps for forced STRAIN segment
    """
    print(f"\n{'=' * 70}")
    print(f"VALIDATION TRAINING RUN (seed={seed})")
    print(f"{'=' * 70}\n")

    # Set seed
    torch.manual_seed(seed)

    # Create model and dataset
    model = TinyModel(vocab_size=1000, n_embd=64, max_experts=8)
    dataset = create_synthetic_dataset(vocab_size=1000, n_samples=1000, seq_len=32, seed=seed)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Track criteria
    criteria = ValidationCriteria()
    timeline: List[TimelineEvent] = []

    # Track all proposals and executions
    all_proposals = []
    all_executions = []

    print(f"Training for {max_steps} steps")
    print(f"STRAIN window: steps {strain_window[0]}-{strain_window[1]}")
    print(f"Starting experts: {model.moe.registry.num_active}\n")

    step = 0
    epoch = 0

    while step < max_steps:
        epoch += 1
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if step >= max_steps:
                break

            model.moe.current_step = step

            # Determine if in forced STRAIN window
            in_strain_window = strain_window[0] <= step < strain_window[1]
            top_k_override = 1 if in_strain_window else None

            # Forward pass
            logits, loss, metadata = model(inputs, targets, top_k_override)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute stress (use loss as proxy)
            stress = loss.item()

            # Force higher stress in STRAIN window
            if in_strain_window:
                stress = stress * 2.0  # Amplify stress

            # Update stress bands
            model.moe.update_stress_bands(stress)

            # Get current state
            current_band = model.moe.stress_bands.current_band
            diagnostics = model.moe.controller.get_diagnostics()

            # Get free energy and bimodality
            f_l = diagnostics.get("free_energy", {}).get("components", {}).get("total", 0.0) if diagnostics.get("free_energy") else 0.0

            bimodality_scores = []
            if diagnostics.get("bimodality", {}).get("by_expert"):
                for expert_id, bim_state in diagnostics["bimodality"]["by_expert"].items():
                    bimodality_scores.append(bim_state.get("bimodality_score", 0.0))
            bimodality_max = max(bimodality_scores) if bimodality_scores else 0.0

            # Process proposals
            results = model.moe.process_controller_proposals(optimizer)

            # Track proposals and actions
            for entry in results.get("log", []):
                action = entry["action"]
                edit_type = entry["type"]
                expert_id = entry["expert_id"]

                all_proposals.append({
                    "step": step,
                    "type": edit_type,
                    "action": action,
                    "expert_id": expert_id,
                    "band": current_band.name,
                    "reason": entry.get("block_reason", ""),
                })

                # Criteria 1: SPLIT proposed
                if edit_type == "split":
                    criteria.split_proposed = True

                    # Criteria 2: SPLIT rejected or queued
                    if action in ["REJECTED", "QUEUED"]:
                        criteria.split_rejected_or_queued = True

                        # Criteria 6: STRAIN blocking
                        if current_band == Band.STRAIN and "COMFORT" in entry.get("block_reason", ""):
                            criteria.strain_blocked_split = True

                    # Criteria 3: SPLIT executed in COMFORT
                    if action == "EXECUTED" and current_band == Band.COMFORT:
                        criteria.split_executed_in_comfort = True
                        criteria.child_a_id = entry.get("child_a_id")
                        criteria.child_b_id = entry.get("child_b_id")
                        criteria.split_step = step

                        all_executions.append({
                            "step": step,
                            "parent": expert_id,
                            "child_a": criteria.child_a_id,
                            "child_b": criteria.child_b_id,
                        })

            # Record timeline event
            event = TimelineEvent(
                step=step,
                band=current_band.name,
                f_l=f_l,
                bimodality_max=bimodality_max,
            )

            # Add proposal info if any
            if results.get("log"):
                last_entry = results["log"][-1]
                event.proposal_type = last_entry["type"]
                event.proposal_action = last_entry["action"]
                event.expert_id = last_entry["expert_id"]

            # Track child utilization if split happened
            if criteria.split_step is not None and step >= criteria.split_step:
                util = diagnostics.get("free_energy", {}).get("utilization_by_expert", {})
                if util:
                    event.child_a_util = util.get(str(criteria.child_a_id), 0.0)
                    event.child_b_util = util.get(str(criteria.child_b_id), 0.0)

                    # Criteria 4: Children take non-trivial load
                    # Check within probation window (500 steps)
                    if step <= criteria.split_step + 500:
                        if event.child_a_util > 1.0 and event.child_b_util > 1.0:
                            criteria.children_took_load = True

            timeline.append(event)

            # Check for bad outcomes (criteria 5)
            # Check if any child got pruned
            if criteria.child_a_id is not None:
                from chronomoe_integration.expert_registry import ExpertState
                if criteria.child_a_id in model.moe.registry.experts:
                    state_a = model.moe.registry.experts[criteria.child_a_id].state
                    if state_a == ExpertState.ARCHIVED:
                        criteria.bad_outcome_handled = True
                        criteria.bad_outcome_type = f"child_{criteria.child_a_id}_pruned"

                if criteria.child_b_id in model.moe.registry.experts:
                    state_b = model.moe.registry.experts[criteria.child_b_id].state
                    if state_b == ExpertState.ARCHIVED:
                        criteria.bad_outcome_handled = True
                        criteria.bad_outcome_type = f"child_{criteria.child_b_id}_pruned"

            # Check if further splits blocked
            if criteria.split_executed_in_comfort and step > criteria.split_step + 100:
                # Check if proposals are being blocked
                recent_blocks = [p for p in all_proposals if p["step"] > criteria.split_step and p["action"] in ["REJECTED", "QUEUED"]]
                if len(recent_blocks) >= 3:  # Multiple blocks = restraint
                    criteria.bad_outcome_handled = True
                    criteria.bad_outcome_type = "further_splits_blocked"

            # Log progress
            if step % 100 == 0:
                print(f"Step {step}: loss={loss.item():.4f}, band={current_band.name}, "
                      f"F_l={f_l:.4f}, bimodality={bimodality_max:.4f}, "
                      f"proposals={len(results.get('log', []))}")

            step += 1

    print(f"\nTraining complete: {step} steps, {epoch} epochs")

    # Generate timeline artifact
    print(f"\n{'=' * 70}")
    print("TIMELINE ARTIFACT")
    print(f"{'=' * 70}\n")

    # Show key events
    print("Key Events:")
    for event in timeline:
        if event.proposal_type:
            print(f"  Step {event.step}: {event.proposal_action} {event.proposal_type} "
                  f"(expert {event.expert_id}, band={event.band})")

    # Show child utilization window
    if criteria.split_step is not None:
        print(f"\nChild Utilization (first 500 steps after split at step {criteria.split_step}):")
        print("Step | Child A | Child B")
        print("-----|---------|--------")

        start = criteria.split_step
        end = min(criteria.split_step + 500, len(timeline))

        for i in range(start, end, 50):
            if i < len(timeline):
                event = timeline[i]
                print(f"{event.step:4d} | {event.child_a_util:7.1f} | {event.child_b_util:7.1f}")

    # Save timeline to file
    timeline_file = f"timeline_seed{seed}.json"
    with open(timeline_file, "w") as f:
        json.dump([{
            "step": e.step,
            "band": e.band,
            "f_l": e.f_l,
            "bimodality_max": e.bimodality_max,
            "proposal_type": e.proposal_type,
            "proposal_action": e.proposal_action,
            "expert_id": e.expert_id,
            "child_a_util": e.child_a_util,
            "child_b_util": e.child_b_util,
        } for e in timeline], f, indent=2)

    print(f"\nTimeline saved to: {timeline_file}")

    # Print criteria report
    print(f"\n{criteria.report()}")

    return criteria.passes(), criteria


def main():
    """Run validation with multiple seeds."""
    seeds = [42, 123, 456]
    results = []

    for seed in seeds:
        passed, criteria = run_validation(
            seed=seed,
            max_steps=1000,
            strain_window=(400, 500)
        )
        results.append((seed, passed, criteria))

        if not passed:
            print(f"\n[FAIL] Seed {seed} did not pass validation criteria")
            print("Stopping - fix issues before continuing")
            break
        else:
            print(f"\n[PASS] Seed {seed} passed all validation criteria")

        # Stop after first pass (Halcyon wants one pass, then two more)
        if len(results) == 1:
            print(f"\nFirst seed passed. Run with additional seeds to validate robustness.")
            break

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")

    for seed, passed, criteria in results:
        status = "PASS" if passed else "FAIL"
        print(f"Seed {seed}: [{status}]")

    all_passed = all(p for _, p, _ in results)
    if all_passed and len(results) >= 2:
        print(f"\n[PASS] Validation complete: {len(results)} seeds passed")
        print("SPLIT operation validated under real training. Ready for MERGE consideration.")
    elif len(results) == 1 and results[0][1]:
        print(f"\n[PASS] First seed passed. Repeat with more seeds to confirm robustness.")
    else:
        print(f"\n[FAIL] Validation incomplete or failed")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
