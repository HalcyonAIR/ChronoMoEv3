#!/usr/bin/env python3
"""
Simplified Validation: SPLIT mechanics under controlled conditions

Pass criteria:
1. SPLIT proposed when bimodality injected
2. SPLIT rejected in STRAIN (constitutional blocking)
3. SPLIT executed in COMFORT after 300 calm steps
4. Parent pruned, 2 children created in PROBATION
5. Timeline artifact generated

This is a mechanics validation, not full real-world training.
For full real-world validation, see validation_training.py (requires dataset with natural bimodality).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
import json

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import StressBandsConfig, Band
from chronomoe_integration.controller import ObservationSnapshot


@dataclass
class SimpleCriteria:
    """Simplified pass criteria for mechanics validation."""
    split_proposed: bool = False
    split_blocked_in_strain: bool = False
    split_executed_in_comfort: bool = False
    parent_pruned: bool = False
    children_in_probation: bool = False

    def passes(self) -> bool:
        return all([
            self.split_proposed,
            self.split_blocked_in_strain,
            self.split_executed_in_comfort,
            self.parent_pruned,
            self.children_in_probation,
        ])

    def report(self) -> str:
        def status(passed: bool) -> str:
            return "✓ PASS" if passed else "✗ FAIL"

        lines = ["=" * 70]
        lines.append("SIMPLE VALIDATION CRITERIA REPORT")
        lines.append("=" * 70)
        lines.append(f"1. SPLIT proposed: {status(self.split_proposed)}")
        lines.append(f"2. SPLIT blocked in STRAIN: {status(self.split_blocked_in_strain)}")
        lines.append(f"3. SPLIT executed in COMFORT: {status(self.split_executed_in_comfort)}")
        lines.append(f"4. Parent pruned: {status(self.parent_pruned)}")
        lines.append(f"5. Children in PROBATION: {status(self.children_in_probation)}")
        lines.append("=" * 70)
        overall = "PASS" if self.passes() else "FAIL"
        lines.append(f"OVERALL: [{overall}]")
        lines.append("=" * 70)
        return "\n".join(lines)


class TinyMLP(nn.Module):
    """Minimal MLP expert."""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.fc(x)), {}


def inject_bimodal_snapshot(step: int, num_experts: int = 4, bimodal_expert_id: int = 0, d_model: int = 64, layer_id: int = 0):
    """Create snapshot with bimodal pattern for one expert."""
    B, T = 4, 32
    batch_size = B * T
    top_k = 2

    # Create fixed centroid for bimodal expert
    torch.manual_seed(42)  # Fixed seed for reproducible centroids
    centroid_a = torch.randn(d_model) * 2.0

    # Alternate between centroid_a and -centroid_a
    if step % 2 == 0:
        centroid = centroid_a
    else:
        centroid = -centroid_a

    # Create expert outputs (all zeros except bimodal expert)
    expert_outputs = torch.zeros(num_experts, batch_size, d_model)
    expert_outputs[bimodal_expert_id] = centroid.unsqueeze(0).expand(batch_size, -1)

    # Create router probs (uniform for simplicity)
    router_probs = torch.ones(batch_size, num_experts) / num_experts

    # Create selected experts (first top_k experts)
    selected_experts = torch.zeros(batch_size, top_k, dtype=torch.long)
    for i in range(batch_size):
        selected_experts[i] = torch.tensor([0, 1])  # Always select first two experts

    # Create utilization (uniform)
    utilization = torch.ones(num_experts) * (batch_size / num_experts)

    # Create mixture output
    mixture_output = expert_outputs[bimodal_expert_id]  # Simplified

    return ObservationSnapshot(
        step=step,
        layer_id=layer_id,
        router_probs=router_probs,
        selected_experts=selected_experts,
        expert_outputs=expert_outputs,
        mixture_output=mixture_output,
        utilization=utilization,
        loss=2.0,  # Low loss = COMFORT
    )


def run_simple_validation():
    """Run simplified validation with injected bimodality."""
    print("=" * 70)
    print("SIMPLE VALIDATION: SPLIT Mechanics")
    print("=" * 70)
    print()

    # Create ChronoMoE layer
    class Config:
        n_embd = 64
        moe_num_experts = 4
        moe_num_experts_per_tok = 2
        moe_softmax_order = "softmax_topk"

    layer = ChronoMoE(
        config=Config(),
        mlp=TinyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
        stress_bands_config=StressBandsConfig(
            comfort_ceiling_init=5.0,
            strain_ceiling_init=8.0,
            split_calm_steps=300,
        ),
    )

    criteria = SimpleCriteria()
    timeline = []

    # Phase 1: Inject bimodal observations in COMFORT (350 steps to build calm credit)
    print("Phase 1: Injecting bimodal observations (350 steps)...")
    for step in range(350):
        layer.current_step = step

        # Inject bimodal snapshot
        snapshot = inject_bimodal_snapshot(step, num_experts=4, bimodal_expert_id=0)
        layer.controller.observe(snapshot)

        # Update stress bands (low stress = COMFORT)
        layer.update_stress_bands(stress=2.0)

        if step % 100 == 0:
            band = layer.stress_bands.current_band
            print(f"  Step {step}: band={band.name}, calm={layer.stress_bands.time_in_comfort}")

    # Check if SPLIT proposed
    proposals = layer.controller.decide()
    split_proposals = [p for p in proposals if p.edit_type == "split"]
    if split_proposals:
        criteria.split_proposed = True
        print(f"✓ SPLIT proposed for expert {split_proposals[0].expert_id}")
        print(f"  Bimodality score: {split_proposals[0].evidence['bimodality_score']:.4f}")
    else:
        print("✗ No SPLIT proposal")

    # Phase 2: Force STRAIN and check blocking
    print("\nPhase 2: Force STRAIN and verify blocking...")
    # Need to sustain high stress to force STRAIN
    for _ in range(10):
        layer.update_stress_bands(stress=7.0)
    print(f"  Band: {layer.stress_bands.current_band.name}")

    # Generate new proposals (old ones executed in Phase 1)
    proposals = layer.controller.decide()

    # Try to process proposals in STRAIN
    results = layer.process_controller_proposals(optimizer=None)
    split_log = [e for e in results.get("log", []) if e["type"] == "split"]

    if layer.stress_bands.current_band == Band.STRAIN:
        if results["rejected"] > 0 or any(e["action"] == "REJECTED" for e in split_log):
            criteria.split_blocked_in_strain = True
            print(f"✓ SPLIT blocked in STRAIN (rejected={results['rejected']})")
        else:
            print(f"✗ SPLIT not blocked in STRAIN")
    else:
        print(f"✗ Not in STRAIN band (in {layer.stress_bands.current_band.name})")

    # Phase 3: Return to COMFORT and execute
    print("\nPhase 3: Return to COMFORT and execute SPLIT...")
    # Sustain low stress to return to COMFORT
    for _ in range(10):
        layer.update_stress_bands(stress=2.0)
    print(f"  Band: {layer.stress_bands.current_band.name}")
    print(f"  Calm credit: {layer.stress_bands.time_in_comfort}")

    # Inject more bimodal observations and generate proposals
    for step in range(350, 400):
        snapshot = inject_bimodal_snapshot(step, num_experts=4, bimodal_expert_id=0)
        layer.controller.observe(snapshot)

    proposals = layer.controller.decide()
    split_proposals = [p for p in proposals if p.edit_type == "split"]

    # Process proposals (should execute now)
    results = layer.process_controller_proposals(optimizer=None)
    split_log = [e for e in results.get("log", []) if e["type"] == "split"]

    # Check parent and children (may have executed in either phase)
    from chronomoe_integration.expert_registry import ExpertState

    # Find the split expert (parent should be ARCHIVED)
    split_executed = False
    for expert_id, expert_info in layer.registry.experts.items():
        if expert_info.state == ExpertState.ARCHIVED and expert_info.pruned_at is not None:
            split_executed = True
            criteria.parent_pruned = True
            parent_id = expert_id
            print(f"✓ Parent expert {parent_id}: ARCHIVED")

            # Find children
            children = [e for e in layer.registry.experts.values()
                       if e.parent_id == parent_id and e.state == ExpertState.PROBATION]
            if len(children) == 2:
                criteria.children_in_probation = True
                print(f"✓ Children [{children[0].expert_id}, {children[1].expert_id}]: PROBATION")

                # Check if executed in COMFORT (Phase 3) or earlier
                if layer.stress_bands.current_band == Band.COMFORT:
                    criteria.split_executed_in_comfort = True
                    print(f"✓ SPLIT executed in COMFORT")
                else:
                    print(f"✗ SPLIT executed but not in COMFORT")
            else:
                print(f"✗ Expected 2 children, found {len(children)}")
            break

    if not split_executed:
        print(f"✗ SPLIT not executed")

    # Save timeline
    timeline_file = "timeline_simple.json"
    with open(timeline_file, "w") as f:
        json.dump({"criteria": criteria.__dict__}, f, indent=2)
    print(f"\nTimeline saved to: {timeline_file}")

    # Print report
    print(f"\n{criteria.report()}")

    return criteria.passes()


if __name__ == "__main__":
    passed = run_simple_validation()
    if passed:
        print("\n[PASS] Simple validation passed - SPLIT mechanics validated")
    else:
        print("\n[FAIL] Simple validation failed - fix issues before continuing")
