#!/usr/bin/env python3
"""
Simplified Validation v2: SPLIT mechanics under controlled conditions

Clear phase separation:
1. Build calm credit in COMFORT (350 steps)
2. Verify SPLIT proposed
3. Force STRAIN, verify SPLIT blocked
4. Return to COMFORT, verify SPLIT executes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from dataclasses import dataclass
import json

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.stress_bands import StressBandsConfig, Band
from chronomoe_integration.controller import ObservationSnapshot
from chronomoe_integration.expert_registry import ExpertState


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
        lines.append("VALIDATION CRITERIA REPORT")
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


# Shared centroid for bimodal expert
CENTROID_A = None


def inject_bimodal_snapshot(step: int, num_experts: int = 4, bimodal_expert_id: int = 0, d_model: int = 64, layer_id: int = 0):
    """Create snapshot with bimodal pattern for one expert."""
    global CENTROID_A

    B, T = 4, 32
    batch_size = B * T
    top_k = 2

    # Create fixed centroid (once)
    if CENTROID_A is None:
        torch.manual_seed(42)
        CENTROID_A = torch.randn(d_model) * 2.0

    # Alternate between CENTROID_A and -CENTROID_A
    if step % 2 == 0:
        centroid = CENTROID_A
    else:
        centroid = -CENTROID_A

    # Create expert outputs (all zeros except bimodal expert)
    expert_outputs = torch.zeros(num_experts, batch_size, d_model)
    expert_outputs[bimodal_expert_id] = centroid.unsqueeze(0).expand(batch_size, -1)

    # Create router probs (uniform)
    router_probs = torch.ones(batch_size, num_experts) / num_experts

    # Create selected experts (first top_k)
    selected_experts = torch.zeros(batch_size, top_k, dtype=torch.long)
    for i in range(batch_size):
        selected_experts[i] = torch.tensor([0, 1])

    # Create utilization (uniform)
    utilization = torch.ones(num_experts) * (batch_size / num_experts)

    # Create mixture output
    mixture_output = expert_outputs[bimodal_expert_id]

    return ObservationSnapshot(
        step=step,
        layer_id=layer_id,
        router_probs=router_probs,
        selected_experts=selected_experts,
        expert_outputs=expert_outputs,
        mixture_output=mixture_output,
        utilization=utilization,
        loss=2.0,
    )


def run_simple_validation():
    """Run simplified validation with injected bimodality."""
    print("=" * 70)
    print("SPLIT MECHANICS VALIDATION")
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

    # PHASE 1: Build calm credit in COMFORT (350 steps)
    print("PHASE 1: Build calm credit (350 steps in COMFORT)")
    print("-" * 70)
    for step in range(350):
        layer.current_step = step
        snapshot = inject_bimodal_snapshot(step, num_experts=4, bimodal_expert_id=0)
        layer.controller.observe(snapshot)
        layer.update_stress_bands(stress=2.0)  # Low stress = COMFORT

        if step % 100 == 0:
            band = layer.stress_bands.current_band
            calm = layer.stress_bands.time_in_comfort
            print(f"  Step {step:3d}: band={band.name:7s}, calm={calm:3d}")

    print(f"  Final: band={layer.stress_bands.current_band.name}, calm={layer.stress_bands.time_in_comfort}")
    print()

    # PHASE 2: Verify SPLIT proposed
    print("PHASE 2: Verify SPLIT proposed")
    print("-" * 70)
    proposals = layer.controller.decide()
    split_proposals = [p for p in proposals if p.edit_type == "split"]

    if split_proposals:
        criteria.split_proposed = True
        print(f"  ✓ SPLIT proposed for expert {split_proposals[0].expert_id}")
        print(f"  Bimodality score: {split_proposals[0].evidence['bimodality_score']:.4f}")
    else:
        print(f"  ✗ No SPLIT proposal")
    print()

    # PHASE 3: Force STRAIN, verify SPLIT blocked
    print("PHASE 3: Force STRAIN, verify SPLIT blocked")
    print("-" * 70)
    # Sustain high stress to force STRAIN
    for i in range(20):
        layer.update_stress_bands(stress=7.5)

    print(f"  Band: {layer.stress_bands.current_band.name}")
    print(f"  Calm: {layer.stress_bands.time_in_comfort}")

    if layer.stress_bands.current_band == Band.STRAIN:
        # Try to process SPLIT in STRAIN
        results = layer.process_controller_proposals(optimizer=None)
        split_log = [e for e in results.get("log", []) if e["type"] == "split"]

        if split_log and split_log[0]["action"] == "REJECTED":
            criteria.split_blocked_in_strain = True
            print(f"  ✓ SPLIT blocked: {split_log[0]['block_reason']}")
        else:
            print(f"  ✗ SPLIT not blocked (action={split_log[0]['action'] if split_log else 'NONE'})")
    else:
        print(f"  ✗ Not in STRAIN (in {layer.stress_bands.current_band.name})")
    print()

    # PHASE 4: Return to COMFORT, execute SPLIT
    print("PHASE 4: Return to COMFORT, execute SPLIT")
    print("-" * 70)
    # Sustain low stress to return to COMFORT and rebuild calm
    for i in range(350):
        layer.update_stress_bands(stress=2.0)

    print(f"  Band: {layer.stress_bands.current_band.name}")
    print(f"  Calm: {layer.stress_bands.time_in_comfort}")

    # Process SPLIT in COMFORT
    results = layer.process_controller_proposals(optimizer=None)
    split_log = [e for e in results.get("log", []) if e["type"] == "split"]

    if split_log and split_log[0]["action"] == "EXECUTED":
        criteria.split_executed_in_comfort = True
        print(f"  ✓ SPLIT executed")

        # Check parent and children
        parent_id = split_log[0]["expert_id"]
        child_a_id = split_log[0]["child_a_id"]
        child_b_id = split_log[0]["child_b_id"]

        if layer.registry.experts[parent_id].state == ExpertState.ARCHIVED:
            criteria.parent_pruned = True
            print(f"  ✓ Parent expert {parent_id}: ARCHIVED")

        if (layer.registry.experts[child_a_id].state == ExpertState.PROBATION and
            layer.registry.experts[child_b_id].state == ExpertState.PROBATION):
            criteria.children_in_probation = True
            print(f"  ✓ Children [{child_a_id}, {child_b_id}]: PROBATION")
        else:
            print(f"  ✗ Children not in PROBATION")
    else:
        print(f"  ✗ SPLIT not executed (action={split_log[0]['action'] if split_log else 'NONE'})")
    print()

    # Save timeline
    timeline_file = "timeline_simple_v2.json"
    with open(timeline_file, "w") as f:
        json.dump({"criteria": criteria.__dict__}, f, indent=2)
    print(f"Timeline saved to: {timeline_file}")
    print()

    # Print report
    print(criteria.report())

    return criteria.passes()


if __name__ == "__main__":
    passed = run_simple_validation()
    if passed:
        print("\n[PASS] SPLIT mechanics validated")
    else:
        print("\n[FAIL] SPLIT mechanics validation failed")
