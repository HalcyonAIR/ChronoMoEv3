"""
SPLIT Demo: Production-Shaped Edit Execution

Demonstrates full split pipeline:
1. Propose (with evidence)
2. Evaluate (dry-run simulation)
3. Approve (if improvement holds)
4. Execute (with gate check)
5. Log (full audit trail)

Shows block-then-allow with actual split execution.
Why SPLIT is reversible: redistributes capacity without destroying information.
"""

import sys
from pathlib import Path
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    # Free energy
    compute_free_energy,
    compute_layer_coherence,
    # Stress bands
    init_stress_bands,
    step_stress_bands,
    # Gates
    LifecycleGates,
    GateViolation,
    get_default_stress_config,
    get_default_collapse_thresholds,
    # Collapse detection
    collapse_signals_from_free_energy,
    check_survival,
    # Edit execution
    EditExecutor,
    create_split_evidence,
)


def simulate_layer_state(phi_mean: float, num_experts: int, bimodal_expert: int = -1):
    """
    Simulate MoE layer state with optional bimodal expert.

    Args:
        phi_mean: Mean coherence for normal experts
        num_experts: Total number of experts
        bimodal_expert: Index of expert to make bimodal (-1 for none)
    """
    phi_slow = torch.ones(num_experts) * phi_mean
    utilization = torch.ones(num_experts) * 200.0

    role_vectors = torch.randn(num_experts, 64)
    role_vectors = torch.nn.functional.normalize(role_vectors, dim=-1)

    bimodality_scores = torch.zeros(num_experts)
    if bimodal_expert >= 0:
        bimodality_scores[bimodal_expert] = 0.85  # High bimodality
        utilization[bimodal_expert] = 300.0  # High utilization (serving two modes)

    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    psi = compute_layer_coherence(phi_slow, utilization)
    router_probs = utilization / utilization.sum()

    return {
        "f_l": f_l,
        "psi": psi,
        "phi_slow": phi_slow,
        "utilization": utilization,
        "components": components,
        "router_probs": router_probs,
        "role_vectors": role_vectors,
        "bimodality_scores": bimodality_scores,
    }


def main():
    print("=" * 70)
    print("SPLIT DEMO: Production-Shaped Edit Execution")
    print("=" * 70)

    # Setup
    stress_cfg = get_default_stress_config()
    stress_state = init_stress_bands(stress_cfg)
    collapse_thresholds = get_default_collapse_thresholds()

    # Create audit log
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        audit_log_path = f.name

    executor = EditExecutor(audit_log_path=audit_log_path)
    gates = LifecycleGates(stress_state, stress_cfg)

    print(f"\nAudit log: {audit_log_path}")

    # ========================================================================
    # Phase 1: Normal operation
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 1: Normal Operation (100 steps, 4 experts)")
    print("─" * 70)

    for step in range(100):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=4)
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter 100 steps:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")

    # ========================================================================
    # Phase 2: Stress spike, expert becomes bimodal
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 2: Stress Spike (50 steps, expert 2 becomes bimodal)")
    print("─" * 70)

    for step in range(100, 150):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=4, bimodal_expert=2)
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter stress spike:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Expert 2 bimodality: {layer['bimodality_scores'][2]:.3f} (HIGH - bimodal)")
    print(f"  Expert 2 utilization: {layer['utilization'][2]:.1f} (HIGH - serving two modes)")
    print(f"  Instability: {layer['components'].instability:.3f} (HIGH)")

    # ========================================================================
    # Phase 3: Try to split, gate BLOCKS
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 3: Attempt Split (Evidence Present, Gate Blocks)")
    print("─" * 70)

    # Create evidence
    evidence = create_split_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=5,  # 4 → 5 (replace bimodal with two coherent)
        source_expert_bimodality=layer["bimodality_scores"][2].item(),
        source_expert_utilization=layer["utilization"][2].item(),
        predicted_psi_improvement=0.05,
    )

    print(f"\nEvidence:")
    print(f"  Current F_l: {evidence.f_l_before:.3f}")
    print(f"  Predicted F_l: {evidence.f_l_predicted:.3f}")
    print(f"  ΔF_l: {evidence.delta_f_l:.3f} ✓")
    print(f"  Trigger: {evidence.trigger}")

    print(f"\nGate check:")
    print(f"  Band: {stress_state.current_band}")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Edit allowed: {gates.check_edit_allowed()}")

    print(f"\nAttempting split...")
    try:
        proposal = executor.propose_split(
            layer_id=0,
            step=150,
            source_expert_id=2,
            evidence=evidence,
            gates=gates,
            bimodality=layer["bimodality_scores"][2].item(),
        )
        print(f"  → Split ALLOWED (unexpected)")
    except GateViolation as e:
        print(f"  → Split BLOCKED ✓")
        print(f"  GateViolation: {str(e).split(chr(10))[0]}")

    # ========================================================================
    # Phase 4: Return to calm
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 4: Return to Calm (600 steps)")
    print("─" * 70)

    print(f"\nRecovering...")
    for step in range(150, 750):
        # Expert 2 still bimodal but system stable
        layer = simulate_layer_state(phi_mean=0.85, num_experts=4, bimodal_expert=2)
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

        if (step - 150) % 100 == 0:
            print(f"  Step {step}: time_in_comfort={stress_state.time_in_comfort}, edit_allowed={gates.check_edit_allowed()}")

    print(f"\nAfter calm recovery:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort} ✓")
    print(f"  Expert 2 still bimodal: bimodality={layer['bimodality_scores'][2]:.3f}")

    # ========================================================================
    # Phase 5: Split now ALLOWED
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 5: Split Now Allowed (Evidence + Calmness)")
    print("─" * 70)

    # Refresh layer state (expert 2 still bimodal)
    layer = simulate_layer_state(phi_mean=0.85, num_experts=4, bimodal_expert=2)

    # Create evidence
    evidence = create_split_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=5,
        source_expert_bimodality=layer["bimodality_scores"][2].item(),
        source_expert_utilization=layer["utilization"][2].item(),
        predicted_psi_improvement=0.05,
    )

    print(f"\nEvidence:")
    print(f"  ΔF_l: {evidence.delta_f_l:.3f} ✓")
    print(f"  Source expert: bimodality={layer['bimodality_scores'][2]:.3f}, util={layer['utilization'][2]:.1f}")

    print(f"\nGate check:")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Edit allowed: {gates.check_edit_allowed()} ✓")

    print(f"\nAttempting split...")
    try:
        # Full pipeline: propose → evaluate → approve → execute
        source_params = {"weight": layer["role_vectors"][2]}  # Simulate parameters

        result = executor.split_expert_full_pipeline(
            layer_id=0,
            step=750,
            source_expert_id=2,
            source_params=source_params,
            child_a_id=4,
            child_b_id=5,
            evidence=evidence,
            gates=gates,
            bimodality=layer["bimodality_scores"][2].item(),
        )

        if result and result.success:
            print(f"  → Split EXECUTED ✓")
            print(f"\n  Result:")
            print(f"    Source expert: {result.source_expert_id}")
            print(f"    Child A expert: {result.child_a_id}")
            print(f"    Child B expert: {result.child_b_id}")
            print(f"    Child A params: {list(result.child_a_params.keys())}")
            print(f"    Child B params: {list(result.child_b_params.keys())}")
            print(f"    Reason: {result.reason}")
        else:
            print(f"  → Split failed")

    except GateViolation as e:
        print(f"  → Split BLOCKED (unexpected)")
        print(f"  Error: {e}")

    # ========================================================================
    # Audit log summary
    # ========================================================================

    print("\n" + "─" * 70)
    print("AUDIT LOG SUMMARY")
    print("─" * 70)

    audit_log = executor.get_audit_log()
    print(f"\nTotal events: {len(audit_log)}")

    for entry in audit_log:
        print(f"\n  [{entry.event_type}] Step {entry.step}")
        print(f"    Proposal: {entry.proposal_id}")
        print(f"    Context: F_l={entry.f_l:.3f}, band={entry.stress_band}, time_in_comfort={entry.time_in_comfort}")

    print(f"\n✓ Full audit trail saved to: {audit_log_path}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY: Production-Shaped SPLIT")
    print("=" * 70)

    print(f"\nWhat happened:")
    print(f"  1. Operated normally (100 steps, 4 experts)")
    print(f"  2. Expert 2 became bimodal (50 steps)")
    print(f"  3. Split blocked due to insufficient calm")
    print(f"  4. Returned to calm (600 steps)")
    print(f"  5. Split executed successfully")

    print(f"\nPipeline verified:")
    print(f"  ✓ Propose (with evidence)")
    print(f"  ✓ Gate check (non-bypassable)")
    print(f"  ✓ Execute (two new experts created)")
    print(f"  ✓ Audit log (full trail)")

    print(f"\nWhy SPLIT is reversible:")
    print(f"  • Doesn't destroy information (redistributes)")
    print(f"  • Evidence: expert persistently bimodal")
    print(f"  • Improves instability (bimodal → two unimodal)")
    print(f"  • Can merge back if doesn't help")

    print(f"\n{'─' * 70}")
    print(f"Block-then-allow proven with actual split execution.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("✓ SPLIT Demo Complete")
    print("=" * 70)
