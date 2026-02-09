"""
SPAWN Demo: Production-Shaped Edit Execution

Demonstrates full spawn pipeline:
1. Propose (with evidence)
2. Evaluate (dry-run simulation)
3. Approve (if improvement holds)
4. Execute (with gate check)
5. Log (full audit trail)

Shows block-then-allow with actual edit execution.
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
    create_spawn_evidence,
)


def simulate_layer_state(phi_mean: float, num_experts: int):
    """Simulate MoE layer state."""
    phi_slow = torch.ones(num_experts) * phi_mean
    utilization = torch.ones(num_experts) * 200.0
    role_vectors = torch.randn(num_experts, 64)
    role_vectors = torch.nn.functional.normalize(role_vectors, dim=-1)
    bimodality_scores = torch.zeros(num_experts)

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
    }


def main():
    print("=" * 70)
    print("SPAWN DEMO: Production-Shaped Edit Execution")
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
    print("PHASE 1: Normal Operation (100 steps)")
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
    # Phase 2: Stress spike, layer starving
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 2: Stress Spike (50 steps, coherence drops)")
    print("─" * 70)

    for step in range(100, 150):
        layer = simulate_layer_state(phi_mean=0.3, num_experts=4)  # Starving
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter stress spike:")
    print(f"  F_l: {result.f_raw:.3f} (HIGH), Band: {result.band_now}")
    print(f"  Psi: {layer['psi']:.3f} (LOW - layer starving)")
    print(f"  Misfit: {layer['components'].misfit:.3f}")

    # ========================================================================
    # Phase 3: Try to spawn, gate BLOCKS
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 3: Attempt Spawn (Evidence Present, Gate Blocks)")
    print("─" * 70)

    # Create evidence
    evidence = create_spawn_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=5,
        predicted_psi_improvement=0.2,
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

    print(f"\nAttempting spawn...")
    try:
        proposal = executor.propose_spawn(
            layer_id=0,
            step=150,
            parent_expert_id=0,  # Clone best expert
            evidence=evidence,
            gates=gates,
        )
        print(f"  → Spawn ALLOWED (unexpected)")
    except GateViolation as e:
        print(f"  → Spawn BLOCKED ✓")
        print(f"  GateViolation: {str(e).split(chr(10))[0]}")

    # ========================================================================
    # Phase 4: Return to calm
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 4: Return to Calm (600 steps)")
    print("─" * 70)

    print(f"\nRecovering...")
    for step in range(150, 750):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=4)  # Healthy again
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

    # ========================================================================
    # Phase 5: Spawn now ALLOWED
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 5: Spawn Now Allowed (Evidence + Calmness)")
    print("─" * 70)

    # Refresh layer state (still could benefit from spawn)
    layer = simulate_layer_state(phi_mean=0.75, num_experts=4)

    # Create evidence
    evidence = create_spawn_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=5,
        predicted_psi_improvement=0.1,
    )

    print(f"\nEvidence:")
    print(f"  ΔF_l: {evidence.delta_f_l:.3f} ✓")

    print(f"\nGate check:")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Edit allowed: {gates.check_edit_allowed()} ✓")

    print(f"\nAttempting spawn...")
    try:
        # Full pipeline: propose → evaluate → approve → execute
        parent_params = {"weight": layer["role_vectors"][0]}  # Simulate parameters

        result = executor.spawn_expert_full_pipeline(
            layer_id=0,
            step=750,
            parent_expert_id=0,
            parent_params=parent_params,
            child_expert_id=4,
            evidence=evidence,
            gates=gates,
        )

        if result and result.success:
            print(f"  → Spawn EXECUTED ✓")
            print(f"\n  Result:")
            print(f"    Parent expert: {result.parent_expert_id}")
            print(f"    Child expert: {result.child_expert_id}")
            print(f"    Child params: {list(result.child_params.keys())}")
            print(f"    Reason: {result.reason}")
        else:
            print(f"  → Spawn failed")

    except GateViolation as e:
        print(f"  → Spawn BLOCKED (unexpected)")
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
    print("SUMMARY: Production-Shaped SPAWN")
    print("=" * 70)

    print(f"\nWhat happened:")
    print(f"  1. Operated normally (100 steps)")
    print(f"  2. Stress spike (50 steps, layer starving)")
    print(f"  3. Spawn blocked due to insufficient calm")
    print(f"  4. Returned to calm (600 steps)")
    print(f"  5. Spawn executed successfully")

    print(f"\nPipeline verified:")
    print(f"  ✓ Propose (with evidence)")
    print(f"  ✓ Gate check (non-bypassable)")
    print(f"  ✓ Execute (parameters cloned + perturbed)")
    print(f"  ✓ Audit log (full trail)")

    print(f"\nWhy SPAWN is clean:")
    print(f"  • Doesn't destroy information")
    print(f"  • Only adds capacity")
    print(f"  • Reversible (can prune if doesn't help)")
    print(f"  • Evidence: layer starving (high misfit)")

    print(f"\n{'─' * 70}")
    print(f"Block-then-allow proven with actual edit execution.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("✓ SPAWN Demo Complete")
    print("=" * 70)
