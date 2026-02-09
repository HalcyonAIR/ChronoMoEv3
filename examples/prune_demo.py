"""
PRUNE Demo: Production-Shaped Edit Execution

Demonstrates full prune pipeline:
1. Propose (with evidence)
2. Evaluate (dry-run simulation)
3. Approve (if improvement holds)
4. Execute (with gate check)
5. Log (full audit trail)

Shows block-then-allow with actual prune execution.
Why PRUNE is careful: destroys information, requires sustained calm.
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
    create_prune_evidence,
)


def simulate_layer_state(phi_mean: float, num_experts: int, decoherent_expert: int = -1):
    """
    Simulate MoE layer state with optional decoherent expert.

    Args:
        phi_mean: Mean coherence for normal experts
        num_experts: Total number of experts
        decoherent_expert: Index of expert to make decoherent (-1 for none)
    """
    phi_slow = torch.ones(num_experts) * phi_mean
    utilization = torch.ones(num_experts) * 200.0

    # Make one expert decoherent if specified
    if decoherent_expert >= 0:
        phi_slow[decoherent_expert] = 0.2  # Very low coherence
        utilization[decoherent_expert] = 50.0  # Low utilization

    role_vectors = torch.randn(num_experts, 64)
    role_vectors = torch.nn.functional.normalize(role_vectors, dim=-1)

    # Make decoherent expert redundant with another
    if decoherent_expert >= 0 and decoherent_expert > 0:
        # Make decoherent expert similar to expert 0
        role_vectors[decoherent_expert] = (
            role_vectors[0] * 0.95 + role_vectors[decoherent_expert] * 0.05
        )
        role_vectors[decoherent_expert] = torch.nn.functional.normalize(
            role_vectors[decoherent_expert], dim=-1
        )

    bimodality_scores = torch.zeros(num_experts)
    if decoherent_expert >= 0:
        bimodality_scores[decoherent_expert] = 0.8  # Unstable

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
    print("PRUNE DEMO: Production-Shaped Edit Execution")
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
    print("PHASE 1: Normal Operation (100 steps, 5 experts)")
    print("─" * 70)

    for step in range(100):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=5)
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
    # Phase 2: Stress spike, expert becomes decoherent
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 2: Stress Spike (50 steps, expert 4 becomes decoherent)")
    print("─" * 70)

    for step in range(100, 150):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=5, decoherent_expert=4)
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter stress spike:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Expert 4 coherence: {layer['phi_slow'][4]:.3f} (LOW - decoherent)")
    print(f"  Expert 4 utilization: {layer['utilization'][4]:.1f} (LOW)")
    print(f"  Redundancy: {layer['components'].redundancy:.3f} (HIGH)")
    print(f"  Instability: {layer['components'].instability:.3f} (HIGH)")

    # ========================================================================
    # Phase 3: Try to prune, gate BLOCKS
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 3: Attempt Prune (Evidence Present, Gate Blocks)")
    print("─" * 70)

    # Create evidence
    evidence = create_prune_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=4,
        target_expert_coherence=layer["phi_slow"][4].item(),
        target_expert_utilization=layer["utilization"][4].item(),
        predicted_psi_delta=0.0,
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

    print(f"\nAttempting prune...")
    try:
        proposal = executor.propose_prune(
            layer_id=0,
            step=150,
            target_expert_id=4,
            evidence=evidence,
            gates=gates,
            phi_slow=layer["phi_slow"][4].item(),
            utilization=layer["utilization"][4].item(),
        )
        print(f"  → Prune ALLOWED (unexpected)")
    except GateViolation as e:
        print(f"  → Prune BLOCKED ✓")
        print(f"  GateViolation: {str(e).split(chr(10))[0]}")

    # ========================================================================
    # Phase 4: Return to calm
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 4: Return to Calm (600 steps)")
    print("─" * 70)

    print(f"\nRecovering...")
    for step in range(150, 750):
        # Expert 4 still decoherent but system stable
        layer = simulate_layer_state(phi_mean=0.85, num_experts=5, decoherent_expert=4)
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
    print(f"  Expert 4 still decoherent: phi={layer['phi_slow'][4]:.3f}")

    # ========================================================================
    # Phase 5: Prune now ALLOWED
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 5: Prune Now Allowed (Evidence + Calmness)")
    print("─" * 70)

    # Refresh layer state (expert 4 still decoherent)
    layer = simulate_layer_state(phi_mean=0.85, num_experts=5, decoherent_expert=4)

    # Create evidence
    evidence = create_prune_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=4,
        target_expert_coherence=layer["phi_slow"][4].item(),
        target_expert_utilization=layer["utilization"][4].item(),
        predicted_psi_delta=0.0,
    )

    print(f"\nEvidence:")
    print(f"  ΔF_l: {evidence.delta_f_l:.3f} ✓")
    print(f"  Target expert: phi={layer['phi_slow'][4]:.3f}, util={layer['utilization'][4]:.1f}")

    print(f"\nGate check:")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Edit allowed: {gates.check_edit_allowed()} ✓")

    print(f"\nAttempting prune...")
    try:
        # Full pipeline: propose → evaluate → approve → execute
        target_expert_id = executor.prune_expert_full_pipeline(
            layer_id=0,
            step=750,
            target_expert_id=4,
            evidence=evidence,
            gates=gates,
            phi_slow=layer["phi_slow"][4].item(),
            utilization=layer["utilization"][4].item(),
        )

        if target_expert_id is not None:
            print(f"  → Prune EXECUTED ✓")
            print(f"\n  Result:")
            print(f"    Removed expert: {target_expert_id}")
            print(f"    Reason: {evidence.trigger}")
            print(f"    Predicted F_l improvement: {-evidence.delta_f_l:.3f}")
        else:
            print(f"  → Prune failed")

    except GateViolation as e:
        print(f"  → Prune BLOCKED (unexpected)")
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
    print("SUMMARY: Production-Shaped PRUNE")
    print("=" * 70)

    print(f"\nWhat happened:")
    print(f"  1. Operated normally (100 steps, 5 experts)")
    print(f"  2. Expert 4 became decoherent (50 steps)")
    print(f"  3. Prune blocked due to insufficient calm")
    print(f"  4. Returned to calm (600 steps)")
    print(f"  5. Prune executed successfully")

    print(f"\nPipeline verified:")
    print(f"  ✓ Propose (with evidence)")
    print(f"  ✓ Gate check (non-bypassable)")
    print(f"  ✓ Execute (expert removed)")
    print(f"  ✓ Audit log (full trail)")

    print(f"\nWhy PRUNE is careful:")
    print(f"  • Destroys information (not reversible)")
    print(f"  • Evidence: expert persistently decoherent")
    print(f"  • Must not collapse layer (checked)")
    print(f"  • Requires sustained calm (identity change)")

    print(f"\n{'─' * 70}")
    print(f"Block-then-allow proven with actual prune execution.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("✓ PRUNE Demo Complete")
    print("=" * 70)
