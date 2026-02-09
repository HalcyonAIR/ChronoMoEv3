"""
MERGE Demo: Production-Shaped Edit Execution (MOST DANGEROUS)

Demonstrates full merge pipeline:
1. Propose (with evidence)
2. Evaluate (dry-run simulation)
3. Approve (if improvement holds)
4. Execute (with gate check)
5. Log (full audit trail)

Shows block-then-allow with actual merge execution.

WARNING: MERGE is destructive and NOT reversible.
This is where you accidentally delete a personality and call it compression.
"""

import sys
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

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
    create_merge_evidence,
)


def simulate_layer_state(
    phi_mean: float, num_experts: int, redundant_pair: tuple = (-1, -1)
):
    """
    Simulate MoE layer state with optional redundant expert pair.

    Args:
        phi_mean: Mean coherence for normal experts
        num_experts: Total number of experts
        redundant_pair: (expert_a, expert_b) to make redundant (-1, -1 for none)
    """
    phi_slow = torch.ones(num_experts) * phi_mean
    utilization = torch.ones(num_experts) * 200.0

    role_vectors = torch.randn(num_experts, 64)
    role_vectors = torch.nn.functional.normalize(role_vectors, dim=-1)

    # Make two experts redundant (highly similar)
    if redundant_pair[0] >= 0 and redundant_pair[1] >= 0:
        a, b = redundant_pair
        # Make expert b very similar to expert a (similarity ~0.95)
        role_vectors[b] = role_vectors[a] * 0.95 + role_vectors[b] * 0.05
        role_vectors[b] = torch.nn.functional.normalize(role_vectors[b], dim=-1)

        # Both have lower utilization (redundant capacity)
        utilization[a] = 150.0
        utilization[b] = 100.0

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

    # Compute similarity between redundant pair
    similarity = 0.0
    if redundant_pair[0] >= 0 and redundant_pair[1] >= 0:
        a, b = redundant_pair
        similarity = F.cosine_similarity(
            role_vectors[a].unsqueeze(0), role_vectors[b].unsqueeze(0)
        ).item()

    return {
        "f_l": f_l,
        "psi": psi,
        "phi_slow": phi_slow,
        "utilization": utilization,
        "components": components,
        "router_probs": router_probs,
        "role_vectors": role_vectors,
        "similarity": similarity,
    }


def main():
    print("=" * 70)
    print("MERGE DEMO: Production-Shaped Edit Execution (MOST DANGEROUS)")
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

    print(f"\n{'!' * 70}")
    print("WARNING: MERGE is destructive and NOT reversible.")
    print("This demo shows the most dangerous edit operation.")
    print(f"{'!' * 70}")

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
    # Phase 2: Experts become redundant
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 2: Redundancy Develops (50 steps, experts 3 and 4 become similar)")
    print("─" * 70)

    for step in range(100, 150):
        layer = simulate_layer_state(phi_mean=0.85, num_experts=5, redundant_pair=(3, 4))
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter redundancy develops:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Experts 3 & 4 similarity: {layer['similarity']:.3f} (HIGH - redundant)")
    print(f"  Expert 3 utilization: {layer['utilization'][3]:.1f}")
    print(f"  Expert 4 utilization: {layer['utilization'][4]:.1f}")
    print(f"  Redundancy: {layer['components'].redundancy:.3f}")

    # ========================================================================
    # Phase 3: Try to merge, gate BLOCKS
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 3: Attempt Merge (Evidence Present, Gate Blocks)")
    print("─" * 70)

    # Create evidence
    evidence = create_merge_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=4,  # 5 → 4 (merge two redundant experts)
        source_a_utilization=layer["utilization"][3].item(),
        source_b_utilization=layer["utilization"][4].item(),
        similarity=layer["similarity"],
        predicted_psi_delta=-0.02,
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

    print(f"\nAttempting merge...")
    try:
        proposal = executor.propose_merge(
            layer_id=0,
            step=150,
            source_a_id=3,
            source_b_id=4,
            evidence=evidence,
            gates=gates,
            similarity=layer["similarity"],
        )
        print(f"  → Merge ALLOWED (unexpected)")
    except GateViolation as e:
        print(f"  → Merge BLOCKED ✓")
        print(f"  GateViolation: {str(e).split(chr(10))[0]}")

    # ========================================================================
    # Phase 4: Return to calm
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 4: Return to Calm (600 steps)")
    print("─" * 70)

    print(f"\nRecovering...")
    for step in range(150, 750):
        # Experts still redundant but system stable
        layer = simulate_layer_state(phi_mean=0.85, num_experts=5, redundant_pair=(3, 4))
        signals = collapse_signals_from_free_energy(
            layer["psi"], layer["utilization"], layer["router_probs"], None
        )
        survived, _ = check_survival(signals, collapse_thresholds)
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

        if (step - 150) % 100 == 0:
            print(
                f"  Step {step}: time_in_comfort={stress_state.time_in_comfort}, edit_allowed={gates.check_edit_allowed()}"
            )

    print(f"\nAfter calm recovery:")
    print(f"  F_l: {result.f_raw:.3f}, Band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort} ✓")
    print(f"  Experts 3 & 4 still redundant: similarity={layer['similarity']:.3f}")

    # ========================================================================
    # Phase 5: Merge now ALLOWED
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 5: Merge Now Allowed (Evidence + Calmness)")
    print("─" * 70)

    # Refresh layer state (experts still redundant)
    layer = simulate_layer_state(phi_mean=0.85, num_experts=5, redundant_pair=(3, 4))

    # Create evidence
    evidence = create_merge_evidence(
        f_l_before=layer["f_l"],
        components_before=layer["components"],
        psi_before=layer["psi"],
        neff_before=None,
        num_experts_after=4,
        source_a_utilization=layer["utilization"][3].item(),
        source_b_utilization=layer["utilization"][4].item(),
        similarity=layer["similarity"],
        predicted_psi_delta=-0.02,
    )

    print(f"\nEvidence:")
    print(f"  ΔF_l: {evidence.delta_f_l:.3f} ✓")
    print(
        f"  Source experts: similarity={layer['similarity']:.3f}, util_a={layer['utilization'][3]:.1f}, util_b={layer['utilization'][4]:.1f}"
    )

    print(f"\nGate check:")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Edit allowed: {gates.check_edit_allowed()} ✓")

    print(f"\nAttempting merge...")
    print(f"  {'!' * 68}")
    print("  WARNING: Destructive operation - NOT reversible")
    print(f"  {'!' * 68}")

    try:
        # Full pipeline: propose → evaluate → approve → execute
        source_a_params = {"weight": layer["role_vectors"][3]}  # Simulate parameters
        source_b_params = {"weight": layer["role_vectors"][4]}

        result = executor.merge_experts_full_pipeline(
            layer_id=0,
            step=750,
            source_a_id=3,
            source_b_id=4,
            source_a_params=source_a_params,
            source_b_params=source_b_params,
            merged_expert_id=3,  # Reuse expert 3 slot
            evidence=evidence,
            gates=gates,
            similarity=layer["similarity"],
        )

        if result and result.success:
            print(f"  → Merge EXECUTED ✓")
            print(f"\n  Result:")
            print(f"    Source expert A: {result.source_a_id}")
            print(f"    Source expert B: {result.source_b_id}")
            print(f"    Merged expert: {result.merged_expert_id}")
            print(f"    Similarity at merge: {result.similarity:.3f}")
            print(f"    Merged params: {list(result.merged_params.keys())}")
            print(f"    Reason: {result.reason}")
        else:
            print(f"  → Merge failed")

    except GateViolation as e:
        print(f"  → Merge BLOCKED (unexpected)")
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
        print(
            f"    Context: F_l={entry.f_l:.3f}, band={entry.stress_band}, time_in_comfort={entry.time_in_comfort}"
        )

    print(f"\n✓ Full audit trail saved to: {audit_log_path}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY: Production-Shaped MERGE (MOST DANGEROUS)")
    print("=" * 70)

    print(f"\nWhat happened:")
    print(f"  1. Operated normally (100 steps, 5 experts)")
    print(f"  2. Experts 3 & 4 became redundant (50 steps)")
    print(f"  3. Merge blocked due to insufficient calm")
    print(f"  4. Returned to calm (600 steps)")
    print(f"  5. Merge executed successfully (DESTRUCTIVE)")

    print(f"\nPipeline verified:")
    print(f"  ✓ Propose (with evidence)")
    print(f"  ✓ Gate check (non-bypassable)")
    print(f"  ✓ Execute (two experts merged, NOT reversible)")
    print(f"  ✓ Audit log (full trail)")

    print(f"\nWhy MERGE is MOST DANGEROUS:")
    print(f"  ⚠  Destroys information (NOT reversible)")
    print(f"  ⚠  Cannot undo if wrong experts merged")
    print(f"  ⚠  Evidence: experts highly redundant (high similarity)")
    print(f"  ⚠  Requires EXTRA sustained calm (destructive identity change)")
    print(f"  ⚠  This is where you accidentally delete a personality")

    print(f"\n{'─' * 70}")
    print(f"Block-then-allow proven with destructive merge execution.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("✓ MERGE Demo Complete (Order: spawn → prune → split → merge)")
    print("=" * 70)
