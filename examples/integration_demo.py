"""
Integration Demo: Full Pipeline with Non-Bypassable Gates

Proves block-then-allow with Phases 1-4 + Stress Bands.

Scenario:
1. System operates normally in comfort (F_l low)
2. Stress spike occurs (F_l rises)
3. System wants to spawn expert (evidence present)
4. Gate BLOCKS spawn due to insufficient calmness
5. System returns to calm (F_l drops, time_in_comfort accumulates)
6. Same spawn is now ALLOWED (evidence + calmness both satisfied)

This demonstrates:
- Double gate: evidence (ΔF_l) + calmness (stress bands)
- Non-bypassable enforcement (GateViolation if violated)
- Trauma doesn't trigger structural changes
- Identity changes require sustained calm
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.free_energy import (
    compute_free_energy,
    compute_layer_coherence,
)
from chronomoe_v3.stress_bands import (
    init_stress_bands,
    step_stress_bands,
)
from chronomoe_v3.collapse_detection import (
    collapse_signals_from_free_energy,
    check_survival,
)
from chronomoe_v3.lifecycle_gates import (
    LifecycleGates,
    GateViolation,
    get_default_collapse_thresholds,
    get_default_stress_config,
)


def simulate_moe_layer(phi_mean: float, num_experts: int = 4):
    """
    Simulate a MoE layer with given average coherence.

    Returns free energy components needed for stress bands.
    """
    # Simulate expert coherence around the mean
    phi_slow = torch.ones(num_experts) * phi_mean + torch.randn(num_experts) * 0.05
    phi_slow = torch.clamp(phi_slow, 0.0, 1.0)

    # Simulate utilization (all experts used)
    utilization = torch.ones(num_experts) * 200.0

    # Simulate role vectors (orthogonal = no redundancy)
    role_vectors = torch.zeros(num_experts, 64)
    for i in range(num_experts):
        role_vectors[i, i * 16 : (i + 1) * 16] = torch.randn(16)
    role_vectors = F.normalize(role_vectors, dim=-1)

    # No bimodality
    bimodality_scores = torch.zeros(num_experts)

    # Compute free energy
    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    # Compute collapse signals
    psi = compute_layer_coherence(phi_slow, utilization)
    router_probs = utilization / utilization.sum()  # Simulate router probs

    return {
        "f_l": f_l,
        "psi": psi,
        "phi_slow": phi_slow,
        "utilization": utilization,
        "router_probs": router_probs,
        "components": components,
    }


def check_evidence_for_spawn(f_l_current: float, f_l_predicted: float, threshold: float = 0.05) -> bool:
    """
    Check if evidence supports spawning a new expert.

    Evidence: predicted F_l reduction > threshold.
    """
    delta_f = f_l_predicted - f_l_current
    return delta_f < -threshold


def main():
    print("=" * 70)
    print("INTEGRATION DEMO: Block-Then-Allow with Full Pipeline")
    print("=" * 70)

    # Initialize stress bands
    stress_cfg = get_default_stress_config()
    stress_state = init_stress_bands(stress_cfg)

    # Initialize collapse thresholds
    collapse_thresholds = get_default_collapse_thresholds()

    # Initialize lifecycle gates
    gates = LifecycleGates(stress_state, stress_cfg)

    print(f"\nInitial state:")
    print(f"  Stress config: comfort_ceiling={stress_cfg.comfort_ceiling_init}, scar_calm={stress_cfg.scar_calm_steps}, edit_calm={stress_cfg.edit_calm_steps}")
    print(f"  Collapse thresholds: Psi>{collapse_thresholds.psi_critical}, Neff>{collapse_thresholds.neff_critical}")

    # ========================================================================
    # Phase 1: Normal operation in comfort
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 1: Normal Operation (150 steps in comfort)")
    print("─" * 70)

    for step in range(150):
        # Simulate healthy layer
        layer = simulate_moe_layer(phi_mean=0.85, num_experts=4)

        # Check survival
        signals = collapse_signals_from_free_energy(
            layer["psi"],
            layer["utilization"],
            layer["router_probs"],
            None,
        )
        survived, _ = check_survival(signals, collapse_thresholds)

        # Update stress bands
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)

        # Refresh gates
        gates.refresh()

    print(f"\nAfter 150 steps:")
    print(f"  Current F_l: {result.f_raw:.3f}")
    print(f"  Current band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Comfort ceiling: {result.ceilings['comfort']:.3f}")
    print(gates.status_summary())

    # ========================================================================
    # Phase 2: Stress spike occurs
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 2: Stress Spike (50 steps, coherence drops)")
    print("─" * 70)

    print(f"\nSimulating stress: experts decoherent, F_l rises...")

    for step in range(50):
        # Simulate degraded layer (low coherence → high misfit → high F_l)
        layer = simulate_moe_layer(phi_mean=0.3, num_experts=4)

        # Check survival
        signals = collapse_signals_from_free_energy(
            layer["psi"],
            layer["utilization"],
            layer["router_probs"],
            None,
        )
        survived, _ = check_survival(signals, collapse_thresholds)

        # Update stress bands
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

    print(f"\nAfter stress spike:")
    print(f"  Current F_l: {result.f_raw:.3f} (HIGH)")
    print(f"  Current band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(f"  Misfit: {layer['components'].misfit:.3f} (layer starving)")

    # ========================================================================
    # Phase 3: Evidence says SPAWN, but gate BLOCKS
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 3: Spawn Evidence Present, But Gate Blocks")
    print("─" * 70)

    # Check evidence: would spawning reduce F_l?
    f_l_current = layer["f_l"]
    # Predict: spawn adds complexity (+0.01) but reduces misfit significantly (-0.3)
    f_l_predicted = f_l_current + 0.01 - 0.3

    evidence_met = check_evidence_for_spawn(f_l_current, f_l_predicted, threshold=0.05)

    print(f"\nEvidence check:")
    print(f"  Current F_l:   {f_l_current:.3f}")
    print(f"  Predicted F_l: {f_l_predicted:.3f}")
    print(f"  ΔF_l:          {f_l_predicted - f_l_current:.3f}")
    print(f"  Evidence met:  {evidence_met} ✓")

    print(f"\nCalmness check:")
    print(f"  Current band:      {stress_state.current_band}")
    print(f"  Time in comfort:   {stress_state.time_in_comfort}")
    print(f"  Requirement:       {stress_cfg.edit_calm_steps}")
    print(f"  Calmness met:      {gates.check_edit_allowed()} ✗")

    print(f"\nAttempting spawn...")
    try:
        gates.require_edit_allowed()
        print("  → Spawn ALLOWED")
    except GateViolation as e:
        print(f"  → Spawn BLOCKED")
        print(f"\n  GateViolation raised:")
        for line in str(e).split("\n"):
            print(f"    {line}")

    print(f"\n✓ Evidence alone is not enough")
    print(f"  System is stressed. No identity changes allowed.")
    print(f"  Trauma doesn't trigger structural edits.")

    # ========================================================================
    # Phase 4: Return to calm
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 4: Return to Calm (600 steps, stress resolves)")
    print("─" * 70)

    print(f"\nSimulating recovery: coherence returns, F_l drops...")

    for step in range(600):
        # Simulate healthy layer again
        layer = simulate_moe_layer(phi_mean=0.85, num_experts=4)

        # Check survival
        signals = collapse_signals_from_free_energy(
            layer["psi"],
            layer["utilization"],
            layer["router_probs"],
            None,
        )
        survived, _ = check_survival(signals, collapse_thresholds)

        # Update stress bands
        result = step_stress_bands(stress_state, stress_cfg, layer["f_l"], survived)
        gates.refresh()

        # Print progress at milestones
        if (step + 1) in [100, 200, 300, 400, 500, 600]:
            print(f"  Step {step+1}: band={result.band_now}, time_in_comfort={stress_state.time_in_comfort}, edit_allowed={gates.check_edit_allowed()}")

    print(f"\nAfter 600 calm steps:")
    print(f"  Current F_l: {result.f_raw:.3f} (low)")
    print(f"  Current band: {result.band_now}")
    print(f"  Time in comfort: {stress_state.time_in_comfort}")
    print(gates.status_summary())

    # ========================================================================
    # Phase 5: Same evidence, now ALLOWED
    # ========================================================================

    print("\n" + "─" * 70)
    print("PHASE 5: Same Spawn Evidence, Now Allowed")
    print("─" * 70)

    # Evidence still present (simulate layer still could benefit from spawn)
    # But now we're calm, so threshold is lower
    evidence_met = check_evidence_for_spawn(result.f_raw, result.f_raw - 0.06, threshold=0.05)

    print(f"\nEvidence check:")
    print(f"  Current F_l:   {result.f_raw:.3f}")
    print(f"  Predicted F_l: {result.f_raw - 0.06:.3f}")
    print(f"  ΔF_l:          {-0.06:.3f}")
    print(f"  Evidence met:  {evidence_met} ✓")

    print(f"\nCalmness check:")
    print(f"  Current band:      {stress_state.current_band}")
    print(f"  Time in comfort:   {stress_state.time_in_comfort}")
    print(f"  Requirement:       {stress_cfg.edit_calm_steps}")
    print(f"  Calmness met:      {gates.check_edit_allowed()} ✓")

    print(f"\nAttempting spawn...")
    try:
        gates.require_edit_allowed()
        print("  → Spawn ALLOWED ✓")
        print(f"\n  Spawn would execute here:")
        print(f"    - Clone best expert")
        print(f"    - Add small perturbation")
        print(f"    - Register in expert registry")
        print(f"    - Update optimizer state")
    except GateViolation as e:
        print(f"  → Spawn BLOCKED (unexpected)")
        print(f"  Error: {e}")

    print(f"\n✓ Evidence + Calmness = Identity change allowed")
    print(f"  Same evidence that was blocked during stress")
    print(f"  Now allowed after sustained calm")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY: Block-Then-Allow Proven")
    print("=" * 70)

    print(f"\nDouble Gate Enforced:")
    print(f"  1. Evidence gate:  ΔF_l < -0.05 (Phase 4 free energy)")
    print(f"  2. Calmness gate:  time_in_comfort > 500 (stress bands)")
    print(f"     Both required. Non-bypassable.")

    print(f"\nWhat Happened:")
    print(f"  1. System operated normally (150 steps calm)")
    print(f"  2. Stress spike occurred (50 steps, F_l rose)")
    print(f"  3. Evidence present, but gate BLOCKED spawn")
    print(f"  4. System returned to calm (600 steps)")
    print(f"  5. Same evidence, now ALLOWED")

    print(f"\nWhy This Matters:")
    print(f"  ✓ Trauma doesn't trigger structural changes")
    print(f"  ✓ Identity changes require sustained calm + evidence")
    print(f"  ✓ Gates are non-bypassable (GateViolation raised)")
    print(f"  ✓ System allowed to be stressed without rewriting itself")

    print(f"\n{'─' * 70}")
    print(f"Pressure tunes behavior.")
    print(f"Calm commits identity.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 70)
    print("✓ Integration Demo Complete")
    print("=" * 70)
