"""
Phase 4 Demo: Free Energy Objective

Demonstrates how F_l unifies all four pathology detectors:
    F_l = (1 - Psi_l) + lambda * N_l + rho * R_l + kappa * I_l

Shows:
- Healthy layer (low F_l)
- High misfit (low coherence)
- High complexity (too many experts)
- High redundancy (duplicate experts)
- High instability (bimodal experts)
- Component tradeoffs (spawn reduces misfit but increases complexity)

Key insight: One scalar replaces the entire rule bag.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.free_energy import (
    compute_free_energy,
    create_free_energy_state,
)


def demo_healthy_layer():
    """Healthy layer: high coherence, few experts, no redundancy/instability."""
    print("=" * 70)
    print("SCENARIO 1: Healthy Layer")
    print("=" * 70)

    # 4 experts, all coherent, all different
    phi_slow = torch.ones(4) * 0.9  # High coherence
    utilization = torch.ones(4) * 200  # Well-used

    # Orthogonal role vectors (no redundancy)
    role_vectors = torch.zeros(4, 64)
    role_vectors[0, 0:16] = torch.randn(16)
    role_vectors[1, 16:32] = torch.randn(16)
    role_vectors[2, 32:48] = torch.randn(16)
    role_vectors[3, 48:64] = torch.randn(16)
    role_vectors = F.normalize(role_vectors, dim=-1)

    bimodality_scores = torch.zeros(4)  # No instability

    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nLayer properties:")
    print(f"  Experts: 4 (all active)")
    print(f"  Coherence: {phi_slow.mean().item():.3f} (healthy)")
    print(f"  Redundancy: None (all orthogonal)")
    print(f"  Instability: None (all unimodal)")

    print(f"\nFree energy components:")
    print(f"  Misfit:      {components.misfit:.4f}  (1 - Psi_l)")
    print(f"  Complexity:  {components.complexity:.4f}  (lambda * N_l)")
    print(f"  Redundancy:  {components.redundancy:.4f}  (rho * R_l)")
    print(f"  Instability: {components.instability:.4f}  (kappa * I_l)")
    print(f"  ─────────────────────────")
    print(f"  Total F_l:   {f_l:.4f}  ✓ LOW (healthy)")


def demo_high_misfit():
    """High misfit: experts decoherent with mixture."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: High Misfit (Low Coherence)")
    print("=" * 70)

    # 4 experts, low coherence
    phi_slow = torch.ones(4) * 0.2  # Decoherent!
    utilization = torch.ones(4) * 200

    role_vectors = torch.randn(4, 64)
    role_vectors = F.normalize(role_vectors, dim=-1)

    bimodality_scores = torch.zeros(4)

    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nLayer properties:")
    print(f"  Experts: 4")
    print(f"  Coherence: {phi_slow.mean().item():.3f} ⚠ DECOHERENT")

    print(f"\nFree energy components:")
    print(f"  Misfit:      {components.misfit:.4f}  ⚠ DOMINATES (1 - {1-components.misfit:.1f})")
    print(f"  Complexity:  {components.complexity:.4f}")
    print(f"  Redundancy:  {components.redundancy:.4f}")
    print(f"  Instability: {components.instability:.4f}")
    print(f"  ─────────────────────────")
    print(f"  Total F_l:   {f_l:.4f}  ⚠ HIGH")

    print(f"\n→ Diagnosis: Layer starving or experts degraded")
    print(f"   Lifecycle action: SPAWN (add capacity) or PRUNE (remove decoherent)")


def demo_high_complexity():
    """High complexity: too many experts."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: High Complexity (Too Many Experts)")
    print("=" * 70)

    # 16 experts (twice as many)
    phi_slow = torch.ones(16) * 0.9
    utilization = torch.ones(16) * 200

    role_vectors = torch.randn(16, 64)
    role_vectors = F.normalize(role_vectors, dim=-1)

    bimodality_scores = torch.zeros(16)

    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nLayer properties:")
    print(f"  Experts: 16 ⚠ TOO MANY")
    print(f"  Coherence: {phi_slow.mean().item():.3f} (healthy)")

    print(f"\nFree energy components:")
    print(f"  Misfit:      {components.misfit:.4f}")
    print(f"  Complexity:  {components.complexity:.4f}  ⚠ HIGH (0.01 * 16)")
    print(f"  Redundancy:  {components.redundancy:.4f}")
    print(f"  Instability: {components.instability:.4f}")
    print(f"  ─────────────────────────")
    print(f"  Total F_l:   {f_l:.4f}  ⚠ ELEVATED")

    print(f"\n→ Diagnosis: Over-parameterized layer")
    print(f"   Lifecycle action: MERGE or PRUNE redundant experts")


def demo_high_redundancy():
    """High redundancy: duplicate experts."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: High Redundancy (Duplicate Experts)")
    print("=" * 70)

    # 6 experts, but 3 pairs of duplicates
    phi_slow = torch.ones(6) * 0.85
    utilization = torch.ones(6) * 200

    # Create 3 pairs of near-identical experts
    role_vectors = torch.zeros(6, 64)

    torch.manual_seed(42)
    direction_1 = torch.randn(64)
    direction_2 = torch.randn(64)
    direction_3 = torch.randn(64)

    # Pair 1: experts 0, 1
    role_vectors[0] = direction_1
    role_vectors[1] = direction_1 + torch.randn(64) * 0.01  # Tiny noise

    # Pair 2: experts 2, 3
    role_vectors[2] = direction_2
    role_vectors[3] = direction_2 + torch.randn(64) * 0.01

    # Pair 3: experts 4, 5
    role_vectors[4] = direction_3
    role_vectors[5] = direction_3 + torch.randn(64) * 0.01

    role_vectors = F.normalize(role_vectors, dim=-1)

    bimodality_scores = torch.zeros(6)

    components, similarity, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nLayer properties:")
    print(f"  Experts: 6")
    print(f"  Coherence: {phi_slow.mean().item():.3f} (healthy)")
    print(f"  Similarity matrix (first 3x3):")
    for i in range(3):
        print(f"    {similarity[i, :3].tolist()}")

    print(f"\nFree energy components:")
    print(f"  Misfit:      {components.misfit:.4f}")
    print(f"  Complexity:  {components.complexity:.4f}")
    print(f"  Redundancy:  {components.redundancy:.4f}  ⚠ HIGH (3 duplicate pairs)")
    print(f"  Instability: {components.instability:.4f}")
    print(f"  ─────────────────────────")
    print(f"  Total F_l:   {f_l:.4f}  ⚠ ELEVATED")

    print(f"\n→ Diagnosis: Duplicate experts wasting capacity")
    print(f"   Lifecycle action: MERGE redundant pairs")


def demo_high_instability():
    """High instability: bimodal experts."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: High Instability (Bimodal Experts)")
    print("=" * 70)

    # 4 experts, 2 are bimodal
    phi_slow = torch.tensor([0.8, 0.7, 0.85, 0.75])  # Decent coherence
    utilization = torch.ones(4) * 200

    role_vectors = torch.randn(4, 64)
    role_vectors = F.normalize(role_vectors, dim=-1)

    # Experts 1 and 3 are bimodal (serving two incompatible basins)
    bimodality_scores = torch.tensor([0.1, 0.9, 0.1, 0.85])

    components, _, f_l = compute_free_energy(
        phi_slow=phi_slow,
        utilization=utilization,
        role_vectors=role_vectors,
        bimodality_scores=bimodality_scores,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nLayer properties:")
    print(f"  Experts: 4")
    print(f"  Coherence: {phi_slow.mean().item():.3f} (looks healthy)")
    print(f"  Bimodality: {bimodality_scores.tolist()}")
    print(f"    → Experts 1 and 3 serving two incompatible basins!")

    print(f"\nFree energy components:")
    print(f"  Misfit:      {components.misfit:.4f}")
    print(f"  Complexity:  {components.complexity:.4f}")
    print(f"  Redundancy:  {components.redundancy:.4f}")
    print(f"  Instability: {components.instability:.4f}  ⚠ HIGH (bimodal experts)")
    print(f"  ─────────────────────────")
    print(f"  Total F_l:   {f_l:.4f}  ⚠ ELEVATED")

    print(f"\n→ Diagnosis: Experts serving phase-incompatible modes")
    print(f"   Lifecycle action: SPLIT bimodal experts")


def demo_tradeoff_spawn():
    """Show tradeoff: spawn reduces misfit but increases complexity."""
    print("\n" + "=" * 70)
    print("SCENARIO 6: Lifecycle Tradeoff (Spawn Decision)")
    print("=" * 70)

    # Before spawn: 3 experts, layer starving (high misfit)
    phi_slow_before = torch.ones(3) * 0.5  # Low coherence
    utilization_before = torch.ones(3) * 200

    role_vectors_before = torch.randn(3, 64)
    role_vectors_before = F.normalize(role_vectors_before, dim=-1)

    bimodality_scores_before = torch.zeros(3)

    components_before, _, f_l_before = compute_free_energy(
        phi_slow=phi_slow_before,
        utilization=utilization_before,
        role_vectors=role_vectors_before,
        bimodality_scores=bimodality_scores_before,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    # After spawn: 4 experts, misfit reduced
    phi_slow_after = torch.ones(4) * 0.7  # Improved coherence
    utilization_after = torch.ones(4) * 150  # Load redistributed

    role_vectors_after = torch.randn(4, 64)
    role_vectors_after = F.normalize(role_vectors_after, dim=-1)

    bimodality_scores_after = torch.zeros(4)

    components_after, _, f_l_after = compute_free_energy(
        phi_slow=phi_slow_after,
        utilization=utilization_after,
        role_vectors=role_vectors_after,
        bimodality_scores=bimodality_scores_after,
        lambda_complexity=0.01,
        rho_redundancy=0.1,
        kappa_instability=0.1,
    )

    print(f"\nBEFORE SPAWN:")
    print(f"  Experts: 3")
    print(f"  Misfit: {components_before.misfit:.4f} ⚠ HIGH (starving)")
    print(f"  Complexity: {components_before.complexity:.4f}")
    print(f"  F_l: {f_l_before:.4f}")

    print(f"\nAFTER SPAWN:")
    print(f"  Experts: 4 (+1)")
    print(f"  Misfit: {components_after.misfit:.4f} ✓ REDUCED")
    print(f"  Complexity: {components_after.complexity:.4f} ⚠ INCREASED")
    print(f"  F_l: {f_l_after:.4f}")

    delta_f = f_l_after - f_l_before
    print(f"\nΔF_l: {delta_f:.4f}")

    if delta_f < 0:
        print(f"  ✓ Free energy DECREASED → Spawn justified")
    else:
        print(f"  ✗ Free energy INCREASED → Spawn not justified")

    print(f"\n→ Key insight: Spawn trades complexity for reduced misfit")
    print(f"   Only worth it if misfit reduction outweighs complexity cost")


def demo_component_summary():
    """Show how each component targets a specific pathology."""
    print("\n" + "=" * 70)
    print("FREE ENERGY COMPONENTS: What Each Term Detects")
    print("=" * 70)

    print(f"\nMisfit = (1 - Psi_l)")
    print(f"  Detects: Layer incoherence")
    print(f"  → High when experts not aligned with mixture")
    print(f"  → Signals: SPAWN (add capacity) or PRUNE (remove decoherent)")

    print(f"\nComplexity = lambda * N_active")
    print(f"  Detects: Over-parameterization")
    print(f"  → High when too many experts")
    print(f"  → Signals: MERGE or PRUNE redundant experts")

    print(f"\nRedundancy = rho * R_l")
    print(f"  Detects: Duplicate experts")
    print(f"  → High when experts have similar outputs")
    print(f"  → Signals: MERGE near-duplicates")

    print(f"\nInstability = kappa * I_l")
    print(f"  Detects: Bimodal experts")
    print(f"  → High when experts serve incompatible basins")
    print(f"  → Signals: SPLIT bimodal experts")

    print(f"\n{'─' * 70}")
    print(f"Total F_l = Misfit + Complexity + Redundancy + Instability")
    print(f"{'─' * 70}")

    print(f"\n✓ One scalar objective replaces the entire rule bag")
    print(f"✓ Slow clock acts only when ΔF_l > threshold")
    print(f"✓ System stops being eager to fiddle")


if __name__ == "__main__":
    demo_healthy_layer()
    demo_high_misfit()
    demo_high_complexity()
    demo_high_redundancy()
    demo_high_instability()
    demo_tradeoff_spawn()
    demo_component_summary()

    print("\n" + "=" * 70)
    print("✓ Phase 4 Demo Complete")
    print("=" * 70)
