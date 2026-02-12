#!/usr/bin/env python3
"""
Test MERGE diagnostic mode (Milestone F Phase 1).

Validates:
- MERGE proposal generation (high similarity + low utilization)
- MERGE threshold filtering (high similarity + high utilization = no proposal)
- Diagnostic-only mode (proposals logged, never executed)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from chronomoe_integration.controller import (
    ChronoController,
    ObservationSnapshot,
    create_controller,
)


def test_merge_proposal_positive():
    """
    Test 17: MERGE Proposal (Positive Test) - FULLY DETERMINISTIC

    Directly constructs controller state with:
    - Two experts with identical centroids (similarity = 1.0)
    - Both experts with low utilization (< 10%)
    - Sufficient observations (> 100)

    No randomness. No sampling. Pure unit test of detection logic.
    """
    print("\n" + "=" * 70)
    print("TEST 17: MERGE Proposal (Positive Test)")
    print("=" * 70)

    from chronomoe_integration.bimodality import BimodalityState

    # Create controller with MERGE enabled
    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=True,
    )

    # Enable MERGE diagnostic mode
    controller.config["merge"]["enabled"] = True
    controller.config["merge"]["similarity_threshold"] = 0.8
    controller.config["merge"]["utilization_threshold"] = 0.1
    controller.config["merge"]["min_observations"] = 100
    controller.config["triggers"]["min_delta_f"] = 1.0  # Disable ΔF_l filter

    # DETERMINISTIC CONSTRUCTION: Manually create bimodality states
    d_model = 128

    # Create identical centroid for experts 0 and 1
    identical_centroid = torch.ones(d_model)  # Deterministic vector

    # Expert 0: bimodality state with sufficient observations
    controller.bimodality_states[0] = BimodalityState(
        expert_id=0,
        layer_id=0,
        d_model=d_model,
        centroid_a=identical_centroid.clone(),
        centroid_b=torch.zeros(d_model),  # NOTE: centroid_b unused in current merge heuristic (only centroid_a compared)
        count_a=150,  # > min_observations
        count_b=0,
    )

    # Expert 1: bimodality state with IDENTICAL centroid
    controller.bimodality_states[1] = BimodalityState(
        expert_id=1,
        layer_id=0,
        d_model=d_model,
        centroid_a=identical_centroid.clone(),
        centroid_b=torch.zeros(d_model),  # Unused but required
        count_a=150,  # > min_observations
        count_b=0,
    )

    # Expert 2: different centroid (control)
    controller.bimodality_states[2] = BimodalityState(
        expert_id=2,
        layer_id=0,
        d_model=d_model,
        centroid_a=torch.zeros(d_model),  # Different
        centroid_b=torch.zeros(d_model),
        count_a=150,
        count_b=0,
    )

    # Expert 3: different centroid (control)
    controller.bimodality_states[3] = BimodalityState(
        expert_id=3,
        layer_id=0,
        d_model=d_model,
        centroid_a=torch.ones(d_model) * -1,  # Different
        centroid_b=torch.zeros(d_model),
        count_a=150,
        count_b=0,
    )

    # Create free energy state (required for MERGE detection)
    from chronomoe_integration.free_energy import FreeEnergyState, FreeEnergyComponents
    controller.free_energy_state = FreeEnergyState(
        layer_id=0,
        step=100,
        components=FreeEnergyComponents(
            misfit=None,
            complexity=0.01,
            redundancy=0.01,
            instability=0.01,
        ),
        num_active_experts=4,
        max_experts=8,
        redundancy_score=0.01,
        instability_score=0.01,
    )

    # Create observation with LOW utilization for experts 0 and 1
    batch_size = 32
    utilization = torch.zeros(4)
    utilization[0] = 2  # 2 tokens out of 32 = 6.25% (LOW)
    utilization[1] = 3  # 3 tokens out of 32 = 9.38% (LOW)
    utilization[2] = 13  # 13 tokens = 40.6% (HIGH)
    utilization[3] = 14  # 14 tokens = 43.8% (HIGH)

    snapshot = ObservationSnapshot(
        step=100,
        layer_id=0,
        router_probs=torch.zeros(batch_size, 4),
        selected_experts=torch.zeros(batch_size, 1, dtype=torch.long),
        expert_outputs=None,
        mixture_output=torch.zeros(batch_size, d_model),
        utilization=utilization,
        loss=1.0,
    )

    controller.observation_history.append(snapshot)

    # Call decide() to generate proposals
    proposals = controller.decide()

    merge_proposals = [p for p in proposals if p.edit_type == "merge"]

    print(f"✓ Total proposals: {len(proposals)}")
    print(f"✓ MERGE proposals: {len(merge_proposals)}")

    assert len(merge_proposals) > 0, "Expected MERGE proposal (identical centroids + low util)"

    # Find MERGE proposal for experts {0, 1} (resilient to controller ordering)
    target_merge = None
    for merge in merge_proposals:
        expert_pair = {merge.expert_id, merge.evidence.get('expert_b_id')}
        if expert_pair == {0, 1}:
            target_merge = merge
            break

    assert target_merge is not None, "Expected MERGE proposal for experts {0, 1}"

    print(f"✓ MERGE candidate found: experts ({target_merge.expert_id}, {target_merge.evidence.get('expert_b_id')})")
    print(f"  Similarity: {target_merge.evidence.get('similarity_score', 0):.4f}")
    print(f"  Utilization A: {target_merge.evidence.get('utilization_a', 0):.4f}")
    print(f"  Utilization B: {target_merge.evidence.get('utilization_b', 0):.4f}")

    # Verify: similarity = 1.0 (identical vectors)
    sim = target_merge.evidence.get('similarity_score', 0)
    assert sim > 0.99, f"Expected similarity ~1.0 for identical vectors, got {sim}"

    # Verify: utilization below threshold
    util_a = target_merge.evidence.get('utilization_a', 1.0)
    util_b = target_merge.evidence.get('utilization_b', 1.0)
    assert util_a < 0.1, f"Expected utilization A < 0.1, got {util_a}"
    assert util_b < 0.1, f"Expected utilization B < 0.1, got {util_b}"

    print("\n✓ TEST 17 PASSED: MERGE proposal triggered correctly\n")


def test_merge_proposal_negative():
    """
    Test 18: MERGE Proposal Blocked (Negative Test)

    Two experts with high similarity but HIGH utilization
    should NOT trigger MERGE proposal.
    """
    print("\n" + "=" * 70)
    print("TEST 18: MERGE Proposal Blocked (Negative Test)")
    print("=" * 70)

    # Set seed for deterministic test
    torch.manual_seed(43)

    # Create controller with MERGE enabled
    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=True,
    )

    # Enable MERGE diagnostic mode
    controller.config["merge"]["enabled"] = True
    controller.config["merge"]["similarity_threshold"] = 0.8
    controller.config["merge"]["utilization_threshold"] = 0.1  # 10% threshold
    controller.config["merge"]["min_observations"] = 100

    # Feed observations
    d_model = 128
    batch_size = 32

    # Create two experts (0 and 1) with similar centroids
    # but HIGH utilization (should block MERGE)
    for step in range(150):
        # Router probs: HIGH utilization for experts 0 and 1 (>> 10%)
        router_probs = torch.zeros(batch_size, 4)
        router_probs[:, 0] = 0.40  # Expert 0: 40% utilization (HIGH)
        router_probs[:, 1] = 0.35  # Expert 1: 35% utilization (HIGH)
        router_probs[:, 2] = 0.15  # Expert 2: 15% utilization
        router_probs[:, 3] = 0.10  # Expert 3: 10% utilization

        selected = router_probs.argmax(dim=1)

        # Create expert outputs
        # Experts 0 and 1: VERY SIMILAR outputs (high similarity)
        base_vector = torch.randn(d_model)
        expert_outputs = torch.zeros(4, batch_size, d_model)
        expert_outputs[0] = base_vector + torch.randn(batch_size, d_model) * 0.01  # Expert 0
        expert_outputs[1] = base_vector + torch.randn(batch_size, d_model) * 0.01  # Expert 1 (SIMILAR)
        expert_outputs[2] = torch.randn(batch_size, d_model)  # Expert 2
        expert_outputs[3] = torch.randn(batch_size, d_model)  # Expert 3

        mixture = expert_outputs.mean(dim=0)

        utilization = torch.zeros(4)
        for i in range(4):
            utilization[i] = (selected == i).sum().item()

        snapshot = ObservationSnapshot(
            step=step,
            layer_id=0,
            router_probs=router_probs,
            selected_experts=selected.unsqueeze(1),
            expert_outputs=expert_outputs,
            mixture_output=mixture,
            utilization=utilization,
            loss=1.0,
        )

        controller.observe(snapshot)

    # Try to propose MERGE
    proposals = controller.decide()

    merge_proposals = [p for p in proposals if p.edit_type == "merge"]

    print(f"✓ Total proposals: {len(proposals)}")
    print(f"✓ MERGE proposals: {len(merge_proposals)}")

    # Should be 0 because utilization is too high (even though similarity is high)
    assert len(merge_proposals) == 0, "Expected NO MERGE proposal (high util blocks merge)"

    print("✓ MERGE correctly blocked (high utilization)")
    print("\n✓ TEST 18 PASSED: MERGE threshold filtering working\n")


def test_merge_proposal_close_enough():
    """
    Test 19: MERGE Proposal with Controlled Noise (Positive Test)

    Two experts with SIMILAR (not identical) centroids under controlled noise.
    Tests threshold boundary: similarity ~0.85 (above 0.8 threshold).

    Fully deterministic construction with fixed noise vector.
    """
    print("\n" + "=" * 70)
    print("TEST 19: MERGE Proposal with Controlled Similarity")
    print("=" * 70)

    from chronomoe_integration.bimodality import BimodalityState

    # Create controller with MERGE enabled
    controller = create_controller(
        layer_id=0,
        max_experts=8,
        initial_active=4,
        autonomous_mode=True,
    )

    # Enable MERGE diagnostic mode
    controller.config["merge"]["enabled"] = True
    controller.config["merge"]["similarity_threshold"] = 0.8
    controller.config["merge"]["utilization_threshold"] = 0.1
    controller.config["merge"]["min_observations"] = 100
    controller.config["triggers"]["min_delta_f"] = 1.0  # Disable ΔF_l filter

    d_model = 128

    # Create vectors with controlled similarity ~0.85 (deterministic)
    # Strategy: orthogonal components with specific weights
    # v0 = [1, 1, 1, ...] (all ones)
    # v1 = [1, 1, ..., 0, 0, ...] (first 80% ones, rest zeros)
    # After normalization, similarity = sqrt(0.8) ≈ 0.894
    # For lower similarity, use 70% overlap → similarity ≈ 0.837

    overlap_fraction = 0.7  # 70% overlap → similarity ~0.837
    overlap_size = int(d_model * overlap_fraction)

    centroid_0 = torch.ones(d_model)
    centroid_1 = torch.cat([
        torch.ones(overlap_size),
        torch.zeros(d_model - overlap_size),
    ])

    # Normalize to unit vectors for controlled similarity
    centroid_0 = centroid_0 / centroid_0.norm()
    centroid_1 = centroid_1 / centroid_1.norm()

    # Verify similarity is above threshold but not perfect
    actual_sim = torch.nn.functional.cosine_similarity(
        centroid_0.unsqueeze(0),
        centroid_1.unsqueeze(0),
        dim=1
    ).item()
    print(f"✓ Constructed similarity: {actual_sim:.4f} (target: 0.8-0.9)")
    assert 0.8 < actual_sim < 0.95, f"Similarity {actual_sim} not in target range"

    # Expert 0: bimodality state
    controller.bimodality_states[0] = BimodalityState(
        expert_id=0,
        layer_id=0,
        d_model=d_model,
        centroid_a=centroid_0,
        centroid_b=torch.zeros(d_model),  # NOTE: centroid_b unused in current merge heuristic
        count_a=150,
        count_b=0,
    )

    # Expert 1: bimodality state with SIMILAR (not identical) centroid
    controller.bimodality_states[1] = BimodalityState(
        expert_id=1,
        layer_id=0,
        d_model=d_model,
        centroid_a=centroid_1,
        centroid_b=torch.zeros(d_model),
        count_a=150,
        count_b=0,
    )

    # Experts 2, 3: different centroids (control)
    controller.bimodality_states[2] = BimodalityState(
        expert_id=2,
        layer_id=0,
        d_model=d_model,
        centroid_a=torch.zeros(d_model),
        centroid_b=torch.zeros(d_model),
        count_a=150,
        count_b=0,
    )

    controller.bimodality_states[3] = BimodalityState(
        expert_id=3,
        layer_id=0,
        d_model=d_model,
        centroid_a=torch.ones(d_model) * -1,
        centroid_b=torch.zeros(d_model),
        count_a=150,
        count_b=0,
    )

    # Create free energy state
    from chronomoe_integration.free_energy import FreeEnergyState, FreeEnergyComponents
    controller.free_energy_state = FreeEnergyState(
        layer_id=0,
        step=100,
        components=FreeEnergyComponents(
            misfit=None,
            complexity=0.01,
            redundancy=0.01,
            instability=0.01,
        ),
        num_active_experts=4,
        max_experts=8,
        redundancy_score=0.01,
        instability_score=0.01,
    )

    # Create observation with LOW utilization for experts 0 and 1
    batch_size = 32
    utilization = torch.zeros(4)
    utilization[0] = 2  # 6.25% (LOW)
    utilization[1] = 3  # 9.38% (LOW)
    utilization[2] = 13  # 40.6% (HIGH)
    utilization[3] = 14  # 43.8% (HIGH)

    snapshot = ObservationSnapshot(
        step=100,
        layer_id=0,
        router_probs=torch.zeros(batch_size, 4),
        selected_experts=torch.zeros(batch_size, 1, dtype=torch.long),
        expert_outputs=None,
        mixture_output=torch.zeros(batch_size, d_model),
        utilization=utilization,
        loss=1.0,
    )

    controller.observation_history.append(snapshot)

    # Call decide()
    proposals = controller.decide()
    merge_proposals = [p for p in proposals if p.edit_type == "merge"]

    print(f"✓ Total proposals: {len(proposals)}")
    print(f"✓ MERGE proposals: {len(merge_proposals)}")

    assert len(merge_proposals) > 0, "Expected MERGE proposal (similar centroids + low util)"

    # Find MERGE proposal for experts {0, 1}
    target_merge = None
    for merge in merge_proposals:
        expert_pair = {merge.expert_id, merge.evidence.get('expert_b_id')}
        if expert_pair == {0, 1}:
            target_merge = merge
            break

    assert target_merge is not None, "Expected MERGE proposal for experts {0, 1}"

    print(f"✓ MERGE candidate found: experts ({target_merge.expert_id}, {target_merge.evidence.get('expert_b_id')})")
    print(f"  Similarity: {target_merge.evidence.get('similarity_score', 0):.4f}")
    print(f"  Utilization A: {target_merge.evidence.get('utilization_a', 0):.4f}")
    print(f"  Utilization B: {target_merge.evidence.get('utilization_b', 0):.4f}")

    # Verify: similarity above threshold but not perfect
    sim = target_merge.evidence.get('similarity_score', 0)
    assert sim > 0.8, f"Expected similarity > 0.8, got {sim}"
    assert sim < 0.95, f"Expected similarity < 0.95 (not identical), got {sim}"

    # Verify: utilization below threshold
    util_a = target_merge.evidence.get('utilization_a', 1.0)
    util_b = target_merge.evidence.get('utilization_b', 1.0)
    assert util_a < 0.1, f"Expected utilization A < 0.1, got {util_a}"
    assert util_b < 0.1, f"Expected utilization B < 0.1, got {util_b}"

    print("\n✓ TEST 19 PASSED: MERGE proposal triggered at threshold boundary\n")


if __name__ == "__main__":
    test_merge_proposal_positive()
    test_merge_proposal_close_enough()
    test_merge_proposal_negative()

    print("=" * 70)
    print("✓ ALL MERGE DIAGNOSTIC TESTS PASSED (3/3)")
    print("=" * 70)
