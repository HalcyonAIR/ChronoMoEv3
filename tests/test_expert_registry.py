"""
Tests for ExpertRegistry (Phase 6).

Validates:
- Fixed-width capacity management
- Active mask correctness
- Optimizer state add/remove/merge
- Lifecycle state transitions
- Checkpoint save/load
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.expert_registry import (
    ExpertRegistry,
    ExpertState,
    ExpertInfo,
)


def create_dummy_expert(d_model: int = 64, d_ff: int = 256) -> nn.Module:
    """Create a simple expert module for testing."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )


def test_initialization():
    """Test registry initialization."""
    print("\n" + "=" * 70)
    print("TEST: Initialization")
    print("=" * 70)

    registry = ExpertRegistry(
        layer_id=0,
        max_experts=16,
        initial_active=4,
    )

    print(f"\nInitial state:")
    print(f"  {registry.status_summary()}")
    print(f"  Active mask: {registry.active_mask.nonzero().squeeze().tolist()}")
    print(f"  Capacity remaining: {registry.capacity_remaining}")

    assert registry.num_active == 4
    assert registry.capacity_remaining == 12
    assert len(registry.active_expert_ids) == 4
    assert registry.active_mask[:4].all()
    assert not registry.active_mask[4:].any()

    print("\n✓ Initialization test passed")
    return True


def test_spawn_expert():
    """Test spawning a new expert."""
    print("\n" + "=" * 70)
    print("TEST: Spawn Expert")
    print("=" * 70)

    registry = ExpertRegistry(layer_id=0, max_experts=16, initial_active=4)
    optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=0.001)

    print(f"\nBefore spawn:")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer param groups: {len(optimizer.param_groups[0]['params'])}")

    # Create new expert and spawn
    new_expert = create_dummy_expert()
    new_id = registry.spawn_expert(
        step=100,
        parent_id=0,
        expert_module=new_expert,
        optimizer=optimizer,
    )

    print(f"\nAfter spawn:")
    print(f"  New expert ID: {new_id}")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer param groups: {len(optimizer.param_groups[0]['params'])}")

    # Verify
    assert new_id == 4  # Next ID after initial 4
    assert registry.num_active == 5
    assert registry.active_mask[new_id]

    info = registry.get_expert_info(new_id)
    assert info is not None
    assert info.state == ExpertState.ACTIVE
    assert info.spawn_parent_id == 0
    assert info.created_at_step == 100

    # Check optimizer registered new params
    expert_param_count = sum(1 for _ in new_expert.parameters())
    assert len(optimizer.param_groups[0]['params']) == 1 + expert_param_count

    print("\n✓ Spawn test passed")
    return True


def test_prune_expert():
    """Test pruning an expert."""
    print("\n" + "=" * 70)
    print("TEST: Prune Expert")
    print("=" * 70)

    registry = ExpertRegistry(layer_id=0, max_experts=16, initial_active=4)

    # Create expert module and optimizer
    expert_to_prune = create_dummy_expert()
    optimizer = torch.optim.Adam(expert_to_prune.parameters(), lr=0.001)

    # Take one optimizer step to initialize state
    loss = sum(p.sum() for p in expert_to_prune.parameters())
    loss.backward()
    optimizer.step()

    print(f"\nBefore prune:")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer state keys: {len(optimizer.state)}")

    # Prune expert 2
    success = registry.prune_expert(
        step=200,
        expert_id=2,
        reason="decoherent",
        optimizer=optimizer,
        expert_module=expert_to_prune,
    )

    print(f"\nAfter prune:")
    print(f"  Prune successful: {success}")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer state keys: {len(optimizer.state)}")

    # Verify
    assert success
    assert registry.num_active == 3
    assert registry.num_cooling == 1
    assert not registry.active_mask[2]

    info = registry.get_expert_info(2)
    assert info.state == ExpertState.COOLING
    assert info.cooling_since_step == 200
    assert info.cooling_reason == "decoherent"

    # Check optimizer state cleaned up
    assert len(optimizer.state) == 0  # All params removed

    print("\n✓ Prune test passed")
    return True


def test_merge_experts():
    """Test merging two experts."""
    print("\n" + "=" * 70)
    print("TEST: Merge Experts")
    print("=" * 70)

    registry = ExpertRegistry(layer_id=0, max_experts=16, initial_active=4)

    # Create two expert modules
    expert_a = create_dummy_expert()
    expert_b = create_dummy_expert()
    merged_expert = create_dummy_expert()

    # Create optimizer with both experts
    all_params = list(expert_a.parameters()) + list(expert_b.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.001)

    # Take one step to initialize state
    loss = sum(p.sum() for p in all_params)
    loss.backward()
    optimizer.step()

    print(f"\nBefore merge:")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer state keys: {len(optimizer.state)}")

    # Merge experts 1 and 2 → keep 1
    merged_id = registry.merge_experts(
        step=300,
        source_a_id=1,
        source_b_id=2,
        merged_module=merged_expert,
        optimizer=optimizer,
        source_a_module=expert_a,
        source_b_module=expert_b,
        optimizer_state_strategy="reset",
    )

    print(f"\nAfter merge:")
    print(f"  Merged ID: {merged_id}")
    print(f"  {registry.status_summary()}")
    print(f"  Optimizer state keys: {len(optimizer.state)}")

    # Verify
    assert merged_id == 1
    assert registry.num_active == 3  # 4 - 1 merged
    assert registry.num_archived == 1
    assert registry.active_mask[1]  # Kept expert still active
    assert not registry.active_mask[2]  # Removed expert inactive

    info_kept = registry.get_expert_info(1)
    info_removed = registry.get_expert_info(2)
    assert info_kept.state == ExpertState.ACTIVE
    assert info_removed.state == ExpertState.ARCHIVED

    # Check optimizer state reset (both removed)
    assert len(optimizer.state) == 0

    print("\n✓ Merge test passed")
    return True


def test_cooling_to_archived():
    """Test automatic archiving of cooling experts."""
    print("\n" + "=" * 70)
    print("TEST: Cooling → Archived")
    print("=" * 70)

    registry = ExpertRegistry(
        layer_id=0,
        max_experts=16,
        initial_active=4,
        cooling_steps=100,  # Short cooling period for test
    )

    # Prune expert 2 at step 100
    expert = create_dummy_expert()
    optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)

    registry.prune_expert(
        step=100,
        expert_id=2,
        reason="test",
        optimizer=optimizer,
        expert_module=expert,
    )

    print(f"\nAfter prune (step 100):")
    print(f"  {registry.status_summary()}")

    # Try archiving at step 150 (50 steps later - not enough)
    archived = registry.archive_cooling_experts(step=150)
    print(f"\nAt step 150 (50 steps cooling):")
    print(f"  Archived: {archived}")
    print(f"  {registry.status_summary()}")

    assert len(archived) == 0
    assert registry.num_cooling == 1

    # Try archiving at step 250 (150 steps later - enough)
    archived = registry.archive_cooling_experts(step=250)
    print(f"\nAt step 250 (150 steps cooling):")
    print(f"  Archived: {archived}")
    print(f"  {registry.status_summary()}")

    assert len(archived) == 1
    assert 2 in archived
    assert registry.num_cooling == 0
    assert registry.num_archived == 1

    info = registry.get_expert_info(2)
    assert info.state == ExpertState.ARCHIVED

    print("\n✓ Cooling → Archived test passed")
    return True


def test_utilization_tracking():
    """Test utilization and coherence EMA updates."""
    print("\n" + "=" * 70)
    print("TEST: Utilization Tracking")
    print("=" * 70)

    registry = ExpertRegistry(layer_id=0, max_experts=16, initial_active=4)

    print(f"\nInitial utilization:")
    for expert_id in range(4):
        info = registry.get_expert_info(expert_id)
        print(f"  Expert {expert_id}: util_ema={info.utilization_ema:.3f}, coherence_ema={info.coherence_ema:.3f}")

    # Update expert 0 with high utilization and coherence
    for step in range(10):
        registry.update_utilization(
            step=step,
            expert_id=0,
            utilization=500.0,
            coherence=0.9,
            alpha=0.95,
        )

    # Update expert 1 with low utilization and coherence
    for step in range(10):
        registry.update_utilization(
            step=step,
            expert_id=1,
            utilization=50.0,
            coherence=0.3,
            alpha=0.95,
        )

    print(f"\nAfter 10 updates:")
    info_0 = registry.get_expert_info(0)
    info_1 = registry.get_expert_info(1)
    print(f"  Expert 0: util_ema={info_0.utilization_ema:.3f}, coherence_ema={info_0.coherence_ema:.3f}")
    print(f"  Expert 1: util_ema={info_1.utilization_ema:.3f}, coherence_ema={info_1.coherence_ema:.3f}")

    # Verify EMAs updated (converge slowly with alpha=0.95)
    assert info_0.utilization_ema > 100  # High utilization
    assert info_0.coherence_ema > 0.3    # Converging toward 0.9
    assert info_1.utilization_ema > 10   # Low utilization
    assert info_1.coherence_ema > 0.05   # Converging toward 0.3

    print("\n✓ Utilization tracking test passed")
    return True


def test_checkpoint_save_load():
    """Test registry checkpoint save/load."""
    print("\n" + "=" * 70)
    print("TEST: Checkpoint Save/Load")
    print("=" * 70)

    # Create registry with some lifecycle events
    registry = ExpertRegistry(layer_id=0, max_experts=16, initial_active=4)

    # Spawn a new expert
    new_expert = create_dummy_expert()
    optimizer = torch.optim.Adam(new_expert.parameters(), lr=0.001)
    new_id = registry.spawn_expert(
        step=100, parent_id=0, expert_module=new_expert, optimizer=optimizer
    )

    # Prune an expert
    expert = create_dummy_expert()
    optimizer2 = torch.optim.Adam(expert.parameters(), lr=0.001)
    registry.prune_expert(
        step=200, expert_id=2, reason="test", optimizer=optimizer2, expert_module=expert
    )

    print(f"\nOriginal registry:")
    print(f"  {registry.status_summary()}")
    print(f"  Events: {len(registry.events)}")

    # Save to dict
    state = registry.to_dict()

    print(f"\nCheckpoint state keys: {list(state.keys())}")
    print(f"  Experts: {len(state['experts'])}")
    print(f"  Events: {len(state['events'])}")

    # Restore from dict
    restored = ExpertRegistry.from_dict(state)

    print(f"\nRestored registry:")
    print(f"  {restored.status_summary()}")
    print(f"  Events: {len(restored.events)}")

    # Verify
    assert restored.layer_id == registry.layer_id
    assert restored.max_experts == registry.max_experts
    assert restored.num_active == registry.num_active
    assert restored.num_cooling == registry.num_cooling
    assert torch.equal(restored.active_mask, registry.active_mask)
    assert len(restored.experts) == len(registry.experts)
    assert len(restored.events) == len(registry.events)

    # Check specific expert info
    original_info = registry.get_expert_info(new_id)
    restored_info = restored.get_expert_info(new_id)
    assert restored_info.state == original_info.state
    assert restored_info.spawn_parent_id == original_info.spawn_parent_id

    print("\n✓ Checkpoint save/load test passed")
    return True


def test_capacity_limits():
    """Test max capacity enforcement."""
    print("\n" + "=" * 70)
    print("TEST: Capacity Limits")
    print("=" * 70)

    registry = ExpertRegistry(layer_id=0, max_experts=8, initial_active=6)

    print(f"\nInitial state:")
    print(f"  {registry.status_summary()}")
    print(f"  Capacity remaining: {registry.capacity_remaining}")

    # Spawn one expert (within capacity)
    expert1 = create_dummy_expert()
    optimizer = torch.optim.Adam(expert1.parameters(), lr=0.001)
    id1 = registry.spawn_expert(
        step=100, parent_id=0, expert_module=expert1, optimizer=optimizer
    )

    print(f"\nAfter spawn 1:")
    print(f"  {registry.status_summary()}")
    print(f"  Capacity remaining: {registry.capacity_remaining}")

    assert id1 == 6
    assert registry.capacity_remaining == 1

    # Spawn another expert (at capacity)
    expert2 = create_dummy_expert()
    id2 = registry.spawn_expert(
        step=101, parent_id=0, expert_module=expert2, optimizer=optimizer
    )

    print(f"\nAfter spawn 2:")
    print(f"  {registry.status_summary()}")
    print(f"  Capacity remaining: {registry.capacity_remaining}")

    assert id2 == 7
    assert registry.capacity_remaining == 0
    assert registry.num_active == 8

    # Try to spawn beyond capacity (should raise)
    expert3 = create_dummy_expert()
    try:
        registry.spawn_expert(
            step=102, parent_id=0, expert_module=expert3, optimizer=optimizer
        )
        print("\n✗ FAIL: Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"\n✓ Correctly raised RuntimeError: {e}")

    print("\n✓ Capacity limits test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("EXPERT REGISTRY TESTS")
    print("=" * 70)

    tests = [
        test_initialization,
        test_spawn_expert,
        test_prune_expert,
        test_merge_experts,
        test_cooling_to_archived,
        test_utilization_tracking,
        test_checkpoint_save_load,
        test_capacity_limits,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
