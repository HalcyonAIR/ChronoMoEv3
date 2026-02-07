"""
Tests for GPU-resident coherence buffer.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.coherence_gpu import CoherenceBuffer, MultiLayerCoherenceBuffer


def test_coherence_buffer_init():
    """Test CoherenceBuffer initialization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )

    # Check tensors on correct device
    assert buffer.phi_fast.device.type == device
    assert buffer.phi_mid.device.type == device
    assert buffer.phi_slow.device.type == device
    assert buffer.total_tokens_seen.device.type == device
    assert buffer.last_update_step.device.type == device

    # Check shapes
    assert buffer.phi_fast.shape == (8,)
    assert buffer.phi_mid.shape == (8,)
    assert buffer.phi_slow.shape == (8,)

    # Check initialization
    assert torch.allclose(buffer.phi_fast, torch.zeros(8))
    assert torch.allclose(buffer.phi_mid, torch.zeros(8))
    assert torch.allclose(buffer.phi_slow, torch.zeros(8))

    print(f"✓ CoherenceBuffer initialization (device={device})")


def test_coherence_buffer_update():
    """Test GPU coherence update."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )

    # Simulate coherence update for experts 0, 2, 5
    phi_raw = torch.tensor([0.8, 0.6, 0.9], device=device)
    active_ids = torch.tensor([0, 2, 5], device=device)
    num_tokens = torch.tensor([10, 5, 8], device=device)

    buffer.update(phi_raw, active_ids, step=1, num_tokens=num_tokens)

    # Check that phi values updated (EMA with alpha=0.9, 0.99, 0.999)
    # First update from zero: phi = (1 - alpha) * phi_raw
    assert buffer.phi_fast[0] > 0  # Should be ~0.08 (0.1 * 0.8)
    assert buffer.phi_fast[2] > 0  # Should be ~0.06 (0.1 * 0.6)
    assert buffer.phi_fast[5] > 0  # Should be ~0.09 (0.1 * 0.9)

    # Inactive experts should remain zero
    assert buffer.phi_fast[1] == 0.0
    assert buffer.phi_fast[3] == 0.0

    # Check observation tracking
    assert buffer.total_tokens_seen[0] == 10
    assert buffer.total_tokens_seen[2] == 5
    assert buffer.total_tokens_seen[5] == 8
    assert buffer.last_update_step[0] == 1
    assert buffer.last_update_step[2] == 1
    assert buffer.last_update_step[5] == 1

    print(f"✓ CoherenceBuffer update (device={device})")


def test_coherence_buffer_three_timescales():
    """Test three-timescale EMA behavior."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0,
        num_experts=8,
        d_model=64,
        device=device,
        alpha_fast=0.5,  # Very fast for testing
        alpha_mid=0.9,
        alpha_slow=0.99,
    )

    # Update with same value 10 times
    phi_raw = torch.tensor([1.0], device=device)
    active_ids = torch.tensor([0], device=device)
    num_tokens = torch.tensor([1], device=device)

    for step in range(10):
        buffer.update(phi_raw, active_ids, step=step, num_tokens=num_tokens)

    # Fast should converge fastest
    fast_val = buffer.phi_fast[0].item()
    mid_val = buffer.phi_mid[0].item()
    slow_val = buffer.phi_slow[0].item()

    assert fast_val > mid_val > slow_val
    assert fast_val > 0.9  # Should be close to 1.0
    assert mid_val > 0.5  # Slower convergence
    assert slow_val < 0.3  # Very slow convergence

    print(
        f"✓ Three-timescale EMA: fast={fast_val:.3f}, "
        f"mid={mid_val:.3f}, slow={slow_val:.3f}"
    )


def test_coherence_buffer_snapshot():
    """Test CPU snapshot generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=2, num_experts=4, d_model=64, device=device
    )

    # Update some experts
    phi_raw = torch.tensor([0.8, 0.6], device=device)
    active_ids = torch.tensor([0, 2], device=device)
    num_tokens = torch.tensor([10, 5], device=device)

    buffer.update(phi_raw, active_ids, step=100, num_tokens=num_tokens)

    # Take snapshot
    snapshot = buffer.snapshot(step=100)

    # Check snapshot structure
    assert len(snapshot) == 4  # All experts, not just active
    assert 0 in snapshot
    assert 1 in snapshot
    assert 2 in snapshot
    assert 3 in snapshot

    # Check active expert state
    state_0 = snapshot[0]
    assert state_0.expert_id == "L2_E0"
    assert state_0.layer_id == 2
    assert state_0.phi_fast > 0
    assert state_0.phi_mid > 0
    assert state_0.phi_slow > 0
    assert state_0.total_tokens_seen == 10
    assert state_0.last_update_step == 100

    # Check inactive expert state
    state_1 = snapshot[1]
    assert state_1.phi_fast == 0.0
    assert state_1.phi_mid == 0.0
    assert state_1.phi_slow == 0.0
    assert state_1.total_tokens_seen == 0

    print(f"✓ CoherenceBuffer snapshot (device={device})")


def test_phi_delta():
    """Test phi_delta computation on GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0,
        num_experts=8,
        d_model=64,
        device=device,
        alpha_fast=0.5,
        alpha_slow=0.99,
    )

    # Update with consistent high value
    phi_raw = torch.tensor([0.9], device=device)
    active_ids = torch.tensor([0], device=device)
    num_tokens = torch.tensor([1], device=device)

    for step in range(10):
        buffer.update(phi_raw, active_ids, step=step, num_tokens=num_tokens)

    # Fast should be higher than slow (faster convergence)
    phi_delta = buffer.get_phi_delta()
    assert phi_delta[0] > 0  # Fast > slow

    # Now introduce sustained degradation (drop coherence to near zero)
    phi_raw_low = torch.tensor([0.0], device=device)
    for step in range(10, 25):  # More steps to ensure fast drops below slow
        buffer.update(phi_raw_low, active_ids, step=step, num_tokens=num_tokens)

    # Fast drops quickly to near zero, slow stays positive → negative delta
    phi_delta = buffer.get_phi_delta()
    assert phi_delta[0] < 0  # Degrading

    print(f"✓ Phi_delta computation (degradation detected)")


def test_reset_expert():
    """Test expert reset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )

    # Update expert 0
    phi_raw = torch.tensor([0.8], device=device)
    active_ids = torch.tensor([0], device=device)
    num_tokens = torch.tensor([10], device=device)

    buffer.update(phi_raw, active_ids, step=1, num_tokens=num_tokens)

    # Verify it has state
    assert buffer.phi_fast[0] > 0
    assert buffer.total_tokens_seen[0] == 10

    # Reset expert 0
    buffer.reset_expert(0)

    # Verify reset
    assert buffer.phi_fast[0] == 0.0
    assert buffer.phi_mid[0] == 0.0
    assert buffer.phi_slow[0] == 0.0
    assert buffer.total_tokens_seen[0] == 0
    assert buffer.last_update_step[0] == 0

    print(f"✓ Expert reset (device={device})")


def test_checkpoint_save_load():
    """Test checkpointing (to_dict/load_from_dict)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=4, d_model=64, device=device
    )

    # Update some state
    phi_raw = torch.tensor([0.8, 0.6], device=device)
    active_ids = torch.tensor([0, 2], device=device)
    num_tokens = torch.tensor([10, 5], device=device)

    buffer.update(phi_raw, active_ids, step=100, num_tokens=num_tokens)

    # Save to dict
    state_dict = buffer.to_dict()

    # Create new buffer and load
    buffer2 = CoherenceBuffer(
        layer_id=0, num_experts=4, d_model=64, device=device
    )
    buffer2.load_from_dict(state_dict)

    # Verify state matches
    assert torch.allclose(buffer.phi_fast, buffer2.phi_fast)
    assert torch.allclose(buffer.phi_mid, buffer2.phi_mid)
    assert torch.allclose(buffer.phi_slow, buffer2.phi_slow)
    assert torch.all(buffer.total_tokens_seen == buffer2.total_tokens_seen)
    assert torch.all(buffer.last_update_step == buffer2.last_update_step)

    print(f"✓ Checkpoint save/load (device={device})")


def test_multi_layer_buffer():
    """Test multi-layer coherence buffer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_buffer = MultiLayerCoherenceBuffer(
        num_layers=3,
        num_experts_per_layer=8,
        d_model=64,
        device=device,
    )

    # Update layer 0, expert 0
    phi_raw = torch.tensor([0.8], device=device)
    active_ids = torch.tensor([0], device=device)
    num_tokens = torch.tensor([10], device=device)
    multi_buffer.update(0, phi_raw, active_ids, step=1, num_tokens=num_tokens)

    # Update layer 2, expert 3
    phi_raw = torch.tensor([0.6], device=device)
    active_ids = torch.tensor([3], device=device)
    num_tokens = torch.tensor([5], device=device)
    multi_buffer.update(2, phi_raw, active_ids, step=1, num_tokens=num_tokens)

    # Take snapshot
    snapshot = multi_buffer.snapshot(step=1)

    # Check that we have states for all layers and experts
    assert "L0_E0" in snapshot
    assert "L2_E3" in snapshot

    # Check state values
    assert snapshot["L0_E0"].phi_fast > 0
    assert snapshot["L0_E0"].total_tokens_seen == 10
    assert snapshot["L2_E3"].phi_fast > 0
    assert snapshot["L2_E3"].total_tokens_seen == 5

    # Check memory usage
    mem_usage = multi_buffer.get_memory_usage()
    assert mem_usage > 0

    print(f"✓ Multi-layer buffer (device={device}, mem={mem_usage} bytes)")


def test_memory_usage():
    """Test memory usage estimation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=64, d_model=128, device=device
    )

    mem = buffer.get_memory_usage()

    # Expected: 3 floats (4 bytes) + 2 longs (8 bytes) per expert
    # = (3*4 + 2*8) * 64 = (12 + 16) * 64 = 1792 bytes
    expected = (3 * 4 + 2 * 8) * 64
    assert mem == expected

    print(f"✓ Memory usage: {mem} bytes for 64 experts")


if __name__ == "__main__":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTesting CoherenceBuffer (Step 2) on device: {device_name}\n")

    test_coherence_buffer_init()
    test_coherence_buffer_update()
    test_coherence_buffer_three_timescales()
    test_coherence_buffer_snapshot()
    test_phi_delta()
    test_reset_expert()
    test_checkpoint_save_load()
    test_multi_layer_buffer()
    test_memory_usage()

    print("\n✅ All Step 2 tests passed!")
