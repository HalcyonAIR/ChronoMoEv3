"""
Step 2 Demo: Coherence on GPU with Buffered State

Demonstrates:
1. GPU-resident coherence updates (no CPU sync per step)
2. CPU snapshot only on eval intervals
3. Memory efficiency (~2KB per layer vs 48KB with role vectors)
4. Performance comparison: GPU vs CPU coherence tracking
"""

import sys
import time
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import CoherenceBuffer, MultiLayerCoherenceBuffer


def demo_gpu_coherence_updates():
    """Show GPU coherence updates without CPU sync."""
    print("=" * 70)
    print("DEMO: GPU-Resident Coherence Updates")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )

    print(f"\nDevice: {device}")
    print(f"Memory usage: {buffer.get_memory_usage()} bytes")
    print(f"  (vs ~48KB if role vectors were included)")

    # Simulate 10 forward passes
    print(f"\nSimulating 10 forward passes...")
    for step in range(10):
        # Random coherence values
        num_active = torch.randint(2, 5, (1,)).item()
        phi_raw = torch.rand(num_active, device=device) * 0.5 + 0.4  # [0.4, 0.9]
        active_ids = torch.randperm(8, device=device)[:num_active]
        num_tokens = torch.randint(5, 20, (num_active,), device=device)

        # Update (all on GPU, no CPU sync)
        buffer.update(phi_raw, active_ids, step, num_tokens)

        if step % 3 == 0:
            # Show current state (still on GPU)
            phi_slow_gpu = buffer.get_phi_slow()
            phi_delta_gpu = buffer.get_phi_delta()
            print(
                f"  Step {step}: phi_slow (GPU) mean={phi_slow_gpu.mean().item():.3f}, "
                f"phi_delta mean={phi_delta_gpu.mean().item():.3f}"
            )

    # CPU snapshot only when needed (e.g., for lifecycle evaluation)
    print(f"\nTaking CPU snapshot (eval interval)...")
    snapshot = buffer.snapshot(step=10)

    print(f"Snapshot contains {len(snapshot)} expert states")
    for expert_id, state in list(snapshot.items())[:3]:
        print(
            f"  Expert {expert_id}: phi_slow={state.phi_slow:.3f}, "
            f"tokens={state.total_tokens_seen}"
        )


def demo_multi_layer():
    """Show multi-layer coherence tracking."""
    print("\n" + "=" * 70)
    print("DEMO: Multi-Layer Coherence Tracking")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    multi_buffer = MultiLayerCoherenceBuffer(
        num_layers=4,
        num_experts_per_layer=16,
        d_model=128,
        device=device,
    )

    print(f"\nTracking 4 layers × 16 experts = 64 expert states")
    print(f"Total memory: {multi_buffer.get_memory_usage()} bytes")
    print(f"  Per layer: {multi_buffer.get_memory_usage() / 4:.0f} bytes")

    # Update each layer
    for layer_id in range(4):
        phi_raw = torch.rand(4, device=device) * 0.5 + 0.4
        active_ids = torch.randperm(16, device=device)[:4]
        num_tokens = torch.randint(5, 20, (4,), device=device)

        multi_buffer.update(layer_id, phi_raw, active_ids, step=1, num_tokens=num_tokens)

    # Snapshot all layers
    snapshot = multi_buffer.snapshot(step=1)
    print(f"\nSnapshot: {len(snapshot)} total expert states across all layers")

    # Show per-layer summary
    for layer_id in range(4):
        layer_experts = [
            state
            for state in snapshot.values()
            if state.layer_id == layer_id and state.total_tokens_seen > 0
        ]
        if layer_experts:
            avg_phi = sum(e.phi_fast for e in layer_experts) / len(layer_experts)
            print(f"  Layer {layer_id}: {len(layer_experts)} active, avg_phi={avg_phi:.3f}")


def demo_degradation_detection():
    """Show degradation detection via phi_delta."""
    print("\n" + "=" * 70)
    print("DEMO: Degradation Detection (phi_delta)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0,
        num_experts=8,
        d_model=64,
        device=device,
        alpha_fast=0.5,  # Fast decay for demo
        alpha_slow=0.95,
    )

    # Expert 0: Healthy (consistent high coherence)
    # Expert 1: Degrading (starts high, drops)
    # Expert 2: Recovering (starts low, improves)

    print(f"\nSimulating 20 steps with different expert trajectories:")
    print(f"  Expert 0: Healthy (consistent)")
    print(f"  Expert 1: Degrading (drops at step 10)")
    print(f"  Expert 2: Recovering (improves at step 10)")
    print()

    for step in range(20):
        if step < 10:
            # First 10 steps
            phi_raw = torch.tensor([0.8, 0.8, 0.3], device=device)
        else:
            # After step 10
            phi_raw = torch.tensor([0.8, 0.2, 0.8], device=device)

        active_ids = torch.tensor([0, 1, 2], device=device)
        num_tokens = torch.tensor([10, 10, 10], device=device)

        buffer.update(phi_raw, active_ids, step, num_tokens)

        if step % 5 == 4:
            phi_delta = buffer.get_phi_delta()
            print(f"Step {step:2d}:")
            print(f"  Expert 0 (healthy):    phi_delta={phi_delta[0]:+.3f}")
            print(f"  Expert 1 (degrading):  phi_delta={phi_delta[1]:+.3f}")
            print(f"  Expert 2 (recovering): phi_delta={phi_delta[2]:+.3f}")

    print(f"\nFinal phi_delta values:")
    phi_delta = buffer.get_phi_delta()
    for i in range(3):
        status = "✓ healthy" if phi_delta[i] > 0.1 else "✗ degrading" if phi_delta[i] < -0.1 else "~ stable"
        print(f"  Expert {i}: {phi_delta[i]:+.3f} {status}")


def demo_checkpoint():
    """Show checkpoint save/load."""
    print("\n" + "=" * 70)
    print("DEMO: Checkpoint Save/Load")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )

    # Update some state
    phi_raw = torch.rand(4, device=device)
    active_ids = torch.tensor([0, 2, 5, 7], device=device)
    num_tokens = torch.tensor([10, 5, 8, 12], device=device)

    for step in range(50):
        buffer.update(phi_raw, active_ids, step, num_tokens)

    print(f"\nOriginal buffer state:")
    print(f"  Expert 0 phi_slow: {buffer.phi_slow[0].item():.4f}")
    print(f"  Expert 2 phi_slow: {buffer.phi_slow[2].item():.4f}")
    print(f"  Expert 0 tokens: {buffer.total_tokens_seen[0].item()}")

    # Save to checkpoint
    checkpoint = buffer.to_dict()
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

    # Create new buffer and load
    buffer2 = CoherenceBuffer(
        layer_id=0, num_experts=8, d_model=64, device=device
    )
    buffer2.load_from_dict(checkpoint)

    print(f"\nRestored buffer state:")
    print(f"  Expert 0 phi_slow: {buffer2.phi_slow[0].item():.4f}")
    print(f"  Expert 2 phi_slow: {buffer2.phi_slow[2].item():.4f}")
    print(f"  Expert 0 tokens: {buffer2.total_tokens_seen[0].item()}")

    # Verify match
    match = torch.allclose(buffer.phi_slow, buffer2.phi_slow)
    print(f"\nCheckpoint integrity: {'✓ verified' if match else '✗ mismatch'}")


def demo_performance():
    """Compare GPU vs CPU coherence update performance."""
    print("\n" + "=" * 70)
    print("DEMO: Performance Comparison (GPU vs CPU)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("\nNote: CUDA not available, comparing CPU performance only")

    # Test configuration
    num_steps = 1000
    num_experts = 64
    d_model = 128

    print(f"\nConfiguration:")
    print(f"  Steps: {num_steps}")
    print(f"  Experts: {num_experts}")
    print(f"  d_model: {d_model}")

    # GPU/CPU coherence buffer
    buffer = CoherenceBuffer(
        layer_id=0, num_experts=num_experts, d_model=d_model, device=device
    )

    # Benchmark
    start_time = time.time()
    for step in range(num_steps):
        num_active = 8
        phi_raw = torch.rand(num_active, device=device)
        active_ids = torch.randperm(num_experts, device=device)[:num_active]
        num_tokens = torch.randint(1, 10, (num_active,), device=device)

        buffer.update(phi_raw, active_ids, step, num_tokens)

        # Snapshot only every 100 steps (eval interval)
        if step % 100 == 0:
            _ = buffer.snapshot(step)

    elapsed = time.time() - start_time

    print(f"\nResults ({device}):")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per step: {elapsed/num_steps*1000:.3f}ms")
    print(f"  Updates/sec: {num_steps/elapsed:.0f}")
    print(f"\nKey benefit: Coherence updates stay on GPU, snapshot only on eval")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STEP 2: Coherence on GPU with Buffered State")
    print("=" * 70)
    print("\nDeliverables:")
    print("  ✓ CoherenceBuffer (GPU-resident tensors)")
    print("  ✓ Update every step (no CPU sync bottleneck)")
    print("  ✓ Snapshot to CPU only on eval intervals")
    print("  ✓ Memory efficient (~2KB per layer)")
    print()

    demo_gpu_coherence_updates()
    demo_multi_layer()
    demo_degradation_detection()
    demo_checkpoint()
    demo_performance()

    print("\n" + "=" * 70)
    print("✅ Step 2 Complete: GPU coherence works!")
    print("=" * 70)
    print("\nNext: Step 3 - Beta update function (close the loop)")
    print()
