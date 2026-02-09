"""
Tests for SwissMoEWrapper (Phase 6 integration).

Validates:
- Forward pass equivalence (wrapped vs unwrapped)
- MoETrace signal capture
- ExpertRegistry masking injection
- Per-expert output extraction
- Multi-layer wrapping
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.integration.swiss_moe_wrapper import (
    SwissMoEWrapper,
    wrap_moe_model,
    extract_traces_from_wrappers,
)
from chronomoe_v3.expert_registry import ExpertRegistry


class MockSwissMoELayer(nn.Module):
    """
    Mock of swiss-ai/MoE layer for testing.

    Mimics the architecture:
    - router: nn.Linear(d_model, num_experts)
    - experts: nn.ModuleList of MLPs
    - forward: topk selection + weighted accumulation
    """

    def __init__(self, d_model: int = 64, d_ff: int = 256, num_experts: int = 8, top_k: int = 2):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Experts (simple 2-layer MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard swiss-ai/MoE forward pass."""
        # Flatten if needed
        original_shape = x.shape
        if x.ndim == 3:
            batch_size, seq_len, d_model = x.shape
            x = x.view(batch_size * seq_len, d_model)
        else:
            batch_size, seq_len = None, None

        # Router
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=1)

        # Top-k
        weights, selected_experts = torch.topk(router_probs, self.top_k, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Expert dispatch
        output = torch.zeros_like(x)
        for expert_id in range(self.num_experts):
            expert_mask = (selected_experts == expert_id).any(dim=1)
            if not expert_mask.any():
                continue

            batch_idx = torch.where(expert_mask)[0]
            expert_positions = (selected_experts[batch_idx] == expert_id).long()
            gate_weights = weights[batch_idx].gather(1, expert_positions.argmax(1, keepdim=True)).squeeze(1)

            expert_output = self.experts[expert_id](x[batch_idx])
            output[batch_idx] += expert_output * gate_weights.unsqueeze(1)

        # Reshape if needed
        if batch_size is not None and seq_len is not None:
            output = output.view(batch_size, seq_len, self.d_model)

        return output


def test_forward_pass_equivalence():
    """Test wrapped forward pass produces same output as unwrapped."""
    print("\n" + "=" * 70)
    print("TEST: Forward Pass Equivalence")
    print("=" * 70)

    torch.manual_seed(42)

    # Create mock MoE layer
    moe_layer = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)

    # Create wrapper
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=None, capture_signals=False)

    # Test input
    x = torch.randn(4, 16, 64)  # [batch, seq_len, d_model]

    print(f"\nInput shape: {x.shape}")

    # Unwrapped forward pass
    with torch.no_grad():
        output_unwrapped = moe_layer(x)

    # Wrapped forward pass (return_trace=False returns just output)
    with torch.no_grad():
        output_wrapped = wrapper(x, return_trace=False)

    print(f"Output shape (unwrapped): {output_unwrapped.shape}")
    print(f"Output shape (wrapped): {output_wrapped.shape}")

    # Compare outputs
    delta = (output_unwrapped - output_wrapped).abs().max().item()
    print(f"\nMax absolute difference: {delta:.2e}")

    assert delta < 1e-5, f"Outputs differ by {delta}"

    print("\n✓ Forward pass equivalence test passed")
    return True


def test_trace_capture():
    """Test MoETrace signal capture."""
    print("\n" + "=" * 70)
    print("TEST: MoETrace Capture")
    print("=" * 70)

    torch.manual_seed(42)

    # Create mock MoE layer
    moe_layer = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)

    # Create wrapper with signal capture
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=None, capture_signals=True)

    # Test input
    x = torch.randn(4, 16, 64)  # [batch, seq_len, d_model]

    print(f"\nInput: batch=4, seq_len=16, d_model=64")

    # Forward pass with trace
    with torch.no_grad():
        output, trace = wrapper(x, return_trace=True)

    print(f"\nTrace captured:")
    print(f"  Mixture shape: {trace.mixture.shape}")
    print(f"  Router logits shape: {trace.router_logits_clean.shape}")
    print(f"  Router probs shape: {trace.router_probs.shape}")
    print(f"  Active experts: {len(trace.active_expert_ids)}")
    print(f"  Expert mean outputs: {len(trace.expert_mean_outputs)}")

    # Verify trace contents
    assert trace is not None
    assert trace.mixture.shape == (64, 64)  # Flattened [batch*seq_len, d_model]
    assert trace.router_logits_clean.shape == (64, 8)
    assert trace.router_probs.shape == (64, 8)
    assert len(trace.active_expert_ids) > 0
    assert len(trace.expert_mean_outputs) == len(trace.active_expert_ids)
    assert len(trace.token_row_indices) == len(trace.active_expert_ids)
    assert len(trace.gate_weights) == len(trace.active_expert_ids)

    # Check per-expert outputs are correct shape
    for expert_output in trace.expert_mean_outputs:
        assert expert_output.shape == (64,), f"Expected (64,), got {expert_output.shape}"

    print("\n✓ Trace capture test passed")
    return True


def test_registry_masking():
    """Test ExpertRegistry active mask injection."""
    print("\n" + "=" * 70)
    print("TEST: ExpertRegistry Masking")
    print("=" * 70)

    torch.manual_seed(42)

    # Create mock MoE layer with 8 experts
    moe_layer = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)

    # Create registry with only 4 experts active
    registry = ExpertRegistry(layer_id=0, max_experts=8, initial_active=4)

    print(f"\nRegistry state:")
    print(f"  {registry.status_summary()}")
    print(f"  Active mask: {registry.active_mask.nonzero().squeeze().tolist()}")

    # Create wrapper with registry
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=registry, capture_signals=True)

    # Test input
    x = torch.randn(4, 16, 64)

    # Forward pass
    with torch.no_grad():
        output, trace = wrapper(x, return_trace=True)

    print(f"\nAfter forward pass:")
    print(f"  Active experts in trace: {trace.active_expert_ids}")
    print(f"  Expected active: [0, 1, 2, 3]")

    # Verify only active experts were used
    for expert_id in trace.active_expert_ids:
        assert expert_id < 4, f"Inactive expert {expert_id} was used!"

    # Verify router probs for inactive experts are zero
    inactive_probs = trace.router_probs[:, 4:].sum().item()
    print(f"  Prob mass on inactive experts: {inactive_probs:.6f}")
    assert inactive_probs < 1e-6, f"Inactive experts got prob mass: {inactive_probs}"

    print("\n✓ Registry masking test passed")
    return True


def test_coherence_from_trace():
    """Test computing coherence from captured trace."""
    print("\n" + "=" * 70)
    print("TEST: Coherence from Trace")
    print("=" * 70)

    torch.manual_seed(42)

    moe_layer = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=None, capture_signals=True)

    x = torch.randn(4, 16, 64)

    with torch.no_grad():
        output, trace = wrapper(x, return_trace=True)

    # Compute coherence using trace
    phi = trace.compute_coherence()

    print(f"\nCoherence computed:")
    print(f"  Shape: {phi.shape}")
    print(f"  Values: {phi.tolist()}")
    print(f"  Range: [{phi.min():.3f}, {phi.max():.3f}]")

    # Verify coherence properties
    assert phi.shape == (len(trace.active_expert_ids),)
    assert (phi >= -1.0).all() and (phi <= 1.0).all(), "Coherence out of [-1, 1] range"

    print("\n✓ Coherence from trace test passed")
    return True


def test_utilization_from_trace():
    """Test computing utilization from captured trace."""
    print("\n" + "=" * 70)
    print("TEST: Utilization from Trace")
    print("=" * 70)

    torch.manual_seed(42)

    moe_layer = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=None, capture_signals=True)

    x = torch.randn(4, 16, 64)  # 64 tokens total

    with torch.no_grad():
        output, trace = wrapper(x, return_trace=True)

    # Compute utilization
    utilization = trace.get_expert_utilization()

    print(f"\nUtilization computed:")
    print(f"  Shape: {utilization.shape}")
    print(f"  Values: {utilization.tolist()}")
    print(f"  Total: {utilization.sum().item()}")

    # Verify utilization properties
    assert utilization.shape == (len(trace.active_expert_ids),)
    assert (utilization > 0).all(), "All active experts should have utilization > 0"

    # With top_k=2 and 64 tokens, total routed = 128 expert-token pairs
    # But some experts might not be selected at all
    print(f"  Expected ~{64 * 2} total expert-token pairs (top_k=2, 64 tokens)")

    print("\n✓ Utilization from trace test passed")
    return True


def test_multi_layer_wrapping():
    """Test wrapping multiple MoE layers in a model."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Layer Wrapping")
    print("=" * 70)

    torch.manual_seed(42)

    # Create a simple model with multiple MoE layers
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)
            self.layer1 = MockSwissMoELayer(d_model=64, num_experts=8, top_k=2)

        def forward(self, x):
            x = self.layer0(x)
            x = self.layer1(x)
            return x

    model = MockModel()

    # Create registries
    registries = {
        0: ExpertRegistry(layer_id=0, max_experts=8, initial_active=6),
        1: ExpertRegistry(layer_id=1, max_experts=8, initial_active=4),
    }

    print(f"\nBefore wrapping:")
    print(f"  layer0 type: {type(model.layer0).__name__}")
    print(f"  layer1 type: {type(model.layer1).__name__}")

    # Wrap both layers
    wrappers = wrap_moe_model(
        model,
        moe_layer_names=["layer0", "layer1"],
        registries=registries,
    )

    print(f"\nAfter wrapping:")
    print(f"  layer0 type: {type(model.layer0).__name__}")
    print(f"  layer1 type: {type(model.layer1).__name__}")
    print(f"  Wrappers: {list(wrappers.keys())}")

    # Verify wrappers installed
    assert isinstance(model.layer0, SwissMoEWrapper)
    assert isinstance(model.layer1, SwissMoEWrapper)
    assert len(wrappers) == 2

    # Forward pass
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        output = model(x)

    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Extract traces
    traces = extract_traces_from_wrappers(wrappers)
    print(f"\nTraces extracted:")
    for layer_name, trace in traces.items():
        if trace:
            print(f"  {layer_name}: {len(trace.active_expert_ids)} active experts")

    assert output.shape == x.shape
    assert len(traces) > 0

    print("\n✓ Multi-layer wrapping test passed")
    return True


def test_wrapped_backward_pass():
    """Test that gradients flow correctly through wrapped layer."""
    print("\n" + "=" * 70)
    print("TEST: Wrapped Backward Pass")
    print("=" * 70)

    torch.manual_seed(42)

    # Create mock MoE layer
    moe_layer = MockSwissMoELayer(d_model=64, num_experts=4, top_k=2)

    # Create wrapper
    wrapper = SwissMoEWrapper(moe_layer, layer_id=0, registry=None, capture_signals=True)

    # Test input (requires_grad)
    x = torch.randn(2, 8, 64, requires_grad=True)

    print(f"\nInput: {x.shape}, requires_grad={x.requires_grad}")

    # Forward pass
    output, trace = wrapper(x, return_trace=True)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Input grad shape: {x.grad.shape}")
    print(f"Input grad norm: {x.grad.norm().item():.4f}")

    # Verify gradients computed
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.norm().item() > 0, "Gradients should be non-zero"

    # Verify expert parameters have gradients
    for expert_id, expert in enumerate(wrapper.moe_layer.experts):
        for param in expert.parameters():
            assert param.grad is not None, f"Expert {expert_id} missing gradients"

    print("\n✓ Wrapped backward pass test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("SWISS-AI/MOE WRAPPER TESTS")
    print("=" * 70)

    tests = [
        test_forward_pass_equivalence,
        test_trace_capture,
        test_registry_masking,
        test_coherence_from_trace,
        test_utilization_from_trace,
        test_multi_layer_wrapping,
        test_wrapped_backward_pass,
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
