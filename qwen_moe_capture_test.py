#!/usr/bin/env python3
"""
Proof-of-concept: Monkeypatch MLX Qwen MoE to capture per-token routing telemetry.

Tests whether we can extract per-token expert indices, scores, and entropy
from Qwen1.5-MoE-A2.7B running on MLX without forward hooks.

Usage:
    source qwen_moe_mlx/bin/activate
    python3 qwen_moe_capture_test.py
"""

import sys
import time
import types
import numpy as np

# MLX imports
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# ---------------------------------------------------------------------------
# Routing capture storage
# ---------------------------------------------------------------------------

class RoutingCapture:
    """Collects per-layer routing data from monkeypatched MoE blocks."""

    def __init__(self):
        self.data = {}  # layer_id -> {gates, inds, scores}
        self.enabled = True

    def clear(self):
        self.data.clear()

    def store(self, layer_id, gates, inds, scores):
        if not self.enabled:
            return
        # Convert to numpy immediately to avoid MLX graph issues
        self.data[layer_id] = {
            "gates": np.array(gates),     # [B*T, num_experts] post-softmax
            "inds": np.array(inds),       # [B*T, top_k] expert indices
            "scores": np.array(scores),   # [B*T, top_k] routing weights
        }

    def summary(self):
        """Print summary of captured routing data."""
        if not self.data:
            print("No routing data captured.")
            return

        print(f"\n{'='*60}")
        print(f"Captured routing from {len(self.data)} MoE layers")
        print(f"{'='*60}")

        for lid in sorted(self.data.keys()):
            d = self.data[lid]
            gates = d["gates"]
            inds = d["inds"]
            scores = d["scores"]

            num_tokens = gates.shape[0]
            num_experts = gates.shape[-1]
            top_k = inds.shape[-1]

            # Entropy
            log_gates = np.log(gates + 1e-8)
            entropy = -(gates * log_gates).sum(axis=-1)
            mean_entropy = entropy.mean()
            max_entropy = np.log(num_experts)

            # Expert usage (selection frequency)
            flat_inds = inds.reshape(-1)
            usage = np.bincount(flat_inds, minlength=num_experts).astype(float)
            usage /= max(num_tokens, 1)  # per-token frequency

            # Neff from usage
            usage_norm = usage / usage.sum()
            usage_sq = (usage_norm ** 2).sum()
            neff = 1.0 / usage_sq if usage_sq > 0 else 0

            # Top experts
            top_experts = np.argsort(-usage)[:5]

            print(f"\nLayer {lid}:")
            print(f"  Tokens: {num_tokens}, Experts: {num_experts}, Top-k: {top_k}")
            print(f"  Mean entropy: {mean_entropy:.4f} (max: {max_entropy:.4f}, ratio: {mean_entropy/max_entropy:.3f})")
            print(f"  Neff: {neff:.2f}")
            print(f"  Mean scores: {scores.mean(axis=0)}")
            print(f"  Top-5 experts by usage: {top_experts} (freq: {usage[top_experts]})")

        print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Monkeypatch
# ---------------------------------------------------------------------------

def patch_moe_blocks(model, capture: RoutingCapture):
    """Patch the MoE block CLASS __call__ to capture routing data.

    Python resolves obj() via type(obj).__call__, not obj.__call__,
    so we must patch the class method, not instance attributes.
    We store layer_id on each instance for the class-level patch to read.
    """

    patched = 0
    moe_class = None

    # Navigate to decoder layers
    # model.model.layers[i].mlp is the MoE block
    layers = model.model.layers

    for i, layer in enumerate(layers):
        moe_block = layer.mlp

        # Check it's a MoE block (has gate, switch_mlp, shared_expert)
        if not hasattr(moe_block, 'gate') or not hasattr(moe_block, 'switch_mlp'):
            continue

        # Store layer_id on instance for class-level patch to read
        moe_block._capture_layer_id = i
        moe_block._routing_capture = capture
        patched += 1

        if moe_class is None:
            moe_class = type(moe_block)

    if moe_class is None:
        print("ERROR: No MoE block class found!")
        return 0

    # Save original class __call__
    moe_class._original_call = moe_class.__call__

    # Replace class-level __call__
    def patched_call(self, x):
        # Compute gates (same as original)
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)

        # Capture routing data
        cap = getattr(self, '_routing_capture', None)
        lid = getattr(self, '_capture_layer_id', -1)
        if cap is not None and cap.enabled:
            mx.eval(gates, inds, scores)
            cap.store(lid, gates, inds, scores)

        # Continue with expert computation
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_expert_output = self.shared_expert(x)
        shared_expert_output = (
            mx.sigmoid(self.shared_expert_gate(x)) * shared_expert_output
        )

        return y + shared_expert_output

    moe_class.__call__ = patched_call

    print(f"Patched {patched} MoE blocks (class: {moe_class.__name__})")
    return patched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = "mlx-community/Qwen1.5-MoE-A2.7B-4bit"

    print(f"Loading {model_path}...")
    t0 = time.time()
    model, tokenizer = load(model_path)
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Set up capture
    capture = RoutingCapture()
    num_patched = patch_moe_blocks(model, capture)

    if num_patched == 0:
        print("ERROR: No MoE blocks found to patch!")
        sys.exit(1)

    # Tokenize a test prompt
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Explain the theory of general relativity in simple terms.",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt[:60]}...")
        print(f"{'='*60}")

        capture.clear()

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        print(f"Input tokens: {len(tokens)}")

        t0 = time.time()
        logits = model(input_ids)
        mx.eval(logits)
        elapsed = time.time() - t0

        print(f"Forward pass: {elapsed:.2f}s")

        # Show summary
        capture.summary()

        # Show prediction
        next_token = mx.argmax(logits[0, -1, :]).item()
        print(f"Predicted next token: '{tokenizer.decode([next_token])}'")

    # Cross-prompt routing comparison
    print(f"\n\n{'='*60}")
    print("CROSS-PROMPT ROUTING COMPARISON")
    print(f"{'='*60}")

    all_captures = {}
    for i, prompt in enumerate(test_prompts):
        capture.clear()
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        all_captures[i] = {lid: d.copy() for lid, d in capture.data.items()}

    # Compare expert usage between prompts at layer 0 and last layer
    for lid in [0, max(all_captures[0].keys())]:
        print(f"\nLayer {lid} - Expert usage comparison:")
        usages = []
        for i in range(len(test_prompts)):
            inds = all_captures[i][lid]["inds"]
            num_tokens = inds.shape[0]
            num_experts = all_captures[i][lid]["gates"].shape[-1]
            flat = inds.reshape(-1)
            usage = np.bincount(flat, minlength=num_experts).astype(float) / num_tokens
            usages.append(usage)
            top3 = np.argsort(-usage)[:3]
            print(f"  Prompt {i}: top-3 experts = {top3} (freq: {np.round(usage[top3], 3)})")

        # KL divergence between prompts
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            ua = usages[a] / usages[a].sum()
            ub = usages[b] / usages[b].sum()
            kl = np.sum(ua * np.log((ua + 1e-10) / (ub + 1e-10)))
            print(f"  KL(prompt_{a} || prompt_{b}) = {kl:.4f}")

    print("\nDone. Routing telemetry is fully accessible via monkeypatch.")


if __name__ == "__main__":
    main()
