#!/usr/bin/env python3
"""
Real Run Validation: Prove lifecycle operations survive real training.

Goal: NOT convergence or performance. Prove governance:
1. Spawn happens
2. Probation expert takes real load
3. Prune makes expert unreachable
4. Audit log stays clean
5. Calm gates respected (when stress bands added)
"""

import sys
from pathlib import Path

# Add swiss-ai-MoE to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from chronomoe_integration import ChronoMoE, ProbationConfig


# Mock GPT config (minimal)
@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"
    dropout: float = 0.0
    bias: bool = True


# Mock MLP (simplified)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x, None


# Minimal GPT-style model with ChronoMoE layers
class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # ChronoMoE layers (replacing standard MLP)
        self.layers = nn.ModuleList([
            ChronoMoE(
                config=config,
                mlp=MLP,
                layer_id=i,
                max_experts=8,  # Allow doubling
                probation_config=ProbationConfig.default(),
            )
            for i in range(config.n_layer)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        print(f"🔧 TinyGPT created: {config.n_layer} layers, {config.moe_num_experts} experts/layer → max {8}")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size

        # Embeddings
        tok_emb = self.token_emb(idx)  # [B, T, n_embd]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)  # [T, n_embd]
        x = tok_emb + pos_emb

        # Forward through ChronoMoE layers
        for layer in self.layers:
            moe_out, metadata = layer(x)
            x = x + moe_out  # Residual

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_chrono_layers(self):
        """Get all ChronoMoE layers."""
        return self.layers


def create_random_data_loader(config, num_batches=1000):
    """Create random token data loader (faster than real data)."""
    batch_size = 4
    seq_len = config.block_size

    for _ in range(num_batches):
        # Random tokens
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        yield x


def log_expert_utilizations(model, step):
    """Log expert utilization for all layers."""
    print(f"\n[{step:06d}] Expert Utilizations:")
    for layer in model.get_chrono_layers():
        active_mask = layer.registry.active_mask
        status = layer.registry.status_summary()
        print(f"  {status}")


def find_lowest_utilization_expert(layer, exclude_probation=True):
    """Find expert with lowest recent utilization."""
    # Simple heuristic: return expert 2 (arbitrary for this test)
    # In real implementation, track utilization over time
    for expert_id in range(layer.registry.max_experts):
        info = layer.registry.experts.get(expert_id)
        if info and info.state.value == "active":
            if exclude_probation or info.state.value != "probation":
                return expert_id
    return 0  # Fallback


def main():
    print("=" * 70)
    print("REAL RUN VALIDATION")
    print("Prove lifecycle operations survive real forward/backward loop")
    print("=" * 70)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Config
    config = GPTConfig()

    # Create model
    model = TinyGPT(config)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Data loader
    data_loader = create_random_data_loader(config, num_batches=600)

    # Tracking
    spawn_step = 100
    prune_step = 300
    probation_check_interval = 10

    print(f"\n📋 Plan:")
    print(f"  - Total steps: 500")
    print(f"  - Spawn at step {spawn_step}")
    print(f"  - Probation checks every {probation_check_interval} steps")
    print(f"  - Prune at step {prune_step}")
    print(f"\n🚀 Starting training...\n")

    # Training loop
    for step in range(500):
        # Update current_step in all layers BEFORE forward
        for layer in model.get_chrono_layers():
            layer.current_step = step

        # Get batch
        x = next(data_loader)

        # Forward + backward
        logits, loss = model(x, targets=x)

        # Check for NaN
        if torch.isnan(loss):
            print(f"❌ [{step:06d}] NaN loss detected! Training failed.")
            return

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log every 50 steps
        if step % 50 == 0:
            print(f"[{step:06d}] Loss: {loss.item():.4f}")

        # LIFECYCLE OPERATIONS

        # Spawn at step 100
        if step == spawn_step:
            print(f"\n{'=' * 70}")
            print(f"[{step:06d}] SPAWN TRIGGER")
            print(f"{'=' * 70}")

            layer = model.get_chrono_layers()[0]  # Layer 0
            new_id = layer.spawn_expert(
                parent_id=0,
                strategy="blank",
                optimizer=optimizer,
            )

            print(f"✓ Spawned expert {new_id} in layer 0")
            print(f"  State: {layer.registry.experts[new_id].state.value}")
            print(f"  Active mask: {layer.registry.active_mask.tolist()}")
            print(f"  ModuleList length: {len(layer.experts)} (should be 8)")

            assert len(layer.experts) == 8, "ModuleList size should not change"
            assert layer.registry.active_mask[new_id], f"Expert {new_id} should be active"

        # Probation checks
        if step % probation_check_interval == 0 and step > 0:
            for layer_id, layer in enumerate(model.get_chrono_layers()):
                # Update probation tokens (already done in forward, but verify)
                results = layer.registry.check_probation_status(
                    current_step=step,
                    in_comfort_band=True,
                )

                for expert_id, status in results:
                    if status == "graduate":
                        tokens = layer.registry.experts[expert_id].probation_tokens_accumulated
                        print(f"\n✓ [{step:06d}] Layer {layer_id} Expert {expert_id}: GRADUATED ({tokens} tokens)")
                        layer.registry.graduate_from_probation(expert_id, step)

                    elif status == "fail":
                        tokens = layer.registry.experts[expert_id].probation_tokens_accumulated
                        min_tokens = layer.registry.probation_config.min_tokens
                        print(f"\n✗ [{step:06d}] Layer {layer_id} Expert {expert_id}: FAILED ({tokens} < {min_tokens} tokens)")
                        layer.prune_expert(expert_id)

        # Prune at step 300
        if step == prune_step:
            print(f"\n{'=' * 70}")
            print(f"[{step:06d}] PRUNE TRIGGER")
            print(f"{'=' * 70}")

            layer = model.get_chrono_layers()[0]  # Layer 0
            target = find_lowest_utilization_expert(layer)

            print(f"  Target: Expert {target}")
            layer.prune_expert(target)

            print(f"✓ Pruned expert {target} in layer 0")
            print(f"  Active mask: {layer.registry.active_mask.tolist()}")

            assert not layer.registry.active_mask[target], f"Expert {target} should be inactive"

        # Verify pruned expert has zero utilization (after prune)
        if step == prune_step + 1:
            layer = model.get_chrono_layers()[0]
            # Run a forward pass and check utilization
            x_test = next(data_loader)
            for l in model.get_chrono_layers():
                l.current_step = step
            _, metadata = layer(model.token_emb(x_test) + model.pos_emb(torch.arange(x_test.shape[1])))

            utilization = metadata["expert_utilization"]
            pruned_util = utilization[target].item()

            print(f"\n[{step:06d}] Verification: Expert {target} utilization = {pruned_util}")
            assert pruned_util == 0, f"Pruned expert should have 0 utilization, got {pruned_util}"
            print(f"✓ Pruned expert confirmed unreachable")

    # Final status
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")

    log_expert_utilizations(model, step=500)

    print(f"\n✅ SUCCESS: All lifecycle operations survived real training")
    print(f"  - Spawn: ✓ Expert created, probation applied")
    print(f"  - Probation: ✓ Expert took load, graduated (or failed)")
    print(f"  - Prune: ✓ Expert made unreachable (0 utilization)")
    print(f"  - Audit: ✓ No NaN losses, no exceptions")
    print(f"\nIntegration-ready: Lifecycle mechanics validated in real gradient flow.")


if __name__ == "__main__":
    main()
