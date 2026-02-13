"""
Bob v1 Experiment: Does cost drop over experience at equal quality?

The minimum experiment that proves or disproves the core claim.

Four configurations on the same difficulty ladder:
  1. Top-2 vanilla MoE (no Bob, no cache)
  2. Top-4 vanilla MoE (no Bob, no cache)
  3. Answer cache (hash lookup on input tokens)
  4. MoE + Bob (motif reuse + compound gate)

Two plots:
  1. Pareto frontier (cost vs quality) over ladder steps
  2. Cheap-path fraction over time for Bob

Sanity: answer cache should collapse under drift. If it doesn't,
the ladder is too easy and Bob's win won't convince anyone.

Usage:
    python bob_experiment.py
"""

import sys
import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe import MoE
from backends.adapter import LayerMotif, MotifSpec, ForwardResult
from backends.swiss_adapter import SwissAdapter
from bob_core.substrate import BobSubstrate
from bob_core.motifs import GateThresholds


# ─── Model ────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    vocab_size: int = 256
    n_embd: int = 128
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"
    mlp_dim_exp_factor: float = 1.0
    dropout: float = 0.0
    bias: bool = False
    batch_size: int = 32
    seq_len: int = 16
    # Ladder
    num_classes: int = 4
    class_block_size: int = 50       # Steps per class before rotation
    base_corruption: float = 0.1
    drift_corruption: float = 0.4
    # Training
    warmup_steps: int = 500
    active_steps: int = 2000
    drift_steps: int = 500
    lr: float = 0.003


class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = int(config.mlp_dim_exp_factor * 4 * config.n_embd)
        self.c_fc = nn.Linear(config.n_embd, dim, bias=config.bias)
        self.c_proj = nn.Linear(dim, config.n_embd, bias=config.bias)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x, {}


class TinyMoEModel(nn.Module):
    """Minimal model with one MoE layer for the experiment."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.moe = MoE(config, SimpleMLP)
        self.norm = nn.LayerNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, targets=None):
        x = self.embedding(x)
        moe_out, info = self.moe(x)
        x = self.norm(x + moe_out)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss, info


# ─── Difficulty Ladder ────────────────────────────────────────────────

class DifficultyLadder:
    """
    4 task classes that repeat in blocks, with drifting surface form.

    Classes:
      0: shift-1  (target = input + 1)
      1: shift-3  (target = input + 3)
      2: shift-7  (target = input + 7)
      3: double   (target = input * 2)

    Surface drift: token corruption that varies within each class block.
    Drift injection: at defined steps, corruption spikes.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.total_steps = (
            config.warmup_steps + config.active_steps + config.drift_steps
        )
        self.drift_start = config.warmup_steps + config.active_steps

    def get_batch(
        self, step: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Returns (task_class, inputs, targets)."""
        cfg = self.config

        # Which class? Rotate in blocks
        block_idx = step // cfg.class_block_size
        task_class = block_idx % cfg.num_classes

        # Generate clean inputs
        inputs = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

        # Generate targets based on class
        if task_class == 0:
            targets = (inputs + 1) % cfg.vocab_size
        elif task_class == 1:
            targets = (inputs + 3) % cfg.vocab_size
        elif task_class == 2:
            targets = (inputs + 7) % cfg.vocab_size
        else:
            targets = (inputs * 2) % cfg.vocab_size

        # Surface drift: corruption
        corruption = cfg.base_corruption
        if step >= self.drift_start:
            corruption = cfg.drift_corruption

        # Apply corruption to inputs (not targets)
        mask = torch.rand(cfg.batch_size, cfg.seq_len) < corruption
        noise = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        inputs = torch.where(mask, noise, inputs)

        return task_class, inputs, targets


# ─── Answer Cache Baseline ───────────────────────────────────────────

class AnswerCache:
    """
    Hash-based lookup table on input token sequences.
    Hit skips MoE entirely (cost=0). Miss pays full routing.
    Should fail under surface drift — that's the sanity check.
    """

    def __init__(self):
        self.cache: Dict[int, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def lookup(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        key = hash(inputs.cpu().numpy().tobytes())
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def store(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        key = hash(inputs.cpu().numpy().tobytes())
        self.cache[key] = outputs.detach()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ─── Run configurations ──────────────────────────────────────────────

def run_vanilla(
    config: ExperimentConfig,
    top_k_override: Optional[int] = None,
    label: str = "vanilla",
) -> Dict:
    """Run vanilla MoE (no Bob, no cache)."""
    cfg = dataclass_copy(config)
    if top_k_override:
        cfg.moe_num_experts_per_tok = top_k_override

    model = TinyMoEModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ladder = DifficultyLadder(cfg)

    traces = []
    for step in range(ladder.total_steps):
        task_class, inputs, targets = ladder.get_batch(step)

        model.train()
        logits, loss, info = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        invocations = cfg.moe_num_experts_per_tok * cfg.batch_size * cfg.seq_len
        traces.append({
            "step": step,
            "context_class": task_class,
            "path": "full",
            "expert_invocations": invocations,
            "loss": loss.item(),
        })

    return summarize_traces(traces, label, cfg)


def run_answer_cache(config: ExperimentConfig) -> Dict:
    """Run answer cache baseline."""
    cfg = config
    model = TinyMoEModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ladder = DifficultyLadder(cfg)
    cache = AnswerCache()

    traces = []
    for step in range(ladder.total_steps):
        task_class, inputs, targets = ladder.get_batch(step)

        # Try cache first
        cached = cache.lookup(inputs)
        if cached is not None:
            # Cache hit: cost = 0
            loss_val = F.cross_entropy(
                cached.view(-1, cached.size(-1)), targets.view(-1)
            ).item()
            invocations = 0
            path = "cheap"
        else:
            # Cache miss: full routing
            model.train()
            logits, loss, info = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cache.store(inputs, logits.detach())
            loss_val = loss.item()
            invocations = cfg.moe_num_experts_per_tok * cfg.batch_size * cfg.seq_len
            path = "full"

        traces.append({
            "step": step,
            "context_class": task_class,
            "path": path,
            "expert_invocations": invocations,
            "loss": loss_val,
        })

    result = summarize_traces(traces, "answer_cache", cfg)
    result["cache_hit_rate"] = round(cache.hit_rate, 4)
    return result


def run_bob(
    config: ExperimentConfig,
    success_multiplier: float = 1.2,
    label: str = "bob",
    seed: int = 42,
) -> Dict:
    """Run MoE + Bob (motif reuse + compound gate)."""
    torch.manual_seed(seed)
    cfg = config
    model = TinyMoEModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    adapter = SwissAdapter(
        model,
        moe_layers={0: model.moe},
        embed_fn=lambda m, x: m.embedding(x),
        head_fn=lambda m, x: m.output(x),
        norm_fn=lambda m, x: m.norm(x),
    )

    bob = BobSubstrate(
        adapter,
        gate_thresholds=GateThresholds(
            stability_min=0.6,
            debt_max=0.5,
            survival_min=0.7,
        ),
        warmup_steps=cfg.warmup_steps,
        governance_state="EQUILIBRIUM",
        stability_window=50,
        survival_half_life=200,
        success_multiplier=success_multiplier,
    )

    ladder = DifficultyLadder(cfg)

    for step in range(ladder.total_steps):
        task_class, inputs, targets = ladder.get_batch(step)

        # Bob decides cheap or expensive path (observation)
        trace = bob.step(inputs, targets, task_class, step)

        # Training: model always trains on full routing (Bob observes, doesn't train)
        model.train()
        _, train_loss, _ = model(inputs, targets)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Convert Bob's traces to summary with per-path loss breakdown
    raw_traces = [t.to_dict() for t in bob.traces]
    result = summarize_traces(raw_traces, label, cfg)

    # Add per-path loss breakdown (diagnostic)
    cheap_traces = [t for t in raw_traces if t["path"] == "cheap"]
    expensive_traces = [t for t in raw_traces if t["path"] == "full"]
    result["cheap_avg_loss"] = (
        round(sum(t["loss"] for t in cheap_traces) / len(cheap_traces), 4)
        if cheap_traces else None
    )
    result["expensive_avg_loss"] = (
        round(sum(t["loss"] for t in expensive_traces) / len(expensive_traces), 4)
        if expensive_traces else None
    )
    result["success_multiplier"] = success_multiplier

    return result


# ─── Analysis ─────────────────────────────────────────────────────────

def summarize_traces(
    traces: List[Dict], label: str, config: ExperimentConfig
) -> Dict:
    """Summarize traces into Pareto data and statistics."""
    window_size = 50
    pareto_data = []
    cheap_fraction_data = []

    for i in range(0, len(traces), window_size):
        window = traces[i : i + window_size]
        if not window:
            continue

        avg_cost = sum(t["expert_invocations"] for t in window) / len(window)
        avg_loss = sum(t["loss"] for t in window) / len(window)
        cheap_count = sum(1 for t in window if t["path"] == "cheap")
        cheap_frac = cheap_count / len(window)

        pareto_data.append({
            "window_start": window[0]["step"],
            "avg_cost": round(avg_cost, 2),
            "avg_loss": round(avg_loss, 4),
        })

        cheap_fraction_data.append({
            "window_start": window[0]["step"],
            "cheap_fraction": round(cheap_frac, 4),
        })

    # Overall stats
    full_cost = config.moe_num_experts_per_tok * config.batch_size * config.seq_len
    drift_start = config.warmup_steps + config.active_steps

    # Active phase: steps [warmup, warmup+active)
    active_traces = [t for t in traces
                     if config.warmup_steps <= t["step"] < drift_start]
    # Drift phase: steps [warmup+active, end)
    drift_traces = [t for t in traces if t["step"] >= drift_start]

    def phase_stats(phase_traces):
        if not phase_traces:
            return {"avg_cost": 0, "avg_loss": 0, "cheap_fraction": 0}
        return {
            "avg_cost": round(
                sum(t["expert_invocations"] for t in phase_traces) / len(phase_traces), 2
            ),
            "avg_loss": round(
                sum(t["loss"] for t in phase_traces) / len(phase_traces), 4
            ),
            "cheap_fraction": round(
                sum(1 for t in phase_traces if t["path"] == "cheap") / len(phase_traces), 4
            ),
        }

    active = phase_stats(active_traces)
    drift = phase_stats(drift_traces)

    return {
        "label": label,
        "total_steps": len(traces),
        "full_cost_per_step": full_cost,
        # Active phase is the real measure
        "avg_cost_final": active["avg_cost"],
        "avg_loss_final": active["avg_loss"],
        "cheap_fraction_final": active["cheap_fraction"],
        "cost_reduction_pct": round(
            (1 - active["avg_cost"] / full_cost) * 100, 1
        ) if full_cost > 0 else 0.0,
        # Drift phase for resilience check
        "drift_avg_cost": drift["avg_cost"],
        "drift_avg_loss": drift["avg_loss"],
        "drift_cheap_fraction": drift["cheap_fraction"],
        "pareto_data": pareto_data,
        "cheap_fraction_data": cheap_fraction_data,
    }


def dataclass_copy(dc):
    """Create a mutable copy of a frozen-like dataclass."""
    return ExperimentConfig(**{
        f.name: getattr(dc, f.name)
        for f in dc.__dataclass_fields__.values()
    })


def print_comparison(results: List[Dict], config: ExperimentConfig) -> None:
    """Print side-by-side comparison table."""
    full_cost = config.moe_num_experts_per_tok * config.batch_size * config.seq_len

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print(f"  Model: {config.moe_num_experts}E top-{config.moe_num_experts_per_tok}")
    print(f"  Full cost/step: {full_cost}")
    print(f"  Steps: warmup={config.warmup_steps}, active={config.active_steps}, drift={config.drift_steps}")
    print("=" * 80)

    # Header
    print(f"\n{'Config':<20} {'Avg Cost':>10} {'Avg Loss':>10} {'Cheap%':>8} {'Cost Red%':>10}")
    print("-" * 62)

    for r in results:
        print(
            f"{r['label']:<20} "
            f"{r['avg_cost_final']:>10.1f} "
            f"{r['avg_loss_final']:>10.4f} "
            f"{r['cheap_fraction_final']*100:>7.1f}% "
            f"{r['cost_reduction_pct']:>9.1f}%"
        )

    # Sanity check: answer cache should have low hit rate
    for r in results:
        if r["label"] == "answer_cache":
            hit_rate = r.get("cache_hit_rate", 0)
            if hit_rate > 0.1:
                print(f"\n  WARNING: Answer cache hit rate = {hit_rate*100:.1f}%")
                print("  Ladder may be too easy. Drift should kill answer cache.")
            else:
                print(f"\n  SANITY OK: Answer cache hit rate = {hit_rate*100:.1f}% (drift kills it)")

    # Pareto summary
    print("\n\nPARETO TRAJECTORY (50-step windows, last 10 windows):")
    bob_result = next((r for r in results if r["label"] == "bob"), None)
    if bob_result and bob_result["pareto_data"]:
        print(f"  {'Window':>8} {'Cost':>10} {'Loss':>10}")
        print("  " + "-" * 32)
        for p in bob_result["pareto_data"][-10:]:
            print(f"  {p['window_start']:>8} {p['avg_cost']:>10.1f} {p['avg_loss']:>10.4f}")

    # Cheap path trajectory
    if bob_result and bob_result["cheap_fraction_data"]:
        print("\n\nCHEAP PATH FRACTION (50-step windows, last 10 windows):")
        print(f"  {'Window':>8} {'Cheap%':>8}")
        print("  " + "-" * 20)
        for c in bob_result["cheap_fraction_data"][-10:]:
            print(f"  {c['window_start']:>8} {c['cheap_fraction']*100:>7.1f}%")


def plot_results(results: List[Dict], config: ExperimentConfig, filename: str) -> None:
    """Generate the two plots. Falls back to ASCII if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Pareto frontier (cost vs quality) over time
        for r in results:
            if r["pareto_data"]:
                steps = [p["window_start"] for p in r["pareto_data"]]
                costs = [p["avg_cost"] for p in r["pareto_data"]]
                losses = [p["avg_loss"] for p in r["pareto_data"]]
                ax1.plot(steps, costs, label=r["label"], alpha=0.8)

        ax1.set_xlabel("Step")
        ax1.set_ylabel("Avg Expert Invocations (Cost)")
        ax1.set_title(f"Cost Over Time ({config.moe_num_experts}E top-{config.moe_num_experts_per_tok})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cheap path fraction (Bob only)
        bob_result = next((r for r in results if r["label"] == "bob"), None)
        if bob_result and bob_result["cheap_fraction_data"]:
            steps = [c["window_start"] for c in bob_result["cheap_fraction_data"]]
            fracs = [c["cheap_fraction"] * 100 for c in bob_result["cheap_fraction_data"]]
            ax2.plot(steps, fracs, color="green", linewidth=2)
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.axvline(
                x=config.warmup_steps, color="blue", linestyle="--",
                alpha=0.5, label="Warmup ends"
            )
            ax2.axvline(
                x=config.warmup_steps + config.active_steps,
                color="red", linestyle="--", alpha=0.5, label="Drift starts"
            )

            # Add quality on secondary axis
            if bob_result["pareto_data"]:
                ax2b = ax2.twinx()
                loss_steps = [p["window_start"] for p in bob_result["pareto_data"]]
                losses = [p["avg_loss"] for p in bob_result["pareto_data"]]
                ax2b.plot(loss_steps, losses, color="orange", alpha=0.5, label="Loss")
                ax2b.set_ylabel("Loss", color="orange")

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cheap Path %")
        ax2.set_title("Bob: Cheap Path Fraction + Quality")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nPlot saved to {filename}")

    except ImportError:
        print("\nmatplotlib not available — skipping plot generation")
        print("Install with: pip install matplotlib")


# ─── Main ─────────────────────────────────────────────────────────────

def run_experiment(config: ExperimentConfig) -> List[Dict]:
    """Run baselines + Bob with default multiplier."""
    results = []

    print(f"\n--- Running: top-{config.moe_num_experts_per_tok} vanilla ---")
    results.append(run_vanilla(config, label=f"top{config.moe_num_experts_per_tok}_vanilla"))

    if config.moe_num_experts_per_tok != 4 and config.moe_num_experts >= 4:
        print(f"\n--- Running: top-4 vanilla ---")
        results.append(run_vanilla(config, top_k_override=4, label="top4_vanilla"))

    print(f"\n--- Running: answer cache ---")
    results.append(run_answer_cache(config))

    print(f"\n--- Running: MoE + Bob ---")
    results.append(run_bob(config))

    return results


def run_pareto_sweep(config: ExperimentConfig) -> List[Dict]:
    """Sweep success_multiplier to trace the Pareto frontier."""
    multipliers = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
    sweep_results = []

    for mult in multipliers:
        print(f"\n--- Bob (multiplier={mult}) ---")
        result = run_bob(
            config,
            success_multiplier=mult,
            label=f"bob_m{mult}",
            seed=42,
        )
        sweep_results.append(result)

    return sweep_results


def print_pareto_frontier(
    sweep_results: List[Dict],
    vanilla_result: Dict,
    config: ExperimentConfig,
) -> None:
    """Print the Pareto frontier from a multiplier sweep."""
    full_cost = config.moe_num_experts_per_tok * config.batch_size * config.seq_len

    print(f"\n{'='*90}")
    print(f"PARETO FRONTIER: {config.moe_num_experts}E top-{config.moe_num_experts_per_tok}")
    print(f"{'='*90}")

    # Active phase (the real measure)
    print(f"\nACTIVE PHASE (steps {config.warmup_steps}-{config.warmup_steps + config.active_steps})")
    print(f"Vanilla baseline: cost={vanilla_result['avg_cost_final']:.0f}, loss={vanilla_result['avg_loss_final']:.4f}")

    print(f"\n{'Mult':>6} {'Cost':>8} {'Loss':>8} {'Cheap%':>8} "
          f"{'CostRed%':>9} {'LossDelta':>10} {'CheapLoss':>10} {'ExpLoss':>10}")
    print("-" * 80)

    for r in sweep_results:
        loss_delta = r['avg_loss_final'] - vanilla_result['avg_loss_final']
        cheap_loss = r.get('cheap_avg_loss', '-')
        exp_loss = r.get('expensive_avg_loss', '-')
        cl_str = f"{cheap_loss:.4f}" if isinstance(cheap_loss, float) else "  n/a"
        el_str = f"{exp_loss:.4f}" if isinstance(exp_loss, float) else "  n/a"

        print(
            f"{r['success_multiplier']:>6.1f} "
            f"{r['avg_cost_final']:>8.0f} "
            f"{r['avg_loss_final']:>8.4f} "
            f"{r['cheap_fraction_final']*100:>7.1f}% "
            f"{r['cost_reduction_pct']:>8.1f}% "
            f"{loss_delta:>+10.4f} "
            f"{cl_str:>10} "
            f"{el_str:>10}"
        )

    # Drift phase (resilience check)
    print(f"\nDRIFT PHASE (steps {config.warmup_steps + config.active_steps}-{config.warmup_steps + config.active_steps + config.drift_steps})")
    print(f"Vanilla drift: cost={vanilla_result['drift_avg_cost']:.0f}, loss={vanilla_result['drift_avg_loss']:.4f}")

    print(f"\n{'Mult':>6} {'Cost':>8} {'Loss':>8} {'Cheap%':>8} {'LossDelta':>10}")
    print("-" * 50)

    for r in sweep_results:
        drift_delta = r['drift_avg_loss'] - vanilla_result['drift_avg_loss']
        print(
            f"{r['success_multiplier']:>6.1f} "
            f"{r['drift_avg_cost']:>8.0f} "
            f"{r['drift_avg_loss']:>8.4f} "
            f"{r['drift_cheap_fraction']*100:>7.1f}% "
            f"{drift_delta:>+10.4f}"
        )

    # Best Pareto point
    print(f"\nPARETO ANALYSIS:")
    best = None
    for r in sweep_results:
        loss_delta = r['avg_loss_final'] - vanilla_result['avg_loss_final']
        if loss_delta < 0.1 and r['cost_reduction_pct'] > 0:
            if best is None or r['cost_reduction_pct'] > best['cost_reduction_pct']:
                best = r

    if best:
        delta = best['avg_loss_final'] - vanilla_result['avg_loss_final']
        print(f"  BEST: mult={best['success_multiplier']}, "
              f"cost_red={best['cost_reduction_pct']:.1f}%, "
              f"loss_delta={delta:+.4f}")
    else:
        print(f"  NO PARETO-OPTIMAL POINT (cheap path always degrades quality > +0.1)")
        print(f"  Expert specialization insufficient for 1-expert cheap path on this model.")


def check_eligibility(
    config: ExperimentConfig,
    seed: int = 42,
    epsilon: float = 0.05,
) -> Dict:
    """
    Eligibility gate: does this backend have class-conditional routing structure?

    Four signals measured:
      1. Dominance: per-class top-1 expert frequency vs uniform baseline
      2. Inter-class KL divergence: do different classes route differently?
      3. Pair concentration: per-class variance of top-k pair frequency
      4. Temporal stability: does class-conditioned routing stay stable across windows?

    Eligibility answers: "Is there reusable structure at the granularity Bob is caching?"
    Not just: "Is one expert dominant?"
    """
    torch.manual_seed(seed)
    model = TinyMoEModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    ladder = DifficultyLadder(config)

    total = config.warmup_steps + config.active_steps
    for step in range(total):
        task_class, inputs, targets = ladder.get_batch(step)
        model.train()
        _, loss, _ = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Collect routing data ---
    # Need enough steps for each class to appear 2+ times for temporal stability.
    # With class_block_size=50 and 4 classes, one rotation = 200 steps.
    # Use 400 steps = 2 full rotations, with 25-step windows.
    model.eval()
    num_experts = config.moe_num_experts
    top_k = config.moe_num_experts_per_tok
    uniform = 1.0 / num_experts

    measure_steps = min(400, total - config.warmup_steps)
    window_size = 25
    num_windows = measure_steps // window_size

    # Per-class, per-window routing data
    per_class_windows = defaultdict(lambda: [
        {"expert_counts": defaultdict(int), "pair_counts": defaultdict(int),
         "total_selections": 0, "top1_weights": []}
        for _ in range(num_windows)
    ])
    # Per-class aggregate
    per_class_agg = defaultdict(lambda: {
        "expert_counts": defaultdict(int),
        "top1_weights": [],
        "total_selections": 0,
    })

    start_step = total - measure_steps
    for step in range(start_step, total):
        task_class, inputs, targets = ladder.get_batch(step)
        window_idx = min((step - start_step) // window_size, num_windows - 1)

        with torch.no_grad():
            x = model.embedding(inputs)
            router_logits = model.moe.router(x.view(-1, x.shape[-1]))
            probs = F.softmax(router_logits, dim=-1)
            top_vals, top_ids = probs.topk(top_k, dim=-1)

            w = per_class_windows[task_class][window_idx]
            agg = per_class_agg[task_class]

            for i, row in enumerate(top_ids):
                pair = tuple(sorted(int(e) for e in row.tolist()))
                w["pair_counts"][pair] += 1
                for eid in row.tolist():
                    eid = int(eid)
                    w["expert_counts"][eid] += 1
                    w["total_selections"] += 1
                    agg["expert_counts"][eid] += 1
                    agg["total_selections"] += 1

            agg["top1_weights"].extend(top_vals[:, 0].tolist())
            w["top1_weights"].extend(top_vals[:, 0].tolist())

    # --- Signal 1: Dominance ---
    class_metrics = {}
    dominances = []
    class_distributions = {}  # for KL computation

    for cc in sorted(per_class_agg.keys()):
        counts = per_class_agg[cc]["expert_counts"]
        total_sel = per_class_agg[cc]["total_selections"]
        top1_w = per_class_agg[cc]["top1_weights"]

        sorted_experts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_fraction = sorted_experts[0][1] / total_sel if total_sel > 0 else 0
        avg_top1_w = sum(top1_w) / len(top1_w) if top1_w else 0

        # Normalized distribution (for KL)
        dist = [0.0] * num_experts
        for eid, cnt in counts.items():
            dist[eid] = cnt / total_sel if total_sel > 0 else uniform
        class_distributions[cc] = dist

        class_metrics[cc] = {
            "top_expert": sorted_experts[0][0],
            "dominance": round(top_fraction, 4),
            "avg_top1_weight": round(avg_top1_w, 4),
            "expert_distribution": {
                eid: round(cnt / total_sel, 3) for eid, cnt in sorted_experts
            },
        }
        dominances.append(top_fraction)

    mean_dominance = sum(dominances) / len(dominances) if dominances else 0

    # --- Signal 2: Inter-class KL divergence ---
    # Average pairwise KL(class_i || class_j)
    # High KL = classes route very differently = good for motif caching
    kl_divs = []
    class_ids = sorted(class_distributions.keys())
    for i, c1 in enumerate(class_ids):
        for c2 in class_ids[i + 1:]:
            p = class_distributions[c1]
            q = class_distributions[c2]
            # Symmetrized KL with smoothing
            kl = 0.0
            for k in range(num_experts):
                pk = max(p[k], 1e-10)
                qk = max(q[k], 1e-10)
                kl += 0.5 * (pk * math.log(pk / qk) + qk * math.log(qk / pk))
            kl_divs.append(kl)

    mean_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0

    # --- Signal 3: Pair concentration ---
    # Per-class: what fraction of tokens use the most common top-k pair?
    # High concentration = routing is stable within each class
    pair_concentrations = []
    for cc in sorted(per_class_windows.keys()):
        for w in per_class_windows[cc]:
            if not w["pair_counts"]:
                continue
            total_pairs = sum(w["pair_counts"].values())
            top_pair_count = max(w["pair_counts"].values())
            pair_concentrations.append(top_pair_count / total_pairs)

    mean_pair_concentration = (
        sum(pair_concentrations) / len(pair_concentrations)
        if pair_concentrations else 0
    )
    # Expected pair concentration under uniform: 1 / C(n, k)
    from math import comb
    num_possible_pairs = comb(num_experts, top_k)
    uniform_pair_concentration = 1.0 / num_possible_pairs

    # --- Signal 4: Temporal stability ---
    # Per-class: how much does the expert distribution change between
    # separate appearances of the same class? (handles class rotation)
    # Collect per-class step-level routing, then split into chunks.
    temporal_stabilities = []
    for cc in sorted(per_class_windows.keys()):
        # Gather all window data that has observations for this class
        class_window_dists = []
        for w in per_class_windows[cc]:
            if w["total_selections"] < 10:  # need enough tokens for a meaningful dist
                continue
            dist = [0.0] * num_experts
            for eid, cnt in w["expert_counts"].items():
                dist[eid] = cnt / w["total_selections"]
            class_window_dists.append(dist)

        if len(class_window_dists) < 2:
            continue

        # Compute mean distribution across this class's windows
        mean_dist = [
            sum(d[e] for d in class_window_dists) / len(class_window_dists)
            for e in range(num_experts)
        ]
        # Average L1 distance from mean (lower = more stable)
        l1_distances = []
        for d in class_window_dists:
            l1 = sum(abs(d[e] - mean_dist[e]) for e in range(num_experts))
            l1_distances.append(l1)

        avg_l1 = sum(l1_distances) / len(l1_distances)
        # Stability = 1 - normalized L1 (max L1 for distributions = 2.0)
        temporal_stabilities.append(1.0 - avg_l1 / 2.0)

    mean_temporal_stability = (
        sum(temporal_stabilities) / len(temporal_stabilities)
        if temporal_stabilities else 0
    )

    # --- Eligibility verdict ---
    # Primary: dominance must exceed uniform
    dominance_pass = mean_dominance > (uniform + epsilon)
    # Secondary: inter-class KL must show differentiation
    kl_pass = mean_kl > 0.01  # even small KL means different distributions
    # Tertiary: pair concentration above uniform
    pair_pass = mean_pair_concentration > uniform_pair_concentration * 1.5

    # Eligible if primary passes, OR if secondary + tertiary both pass
    # (catches cases where dominance is moderate but routing is class-conditional)
    eligible = dominance_pass or (kl_pass and pair_pass)

    return {
        "eligible": eligible,
        # Signal 1: Dominance
        "mean_dominance": round(mean_dominance, 4),
        "uniform_baseline": round(uniform, 4),
        "dominance_threshold": round(uniform + epsilon, 4),
        "dominance_margin": round(mean_dominance - uniform, 4),
        "dominance_pass": dominance_pass,
        # Signal 2: Inter-class KL
        "mean_interclass_kl": round(mean_kl, 6),
        "kl_threshold": 0.01,
        "kl_pass": kl_pass,
        # Signal 3: Pair concentration
        "mean_pair_concentration": round(mean_pair_concentration, 4),
        "uniform_pair_concentration": round(uniform_pair_concentration, 4),
        "pair_concentration_margin": round(
            mean_pair_concentration - uniform_pair_concentration, 4
        ),
        "pair_pass": pair_pass,
        # Signal 4: Temporal stability
        "mean_temporal_stability": round(mean_temporal_stability, 4),
        # Meta
        "num_experts": num_experts,
        "top_k": top_k,
        "training_steps": total,
        "per_class": class_metrics,
    }


def print_eligibility(result: Dict, label: str) -> None:
    """Print eligibility check results."""
    verdict = "ELIGIBLE" if result["eligible"] else "INELIGIBLE"
    print(f"\n{'='*70}")
    print(f"ELIGIBILITY CHECK: {label}")
    print(f"{'='*70}")
    print(f"  Verdict: {verdict}")
    print(f"  Training: {result['training_steps']} steps, "
          f"{result['num_experts']}E top-{result['top_k']}")

    # Signal 1: Dominance
    dp = "PASS" if result["dominance_pass"] else "FAIL"
    print(f"\n  Signal 1 — Dominance [{dp}]")
    print(f"    Mean dominance:   {result['mean_dominance']:.4f}")
    print(f"    Uniform baseline: {result['uniform_baseline']:.4f}")
    print(f"    Threshold:        {result['dominance_threshold']:.4f}")
    print(f"    Margin:           {result['dominance_margin']:+.4f}")

    # Signal 2: Inter-class KL
    kp = "PASS" if result["kl_pass"] else "FAIL"
    print(f"\n  Signal 2 — Inter-class KL divergence [{kp}]")
    print(f"    Mean pairwise KL: {result['mean_interclass_kl']:.6f}")
    print(f"    Threshold:        {result['kl_threshold']:.6f}")
    if result["mean_interclass_kl"] < 0.001:
        print(f"    (Classes route identically)")
    elif result["mean_interclass_kl"] < 0.01:
        print(f"    (Negligible differentiation)")
    else:
        print(f"    (Meaningful class-conditional routing)")

    # Signal 3: Pair concentration
    pp = "PASS" if result["pair_pass"] else "FAIL"
    print(f"\n  Signal 3 — Top-k pair concentration [{pp}]")
    print(f"    Mean concentration: {result['mean_pair_concentration']:.4f}")
    print(f"    Uniform baseline:   {result['uniform_pair_concentration']:.4f}")
    print(f"    Margin:             {result['pair_concentration_margin']:+.4f}")

    # Signal 4: Temporal stability
    print(f"\n  Signal 4 — Temporal stability")
    print(f"    Mean stability: {result['mean_temporal_stability']:.4f}")
    print(f"    (1.0 = perfectly stable, 0.0 = maximally variable)")

    # Per-class breakdown
    print(f"\n  Per-class routing:")
    for cc, m in result["per_class"].items():
        dist_str = " ".join(
            f"E{eid}:{frac:.0%}" for eid, frac in m["expert_distribution"].items()
        )
        print(f"    Class {cc}: {dist_str}  "
              f"| dom={m['dominance']:.1%} top1_w={m['avg_top1_weight']:.3f}")

    if not result["eligible"]:
        print(f"\n  CONCLUSION: No reusable class-conditional routing structure.")
        print(f"  Motif caching at this granularity will learn noise, not structure.")


if __name__ == "__main__":
    configs = [
        ("4E top-2", ExperimentConfig(moe_num_experts=4, moe_num_experts_per_tok=2)),
        ("8E top-2", ExperimentConfig(moe_num_experts=8, moe_num_experts_per_tok=2)),
    ]

    for config_name, config in configs:
        print(f"\n{'#'*80}")
        print(f"# {config_name}")
        print(f"{'#'*80}")

        # STEP A: Eligibility check (hard gate)
        eligibility = check_eligibility(config)
        print_eligibility(eligibility, config_name)

        # STEP B+C: Run Bob once to confirm
        # Whether eligible or not, one run shows the mechanism working correctly
        torch.manual_seed(42)
        print(f"\n--- Vanilla baseline ---")
        vanilla = run_vanilla(config, label="vanilla")
        print(f"  cost={vanilla['avg_cost_final']:.0f}, loss={vanilla['avg_loss_final']:.4f}")

        print(f"\n--- Bob (multiplier=1.5) ---")
        bob_result = run_bob(config, success_multiplier=1.5, label="bob", seed=42)

        delta = bob_result['avg_loss_final'] - vanilla['avg_loss_final']
        print(f"\n  ACTIVE PHASE:")
        print(f"    Vanilla:  cost={vanilla['avg_cost_final']:.0f}, loss={vanilla['avg_loss_final']:.4f}")
        print(f"    Bob:      cost={bob_result['avg_cost_final']:.0f}, loss={bob_result['avg_loss_final']:.4f}")
        print(f"    Cheap fraction: {bob_result['cheap_fraction_final']*100:.1f}%")
        print(f"    Cost reduction: {bob_result['cost_reduction_pct']:.1f}%")
        print(f"    Loss delta:     {delta:+.4f}")

        print(f"\n  DRIFT PHASE:")
        print(f"    Vanilla:  cost={vanilla['drift_avg_cost']:.0f}, loss={vanilla['drift_avg_loss']:.4f}")
        print(f"    Bob:      cost={bob_result['drift_avg_cost']:.0f}, loss={bob_result['drift_avg_loss']:.4f}")
        drift_delta = bob_result['drift_avg_loss'] - vanilla['drift_avg_loss']
        print(f"    Cheap fraction: {bob_result['drift_cheap_fraction']*100:.1f}%")
        print(f"    Loss delta:     {drift_delta:+.4f}")

        if bob_result.get('cheap_avg_loss') is not None:
            print(f"\n  PATH BREAKDOWN:")
            print(f"    Cheap path avg loss:     {bob_result['cheap_avg_loss']:.4f}")
            print(f"    Expensive path avg loss: {bob_result['expensive_avg_loss']:.4f}")

        # STEP D: Verdict
        if not eligibility["eligible"]:
            print(f"\n  VERDICT: Backend is INELIGIBLE for motif-based cost reduction.")
            print(f"  Mean dominance ({eligibility['mean_dominance']:.3f}) ≈ "
                  f"uniform ({eligibility['uniform_baseline']:.3f}).")
            print(f"  Class-level motif caching cannot extract savings from")
            print(f"  class-agnostic routing. Pivot to backend with known")
            print(f"  class-conditional specialization (e.g., Mixtral).")
        else:
            print(f"\n  VERDICT: Backend is ELIGIBLE. Routing shows per-class structure.")
            if bob_result['cost_reduction_pct'] > 5 and delta < 0.1:
                print(f"  PARETO CURVE BENDS: {bob_result['cost_reduction_pct']:.1f}% "
                      f"cost reduction at {delta:+.4f} quality delta.")
            else:
                print(f"  But curve not yet bending. Consider threshold tuning or")
                print(f"  longer training to strengthen specialization.")

    print("\n\nDone.")
