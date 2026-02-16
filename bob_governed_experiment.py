"""
Bob Phase 1 Governed Experiment: halting + ledgers + medium clock.

Runs identical seeds with governor enabled vs disabled,
then diffs the results to prove halting metrics move.

Metrics:
- commits_blocked: governor BLOCK/ESCALATE count (must be > 0)
- commit_then_violate_within_K: fraction of commits followed by
  a violation within K same-class steps
- escalation_rate: fraction of steps that are governor blocks

Definitions (crisp):
- A "commit" is any step where path == "cheap" AND governor_decision == "allow"
- A "violation" is loss > baseline * success_multiplier for that context class
- "within K" means K subsequent steps in the SAME context_class
"""

import sys
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moe import MoE
from backends.swiss_adapter import SwissAdapter
from bob_core.substrate import BobSubstrate
from bob_core.motifs import GateThresholds
from bob_core.ledgers import BobCore
from bob_core.medium_clock import MediumClock
from bob_core.governor import BobGovernor
from bob_core.promotion import PromotionGate


# ─── Model (same as bob_experiment.py) ──────────────────────────────

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
    num_classes: int = 4
    class_block_size: int = 50
    base_corruption: float = 0.1
    drift_corruption: float = 0.4
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


class DifficultyLadder:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.total_steps = config.warmup_steps + config.active_steps + config.drift_steps
        self.drift_start = config.warmup_steps + config.active_steps

    def get_batch(self, step: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        cfg = self.config
        block_idx = step // cfg.class_block_size
        task_class = block_idx % cfg.num_classes
        inputs = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        if task_class == 0:
            targets = (inputs + 1) % cfg.vocab_size
        elif task_class == 1:
            targets = (inputs + 3) % cfg.vocab_size
        elif task_class == 2:
            targets = (inputs + 7) % cfg.vocab_size
        else:
            targets = (inputs * 2) % cfg.vocab_size
        corruption = cfg.base_corruption
        if step >= self.drift_start:
            corruption = cfg.drift_corruption
        mask = torch.rand(cfg.batch_size, cfg.seq_len) < corruption
        noise = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        inputs = torch.where(mask, noise, inputs)
        return task_class, inputs, targets


# ─── Halting Metrics ─────────────────────────────────────────────────

def compute_commit_then_violate(
    traces: List[Dict], success_multiplier: float = 1.2, K: int = 5,
) -> Dict:
    """Compute commit_then_violate_within_K.

    Definitions:
    - A "commit" is any step where path == "cheap" AND governor_decision == "allow"
      (or path == "cheap" when no governor is present)
    - A "violation" is loss > baseline_loss * success_multiplier
      Baseline = running average of expensive-path losses for that context class
    - "within K" means K subsequent steps in the SAME context_class
    """
    # Group traces by context_class, preserving order
    by_class: Dict[int, List[Dict]] = defaultdict(list)
    for t in traces:
        by_class[t["context_class"]].append(t)

    # Track expensive-path running averages per class
    expensive_sums: Dict[int, float] = defaultdict(float)
    expensive_counts: Dict[int, int] = defaultdict(int)

    total_commits = 0
    violated_commits = 0

    for cc, class_traces in by_class.items():
        for i, t in enumerate(class_traces):
            # Update expensive baseline
            if t["path"] == "full":
                expensive_sums[cc] += t["loss"]
                expensive_counts[cc] += 1

            # Check if this is a commit
            is_commit = t["path"] == "cheap"
            if t.get("governor_decision") is not None:
                is_commit = is_commit and t["governor_decision"] == "allow"

            if not is_commit:
                continue

            total_commits += 1

            # Look ahead K same-class steps for violation
            if expensive_counts[cc] == 0:
                continue
            baseline = expensive_sums[cc] / expensive_counts[cc]
            threshold = baseline * success_multiplier

            for j in range(i + 1, min(i + 1 + K, len(class_traces))):
                future = class_traces[j]
                if future["loss"] > threshold:
                    violated_commits += 1
                    break

    return {
        "total_commits": total_commits,
        "violated_commits": violated_commits,
        "commit_then_violate_rate": (
            round(violated_commits / total_commits, 4)
            if total_commits > 0 else 0.0
        ),
        "K": K,
        "success_multiplier": success_multiplier,
    }


# ─── Run Functions ───────────────────────────────────────────────────

def run_bob_governed(
    config: ExperimentConfig,
    seed: int = 42,
    enable_governor: bool = True,
    label: str = "governed",
) -> Dict:
    """Run Bob with or without governor, same seed."""
    torch.manual_seed(seed)
    model = TinyMoEModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    adapter = SwissAdapter(
        model,
        moe_layers={0: model.moe},
        embed_fn=lambda m, x: m.embedding(x),
        head_fn=lambda m, x: m.output(x),
        norm_fn=lambda m, x: m.norm(x),
    )

    # Build optional components
    bob_core = None
    governor = None
    medium_clock = None
    promotion_gate = None

    if enable_governor:
        bob_core = BobCore(success_multiplier=1.2)
        medium_clock = MediumClock(ema_alpha=0.1, instability_threshold=0.5)
        governor = BobGovernor(
            bob_core, medium_clock,
            fast_threshold=0.8,
            medium_threshold=0.5,
            debt_threshold=0.7,
        )
        promotion_gate = PromotionGate(stability_window=20)

    bob = BobSubstrate(
        adapter,
        gate_thresholds=GateThresholds(
            stability_min=0.6,
            debt_max=0.5,
            survival_min=0.7,
        ),
        warmup_steps=config.warmup_steps,
        governance_state="EQUILIBRIUM",
        bob_core=bob_core,
        governor=governor,
        medium_clock=medium_clock,
        promotion_gate=promotion_gate,
        stability_window=50,
        survival_half_life=200,
        success_multiplier=1.5,
    )

    ladder = DifficultyLadder(config)

    for step in range(ladder.total_steps):
        task_class, inputs, targets = ladder.get_batch(step)
        trace = bob.step(inputs, targets, task_class, step)

        model.train()
        _, train_loss, _ = model(inputs, targets)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    raw_traces = [t.to_dict() for t in bob.traces]

    # Compute metrics
    drift_start = config.warmup_steps + config.active_steps
    active_traces = [t for t in raw_traces if config.warmup_steps <= t["step"] < drift_start]

    cheap_count = sum(1 for t in active_traces if t["path"] == "cheap")
    total_active = len(active_traces)

    # Proposed/blocked/authorized commit counts
    proposed_commits = 0  # gate passed (would have been cheap without governor)
    blocked_commits = 0
    authorized_commits = 0
    authorized_losses = []

    commits_blocked = 0
    governor_allows = 0
    if enable_governor:
        commits_blocked = governor.blocks_count
        for t in raw_traces:
            gd = t.get("governor_decision")
            if gd is not None:
                proposed_commits += 1
                if gd == "allow":
                    governor_allows += 1
                    authorized_commits += 1
                    authorized_losses.append(t["loss"])
                else:
                    blocked_commits += 1

    ctv = compute_commit_then_violate(raw_traces, success_multiplier=1.2, K=5)

    # Authorized commit quality (avg loss/baseline on authorized commits)
    authorized_commit_quality = None
    if authorized_losses and bob_core:
        # Average quality relative to baseline across all context classes
        qualities = []
        for t in raw_traces:
            if t.get("governor_decision") == "allow":
                bl = bob_core.commitments.baseline_loss(t["context_class"])
                if bl != float("inf") and bl > 0:
                    qualities.append(t["loss"] / bl)
        authorized_commit_quality = round(
            sum(qualities) / len(qualities), 4
        ) if qualities else None

    # Routing region tracking
    all_regions_visited = set()
    for t in raw_traces:
        region = tuple(sorted(t["expert_ids"]))
        if region:
            all_regions_visited.add(region)

    # Escalation rate per context class
    escalation_by_class: Dict[int, Dict] = {}
    for cc in range(config.num_classes):
        class_traces = [t for t in active_traces if t["context_class"] == cc]
        if not class_traces:
            continue
        blocked = sum(
            1 for t in class_traces
            if t.get("governor_decision") in ("block", "escalate")
        )
        escalation_by_class[cc] = {
            "total": len(class_traces),
            "blocked": blocked,
            "rate": round(blocked / len(class_traces), 4) if class_traces else 0.0,
        }

    # Promotion eligibility
    promo_eligible = {}
    if promotion_gate:
        for cc in range(config.num_classes):
            promo_eligible[cc] = {
                "eligible": promotion_gate.is_eligible(cc),
                "score": round(promotion_gate.eligibility_score(cc), 4),
            }

    # Medium clock final state
    medium_state = None
    if medium_clock:
        s = medium_clock.state
        medium_state = {
            "churn_ema": round(s.churn_ema, 4),
            "flipflop_ema": round(s.flipflop_ema, 4),
            "outcome_var_ema": round(s.outcome_var_ema, 4),
            "escalation_ema": round(s.escalation_ema, 4),
            "activation": round(medium_clock.activation, 4),
        }

    # Scar summary
    scar_summary = None
    if bob_core:
        scars = bob_core.scars._scars
        scarred_regions = len(scars)
        visited_regions = len(all_regions_visited)
        scar_summary = {
            "total_scars": scarred_regions,
            "total_visited": visited_regions,
            "scar_saturation": round(
                scarred_regions / visited_regions, 4
            ) if visited_regions > 0 else 0.0,
            "total_debt": round(bob_core.scars.total_debt(ladder.total_steps), 4),
            "regions": {
                str(region): {
                    "severity": round(s.severity, 4),
                    "triggers": s.trigger_count,
                    "decayed": round(s.decayed_severity(ladder.total_steps), 4),
                }
                for region, s in scars.items()
            },
        }

    avg_loss = sum(t["loss"] for t in active_traces) / total_active if total_active else 0
    full_cost = config.moe_num_experts_per_tok * config.batch_size * config.seq_len

    return {
        "label": label,
        "governor_enabled": enable_governor,
        "seed": seed,
        "total_steps": len(raw_traces),
        "active_cheap_count": cheap_count,
        "active_cheap_fraction": round(cheap_count / total_active, 4) if total_active else 0,
        "avg_loss": round(avg_loss, 4),
        "full_cost": full_cost,
        # Halting metrics
        "commits_blocked": commits_blocked,
        "governor_allows": governor_allows,
        "proposed_commits": proposed_commits,
        "authorized_commits": authorized_commits,
        "blocked_commits": blocked_commits,
        "authorized_commit_quality": authorized_commit_quality,
        "commit_then_violate": ctv,
        "escalation_by_class": escalation_by_class,
        # Region tracking
        "unique_regions_visited": len(all_regions_visited),
        # Components
        "promotion": promo_eligible,
        "medium_clock": medium_state,
        "scars": scar_summary,
    }


# ─── Diff Report ─────────────────────────────────────────────────────

def diff_report(baseline: Dict, governed: Dict) -> Dict:
    """Produce JSON diff between baseline (no governor) and governed runs."""
    report = {
        "baseline_seed": baseline["seed"],
        "governed_seed": governed["seed"],
        "same_seed": baseline["seed"] == governed["seed"],
    }

    # Core acceptance criteria
    report["commits_blocked"] = governed["commits_blocked"]
    report["commits_blocked_pass"] = governed["commits_blocked"] > 0

    b_ctv = baseline["commit_then_violate"]["commit_then_violate_rate"]
    g_ctv = governed["commit_then_violate"]["commit_then_violate_rate"]
    report["baseline_ctv_rate"] = b_ctv
    report["governed_ctv_rate"] = g_ctv
    report["ctv_improved"] = g_ctv < b_ctv
    report["ctv_delta"] = round(g_ctv - b_ctv, 4)

    # CTV sample size check
    g_authorized = governed["authorized_commits"]
    report["governed_ctv_sufficient_sample"] = g_authorized >= 20
    if g_authorized < 20:
        report["governed_ctv_note"] = (
            f"insufficient sample: {g_authorized} authorized commits (need >= 20)"
        )

    # Proposed/blocked/authorized commit counts
    report["proposed_commits"] = governed["proposed_commits"]
    report["authorized_commits"] = governed["authorized_commits"]
    report["blocked_commits"] = governed["blocked_commits"]
    report["blocked_commit_rate"] = round(
        governed["blocked_commits"] / governed["proposed_commits"], 4
    ) if governed["proposed_commits"] > 0 else 0.0
    report["authorized_commit_quality"] = governed["authorized_commit_quality"]

    # Loss comparison
    report["baseline_avg_loss"] = baseline["avg_loss"]
    report["governed_avg_loss"] = governed["avg_loss"]
    report["loss_delta"] = round(governed["avg_loss"] - baseline["avg_loss"], 4)

    # Cheap fraction comparison
    report["baseline_cheap_fraction"] = baseline["active_cheap_fraction"]
    report["governed_cheap_fraction"] = governed["active_cheap_fraction"]

    # Region tracking
    report["unique_regions_visited"] = governed["unique_regions_visited"]
    if governed["scars"]:
        report["scar_saturation"] = governed["scars"]["scar_saturation"]
        report["total_debt"] = governed["scars"]["total_debt"]

    # Escalation rate per class
    report["escalation_by_class"] = governed["escalation_by_class"]

    # Governor components
    report["scars"] = governed["scars"]
    report["medium_clock"] = governed["medium_clock"]
    report["promotion"] = governed["promotion"]

    return report


# ─── Multi-seed runner ───────────────────────────────────────────────

def run_multi_seed(config: ExperimentConfig, num_seeds: int = 20) -> Dict:
    """Run baseline vs governed across N seeds. Report distributions."""
    import statistics

    all_baseline = []
    all_governed = []
    all_diffs = []

    for i in range(num_seeds):
        seed = i * 7 + 1  # Deterministic, spread out
        print(f"  seed {seed:>4} ({i+1}/{num_seeds}) ... ", end="", flush=True)

        baseline = run_bob_governed(config, seed=seed, enable_governor=False, label=f"base_{seed}")
        governed = run_bob_governed(config, seed=seed, enable_governor=True, label=f"gov_{seed}")
        report = diff_report(baseline, governed)

        all_baseline.append(baseline)
        all_governed.append(governed)
        all_diffs.append(report)

        scar_sat = report.get("scar_saturation", "n/a")
        scar_str = f"{scar_sat:.2f}" if isinstance(scar_sat, float) else scar_sat
        auth = report["authorized_commits"]
        print(f"blocked={report['commits_blocked']:>3}  "
              f"prop/auth/blk={report['proposed_commits']}/{auth}/{report['blocked_commits']}  "
              f"ctv: {report['baseline_ctv_rate']:.3f}->{report['governed_ctv_rate']:.3f}  "
              f"scar_sat={scar_str}  "
              f"loss_d={report['loss_delta']:+.4f}")

    # Extract distributions
    blocks = [d["commits_blocked"] for d in all_diffs]
    ctv_base = [d["baseline_ctv_rate"] for d in all_diffs]
    ctv_gov = [d["governed_ctv_rate"] for d in all_diffs]
    ctv_delta = [d["ctv_delta"] for d in all_diffs]
    loss_delta = [d["loss_delta"] for d in all_diffs]

    # Also track: baseline commit counts and governed commit counts
    base_commits = [b["commit_then_violate"]["total_commits"] for b in all_baseline]
    gov_commits = [g["commit_then_violate"]["total_commits"] for g in all_governed]

    # Proposed/authorized/blocked
    proposed = [d["proposed_commits"] for d in all_diffs]
    authorized = [d["authorized_commits"] for d in all_diffs]
    blocked = [d["blocked_commits"] for d in all_diffs]
    blocked_rate = [d["blocked_commit_rate"] for d in all_diffs]

    # Region tracking
    regions_visited = [float(d["unique_regions_visited"]) for d in all_diffs]
    scar_saturations = [d.get("scar_saturation", 0.0) for d in all_diffs]

    # Scar counts
    scar_counts = [
        g["scars"]["total_scars"] if g["scars"] else 0
        for g in all_governed
    ]
    scar_debts = [
        g["scars"]["total_debt"] if g["scars"] else 0.0
        for g in all_governed
    ]

    # Medium clock activations
    mc_activations = [
        g["medium_clock"]["activation"] if g["medium_clock"] else 0.0
        for g in all_governed
    ]

    def dist_stats(vals):
        s = sorted(vals)
        n = len(s)
        return {
            "min": round(s[0], 4),
            "p25": round(s[n // 4], 4),
            "median": round(statistics.median(s), 4),
            "p75": round(s[3 * n // 4], 4),
            "max": round(s[-1], 4),
            "mean": round(statistics.mean(s), 4),
            "stdev": round(statistics.stdev(s), 4) if n > 1 else 0.0,
        }

    # Acceptance criteria across all seeds
    all_blocked_pass = all(b > 0 for b in blocks)
    ctv_improved_count = sum(1 for d in ctv_delta if d < 0)
    ctv_improved_fraction = ctv_improved_count / num_seeds
    # Only count CTV improvement on seeds with sufficient sample
    ctv_sufficient = [d for d in all_diffs if d["authorized_commits"] >= 20]
    ctv_sufficient_improved = sum(1 for d in ctv_sufficient if d["ctv_delta"] < 0)
    loss_ok_count = sum(1 for d in loss_delta if d <= 0.05)
    loss_ok_fraction = loss_ok_count / num_seeds

    summary = {
        "num_seeds": num_seeds,
        "num_experts": config.moe_num_experts,
        "top_k": config.moe_num_experts_per_tok,
        "seeds": [i * 7 + 1 for i in range(num_seeds)],
        "distributions": {
            "commits_blocked": dist_stats(blocks),
            "baseline_ctv_rate": dist_stats(ctv_base),
            "governed_ctv_rate": dist_stats(ctv_gov),
            "ctv_delta": dist_stats(ctv_delta),
            "loss_delta": dist_stats(loss_delta),
            "baseline_commits": dist_stats([float(x) for x in base_commits]),
            "governed_commits": dist_stats([float(x) for x in gov_commits]),
            "proposed_commits": dist_stats([float(x) for x in proposed]),
            "authorized_commits": dist_stats([float(x) for x in authorized]),
            "blocked_commits": dist_stats([float(x) for x in blocked]),
            "blocked_commit_rate": dist_stats(blocked_rate),
            "unique_regions_visited": dist_stats(regions_visited),
            "scar_saturation": dist_stats(scar_saturations),
            "scar_count": dist_stats([float(x) for x in scar_counts]),
            "scar_debt": dist_stats(scar_debts),
            "medium_clock_activation": dist_stats(mc_activations),
        },
        "acceptance": {
            "commits_blocked_all_pass": all_blocked_pass,
            "ctv_improved_fraction": round(ctv_improved_fraction, 4),
            "ctv_improved_count": ctv_improved_count,
            "ctv_sufficient_sample_count": len(ctv_sufficient),
            "ctv_sufficient_improved": ctv_sufficient_improved,
            "loss_ok_fraction": round(loss_ok_fraction, 4),
            "loss_ok_count": loss_ok_count,
        },
        "per_seed": all_diffs,
    }

    return summary


def print_multi_seed_summary(summary: Dict) -> None:
    """Print the multi-seed distribution report."""
    n = summary["num_seeds"]
    acc = summary["acceptance"]
    dist = summary["distributions"]

    print(f"\n{'='*70}")
    print(f"MULTI-SEED DISTRIBUTION REPORT ({n} seeds, "
          f"{summary.get('num_experts', '?')}E top-{summary.get('top_k', '?')})")
    print(f"{'='*70}")

    def fmt_dist(d):
        return f"median={d['median']:.4f}  IQR=[{d['p25']:.4f}, {d['p75']:.4f}]  range=[{d['min']:.4f}, {d['max']:.4f}]"

    print(f"\n  COMMITS BLOCKED:")
    print(f"    {fmt_dist(dist['commits_blocked'])}")

    print(f"\n  PROPOSED / AUTHORIZED / BLOCKED:")
    print(f"    Proposed:   {fmt_dist(dist['proposed_commits'])}")
    print(f"    Authorized: {fmt_dist(dist['authorized_commits'])}")
    print(f"    Blocked:    {fmt_dist(dist['blocked_commits'])}")
    print(f"    Block rate: {fmt_dist(dist['blocked_commit_rate'])}")

    print(f"\n  COMMIT-THEN-VIOLATE RATE:")
    print(f"    Baseline:  {fmt_dist(dist['baseline_ctv_rate'])}")
    print(f"    Governed:  {fmt_dist(dist['governed_ctv_rate'])}")
    print(f"    Delta:     {fmt_dist(dist['ctv_delta'])}")
    if acc['ctv_sufficient_sample_count'] < n:
        print(f"    NOTE: Only {acc['ctv_sufficient_sample_count']}/{n} seeds had >= 20 "
              f"authorized commits (sufficient sample)")
        if acc['ctv_sufficient_sample_count'] > 0:
            print(f"    CTV improved in {acc['ctv_sufficient_improved']}/"
                  f"{acc['ctv_sufficient_sample_count']} sufficient-sample seeds")
    else:
        print(f"    All seeds have sufficient sample (>= 20 authorized commits)")

    print(f"\n  LOSS DELTA (governed - baseline):")
    print(f"    {fmt_dist(dist['loss_delta'])}")

    print(f"\n  COMMIT COUNTS:")
    print(f"    Baseline:  {fmt_dist(dist['baseline_commits'])}")
    print(f"    Governed:  {fmt_dist(dist['governed_commits'])}")

    print(f"\n  ROUTING REGIONS:")
    print(f"    Visited:   {fmt_dist(dist['unique_regions_visited'])}")
    print(f"    Scar sat:  {fmt_dist(dist['scar_saturation'])}")

    print(f"\n  SCARS:")
    print(f"    Count:     {fmt_dist(dist['scar_count'])}")
    print(f"    Total debt:{fmt_dist(dist['scar_debt'])}")

    print(f"\n  MEDIUM CLOCK:")
    print(f"    Activation:{fmt_dist(dist['medium_clock_activation'])}")

    print(f"\n  ACCEPTANCE CRITERIA (must hold across distribution):")
    print(f"    1. commits_blocked > 0 (ALL seeds):  "
          f"{'PASS' if acc['commits_blocked_all_pass'] else 'FAIL'}")
    print(f"    2. ctv_delta < 0 (improved):          "
          f"{acc['ctv_improved_count']}/{n} seeds = {acc['ctv_improved_fraction']:.0%}")
    if acc['ctv_sufficient_sample_count'] > 0:
        print(f"       (sufficient-sample only):          "
              f"{acc['ctv_sufficient_improved']}/{acc['ctv_sufficient_sample_count']} seeds")
    print(f"    3. loss_delta <= 0.05 (not degraded): "
          f"{acc['loss_ok_count']}/{n} seeds = {acc['loss_ok_fraction']:.0%}")

    # Flag if the result is fragile
    if acc['ctv_improved_fraction'] < 0.8:
        print(f"\n  WARNING: CTV improvement holds in <80% of seeds.")
        if acc['ctv_sufficient_sample_count'] == 0:
            print(f"  NO seeds had >= 20 authorized commits. CTV metric is meaningless.")
            print(f"  Governor is blocking too aggressively — need wider region space or softer scars.")
        else:
            print(f"  The governor effect may be seed-dependent.")
    if not acc['commits_blocked_all_pass']:
        print(f"\n  WARNING: Some seeds had zero blocks.")
        print(f"  Governor may not be firing consistently.")


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bob Phase 1 governed experiment")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds (1=single run, >1=multi-seed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for single run mode")
    parser.add_argument("--experts", type=int, default=4,
                        help="Number of MoE experts")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Top-k experts per token")
    args = parser.parse_args()

    config = ExperimentConfig(
        moe_num_experts=args.experts,
        moe_num_experts_per_tok=args.top_k,
    )

    if args.seeds > 1:
        # Multi-seed mode
        print("=" * 70)
        print(f"BOB PHASE 1: MULTI-SEED EVALUATION ({args.seeds} seeds)")
        print(f"  Model: {config.moe_num_experts}E top-{config.moe_num_experts_per_tok}")
        print(f"  Steps: warmup={config.warmup_steps}, active={config.active_steps}, drift={config.drift_steps}")
        print("=" * 70)
        print()

        summary = run_multi_seed(config, num_seeds=args.seeds)
        print_multi_seed_summary(summary)

        report_path = "bob_governed_multiseed.json"
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Full report saved to {report_path}")

    else:
        # Single-seed mode (original behavior)
        seed = args.seed

        print("=" * 70)
        print("BOB PHASE 1: GOVERNED EXPERIMENT")
        print(f"  Model: {config.moe_num_experts}E top-{config.moe_num_experts_per_tok}")
        print(f"  Seed: {seed}")
        print(f"  Steps: warmup={config.warmup_steps}, active={config.active_steps}, drift={config.drift_steps}")
        print("=" * 70)

        print("\n--- Running baseline (no governor) ---")
        baseline = run_bob_governed(config, seed=seed, enable_governor=False, label="baseline")
        print(f"  Cheap fraction: {baseline['active_cheap_fraction']*100:.1f}%")
        print(f"  Avg loss: {baseline['avg_loss']:.4f}")
        print(f"  Commits: {baseline['commit_then_violate']['total_commits']}")
        print(f"  Commit-then-violate rate: {baseline['commit_then_violate']['commit_then_violate_rate']:.4f}")

        print("\n--- Running governed (with governor) ---")
        governed = run_bob_governed(config, seed=seed, enable_governor=True, label="governed")
        print(f"  Cheap fraction: {governed['active_cheap_fraction']*100:.1f}%")
        print(f"  Avg loss: {governed['avg_loss']:.4f}")
        print(f"  Commits: {governed['commit_then_violate']['total_commits']}")
        print(f"  Commit-then-violate rate: {governed['commit_then_violate']['commit_then_violate_rate']:.4f}")
        print(f"  Commits blocked: {governed['commits_blocked']}")

        if governed["scars"]:
            print(f"  Scars: {governed['scars']['total_scars']} regions, "
                  f"debt={governed['scars']['total_debt']:.4f}")

        if governed["medium_clock"]:
            mc = governed["medium_clock"]
            print(f"  Medium clock: activation={mc['activation']:.4f}, "
                  f"flipflop={mc['flipflop_ema']:.4f}, churn={mc['churn_ema']:.4f}")

        if governed["promotion"]:
            for cc, p in governed["promotion"].items():
                print(f"  Promotion class {cc}: eligible={p['eligible']}, score={p['score']:.3f}")

        report = diff_report(baseline, governed)

        print("\n" + "=" * 70)
        print("DIFF REPORT")
        print("=" * 70)
        print(f"  Same seed: {report['same_seed']}")
        print(f"\n  ACCEPTANCE CRITERIA:")
        print(f"    1. commits_blocked > 0:  {'PASS' if report['commits_blocked_pass'] else 'FAIL'} "
              f"({report['commits_blocked']} blocks)")
        print(f"    2. commit_then_violate drops: {'PASS' if report['ctv_improved'] else 'FAIL'} "
              f"(baseline={report['baseline_ctv_rate']:.4f}, governed={report['governed_ctv_rate']:.4f}, "
              f"delta={report['ctv_delta']:+.4f})")
        print(f"    3. Loss not degraded: {'PASS' if report['loss_delta'] <= 0.05 else 'WARN'} "
              f"(delta={report['loss_delta']:+.4f})")

        if report["escalation_by_class"]:
            print(f"\n  ESCALATION BY CLASS:")
            for cc, e in report["escalation_by_class"].items():
                print(f"    Class {cc}: {e['blocked']}/{e['total']} blocked = {e['rate']:.1%}")

        report_path = "bob_governed_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to {report_path}")

    print("\nDone.")
