#!/usr/bin/env python3
"""
Probe Battery: Check protected deltas after merge.

Runs a set of metrics immediately after merge to detect capability loss.
If any protected delta regresses beyond tolerance, triggers rollback.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProbeBatteryResult:
    """
    Result of running probe battery.

    Contains pre/post metrics and rollback verdict.
    """
    # Primary signal: Loss
    loss_pre: float
    loss_post: float
    loss_delta: float

    # Secondary signal: Perplexity
    perplexity_pre: float
    perplexity_post: float
    perplexity_delta: float

    # Structural signal: Free energy
    f_l_pre: float
    f_l_post: float
    f_l_delta: float

    # Structural signal: Coherence
    coherence_pre: float
    coherence_post: float
    coherence_delta: float

    # Structural signal: Neff
    neff_pre: float
    neff_post: float
    neff_delta: float

    # Verdict
    rollback_triggered: bool
    rollback_signals: List[str]

    # Tolerances (with defaults)
    loss_tolerance: float = 0.05
    perplexity_tolerance: float = 0.10  # 10% relative increase
    f_l_tolerance: float = 0.01
    coherence_tolerance: float = -0.05  # Max 5% drop (negative)
    neff_tolerance: float = -0.20  # Max 20% drop (negative)


def run_probe_battery(
    model,
    dataloader,
    criterion,
    num_steps: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Run probe battery to collect metrics.

    Args:
        model: Model with ChronoMoE layer
        dataloader: Data iterator
        criterion: Loss function
        num_steps: Number of steps to run
        device: Device to run on

    Returns:
        Dict of average metrics: loss, perplexity, f_l, coherence, neff
    """
    model.eval()  # Evaluation mode (no dropout)

    metrics = {
        "loss": [],
        "perplexity": [],
        "f_l": [],
        "coherence": [],
        "neff": [],
    }

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            # Move to device
            inputs = batch.to(device)
            targets = inputs[:, 1:].contiguous()  # Next token prediction
            inputs = inputs[:, :-1].contiguous()

            # Forward pass
            logits, _ = model(inputs)

            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            metrics["loss"].append(loss.item())
            metrics["perplexity"].append(torch.exp(loss).item())

            # Get controller diagnostics (if available)
            if hasattr(model, 'moe') and hasattr(model.moe, 'controller'):
                diag = model.moe.controller.get_diagnostics()

                if "free_energy" in diag and diag["free_energy"] and "F_l" in diag["free_energy"]:
                    metrics["f_l"].append(diag["free_energy"]["F_l"])

                if "coherence" in diag and diag["coherence"]:
                    if "avg_coherence" in diag["coherence"]:
                        metrics["coherence"].append(diag["coherence"]["avg_coherence"])
                    if "Neff" in diag["coherence"]:
                        metrics["neff"].append(diag["coherence"]["Neff"])

    model.train()  # Back to training mode

    # Compute averages
    return {
        "loss": sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0.0,
        "perplexity": sum(metrics["perplexity"]) / len(metrics["perplexity"]) if metrics["perplexity"] else 0.0,
        "f_l": sum(metrics["f_l"]) / len(metrics["f_l"]) if metrics["f_l"] else 0.0,
        "coherence": sum(metrics["coherence"]) / len(metrics["coherence"]) if metrics["coherence"] else 0.0,
        "neff": sum(metrics["neff"]) / len(metrics["neff"]) if metrics["neff"] else 0.0,
    }


def check_protected_deltas(
    pre_metrics: Dict[str, float],
    post_metrics: Dict[str, float],
    tolerances: Optional[Dict[str, float]] = None,
) -> ProbeBatteryResult:
    """
    Check if any protected delta regressed beyond tolerance.

    Args:
        pre_metrics: Metrics before merge
        post_metrics: Metrics after merge
        tolerances: Custom tolerances (optional)

    Returns:
        ProbeBatteryResult with verdict
    """
    # Default tolerances
    if tolerances is None:
        tolerances = {
            "loss": 0.05,
            "perplexity": 0.10,
            "f_l": 0.01,
            "coherence": -0.05,  # Negative (drop)
            "neff": -0.20,  # Negative (drop)
        }

    rollback_signals = []

    # 1. Loss delta (absolute increase)
    loss_delta = post_metrics["loss"] - pre_metrics["loss"]
    if loss_delta > tolerances["loss"]:
        rollback_signals.append(
            f"loss_regression (delta={loss_delta:+.4f} > {tolerances['loss']})"
        )

    # 2. Perplexity delta (relative increase)
    if pre_metrics["perplexity"] > 0:
        ppl_delta = (post_metrics["perplexity"] - pre_metrics["perplexity"]) / pre_metrics["perplexity"]
        if ppl_delta > tolerances["perplexity"]:
            rollback_signals.append(
                f"perplexity_regression (delta={ppl_delta:+.4f} > {tolerances['perplexity']})"
            )
    else:
        ppl_delta = 0.0

    # 3. F_l delta (absolute increase)
    f_l_delta = post_metrics["f_l"] - pre_metrics["f_l"]
    if post_metrics["f_l"] > 0 and f_l_delta > tolerances["f_l"]:
        rollback_signals.append(
            f"f_l_spike (delta={f_l_delta:+.4f} > {tolerances['f_l']})"
        )

    # 4. Coherence delta (absolute drop, negative tolerance)
    coherence_delta = post_metrics["coherence"] - pre_metrics["coherence"]
    if post_metrics["coherence"] > 0 and coherence_delta < tolerances["coherence"]:
        rollback_signals.append(
            f"coherence_drop (delta={coherence_delta:+.4f} < {tolerances['coherence']})"
        )

    # 5. Neff delta (relative drop, negative tolerance)
    if pre_metrics["neff"] > 0:
        neff_delta = (post_metrics["neff"] - pre_metrics["neff"]) / pre_metrics["neff"]
        if neff_delta < tolerances["neff"]:
            rollback_signals.append(
                f"neff_collapse (delta={neff_delta:+.4f} < {tolerances['neff']})"
            )
    else:
        neff_delta = 0.0

    # Verdict: rollback if any signal triggered
    rollback_triggered = len(rollback_signals) > 0

    return ProbeBatteryResult(
        loss_pre=pre_metrics["loss"],
        loss_post=post_metrics["loss"],
        loss_delta=loss_delta,
        loss_tolerance=tolerances["loss"],
        perplexity_pre=pre_metrics["perplexity"],
        perplexity_post=post_metrics["perplexity"],
        perplexity_delta=ppl_delta,
        perplexity_tolerance=tolerances["perplexity"],
        f_l_pre=pre_metrics["f_l"],
        f_l_post=post_metrics["f_l"],
        f_l_delta=f_l_delta,
        f_l_tolerance=tolerances["f_l"],
        coherence_pre=pre_metrics["coherence"],
        coherence_post=post_metrics["coherence"],
        coherence_delta=coherence_delta,
        coherence_tolerance=tolerances["coherence"],
        neff_pre=pre_metrics["neff"],
        neff_post=post_metrics["neff"],
        neff_delta=neff_delta,
        neff_tolerance=tolerances["neff"],
        rollback_triggered=rollback_triggered,
        rollback_signals=rollback_signals,
    )


def print_probe_battery_result(result: ProbeBatteryResult) -> None:
    """
    Pretty-print probe battery result.

    Args:
        result: ProbeBatteryResult
    """
    print("\n  Probe Battery Results:")
    print("  " + "-" * 68)

    # Loss
    status = "✗" if "loss" in str(result.rollback_signals) else "✓"
    print(f"  {status} Loss:       {result.loss_pre:.4f} → {result.loss_post:.4f} "
          f"(delta={result.loss_delta:+.4f}, tol={result.loss_tolerance})")

    # Perplexity
    status = "✗" if "perplexity" in str(result.rollback_signals) else "✓"
    print(f"  {status} Perplexity: {result.perplexity_pre:.2f} → {result.perplexity_post:.2f} "
          f"(delta={result.perplexity_delta:+.4f}, tol={result.perplexity_tolerance})")

    # F_l
    status = "✗" if "f_l" in str(result.rollback_signals) else "✓"
    print(f"  {status} F_l:        {result.f_l_pre:.4f} → {result.f_l_post:.4f} "
          f"(delta={result.f_l_delta:+.4f}, tol={result.f_l_tolerance})")

    # Coherence
    status = "✗" if "coherence" in str(result.rollback_signals) else "✓"
    print(f"  {status} Coherence:  {result.coherence_pre:.4f} → {result.coherence_post:.4f} "
          f"(delta={result.coherence_delta:+.4f}, tol={result.coherence_tolerance})")

    # Neff
    status = "✗" if "neff" in str(result.rollback_signals) else "✓"
    print(f"  {status} Neff:       {result.neff_pre:.2f} → {result.neff_post:.2f} "
          f"(delta={result.neff_delta:+.4f}, tol={result.neff_tolerance})")

    print("  " + "-" * 68)

    if result.rollback_triggered:
        print(f"  VERDICT: ROLLBACK TRIGGERED")
        for signal in result.rollback_signals:
            print(f"    - {signal}")
    else:
        print(f"  VERDICT: PROTECTED DELTAS SATISFIED")

    print()
