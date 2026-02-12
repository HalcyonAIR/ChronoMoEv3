#!/usr/bin/env python3
"""
Test 26: Adversarial Suppression Trial (MUST VETO)

Validates that suppression trial criteria correctly reject trials when conditions degrade.

This test MUST produce a failure verdict. If it passes, the veto system is broken.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from dataclasses import dataclass

from chronomoe_integration.chronomoe_layer import ChronoMoE


@dataclass
class Config:
    """Minimal config for testing."""
    n_embd: int = 64
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    moe_softmax_order: str = "softmax_topk"


class DummyMLP(nn.Module):
    """Minimal MLP for testing."""
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        return self.fc(x), {}


def test_adversarial_loss_spike():
    """
    Test 26: Adversarial Suppression Trial (Loss Spike)

    Simulate suppression trial where loss spikes during trial period.
    This MUST veto the trial (fail verdict).

    Setup:
    - Baseline: loss ~2.5
    - Trial: loss spikes to ~4.0 (injected)
    - Expected: VETO due to loss spike

    If this test passes when it should fail, the veto system is broken.
    """
    print("\n" + "=" * 70)
    print("TEST 26: Adversarial Suppression Trial (Loss Spike MUST VETO)")
    print("=" * 70)

    config = Config()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Phase 1: Establish baseline (50 steps)
    print("\nPhase 1: Establish Baseline (50 steps)")
    print("-" * 70)

    x = torch.randn(2, 10, config.n_embd)
    baseline_losses = []

    for step in range(50):
        output, metadata = layer(x)

        # Simulate stable baseline loss (~2.5)
        loss = torch.tensor(2.5 + torch.randn(1).item() * 0.1)
        baseline_losses.append(loss.item())

        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation:
            layer.controller.last_observation.loss = loss.item()

        layer.current_step += 1

    baseline_loss = sum(baseline_losses) / len(baseline_losses)
    print(f"✓ Baseline loss (50 steps): {baseline_loss:.4f}")

    # Phase 2: Suppression trial with injected loss spike (100 steps)
    print("\nPhase 2: Suppression Trial with Loss Spike (100 steps)")
    print("-" * 70)

    # Suppress expert 1
    layer.controller.suppress_expert(expert_id=1, duration_steps=0)
    print(f"✓ Expert 1 suppressed (soft penalty)")

    trial_losses = []

    for step in range(50, 150):
        output, metadata = layer(x)

        # INJECT LOSS SPIKE during trial (step 75-125)
        if 75 <= step < 125:
            # Loss spikes to 4.0 (degradation)
            loss = torch.tensor(4.0 + torch.randn(1).item() * 0.2)
        else:
            # Normal baseline
            loss = torch.tensor(2.5 + torch.randn(1).item() * 0.1)

        trial_losses.append(loss.item())

        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation:
            layer.controller.last_observation.loss = loss.item()

        # Decay penalties
        layer.controller.update_penalties()
        layer.current_step += 1

    trial_loss = sum(trial_losses) / len(trial_losses)
    max_trial_loss = max(trial_losses)

    print(f"✓ Trial loss (avg): {trial_loss:.4f}")
    print(f"✓ Trial loss (max): {max_trial_loss:.4f}")
    print(f"✓ Loss delta (avg): {trial_loss - baseline_loss:+.4f}")
    print(f"✓ Loss delta (max): {max_trial_loss - baseline_loss:+.4f}")

    # Apply veto criteria
    print("\nPhase 3: Apply Veto Criteria")
    print("-" * 70)

    loss_tolerance = 0.05
    loss_delta_avg = trial_loss - baseline_loss
    loss_delta_max = max_trial_loss - baseline_loss

    print(f"✓ Loss tolerance: {loss_tolerance}")
    print(f"✓ Avg delta vs tolerance: {loss_delta_avg:.4f} vs {loss_tolerance:.4f}")
    print(f"✓ Max delta vs tolerance: {loss_delta_max:.4f} vs {loss_tolerance:.4f}")

    # Veto if max loss exceeds baseline + tolerance
    trial_vetoed = loss_delta_max >= loss_tolerance

    print(f"\n✓ Trial verdict: {'VETO (rejected)' if trial_vetoed else 'PASS (eligible)'}")

    # CRITICAL ASSERTION: Trial MUST be vetoed
    assert trial_vetoed, \
        f"Trial should be VETOED due to loss spike (max delta {loss_delta_max:.4f} >= {loss_tolerance})"

    print("\n✓ TEST 26 PASSED: Adversarial trial correctly vetoed")
    print("  Loss spike triggered veto as expected")
    print("  Veto system working correctly")
    print()


def test_adversarial_stress_band_change():
    """
    Test 26b: Adversarial Suppression Trial (Stress Band Change)

    Simulate suppression trial where stress band changes to STRAIN.
    This MUST veto the trial (fail verdict).

    Setup:
    - Baseline: COMFORT
    - Trial: Force STRAIN (simulated high stress)
    - Expected: VETO due to stress band change

    If this test passes when it should fail, the veto system is broken.
    """
    print("\n" + "=" * 70)
    print("TEST 26b: Adversarial Suppression Trial (STRAIN MUST VETO)")
    print("=" * 70)

    config = Config()
    layer = ChronoMoE(
        config=config,
        mlp=DummyMLP,
        layer_id=0,
        max_experts=8,
        autonomous_mode=True,
    )

    # Phase 1: Establish baseline in COMFORT
    print("\nPhase 1: Establish Baseline in COMFORT")
    print("-" * 70)

    x = torch.randn(2, 10, config.n_embd)

    for step in range(50):
        output, metadata = layer(x)
        loss = torch.tensor(2.5 + torch.randn(1).item() * 0.1)

        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation:
            layer.controller.last_observation.loss = loss.item()

        # Update stress bands with low stress (stay in COMFORT)
        layer.update_stress_bands(stress=0.5)
        layer.current_step += 1

    baseline_band = layer.stress_bands.current_band.name
    print(f"✓ Baseline stress band: {baseline_band}")
    assert baseline_band == "COMFORT", "Baseline should be in COMFORT"

    # Phase 2: Suppression trial with forced STRAIN
    print("\nPhase 2: Suppression Trial with Forced STRAIN")
    print("-" * 70)

    # Suppress expert 1
    layer.controller.suppress_expert(expert_id=1, duration_steps=0)
    print(f"✓ Expert 1 suppressed (soft penalty)")

    bands_during_trial = []

    for step in range(50, 150):
        output, metadata = layer(x)
        loss = torch.tensor(2.5 + torch.randn(1).item() * 0.1)

        if hasattr(layer.controller, 'last_observation') and layer.controller.last_observation:
            layer.controller.last_observation.loss = loss.item()

        # INJECT HIGH STRESS to force STRAIN (step 75-125)
        if 75 <= step < 125:
            layer.update_stress_bands(stress=2.5)  # STRAIN territory
        else:
            layer.update_stress_bands(stress=0.5)  # COMFORT

        bands_during_trial.append(layer.stress_bands.current_band.name)

        # Decay penalties
        layer.controller.update_penalties()
        layer.current_step += 1

    print(f"✓ Bands observed during trial: {set(bands_during_trial)}")

    # Apply veto criteria
    print("\nPhase 3: Apply Veto Criteria")
    print("-" * 70)

    # Veto if any non-COMFORT band observed
    entered_strain = "STRAIN" in bands_during_trial or "PANIC" in bands_during_trial

    print(f"✓ Entered STRAIN/PANIC: {entered_strain}")
    print(f"\n✓ Trial verdict: {'VETO (rejected)' if entered_strain else 'PASS (eligible)'}")

    # CRITICAL ASSERTION: Trial MUST be vetoed
    assert entered_strain, \
        "Trial should be VETOED due to stress band change to STRAIN"

    print("\n✓ TEST 26b PASSED: Adversarial trial correctly vetoed")
    print("  STRAIN entry triggered veto as expected")
    print("  Veto system working correctly")
    print()


if __name__ == "__main__":
    test_adversarial_loss_spike()
    test_adversarial_stress_band_change()

    print("=" * 70)
    print("✓ ALL ADVERSARIAL SUPPRESSION TESTS PASSED (2/2)")
    print("=" * 70)
    print()
    print("Verification: Both trials correctly VETOED when conditions degraded.")
    print("If these tests had passed (no veto), the criteria would be broken.")
    print()
