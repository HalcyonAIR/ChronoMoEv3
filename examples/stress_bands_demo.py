"""
Stress Bands Demo: Autonomic Regulation

Demonstrates the key principle: Pressure tunes behavior. Calm commits identity.

Shows:
1. How stress bands learn from experience (boundary adaptation)
2. How irreversible gates work (double gate: evidence + calmness)
3. How different stress profiles create different personalities
4. How the system responds to pressure vs calm
5. Why trauma doesn't make good reflexes

Key insight: F_l is a sensor, not an objective.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3.stress_bands import (
    StressBandsConfig,
    init_stress_bands,
    step_stress_bands,
)
from chronomoe_v3.collapse_detection import (
    CollapseThresholds,
    CollapseSignals,
    check_survival,
)


def demo_band_learning():
    """
    Show how stress bands learn from operational experience.

    Two systems:
    - System A: Tolerates stress well (survives under pressure)
    - System B: Degrades quickly (collapses under pressure)

    After 1000 steps, they should have different band boundaries.
    """
    print("=" * 70)
    print("DEMO 1: Band Learning from Experience")
    print("=" * 70)

    cfg = StressBandsConfig(
        comfort_ceiling_init=0.5,
        strain_ceiling_init=1.0,
        lr_widen=0.001,
        lr_narrow=0.001,
    )

    state_a = init_stress_bands(cfg)
    state_b = init_stress_bands(cfg)

    print(f"\nInitial boundaries (both systems):")
    print(f"  Comfort ceiling: {cfg.comfort_ceiling_init:.3f}")
    print(f"  Strain ceiling:  {cfg.strain_ceiling_init:.3f}")

    # Simulate 1000 steps with different stress profiles
    for step in range(1000):
        # System A: moderate stress, always survives
        f_a = 0.4 + 0.2 * torch.rand(1).item()  # [0.4, 0.6]
        survived_a = True  # Robust system

        # System B: same stress level, but collapses often
        f_b = 0.4 + 0.2 * torch.rand(1).item()
        # Collapses when F_l > 0.5
        survived_b = f_b < 0.5

        step_stress_bands(state_a, cfg, f_a, survived_a)
        step_stress_bands(state_b, cfg, f_b, survived_b)

    print(f"\nAfter 1000 steps:")
    print(f"\nSystem A (robust, always survives):")
    print(f"  Comfort ceiling: {state_a.comfort_ceiling:.3f} (widened)")
    print(f"  Strain ceiling:  {state_a.strain_ceiling:.3f}")
    print(f"  Time distribution: comfort={state_a.time_in_comfort}, strain={state_a.time_in_strain}, panic={state_a.time_in_panic}")
    print(f"  Comfort survival: {state_a.comfort_survival.value:.3f}")

    print(f"\nSystem B (fragile, collapses often):")
    print(f"  Comfort ceiling: {state_b.comfort_ceiling:.3f} (narrowed)")
    print(f"  Strain ceiling:  {state_b.strain_ceiling:.3f}")
    print(f"  Time distribution: comfort={state_b.time_in_comfort}, strain={state_b.time_in_strain}, panic={state_b.time_in_panic}")
    print(f"  Comfort survival: {state_b.comfort_survival.value:.3f}")

    print(f"\n✓ Different stress tolerance → different band boundaries")
    print(f"  No hardcoded personality. Just calibration.")


def demo_irreversible_gates():
    """
    Show how irreversible gates work: evidence AND calmness required.

    Three scenarios:
    1. Calm + evidence → irreversible allowed
    2. Stressed + evidence → irreversible BLOCKED
    3. Panic + evidence → irreversible BLOCKED
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Irreversible Gates (Evidence + Calmness)")
    print("=" * 70)

    cfg = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
        scar_calm_steps=100,
        crystallize_calm_steps=500,
        edit_calm_steps=200,
    )

    state = init_stress_bands(cfg)

    # Scenario 1: Operate in comfort for 150 steps
    print(f"\nScenario 1: Calm operation (F_l = 0.3, 150 steps)")
    for _ in range(150):
        result = step_stress_bands(state, cfg, f_l=0.3, survived=True)

    print(f"  Band: {result.band_now}")
    print(f"  Time in comfort: {state.time_in_comfort}")
    print(f"  Gates:")
    print(f"    Scar formation:    {result.gates.allow_scar} ✓ (need {cfg.scar_calm_steps})")
    print(f"    Structural edits:  {result.gates.allow_structural_edits} ✗ (need {cfg.edit_calm_steps})")
    print(f"    Crystallization:   {result.gates.allow_crystallize} ✗ (need {cfg.crystallize_calm_steps})")
    print(f"  → Evidence alone isn't enough. Need sustained calm.")

    # Scenario 2: Enter strain band
    print(f"\nScenario 2: Stress spike (F_l = 1.5, 50 steps)")
    for _ in range(50):
        result = step_stress_bands(state, cfg, f_l=1.5, survived=True)

    print(f"  Band: {result.band_now}")
    print(f"  Time in comfort: {state.time_in_comfort} (reset on band change)")
    print(f"  Gates:")
    print(f"    Scar formation:    {result.gates.allow_scar}")
    print(f"    Structural edits:  {result.gates.allow_structural_edits} ✗ FROZEN")
    print(f"    Crystallization:   {result.gates.allow_crystallize} ✗ FROZEN")
    print(f"  → Under strain: irreversibles HARDER, not easier")

    # Scenario 3: Panic
    print(f"\nScenario 3: Panic (F_l = 3.0, 10 steps)")
    for _ in range(10):
        result = step_stress_bands(state, cfg, f_l=3.0, survived=True)

    print(f"  Band: {result.band_now}")
    print(f"  Gates:")
    print(f"    Scar formation:    {result.gates.allow_scar} ✗ BLOCKED")
    print(f"    Structural edits:  {result.gates.allow_structural_edits} ✗ BLOCKED")
    print(f"    Crystallization:   {result.gates.allow_crystallize} ✗ BLOCKED")
    print(f"  Reason: {result.gates.reason}")
    print(f"  → Panic = preservation mode. Zero irreversibles.")

    print(f"\n✓ Trauma doesn't make good reflexes. It makes scars.")
    print(f"  Reflexes crystallize in calm, not crisis.")


def demo_hysteresis():
    """
    Show hysteresis: entering a worse band is easier than exiting it.

    Prevents thrashing at band boundaries.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Hysteresis (Sticky Bands)")
    print("=" * 70)

    cfg = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
        hysteresis_margin=0.05,  # 5% margin
    )

    state = init_stress_bands(cfg)

    # Start in comfort
    result = step_stress_bands(state, cfg, f_l=0.5, survived=True)
    print(f"\nStep 1: F_l = 0.5")
    print(f"  Band: {result.band_now} (comfort)")

    # Slowly increase stress
    print(f"\nSlowly increasing stress...")
    for f in [0.8, 0.95, 1.05, 1.1]:
        result = step_stress_bands(state, cfg, f_l=f, survived=True)
        print(f"  F_l = {f:.2f} → Band: {result.band_now}")

    # Now decrease stress
    print(f"\nSlowly decreasing stress...")
    for f in [1.0, 0.95, 0.90, 0.85]:
        result = step_stress_bands(state, cfg, f_l=f, survived=True)
        print(f"  F_l = {f:.2f} → Band: {result.band_now}")

    print(f"\n✓ Hysteresis prevents thrashing")
    print(f"  Entering strain at F_l=1.0, but staying in strain until F_l<0.95")


def demo_stress_tolerance_profiles():
    """
    Show how different operational histories create different profiles.

    Three systems:
    - Stable: operates at low stress, narrow comfort band
    - Resilient: operates at high stress, wide comfort band
    - Volatile: alternates between calm and panic, learned caution
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Stress Tolerance Profiles Emerge")
    print("=" * 70)

    cfg = StressBandsConfig(
        comfort_ceiling_init=1.0,
        strain_ceiling_init=2.0,
        lr_widen=0.002,
        lr_narrow=0.002,
    )

    stable = init_stress_bands(cfg)
    resilient = init_stress_bands(cfg)
    volatile = init_stress_bands(cfg)

    print(f"\nSimulating 2000 steps with different stress profiles...")

    for step in range(2000):
        # Stable: low stress, always survives
        f_stable = 0.2 + 0.1 * torch.rand(1).item()
        step_stress_bands(stable, cfg, f_stable, survived=True)

        # Resilient: high stress, always survives
        f_resilient = 0.8 + 0.4 * torch.rand(1).item()
        step_stress_bands(resilient, cfg, f_resilient, survived=True)

        # Volatile: alternates between calm and panic
        if step % 200 < 100:
            f_volatile = 0.3  # Calm period
            survived_volatile = True
        else:
            f_volatile = 2.5  # Panic period
            survived_volatile = torch.rand(1).item() > 0.3  # Often collapses

        step_stress_bands(volatile, cfg, f_volatile, survived_volatile)

    print(f"\nStable System (low stress, always survives):")
    print(f"  Comfort ceiling: {stable.comfort_ceiling:.3f}")
    print(f"  Strain ceiling:  {stable.strain_ceiling:.3f}")
    print(f"  Distribution: {stable.time_in_comfort}/{stable.time_in_strain}/{stable.time_in_panic}")
    print(f"  → Narrow comfort band (hasn't needed to widen)")

    print(f"\nResilient System (high stress, always survives):")
    print(f"  Comfort ceiling: {resilient.comfort_ceiling:.3f} (widened!)")
    print(f"  Strain ceiling:  {resilient.strain_ceiling:.3f}")
    print(f"  Distribution: {resilient.time_in_comfort}/{resilient.time_in_strain}/{resilient.time_in_panic}")
    print(f"  → Wide comfort band (learned tolerance)")

    print(f"\nVolatile System (alternates calm/panic, often collapses):")
    print(f"  Comfort ceiling: {volatile.comfort_ceiling:.3f}")
    print(f"  Strain ceiling:  {volatile.strain_ceiling:.3f}")
    print(f"  Distribution: {volatile.time_in_comfort}/{volatile.time_in_strain}/{volatile.time_in_panic}")
    print(f"  Comfort survival: {volatile.comfort_survival.value:.3f}")
    print(f"  → Learned caution (narrow bands)")

    print(f"\n✓ Three systems, three personalities")
    print(f"  Not hardcoded. Emerged from experience.")


def demo_collapse_detection():
    """
    Show how collapse detection works with stress bands.

    The "survived" signal comes from internal integrity checks.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Collapse Detection (Survival Signal)")
    print("=" * 70)

    thresholds = CollapseThresholds(
        psi_critical=0.1,  # Coherence below 0.1 = collapse
        neff_critical=1.5,  # Routing to <1.5 experts = collapse
        saturation_critical=0.9,  # One expert >90% load = collapse
    )

    print(f"\nCollapse thresholds:")
    print(f"  Psi (coherence):   < {thresholds.psi_critical}")
    print(f"  Neff (diversity):  < {thresholds.neff_critical}")
    print(f"  Saturation:        > {thresholds.saturation_critical}")

    # Scenario 1: Healthy system
    print(f"\nScenario 1: Healthy system")
    signals_healthy = CollapseSignals(
        psi=0.8,  # High coherence
        neff=6.5,  # Good diversity
        saturation=0.2,  # Well-distributed load
    )
    survived, reason = check_survival(signals_healthy, thresholds)
    print(f"  Psi={signals_healthy.psi:.2f}, Neff={signals_healthy.neff:.2f}, Sat={signals_healthy.saturation:.2f}")
    print(f"  Survived: {survived} ({reason})")

    # Scenario 2: Coherence collapse
    print(f"\nScenario 2: Coherence collapse")
    signals_coh_collapse = CollapseSignals(
        psi=0.05,  # Very low coherence
        neff=6.0,
        saturation=0.3,
    )
    survived, reason = check_survival(signals_coh_collapse, thresholds)
    print(f"  Psi={signals_coh_collapse.psi:.2f}, Neff={signals_coh_collapse.neff:.2f}, Sat={signals_coh_collapse.saturation:.2f}")
    print(f"  Survived: {survived} ({reason})")

    # Scenario 3: Routing collapse
    print(f"\nScenario 3: Routing collapse (all load to one expert)")
    signals_route_collapse = CollapseSignals(
        psi=0.7,
        neff=1.2,  # Collapsed to single expert
        saturation=0.95,  # One expert has 95% of load
    )
    survived, reason = check_survival(signals_route_collapse, thresholds)
    print(f"  Psi={signals_route_collapse.psi:.2f}, Neff={signals_route_collapse.neff:.2f}, Sat={signals_route_collapse.saturation:.2f}")
    print(f"  Survived: {survived} ({reason})")

    print(f"\n✓ Survival = internal integrity, not task performance")
    print(f"  Stress bands learn from whether the system held together,")
    print(f"  not from whether it got the 'right' answer.")


def demo_principle():
    """
    Summarize the core principle with a concrete example.
    """
    print("\n" + "=" * 70)
    print("THE PRINCIPLE: Pressure Tunes Behavior, Calm Commits Identity")
    print("=" * 70)

    print(f"\nPressure (high F_l):")
    print(f"  → Exploration temperature drops or rises (style-dependent)")
    print(f"  → Proposal budgets tighten")
    print(f"  → Deliberation slows")
    print(f"  → Irreversible thresholds GO UP")
    print(f"  → You're allowed to be stressed")

    print(f"\nCalm (low F_l, sustained):")
    print(f"  → Scar formation allowed (after {200} steps calm)")
    print(f"  → Crystallization allowed (after {2000} steps calm)")
    print(f"  → Structural edits allowed (after {500} steps calm)")
    print(f"  → Identity changes require stability + time + evidence")

    print(f"\nWhat this prevents:")
    print(f"  ✗ Panic-driven crystallization (bad reflexes)")
    print(f"  ✗ Stress-driven structural changes (identity churn)")
    print(f"  ✗ Trauma becoming character")

    print(f"\nWhat this allows:")
    print(f"  ✓ Different systems tolerate different stress levels")
    print(f"  ✓ Boundaries learned from experience, not hardcoded")
    print(f"  ✓ Stress modulates behavior without rewriting who you are")

    print(f"\n{'─' * 70}")
    print(f"Stress is allowed to exist.")
    print(f"It's not allowed to decide who you become.")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    demo_band_learning()
    demo_irreversible_gates()
    demo_hysteresis()
    demo_stress_tolerance_profiles()
    demo_collapse_detection()
    demo_principle()

    print("\n" + "=" * 70)
    print("✓ Stress Bands Demo Complete")
    print("=" * 70)
