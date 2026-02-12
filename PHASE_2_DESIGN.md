# Phase 2: Convergence Definition (Rigorous)

**Date:** 2026-02-12
**Status:** Implementation complete, instrumentation running
**Purpose:** Define convergence mathematically, not poetically

---

## Design Principles

### 1. State vs. Event

**Convergence State:** Sustained condition where:
1. Bob and Maniac agree on high-impact claims
2. No successful downgrades for N consecutive evaluations
3. Routing diversity below regime-specific threshold

**Convergence Event:** Transition from non-converged → converged state

Both are necessary. State is current condition. Event is moment of transition.

### 2. Regime-Aware Thresholds

**From Phase 1:** Three deformation regimes identified.

**Critical insight:** Cannot use single diversity threshold across regimes.

| Regime | Scar Range | Expected Diversity | Threshold Strategy |
|--------|-----------|-------------------|-------------------|
| Exploration | 0.0-2.0 | High | **Strict** (low diversity suspicious) |
| Transition | 2.0-4.0 | Medium | **Moderate** (low diversity expected) |
| Saturation | 4.0+ | Low | **Lenient** (diversity already collapsed) |

**Why:** If you ignore regime, you'll mistake saturation for convergence.

### 3. Three-Condition AND

**All three must hold:**

```python
convergence = (
    agreement_window >= W_min AND
    downgrade_silence >= S_min AND
    diversity_normalized <= D_threshold(regime)
)
```

**Not two. All three.**

### 4. Persistence Required

**Require minimum time-in-state:**

Convergence must hold for K consecutive intervals (default: K=5).

**Why:** Avoid detecting transient alignment spikes.

### 5. Baseline-Derived Thresholds

**DO NOT use intuition. Use percentiles from natural behavior.**

```python
W_min = 90th percentile of natural agreement lengths
S_min = 90th percentile of natural silence periods
D_threshold = 10th percentile (low end) of diversity per regime
```

**Otherwise:** Confirmation bias wrapped in ROC curves.

### 6. Instrument Before Policy

**Measure first:**
- Time to convergence after entering transition regime
- Scar strength at convergence
- Regime distribution of convergence events
- Divergence frequency (premature convergence)

**DO NOT attach interventions yet.** That's Phase 4.

---

## Operational Definition

### Convergence Conditions

```python
@dataclass
class ConvergenceThresholds:
    W_min: int                  # Agreement window (consecutive agreements)
    S_min: int                  # Downgrade silence (cycles since last downgrade)
    D_exploration: float        # Diversity threshold (exploration regime)
    D_transition: float         # Diversity threshold (transition regime)
    D_saturation: float         # Diversity threshold (saturation regime)
    K_persistence: int          # Persistence intervals required
    impact_threshold: float     # "High impact" claim threshold
```

### Detection Algorithm

```python
for each step:
    # 1. Compute regime
    regime = compute_deformation_regime(scar_strength_total)

    # 2. Update agreement window
    if bob_agreed AND claim_impact >= impact_threshold:
        agreement_window += 1
    else:
        agreement_window = 0  # Reset on disagreement

    # 3. Update downgrade silence
    if maniac_downgraded:
        downgrade_silence = 0
        # Log divergence event (if was converged)
    else:
        downgrade_silence += 1

    # 4. Normalize diversity by regime baseline
    diversity_normalized = diversity_current / baseline_diversity[regime]

    # 5. Check convergence conditions
    conditions_met = (
        agreement_window >= W_min AND
        downgrade_silence >= S_min AND
        diversity_normalized <= D_threshold[regime]
    )

    # 6. Update persistence
    if conditions_met:
        persistence_count += 1
    else:
        persistence_count = 0

    # 7. State transition (with persistence)
    if persistence_count >= K_persistence:
        converged = True
        log_convergence_event()
```

---

## Instrumentation (No Policy)

### Questions to Answer

**1. Does convergence cluster in transition regime?**
- Expect: Yes, because that's where deformation accelerates (Phase 1)

**2. Does it appear mostly in saturation?**
- Expect: Some, but should also detect onset during transition

**3. Is it domain-dependent?**
- Not tested yet (Phase 2 uses single domain)
- Future work: Multi-domain experiments

**4. Time-to-convergence after transition entry?**
- Critical for early warning systems (Phase 3)
- If slow: Intervention has time
- If fast: Need preemptive detection

### Statistics Logged

```python
{
    "num_convergence_events": int,
    "num_divergence_events": int,

    "convergence_by_regime": {
        "exploration": int,
        "transition": int,
        "saturation": int,
    },

    "scar_strength_at_convergence": {
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
    },

    "divergence_durations": {
        "mean": float,
        "median": float,
    },
}
```

---

## Divergence Events

**Definition:** Successful maniac downgrade after convergence state.

**Indicates:** Convergence was premature.

**Log:**
- Converged duration (how long before divergence?)
- Scar strength at divergence
- Diversity at divergence
- Domain volatility (if available)

**Purpose:** Validate that convergence detection isn't just measuring saturation.

If divergence_rate > 50%, convergence detection is broken (detecting wrong thing).

---

## Baseline Measurement Protocol

### Phase 1: Measure Natural Distributions

Run system **without convergence detection** for N steps (5000+).

Simulate scar accumulation:
- Steps 0-1000: Exploration (scar < 2.0)
- Steps 1000-2000: Transition (scar 2.0-4.0)
- Steps 2000+: Saturation (scar > 4.0)

Measure:
- Agreement window lengths (every agreement-to-disagreement sequence)
- Silence period durations (downgrade-to-downgrade intervals)
- Diversity by regime (sample every 100 steps)

### Phase 2: Derive Thresholds

```python
W_min = 90th percentile(agreement_lengths)
S_min = 90th percentile(silence_periods)
D_exploration = 10th percentile(diversity in exploration)
D_transition = 10th percentile(diversity in transition)
D_saturation = 10th percentile(diversity in saturation)
```

**Why 90th/10th percentiles?**
- 90th for W_min/S_min: Capture unusually long agreements/silences
- 10th for D: Capture unusually low diversity
- Not median (50th): Too lenient, would detect convergence too often
- Not extreme (99th): Too strict, would never detect

### Phase 3: Run Detection

Run detector with derived thresholds, **no policy attached**.

Log convergence events, divergence events, statistics.

Analyze:
- Regime distribution (where does convergence happen?)
- Time-to-convergence (how quickly after transition entry?)
- Divergence rate (how often is convergence premature?)

---

## What Phase 2 Does NOT Do

**✗ Attach interventions** - That's Phase 4
**✗ Build ROC curves** - That's Phase 3 (after attack surface exhaustion)
**✗ Tune for specific outcomes** - Thresholds from data, not goals
**✗ Define policy** - Detection only, no response yet

---

## Success Criteria

**Phase 2 succeeds if:**

1. ✓ Convergence defined operationally (not poetically)
2. ✓ Regime-aware thresholds (not single threshold)
3. ✓ Baseline-derived (not intuition)
4. ✓ Persistence-required (not instant)
5. ✓ Instrumented (statistics logged)
6. ✓ Divergence detected (premature convergence caught)

**Phase 2 fails if:**

- ✗ Thresholds come from intuition, not data
- ✗ Single diversity threshold across regimes
- ✗ No persistence requirement (instant detection)
- ✗ Policy attached before validation

---

## Files Created

```
chronomoe_integration/convergence.py         ~500 lines (definition + detector)
experiment_convergence_baseline.py           ~400 lines (measurement + instrumentation)
```

---

## Next: Phase 3 (Attack Surface Exhaustion)

**After Phase 2 validates convergence detection:**

1. Implement challenge receipt logging (behavioral, not self-report)
2. Measure method-space diversity over time
3. Detect clustering (receipts concentrating in small method family)
4. Test: Does clustering precede convergence? Predictive power?
5. Build ROC curve (precision/recall against convergence events)

**Foundation:** Phase 1 (deformation curve) + Phase 2 (convergence definition)

---

**Status:** Design complete. Instrumentation running. Baseline measurement in progress.
