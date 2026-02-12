# Convergence Characteristics: Measurement Report

**Goal:** Sharpen hypotheses before swiss-ai/MoE real training validation.

**Method:** Measure, don't tune. K=5 vs K=3 comparison on synthetic data.

---

## Executive Summary

### Key Findings

1. **K=5 (Production):** Convergence is extremely rare (0-0.12% of time, 0-1 episodes in 5000 steps)
2. **K=3 (Validation):** Convergence occurs 2-3x more frequently (0.24-0.28% of time, 3 episodes)
3. **Lead time:** Highly variable (13-452 steps, mean 114-266 steps depending on seed/K)
4. **All convergence in saturation regime** (scar 4.7-4.9) - none in exploration/transition
5. **Volatility unchanged** after convergence (0.90-0.91x) - not a performance collapse signal

### Implications

- **K=5 requires deep structural stability** that synthetic random walk rarely achieves
- **Lead time is long but variable** (100-300 steps typical, but can be as short as 13 or as long as 450)
- **Convergence ≠ performance collapse** (at least in routing volatility proxy)
- **Saturation-only convergence** suggests transition regime exhaustion doesn't lead to convergence

---

## 1. Base Rate Analysis

### Seed 42

| Metric | K=5 (Production) | K=3 (Validation) |
|--------|------------------|------------------|
| Convergence episodes | 1 | 3 |
| % time converged | 0.12% | 0.28% |
| Mean episode duration | 6.0 steps | 4.7 ± 0.9 steps |
| Max persistence reached | 5 | 5 |

### Seed 123

| Metric | K=5 (Production) | K=3 (Validation) |
|--------|------------------|------------------|
| Convergence episodes | 0 | 3 |
| % time converged | 0.00% | 0.24% |
| Mean episode duration | N/A | 4.0 steps |
| Max persistence reached | 3 | 3 |

### Persistence Distribution (Seed 42)

Both K=5 and K=3 show identical persistence distribution (detector runs same way, threshold only affects final decision):

- Persistence 1: 97 occurrences (80%)
- Persistence 2: 19 occurrences (16%)
- Persistence 3: 3 occurrences (2.5%)
- Persistence 4: 1 occurrence (0.8%)
- Persistence 5: 1 occurrence (0.8%)

**Interpretation:** Most of the time (80%), conditions hold for only 1 step before one breaks. Persistence ≥3 is rare (~3% of time). K=5 requires an extremely rare event (0.8% of observations).

### Key Insight: Convergence is Rare by Design

Even in **extreme saturation mode** (99% agreement, 0.5% downgrade rate, 96% input clustering):
- K=5: 0-0.12% of time converged
- K=3: 0.24-0.28% of time converged

This is **expected behavior** for a conservative detector with persistence requirements. Convergence should be rare - it's not a default state but a pathological one.

---

## 2. Lead Time by Regime

### Seed 42

| Regime | K=5 | K=3 |
|--------|-----|-----|
| Exploration | No events | No events |
| Transition | No events | No events |
| Saturation | 452 steps (n=1) | 266 ± 174 steps (n=3) |

### Seed 123

| Regime | K=5 | K=3 |
|--------|-----|-----|
| Exploration | No events | No events |
| Transition | No events | No events |
| Saturation | No events | 114 ± 125 steps (n=3) |

### Key Insights

**1. Saturation-only convergence**
- 100% of convergence events occur in saturation regime (scar ≥ 4.0)
- Transition regime exhaustion does NOT lead to convergence in this setup
- Hypothesis: Convergence requires both method exhaustion AND deep scar accumulation

**2. Lead time high variability**
- Range: 13-452 steps (35x variation)
- Mean: 114-266 steps depending on seed/K
- Std: 124-174 steps (comparable to mean)

**Interpretation:** Lead time is NOT a tight constant. Some convergence happens quickly after exhaustion (13 steps), others take hundreds of steps (452). This suggests exhaustion creates a **precondition** but convergence timing depends on additional factors (domain stability, randomness in agreement runs).

**3. K=5 vs K=3 lead time**
- K=5: Longer lead times (452 steps, n=1)
- K=3: Shorter lead times (114-266 steps, n=3)

This makes sense: K=5 requires more stable conditions, which only occur deep into saturation after long periods of exhaustion. K=3 can trigger earlier when conditions briefly align.

---

## 3. Convergence Proximity

### Conditions at Convergence

| Metric | K=5 (Seed 42) | K=3 (Seed 42) | K=3 (Seed 123) |
|--------|---------------|---------------|----------------|
| Average diversity | 0.9265 | 0.9269 ± 0.0139 | 0.9278 ± 0.0182 |
| Average scar strength | 4.93 | 4.74 ± 0.17 | 4.71 ± 0.13 |
| Regime distribution | 100% saturation | 100% saturation | 100% saturation |

### Key Insights

**1. Deep saturation requirement**
- Scar strength: 4.7-4.9 (well above saturation threshold of 4.0)
- Diversity (routing entropy): ~0.93 (low, but not zero)
- All events in saturation regime

**2. Tight clustering**
- Diversity std is very small (0.01-0.02)
- Convergence happens under similar conditions each time
- Not scattered across regime - concentrated in deep saturation

**3. No exploration/transition convergence**
- Despite exhaustion events in transition regime (from Phase 3 results)
- Convergence requires BOTH exhaustion AND saturation
- Validates two-signal architecture

---

## 4. Performance Proxy

### Routing Volatility Before/After Convergence

| Metric | K=5 (Seed 42) | K=3 (Seed 42) | K=3 (Seed 123) |
|--------|---------------|---------------|----------------|
| Volatility before | 0.1649 | 0.1603 | 0.1595 |
| Volatility after | 0.1499 | 0.1447 | 0.1453 |
| Ratio (after/before) | 0.91x | 0.90x | 0.91x |
| Interpretation | Unchanged | Unchanged | Unchanged |

### Key Insights

**1. Volatility slightly decreases (not increases)**
- Ratio: 0.90-0.91x (9-10% decrease)
- Direction: After convergence, routing becomes slightly MORE stable
- Magnitude: Small (within ~10%)

**2. NOT a performance collapse signal**
- If convergence caused performance degradation, volatility would INCREASE (model thrashing)
- Instead, volatility decreases or stays flat
- Suggests convergence is "locked in" state, not chaotic failure

**3. Synthetic limitation**
- This is routing volatility, not actual task performance
- Real performance metrics would require:
  - Task loss (not available in synthetic setup)
  - Adaptation speed under domain shift
  - Generalization error

---

## Cross-Seed Stability

### Seed 42 vs Seed 123

| Metric | Seed 42 (K=3) | Seed 123 (K=3) |
|--------|---------------|----------------|
| Convergence episodes | 3 | 3 |
| % time converged | 0.28% | 0.24% |
| Mean lead time | 266 ± 174 steps | 114 ± 125 steps |
| Scar at convergence | 4.74 ± 0.17 | 4.71 ± 0.13 |
| Volatility ratio | 0.90x | 0.91x |

### Consistency Across Seeds

**Stable metrics:**
- Number of episodes (3 both seeds)
- % time converged (~0.25%)
- Scar strength at convergence (~4.7)
- Volatility ratio (~0.90x)

**Variable metrics:**
- Lead time (2x variation: 114 vs 266 steps)
- Lead time std (also high: 125-174 steps)

**Interpretation:** Core patterns (convergence rarity, saturation requirement, locked-in behavior) are consistent. Lead time timing is variable but order of magnitude (~100-300 steps) is stable.

---

## Hypotheses Sharpened

### H1: K=5 Requires Structural Momentum (Confirmed)

**Evidence:**
- K=5: 0-1 episodes (max persistence 3-5)
- K=3: 3 episodes (max persistence 3-5)
- Seed 123 with K=5: Zero convergence (max persistence only 3)

**Conclusion:** Random walk without optimizer inertia cannot sustain 5 consecutive intervals of alignment. K=5 is appropriate for production (prevents transient spikes) but too strict for synthetic validation.

### H2: Lead Time Varies by Regime (Partially Confirmed)

**Evidence:**
- All convergence in saturation (scar 4.7-4.9)
- No convergence in exploration or transition
- Lead times: 13-452 steps (high variance)

**Conclusion:** Lead time is saturation-only, not regime-stratified. But within saturation, lead time is highly variable (13-452 steps). Exhaustion creates precondition, but convergence timing depends on stochastic alignment runs.

### H3: Convergence ≠ Performance Collapse (Supported)

**Evidence:**
- Volatility decreases 9-10% after convergence (not increases)
- Routing becomes more stable, not chaotic
- "Locked in" pattern, not thrashing

**Conclusion:** Convergence is stable agreement, not performance degradation. However, this is synthetic routing volatility - real performance metrics needed for conclusive evidence.

### H4: Exhaustion is Necessary but Not Sufficient (Confirmed)

**Evidence:**
- 100% of convergence preceded by exhaustion
- But most exhaustion doesn't lead to convergence (precision 6-27%)
- Convergence requires deep saturation (scar ~4.9) beyond just exhaustion

**Conclusion:** Two-signal architecture validated. Exhaustion alone isn't enough - need deep scar accumulation too.

---

## Recommendations for Real Training

### 1. Restore K=5 as Production Default

**Rationale:** K=5 is theoretically grounded (persistence prevents transient spikes). Synthetic validation failure is expected due to lack of optimizer momentum.

**Action:** Update convergence.py to K_persistence=5 with comment noting synthetic validation used K=3.

### 2. Test on swiss-ai/MoE with Real Optimizer

**Expected:** Real training with Adam momentum should sustain K=5 more easily than random walk.

**If K=5 fails:** Investigate gradient memory patterns, not immediately recalibrate threshold.

### 3. Collect Real Performance Metrics

**Synthetic proxy limitations:** Routing volatility is not task performance.

**Real metrics needed:**
- Task loss before/after convergence
- Adaptation speed under domain shift
- Generalization error on held-out data

### 4. Stratify by Training Phase (Not Just Scar Regime)

**Observation:** All convergence in deep saturation (scar 4.7-4.9).

**Implication:** Early training (scar < 4.0) won't see convergence regardless of exhaustion.

**Action:** Phase-aware intervention policy - don't expect convergence signals in early training.

### 5. Accept High Lead Time Variance

**Observation:** Lead time range 13-452 steps (mean ~100-300).

**Implication:** Early warning window is ~100-300 steps, but can be as short as 13.

**Action:** Policy should trigger on exhaustion + 50-step persistence, not wait for convergence confirmation.

---

## Conclusion

**What Synthetic Data Validated:**
- ✓ Detector plumbing works (both K=5 and K=3)
- ✓ Convergence is rare by design (0-0.28% of time)
- ✓ Lead time exists (13-452 steps, mean ~100-300)
- ✓ Saturation-only convergence (all events at scar ~4.7-4.9)
- ✓ Locked-in pattern (volatility decreases, not increases)
- ✓ Cross-seed consistency (patterns stable, timing variable)

**What Synthetic Data Cannot Validate:**
- ✗ K=5 persistence threshold (requires optimizer momentum)
- ✗ Performance consequences (routing volatility ≠ task loss)
- ✗ Regime-stratified lead times (all convergence in saturation)
- ✗ Intervention effectiveness (no policy integration)

**Next Step:**
Move to swiss-ai/MoE real training for:
1. K=5 persistence validation with optimizer inertia
2. Real performance metrics (loss, adaptation speed)
3. Phase-aware policy integration
4. Exhaustion→intervention→outcome causality

**No Further Tuning on Synthetic Data.**

Restore K=5, document synthetic limitations, proceed to real training.
