# Geometric Deformation Experiment

**Date:** 2026-02-12
**Status:** Running
**Purpose:** Measure whether scars actually contract reachable manifold (architecture vs. metaphysics)

---

## The Question

Halcyon: "Until you can show scars → measurable contraction and epoch review → measurable expansion, it's still metaphysics."

This experiment answers that question with numbers.

---

## Hypothesis

**Scars contract the reachable solution manifold.**

If true:
- Motif diversity drops by ≥5% under domain shift with scars active
- Epoch review (removing obsolete scars) expands motif diversity by ≥5%

If false:
- Scars are bookkeeping, not architecture
- Return to design phase

---

## Experimental Design

### Phase 1: Baseline Reachability
- Dataset: Arithmetic progressions (1,2,3,4,5,...)
- Scars: None
- Measure: Motif diversity, effective rank, router entropy

### Phase 2: Train with Scar Accumulation
- Dataset: Arithmetic progressions
- Steps: 5000
- Simulate: Mark half of experts as "scarred" (high penalties)
- Mechanism: Use existing `expert_penalties` dict in controller

### Phase 3: Domain Shift with Scars Active
- Dataset: Geometric progressions (1,2,4,8,16,...)
- Scars: Active (penalties applied to scarred experts)
- Measure: Motif diversity, effective rank, router entropy
- **Critical**: Contraction = baseline - scarred_new

### Phase 4: Epoch Review (Remove Obsolete Scars)
- Simulate: Remove penalties from all scarred experts
- Rationale: Scars formed under arithmetic don't apply to geometric

### Phase 5: Post-Review Reachability
- Dataset: Geometric progressions
- Scars: Removed
- Measure: Motif diversity, effective rank, router entropy
- **Critical**: Expansion = recovered - scarred_new

---

## Metrics

### Core Metric: Motif Diversity
- Ratio of unique routing patterns (4-token windows)
- Sensitive to manifold contraction (collapses → fewer unique patterns)
- **Falsifiability**: Must drop ≥5% with scars, recover ≥5% after review

### Supporting Metrics
- **Effective rank**: Participation ratio over routing logits (normalized to avoid amplitude drift)
- **Router entropy**: Shannon entropy of routing distribution
- **T_90 recovery**: Time to reach 90% of asymptotic entropy (adaptation speed)

### Why Motif Diversity?
Entropy alone can lie - system can jitter in small basin and look diverse. Motif diversity captures temporal structure and detects collapse even when entropy stays high.

---

## Domain Shift: Arithmetic → Geometric

**Not "different random seed" - genuine structural shift:**

Arithmetic progressions:
- [1, 2, 3, 4, 5, ...]
- [10, 12, 14, 16, 18, ...]
- Linear pattern detection required

Geometric progressions:
- [1, 2, 4, 8, 16, ...]
- [3, 9, 27, 81, ...]
- Multiplicative pattern detection required

Experts specialized for linear growth fail on exponential growth. Scars formed under arithmetic should become obsolete under geometric.

---

## Implementation Notes

### Scar Simulation
Using existing controller infrastructure:
```python
# Apply scars (penalties)
for expert_id in scarred_experts:
    model.moe.controller.expert_penalties[expert_id] = 5.0

# Remove scars (epoch review)
for expert_id in scarred_experts:
    del model.moe.controller.expert_penalties[expert_id]
```

This tests the measurement harness without requiring full scar system implementation.

### Routing Geometry Measurement
```python
def measure_routing_geometry(layer, dataset, num_steps=1000):
    """
    Measure routing behavior distribution.

    Returns:
        - motif_diversity: Ratio of unique routing patterns
        - effective_rank: Participation ratio (SVD of normalized logits)
        - router_entropy: Shannon entropy
        - num_unique_motifs: Absolute count of unique patterns
    """
```

### Critical Refinements (from Halcyon)
1. **Effective rank**: Normalize logits to avoid measuring amplitude drift
2. **Motif diversity**: Believe this over entropy if they disagree
3. **State restoration**: Snapshot both weights AND optimizer between scar tests

---

## Falsifiability Criteria

**Hypothesis CONFIRMED if:**
- Contraction ≥ 5% (motif diversity drops under shift with scars)
- Expansion ≥ 5% (motif diversity recovers after epoch review)

**Hypothesis FALSIFIED if:**
- Contraction < 5% → Scars don't contract manifold → bookkeeping, not architecture
- Expansion < 5% → Epoch review doesn't restore reachability → mechanism ineffective

**If falsified:** Return to design phase. Governance is metaphysics without deformation.

---

## Why This Matters

From Halcyon:
> Identity = constraint accumulation
> Judgment = artifact-gated commitment
> Skepticism = adversarial perturbation
> Wisdom = periodic scar audit

Clean framing, but **clean framing doesn't make it real.**

If scars don't contract the manifold, then:
- "Identity = constraint accumulation" is just poetry
- "Wisdom = periodic scar audit" is governance theater
- The whole dialectical engine is institutional metaphor applied to routers

This experiment separates architecture from philosophy.

---

## Expected Runtime

- Training: 5000 steps (~10-15 minutes)
- Measurement: 5 phases × 500-1000 steps each (~5-10 minutes)
- Total: ~15-25 minutes

---

## Artifacts

- **Code**: `experiment_domain_shift.py`, `routing_geometry.py`, `synthetic_datasets.py`
- **Output**: `domain_shift_results.json` (contraction/expansion measurements)
- **Verdict**: Exit code 0 (confirmed) or 1 (falsified)

---

## One-Sentence Summary (for reviewers)

"Scars accumulated under arithmetic patterns contract routing diversity by [X]%, epoch review detects obsolescence under geometric patterns and restores reachability by [Y]%."

Fill in [X] and [Y] when experiment completes.

---

**Status:** Experiment running. Waiting for measurements.
