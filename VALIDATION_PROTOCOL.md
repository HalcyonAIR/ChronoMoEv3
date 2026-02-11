# SPLIT Operation Validation Protocol

**Date:** 2026-02-09
**Status:** FROZEN
**Milestone:** E (SPLIT operation)

---

## Overview

This document defines the repeatable validation protocol for the SPLIT operation in ChronoMoE. Two validation harnesses confirm that SPLIT works correctly under both controlled invariant testing and semi-realistic controller-driven dynamics.

---

## Validation Harnesses

### 1. Simplified Mechanics Validation

**File:** `validation_training_simple_v2.py`

**Purpose:** Unit-level mechanics test that validates core invariants with manual state forcing.

**Pass Criteria:**
1. SPLIT proposed (when bimodality injected)
2. SPLIT blocked in STRAIN (constitutional blocking)
3. SPLIT executed in COMFORT (after calm credit accumulates)
4. Parent expert pruned (ARCHIVED state)
5. Children in PROBATION

**Implementation:**
- Manually injects bimodal observations (alternating +centroid and -centroid)
- Manually forces band transitions (sustained high/low stress updates)
- Verifies guards work when explicitly manipulating states

**Result:** ✅ PASS (single seed validation)

**Usage:**
```bash
python3 validation_training_simple_v2.py
```

---

### 2. Semi-Real Validation (PRIMARY PROTOCOL)

**File:** `validation_training_semireal.py`

**Purpose:** Controller-driven dynamics validation where stress rises/falls naturally and band transitions emerge from F_l changes.

**Pass Criteria:**
1. SPLIT proposed (based on bimodality signal)
2. SPLIT blocked during high-stress segment (band != COMFORT)
3. SPLIT executed after stress drops and calm accumulates
4. Parent expert pruned (ARCHIVED state)
5. Children in PROBATION

**Guards (Enforceable Assertions):**
6. Band must transition to STRAIN within 20 steps of high-stress window start
7. Band must return to COMFORT within 50 steps of high-stress window end
8. Capacity must never hit zero (prevents masking behavior)

**Configuration:**
- **Seeds:** 42, 7, 1337
- **Max steps:** 500
- **High-stress window:** steps 90-190
- **Stress amplification:** 20× during window
- **Max experts:** 12 (prevents capacity exhaustion)
- **COMFORT ceiling:** 3.0
- **STRAIN ceiling:** 5.0
- **Split calm steps:** 300

**Validated Results (2026-02-09):**
```
SEMIREAL_PASS seed=42 split_proposed=1 blocked_steps=95 executed_step=493 calm_at_exec=300
SEMIREAL_PASS seed=7 split_proposed=1 blocked_steps=99 executed_step=497 calm_at_exec=300
SEMIREAL_PASS seed=1337 split_proposed=1 blocked_steps=96 executed_step=494 calm_at_exec=300
```

**Key Dynamics Verified:**
- Stress rose because signals rose (loss amplified 20× during window)
- Band changed because stress crossed thresholds (COMFORT → STRAIN at step ~90)
- SPLIT blocked because band != COMFORT (steps 90-190)
- SPLIT allowed after stress dropped and calm accumulated (step ~493-497)
- No capacity exhaustion masking behavior

**Usage:**
```bash
python3 validation_training_semireal.py
```

**Expected Output:**
- All 3 seeds PASS
- One-line summary per seed (grep-friendly)
- Guards all verified

---

## Validation Philosophy

### What Simplified Validation Tests
- **Invariants:** Do the gates work when manually forced?
- **Mechanics:** Can the system execute SPLIT given the right conditions?
- **Scope:** Tightly coupled to stress-band update mechanics

**Limitation:** Not validating real interaction between observation stream, F_l dynamics, band transitions, and execution gating.

### What Semi-Real Validation Tests
- **Constitutional Behavior:** Does the system behave correctly under load?
- **Emergence:** Do state transitions emerge from controller observations?
- **Guards:** Are the transitions actually happening (not scripted)?

**Limitation:** Still uses injected bimodality (not fully natural). Loss amplification is artificial.

---

## Next Steps

After semi-real validation passes across all seeds:
1. ✅ DONE: Freeze this protocol in repo docs
2. ⏭️ NEXT: Run real training on real dataset
   - Use actual language modeling task
   - Let bimodality emerge naturally (if it does)
   - No artificial stress amplification
   - Observe whether SPLIT fires in practice

**Critical:** Do not move to real training until semi-real validation is repeatable across seeds.

---

## Frozen Configuration

**DO NOT MODIFY** these parameters without re-validating:
- High-stress window: `(90, 190)` steps
- Stress amplification: `20.0×`
- Max experts: `12`
- COMFORT ceiling: `3.0`
- STRAIN ceiling: `5.0`
- Split calm steps: `300`
- Seeds: `[42, 7, 1337]`

If any parameter changes, re-run full validation and update this document.

---

## Validation Artifacts

**Timeline files (preserved):**
- `timeline_simple_v2.json` - Simplified mechanics validation
- `timeline_semireal.json` - Semi-real validation (seed 42)

**Code files (frozen):**
- `validation_training_simple_v2.py` - Mechanics test
- `validation_training_semireal.py` - Semi-real test (PRIMARY)

---

## Grep-Friendly Summary Format

```
SEMIREAL_PASS seed={seed} split_proposed={0|1} blocked_steps={count} executed_step={step|NONE} calm_at_exec={calm|NONE}
```

**Example:**
```
SEMIREAL_PASS seed=42 split_proposed=1 blocked_steps=95 executed_step=493 calm_at_exec=300
```

Use this format for CI/CD integration and automated validation runs.

---

## Validation Complete

**Status:** ✅ ALL SEEDS PASSED
**Date:** 2026-02-09
**Validated By:** Claude Code (CC)
**Approved By:** Halcyon

SPLIT operation validated under controller-driven dynamics. Ready to proceed to real training validation.
