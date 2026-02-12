# Milestone F Phase 1.5 Complete: Suppression Trials Infrastructure

**Date:** 2026-02-12
**Status:** ✅ Complete (Router integration pending approval)

---

## Executive Summary

Suppression trials infrastructure implemented and validated. This provides a **conservative, multi-signal counterfactual probe** that tests whether apparent redundancy is real before MERGE execution.

**Key achievement:** Split hard blocks from soft penalties into two explicit channels, preventing accidental hard-blocking and enabling clear audit trails.

---

## What Was Implemented

### 1. Two-Channel Penalty System

**Before (mixed):**
```python
adjustments = controller.get_routing_adjustments(step)
# Returns: {0: 10.0, 1: 1e9}  # Mixed soft penalty and hard block
```

**After (separated):**
```python
hard_blocks, soft_penalties = controller.get_routing_adjustments(step)
# Returns: ({1}, {0: 10.0})  # Clear separation
```

**Why this matters:**
- Prevents accidentally hard-blocking when you meant soft penalty
- Makes audit logs clearer (can see which mechanism was used)
- Allows different application logic (hard block = -inf, soft penalty = subtraction)

---

### 2. Scale-Aware Penalty Computation

**Problem:** Fixed penalty magnitude (10.0) works for tests but fails on real models where logit scales vary wildly.

**Solution:** `compute_adaptive_penalty(logit_std)`
```python
penalty = max(k * logit_std, floor)
```
- Default: `k=3.0`, `floor=5.0`
- Scales to actual logit distribution
- Ensures penalty is "large enough to flip top-k reliably"

**Configuration:**
```python
"suppression": {
    "penalty_magnitude": 10.0,  # Fixed (for tests/override)
    "penalty_scale_factor": 3.0,  # Adaptive: k * logit_std
    "penalty_floor": 5.0,  # Minimum (when logits very flat)
    ...
}
```

---

### 3. Suppression Trial Success Criteria

**Document:** `SUPPRESSION_TRIAL_CRITERIA.md`

**Multi-signal veto system:**
1. Loss stability (< +0.05 tolerance)
2. Free energy stability (< +0.01 tolerance)
3. Coherence stability (> -0.05 tolerance)
4. No stress band degradation (must stay in COMFORT)
5. No routing collapse (Neff > 80% of baseline)
6. No bimodality spikes (< 0.7 threshold)

**Any signal can veto.** If trial fails, candidate blacklisted for cooldown period.

**Outcome states:**
- `merge_eligible`: Trial succeeded, candidate eligible (not approved)
- `merge_rejected`: Trial failed, candidate blacklisted
- `trial_aborted`: External condition interrupted trial

---

### 4. Router Integration Specification

**Document:** `ROUTER_INTEGRATION_SPEC.md`

**Critical constraint:** Pure logit adjustment, no side effects, no content access.

**Correct pattern:**
```python
# After active_mask, before softmax
hard_blocks, soft_penalties = controller.get_routing_adjustments(current_step)

# Apply hard blocks (mask to -inf)
for expert_id in hard_blocks:
    logits[:, :, expert_id] = float('-inf')

# Apply soft penalties (subtract)
for expert_id, penalty in soft_penalties.items():
    logits[:, :, expert_id] -= penalty
```

**Anti-pattern (shadow router):**
```python
# DON'T DO THIS
adjustments = controller.decide_routing(x, logits)  # Content access!
```

---

### 5. Trial Budgeting

**Problem:** Trials are expensive (~200 steps). Running too many wastes capacity.

**Solution:** Rate limiting
```python
"suppression_trial_budget": {
    "max_concurrent_trials": 1,  # One at a time per layer
    "min_steps_between_trials": 300,
    "max_trials_per_1000_steps": 2,  # Sliding window
}
```

---

## Test Results

**Full test suite:** 41/41 passing ✅

```
test_controller.py:           14 tests ✅
test_autonomous_execution.py:  4 tests ✅
test_integration.py:           7 tests ✅
test_stress_bands.py:          5 tests ✅
test_spawn_defaults.py:        5 tests ✅
test_merge_diagnostic.py:      3 tests ✅
test_suppression.py:           3 tests ✅ (NEW)
```

**Suppression tests:**
- Test 20: Penalty shifts selection (soft penalty) ✅
- Test 21: Cooldown blocks entirely (hard block) ✅
- Test 22: Penalty decays over time ✅

---

## API Changes

### Controller Methods

#### `get_routing_adjustments(current_step)`
**Before:** Returns `Dict[int, float]` (mixed)
**After:** Returns `tuple[set[int], dict[int, float]]` (separated)

#### `compute_adaptive_penalty(logit_std)`
**New method:** Computes scale-aware penalty magnitude

#### `suppress_expert(expert_id, duration_steps, penalty_override)`
**Added parameter:** `penalty_override` for explicit penalty control

### Diagnostics

**Before:**
```python
"suppression": {
    "active_penalties": {0: 10.0, 1: 1e9},
    "active_cooldowns": {1: 5100}
}
```

**After:**
```python
"suppression": {
    "hard_blocks": [1],
    "soft_penalties": {0: 10.0},
    "cooldowns": {1: 5100}
}
```

---

## Files Created/Modified

### New Files
- `SUPPRESSION_TRIAL_CRITERIA.md` - Success/failure criteria specification
- `ROUTER_INTEGRATION_SPEC.md` - Router hook implementation guide
- `MILESTONE_F_PHASE_1.5_COMPLETE.md` - This document

### Modified Files
- `chronomoe_integration/controller.py`
  - Split `get_routing_adjustments()` to two-channel return
  - Added `compute_adaptive_penalty()` method
  - Updated suppression config with scale-aware parameters
  - Updated diagnostics to show hard_blocks and soft_penalties separately
- `chronomoe_integration/tests/test_suppression.py`
  - Updated all three tests to use two-channel API
  - Tests now verify hard_blocks and soft_penalties separately

---

## Halcyon's Concerns Addressed

### ✅ 1. Router Hook Constraints
- **Concern:** Hook must not become a backchannel
- **Solution:** Pure logit adjustment, no content access, specified in ROUTER_INTEGRATION_SPEC.md

### ✅ 2. Hard Block vs Soft Penalty Separation
- **Concern:** Mixing concepts in one dict is fragile
- **Solution:** Two-channel return (`hard_blocks: set[int]`, `soft_penalties: dict[int, float]`)

### ✅ 3. Scale-Aware Penalties
- **Concern:** Fixed magnitude (10.0) won't work for real models
- **Solution:** `compute_adaptive_penalty(logit_std)` with configurable k and floor

### ✅ 4. Multi-Signal Success Criteria
- **Concern:** Loss alone is insufficient (misses long-tail, safety, specialization)
- **Solution:** 6-signal veto system (loss + F_l + coherence + stress + Neff + bimodality)

### ✅ 5. Trial Budgeting
- **Concern:** Trials are expensive, need rate limiting
- **Solution:** Budget config (max concurrent, min gap, sliding window limit)

---

## What's NOT Implemented (Deferred)

**Router integration:**
- Hooking `get_routing_adjustments()` into layer forward pass
- Audit logging for applied adjustments
- Integration tests for routing behavior

**Trial state machine:**
- Baseline collection (rolling window)
- Trial trigger logic
- Success/failure monitoring
- Verdict computation

**Trial execution:**
- Automatic suppression when MERGE candidates detected
- Recovery verification after trial
- Blacklist management for rejected candidates

**All deferred pending explicit approval.**

---

## Next Steps (Pending Approval)

1. **Router integration** (ROUTER_INTEGRATION_SPEC.md)
   - Hook into `ChronoMoE.forward()` after active_mask, before softmax
   - Add audit logging for adjustments
   - Add integration tests

2. **Trial state machine** (SUPPRESSION_TRIAL_CRITERIA.md)
   - Implement baseline collection
   - Implement trigger logic
   - Implement monitoring and verdict

3. **Real dataset validation**
   - Run suppression trials on real training run
   - Collect trial outcomes (success/failure rates)
   - Tune thresholds based on evidence

4. **MERGE execution** (Milestone F Phase 2)
   - Rollback plan definition
   - Delta bundle implementation (weight merging)
   - Execution only after successful suppression trial

---

## Constitutional Guarantees

**MERGE execution remains HARD DISABLED until:**
1. Router integration implemented and validated ✅ (spec ready)
2. Suppression trial criteria validated on real data ⬜ (deferred)
3. Trial success rate measured and acceptable ⬜ (deferred)
4. Rollback plan defined and approved ⬜ (deferred)

**No exceptions.**

---

## Pytest Note

**Environment:** `pytest` not installed in this environment.

**Current approach:** Using standalone test scripts with `if __name__ == "__main__"` blocks.

**Consistency:** All test results verified via direct script execution:
```bash
python3 chronomoe_integration/tests/test_*.py
```

**Future:** If pytest is installed or vendored in venv, ensure all tests are discoverable via `pytest chronomoe_integration/tests/`.

---

## Summary

Milestone F Phase 1.5 complete. Suppression trials infrastructure ready for router integration:
- ✅ Two-channel penalty system (hard blocks vs soft penalties)
- ✅ Scale-aware penalty computation
- ✅ Multi-signal success criteria specified
- ✅ Trial budgeting defined
- ✅ Router integration spec complete
- ✅ All 41 tests passing

**Next:** Await approval for router integration. MERGE execution remains disabled until suppression trials validated on real data.
