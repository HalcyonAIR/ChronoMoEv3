# SPLIT Logic - FROZEN (Milestone E Complete)

**Status:** 🔒 **FROZEN - DO NOT MODIFY**
**Date:** 2026-02-11
**Tag:** `milestone-e-complete`

---

## Policy

SPLIT logic is frozen effective immediately. No changes allowed except:

### ✅ ALLOWED
- **Bugfixes** - Critical correctness issues only
- **Instrumentation** - Adding logging/diagnostics for observability
- **Test clarifications** - Improving test documentation or assertions

### ❌ NOT ALLOWED
- Threshold tuning (split_threshold, split_calm_steps, etc.)
- Refactoring or code reorganization
- Feature additions or enhancements
- Algorithm changes (bimodality detection, latch logic, etc.)

---

## Rationale

SPLIT has been validated across all scenarios:
- 35/35 unit tests passing
- 3/3 semi-real validation seeds passing
- 3/3 real dataset validation seeds passing
- Children outcomes consistently resolved (both_graduated)
- Pending latch working correctly across band transitions
- Lineage cooldown preventing split cascades

**The SPLIT era is complete. No heroics.**

---

## Frozen Components

### Files (SPLIT-specific sections)
- `chronomoe_integration/controller.py`
  - `_try_propose_split()` method
  - `_check_pending_split_latch()` method
  - `PendingSplitLatch` dataclass
  - SPLIT-related config parameters
  - Split lineage tracking

- `chronomoe_integration/chronomoe_layer.py`
  - `split_expert()` method
  - SPLIT execution logic in `process_controller_proposals()`

- `chronomoe_integration/stress_bands.py`
  - `allow_split` gate logic
  - `split_calm_steps` configuration

### Configuration Parameters (FROZEN)
```python
"bimodality": {
    "split_threshold": 0.5,  # FROZEN - do not tune
},
"triggers": {
    "split_calm_steps": 300,  # FROZEN - do not tune
    "split_latch_ttl": 500,  # FROZEN - do not tune
    "split_lineage_cooldown": 500,  # FROZEN - do not tune
}
```

---

## If You Need to Change SPLIT Logic

1. Open an issue documenting the problem
2. Provide evidence (test failures, validation breakage, etc.)
3. Get explicit approval from Halcyon
4. Document the change in `MILESTONE_E_ENHANCEMENTS.md`
5. Update validation harness if needed
6. Rerun full validation suite (35 tests + multi-seed validation)

**No exceptions. The constitution is weight-bearing.**

---

## Current Focus

**Milestone F Phase 1:** MERGE diagnostic-only mode

SPLIT is done. Move forward.
