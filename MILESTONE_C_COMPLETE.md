# Milestone C Complete: Free Energy Sensor via Controller

**Date:** 2026-02-10
**Status:** ✅ Complete - All success criteria met
**Tests:** 26/26 passing (24 existing + 2 new)

---

## Summary

Issue #3C complete: Free energy computation integrated via the controller boundary.

**Scope:** Diagnostics only. No triggers, no edits, no causal effects.

**Key principle:** Signal layer computes F_l but does NOT affect routing or lifecycle decisions.

**Partial F_l:** Misfit term left as optional/None until canonical misfit proxy exists in swiss-ai/MoE.

---

## What Was Implemented

### 1. Free Energy Module (`chronomoe_integration/free_energy.py`)

**Purpose:** Compute layer-level free energy F_l and its components (Phase 3 ported from ChronoMoEv3).

**Components:**
- `FreeEnergyComponents`: Dataclass for decomposed F_l terms
  - `misfit`: Optional (None for partial F_l)
  - `complexity`: λ * (N_active / N_max) - wasteful capacity penalty
  - `redundancy`: ρ * R_l - duplicate expert penalty
  - `instability`: κ * I_l - coherence variance penalty
  - `total`: Sum of all components (partial if misfit is None)
  - `is_partial`: True if misfit term missing

- `FreeEnergyState`: Per-layer state with components and raw scores
  - Tracks all inputs needed to compute F_l
  - Serializes to dict for diagnostics

- `compute_complexity_term()`: Capacity waste metric
  - Formula: λ * (num_active / max_experts)
  - Normalized by max capacity → [0, λ]
  - Penalizes having too many experts

- `compute_redundancy_term()`: Duplicate expert detection
  - Computes mean output per expert [num_experts, d_model]
  - Pairwise cosine similarity (scale-invariant)
  - R_l = mean pairwise similarity (1 = all identical)
  - Returns (weighted_redundancy, raw_redundancy)

- `compute_instability_term()`: Coherence variance
  - Formula: variance(|phi_fast - phi_slow|) across active experts
  - High variance = expert degrading or oscillating
  - Requires at least 2 active experts (avoid variance warning)
  - Returns (weighted_instability, raw_instability)

- `compute_free_energy()`: Full F_l computation
  - F_l = [misfit] + complexity + redundancy + instability
  - If misfit is None, returns partial F_l
  - Returns (components, free_energy)

- `create_free_energy_state()`: Factory for FreeEnergyState
  - Computes all components
  - Stores raw scores for diagnostics

**File:** `chronomoe_integration/free_energy.py` (~330 lines)

### 2. Controller Integration

**Purpose:** Free energy tracking inside the controller boundary.

**Changes to `controller.py`:**
- Added `free_energy_state: Optional[FreeEnergyState]` to __init__
- Implemented `_update_free_energy(snapshot)` method:
  - Builds active mask from utilization (min_tokens_active threshold)
  - Extracts coherence_fast/slow from coherence states
  - Calls create_free_energy_state() with all inputs
  - Explicitly passes misfit=None (partial F_l)

- Updated `get_diagnostics()` to return:
  - `free_energy`: FreeEnergyState.to_dict() if state exists, else None
  - Includes components, raw scores, num_active, max_experts

- Config already had free energy weights (unchanged):
  - `lambda_complexity`: 0.01
  - `rho_redundancy`: 0.01
  - `kappa_instability`: 0.01
  - `min_tokens_active`: 1
  - `min_tokens_redundancy`: 100 (for future use)

**Key insight:** All free energy computation happens INSIDE the controller. swiss-ai/MoE code only calls `observe()` and `get_diagnostics()`.

### 3. Test Suite (`tests/test_controller.py`)

**2 new deterministic pathology tests:**

**Test 8: Redundancy Injection**
- Creates 4 experts producing nearly identical outputs (tiny noise)
- Uses torch.manual_seed(42) for deterministic results
- Runs 50 observations
- Verifies high redundancy score (> 0.9)
- **Result:** redundancy_score=0.9999 ✓
- Verifies NO edit proposals

**Test 9: Instability Injection**
- Expert 0 oscillates between aligned and orthogonal to mixture
- Other 3 experts stable (for contrast)
- Runs 100 observations (allow EMA to capture variance)
- Verifies high instability score (> 0.001)
- **Result:** instability_score=0.0017 ✓
- Verifies NO edit proposals

**File:** `chronomoe_integration/tests/test_controller.py` (~600 lines)

---

## Success Criteria: All Met ✅

- [x] `free_energy.py` created with component computation
- [x] FreeEnergyComponents tracks complexity, redundancy, instability
- [x] FreeEnergyState stores F_l and raw scores
- [x] `controller._update_free_energy()` integrates free energy tracking
- [x] `controller.get_diagnostics()` returns F_l and components
- [x] Misfit term left as optional/None (partial F_l)
- [x] Test 8 verifies high redundancy detection (duplicate experts)
- [x] Test 9 verifies high instability detection (coherence oscillation)
- [x] Both tests verify NO edit proposals
- [x] Overhead minimal (cosine similarities, variance)
- [x] **24/24 existing tests still pass** (green invariant: now 26/26)

---

## Green Invariant: Maintained ✅

**Before Milestone C:** 24/24 tests passing
- 7 integration tests
- 5 stress band tests
- 5 spawn default tests
- 7 controller tests (5 Milestone A + 2 Milestone B)

**After Milestone C:** 26/26 tests passing
- 7 integration tests (unchanged)
- 5 stress band tests (unchanged)
- 5 spawn default tests (unchanged)
- 9 controller tests (5 Milestone A + 2 Milestone B + 2 Milestone C)

**No regressions.** All old tests still pass.

---

## Git Commits

```
[To be added after commit]
```

**Files created:**
- `chronomoe_integration/free_energy.py`

**Files modified:**
- `chronomoe_integration/controller.py` (integrated free energy tracking)
- `chronomoe_integration/__init__.py` (export FreeEnergyState, FreeEnergyComponents)
- `chronomoe_integration/tests/test_controller.py` (2 new tests)

---

## Validation: No Edit Proposals ✅

**Most important test:** Both Test 8 and Test 9 verify:

```python
proposals = controller.decide()
assert len(proposals) == 0, f"Expected 0 proposals in Milestone C, got {len(proposals)}"
```

**Result:** PASS

**Meaning:** Controller is purely diagnostic. No triggers, no edits, no causal effects. Milestone C scope maintained.

---

## Architecture Boundary: Enforced ✅

**Contract:**
- swiss-ai/MoE code only calls `controller.observe(snapshot)` and `controller.get_diagnostics()`
- No direct imports from `free_energy.py`
- No direct access to free energy state
- Controller is the ONLY interface

**Validation:** Check imports in `chronomoe_layer.py`:
```python
from chronomoe_integration.controller import (
    ChronoController,
    ObservationSnapshot,
    create_controller,
)
# NO import from free_energy.py
```

**Status:** Boundary enforced. ✅

---

## Sample Output

**Controller diagnostics:**
```python
diagnostics = controller.get_diagnostics()

{
    "layer_id": 0,
    "max_experts": 8,
    "observations_count": 50,
    "edits_count": 0,
    "coherence": {
        # ... Milestone A coherence data ...
    },
    "bimodality": {
        # ... Milestone B bimodality data ...
    },
    "free_energy": {
        "layer_id": 0,
        "step": 49,
        "num_active_experts": 4,
        "max_experts": 8,
        "redundancy_score": 0.9999,
        "instability_score": 0.0000,
        "components": {
            "complexity": 0.0050,
            "redundancy": 0.0100,
            "instability": 0.0000,
            "total": 0.0150,
            "is_partial": True,  # No misfit term
        },
    },
}
```

---

## Technical Details

### Free Energy Formulation

**F_l = [misfit] + λ·complexity + ρ·redundancy + κ·instability**

**Components:**

1. **Misfit (optional):** `(1 - Psi_l)` - layer incoherence
   - **Status:** Left as None (partial F_l)
   - **Rationale:** No canonical misfit proxy in swiss-ai/MoE yet
   - **Future:** Could use loss, perplexity, or other training signal

2. **Complexity:** `λ * (N_active / N_max)` - wasteful capacity
   - **Formula:** Normalized by max experts → [0, λ]
   - **Penalizes:** Having too many experts (parsimony)
   - **Weight:** λ = 0.01 (configurable)

3. **Redundancy:** `ρ * R_l` - duplicate expert work
   - **Formula:** R_l = mean pairwise cosine similarity of expert mean outputs
   - **Range:** [0, 1] (0 = all experts diverse, 1 = all identical)
   - **Detects:** Near-duplicate experts (should merge)
   - **Weight:** ρ = 0.01 (configurable)

4. **Instability:** `κ * I_l` - coherence variance
   - **Formula:** I_l = variance(|phi_fast - phi_slow|) across active experts
   - **Measures:** Coherence oscillation (expert degrading)
   - **Requires:** At least 2 active experts (variance needs 2+ samples)
   - **Weight:** κ = 0.01 (configurable)

**Interpretation:**
- **Low F_l:** Layer is efficient (few experts, diverse work, stable coherence)
- **High F_l:** Layer is wasteful (many experts, redundant work, volatile coherence)

**Partial F_l:**
- Without misfit term, F_l measures internal efficiency only
- Still useful for detecting redundancy and instability pathologies
- Can guide lifecycle decisions even without external training loss

### Weights Configuration

**All weights exposed in public config:**
```python
controller = create_controller(
    layer_id=0,
    max_experts=8,
    initial_active=4,
    config={
        "free_energy": {
            "lambda_complexity": 0.01,  # Default
            "rho_redundancy": 0.01,     # Default
            "kappa_instability": 0.01,  # Default
            "min_tokens_active": 1,
        },
    },
)
```

**Rationale for weights:**
- **Equal weighting (0.01):** Treats all three terms as comparable
- **Low magnitude:** Prevents free energy from dominating other signals
- **Configurable:** Can be tuned empirically based on training runs

**Not universal constants:** These are defaults for validation, not absolute truth. Future work may adjust weights based on empirical distribution.

---

## Next Steps: STOP AND REPORT

**Per hard stop rule:** When Milestone C meets success criteria, stop and report.

**Do NOT proceed with Milestone D (autonomous triggers) until approved.**

**Waiting for:** User approval to proceed with Issue #3D.

---

## What's Blocked

- ❌ Issue #3D (autonomous triggers) - blocked until #3C approved
- ❌ Issue #3E (SPLIT/MERGE) - blocked until #3D proven stable

---

## Language Alignment (Used Consistently)

- **Layer 1:** Lifecycle mechanics (frozen except bugfixes)
- **Layer 2:** Decision intelligence (incremental integration)
- **Controller:** The only bridge between layers
- **Signal:** Diagnostic only, not causal (Milestone A + B + C)
- **Trigger:** Causal, forbidden until Milestone D
- **Partial F_l:** Free energy without misfit term (acceptable for internal efficiency)

**Terminology enforced in:**
- Code comments
- Commit messages
- Documentation
- Test names

---

## Reversibility of Understanding: Optimized

**Clarity > cleverness:**
- Simple capacity ratio (N_active / N_max)
- Pairwise cosine similarity (standard metric)
- Variance of coherence delta (basic statistics)
- Explicit partial F_l handling (not hidden)

**Logs > heuristics:**
- `get_diagnostics()` returns full free energy breakdown
- Raw scores (before weighting) logged
- Partial F_l status explicitly tracked

**Boring > impressive:**
- No clever abstractions
- No premature optimizations
- Straightforward port from ChronoMoEv3
- Explicit rather than implicit

---

## Milestone C: Complete ✅

**Status:** All success criteria met. Tests passing. No edit proposals. Boundary enforced.

**Philosophy:** Signal layer is diagnostic, not causal. Partial F_l is acceptable.

**Next:** Stop and report. Await approval for Milestone D.

**Test Coverage:** 26/26 passing
- 7 integration tests (Layer 1 mechanics)
- 5 stress band tests (Layer 1 mechanics)
- 5 spawn default tests (Layer 1 mechanics)
- 9 controller tests (5 Milestone A + 2 Milestone B + 2 Milestone C)

**Completion Date:** 2026-02-10
**Integration:** Incremental, controlled, reversible
**Scope:** Free energy logging only (no triggers, no edits)
