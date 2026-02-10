# Routing Paths and Active Mask Enforcement

## Overview

This document identifies all routing paths in swiss-ai/MoE and verifies that `active_mask` is enforced in every path where experts can be selected.

---

## ChronoMoE Integration (chronomoe_layer.py)

**Status:** ✅ Active mask enforced

**Routing path:**
1. Router outputs logits for all max_experts
2. Probation boost applied
3. **Inactive experts hard-masked to -inf** ← Enforcement point
4. Softmax + top-k
5. Expert dispatch

**Enforcement:** Line ~93-95 in `chronomoe_layer.py`:
```python
# 3. Hard mask inactive experts to -inf (BEFORE softmax)
inactive_mask = ~active_mask
router_logits[:, inactive_mask] = float('-inf')
```

**Validation:** Test `test_ghost_routing_assertion()` verifies:
- Inactive logits = -inf (exact)
- Inactive probabilities = 0.0 (exact)
- Inactive utilization = 0 (exact)

---

## Standard MoE (moe.py - MoE class)

**Status:** ⚠️ No lifecycle support (original swiss-ai implementation)

**Routing path:**
1. Router outputs logits
2. Softmax + top-k OR top-k + softmax
3. Expert dispatch

**Active mask:** Not enforced (standard implementation has no lifecycle)

**Usage:** Replaced by ChronoMoE when lifecycle operations are needed.

**Note:** If using standard MoE without ChronoMoE wrapper, lifecycle operations (spawn/prune) are not available.

---

## ExpertChoiceMoE (moe.py - ExpertChoiceMoE class)

**Status:** ⚠️ Not integrated with ChronoMoE

**Routing path:**
1. Router outputs logits
2. Top-k over tokens (experts choose tokens, not vice versa)
3. Expert dispatch

**Active mask:** Not enforced (not integrated)

**Compatibility:** ExpertChoiceMoE uses different routing logic (expert-choice vs token-choice). Would require separate ChronoMoE wrapper if lifecycle operations needed.

**Recommendation:** If using ExpertChoiceMoE, do NOT attempt lifecycle operations until a ChronoExpertChoiceMoE wrapper is created with proper active_mask enforcement.

---

## Bypass Risk Assessment

### Low Risk (Already Integrated)

- **ChronoMoE:** Active mask enforced, tested, validated ✅

### Medium Risk (Original Implementations)

- **Standard MoE (moe.py):** No active_mask, but also no lifecycle operations
  - **Mitigation:** Use ChronoMoE instead of MoE for lifecycle-enabled layers

- **ExpertChoiceMoE (moe.py):** No active_mask, no lifecycle operations
  - **Mitigation:** Do not attempt lifecycle operations with ExpertChoiceMoE
  - **Future work:** Create ChronoExpertChoiceMoE if needed

### High Risk (None Identified)

No high-risk bypass paths found.

---

## Enforcement Checklist

When adding new routing strategies:

- [ ] Router outputs max_experts logits (not moe_num_experts)
- [ ] Active mask applied before softmax
- [ ] Inactive experts get -inf logits
- [ ] Test verifies inactive expert probability = 0.0
- [ ] Test verifies inactive expert utilization = 0
- [ ] Add test to validation suite

---

## Usage Guidelines

### ✅ Safe: Use ChronoMoE for lifecycle operations

```python
from chronomoe_integration import ChronoMoE

layer = ChronoMoE(config, mlp, layer_id=0, max_experts=8)
layer.spawn_expert(parent_id=0, strategy="blank")
layer.prune_expert(expert_id=2)
```

### ⚠️ Unsafe: Do not use standard MoE with lifecycle operations

```python
from moe import MoE

layer = MoE(config, mlp)
# DO NOT attempt spawn/prune - no active_mask enforcement
```

### ⚠️ Unsupported: ExpertChoiceMoE not integrated

```python
from moe import ExpertChoiceMoE

layer = ExpertChoiceMoE(config, mlp)
# Lifecycle operations not supported
# Would require ChronoExpertChoiceMoE wrapper
```

---

## Future Work

If ExpertChoiceMoE needs lifecycle support:

1. Create `ChronoExpertChoiceMoE` class
2. Apply same fixed-width pattern
3. Enforce active_mask before expert selection
4. Add validation tests
5. Update this document

---

## Verification

Run validation suite to verify no bypass paths:

```bash
make test
```

Expected: 7/7 tests pass, including ghost routing assertion.

Last verified: 2026-02-10
