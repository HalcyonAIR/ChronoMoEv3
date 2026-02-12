# Router Integration Specification: Suppression Trials

**Date:** 2026-02-12
**Status:** Design specification (implementation pending approval)

---

## Purpose

Hook `controller.get_routing_adjustments()` into the router forward pass to enable suppression trials. **Critical constraint:** This must be a pure logit adjustment with no side effects, no extra controller queries, and no access to content.

---

## Anti-Pattern: Shadow Router

**DO NOT:**
```python
# BAD: Layer queries controller during routing
def forward(self, x):
    logits = self.router(x)

    # DON'T DO THIS - creates shadow router
    adjustments = self.controller.decide_routing(x, logits)
    logits = logits + adjustments  # Controller influencing every routing decision

    probs = F.softmax(logits, dim=-1)
```

**Why this is bad:**
- Controller gets access to content (x)
- Controller can inspect every routing decision
- Becomes a second router (backchannel)
- Defeats the purpose of having a router module

---

## Correct Pattern: Pure Logit Adjustment

**DO:**
```python
# GOOD: Pure logit adjustment, no content access
def forward(self, x):
    # Normal routing
    logits = self.router(x)  # Shape: (batch, seq_len, max_experts)

    # Apply active mask (prevent ghost routing)
    active_mask = self.registry.get_active_mask()  # Shape: (max_experts,)
    logits = logits + torch.where(
        active_mask.unsqueeze(0).unsqueeze(0),
        torch.zeros_like(logits),
        torch.full_like(logits, -1e9)
    )

    # Apply suppression adjustments (if any)
    if hasattr(self, 'controller') and self.controller is not None:
        current_step = self.current_step
        hard_blocks, soft_penalties = self.controller.get_routing_adjustments(current_step)

        # Apply hard blocks (mask to -inf)
        for expert_id in hard_blocks:
            logits[:, :, expert_id] = float('-inf')

        # Apply soft penalties (subtract from logits)
        for expert_id, penalty in soft_penalties.items():
            logits[:, :, expert_id] -= penalty

    # Softmax as normal
    probs = F.softmax(logits, dim=-1)
```

**Why this is correct:**
- Controller never sees content (x)
- Controller never sees original logits
- Controller provides only `{expert_id: penalty}` map
- Layer applies adjustments mechanically
- No feedback loop, no backchannel

---

## Integration Checklist

### 1. Call Site
- Insert adjustment **after active_mask** (don't resurrect inactive experts)
- Insert adjustment **before softmax** (penalties affect probabilities correctly)
- Call **only once per forward pass** (don't query multiple times)

### 2. Error Handling
- If `controller` is None, skip adjustments (no-op)
- If `get_routing_adjustments()` returns empty, skip (no-op)
- Never fail forward pass due to adjustment errors

### 3. Audit Logging
Log adjustments applied (for debugging):
```python
if hard_blocks or soft_penalties:
    self.suppression_log.append({
        "step": current_step,
        "hard_blocks": list(hard_blocks),
        "soft_penalties": dict(soft_penalties),
    })
```

**DO NOT log:**
- Content (x)
- Original logits
- Probabilities
- Expert outputs

**Only log:** Which adjustments were applied (for audit trail)

### 4. Constraint Enforcement

**Verify adjustments only affect active experts:**
```python
# Assert: hard_blocks and soft_penalties only reference active experts
active_expert_ids = self.registry.get_active_expert_ids()
for expert_id in hard_blocks:
    assert expert_id in active_expert_ids, \
        f"Hard block on inactive expert {expert_id}"

for expert_id in soft_penalties:
    assert expert_id in active_expert_ids, \
        f"Soft penalty on inactive expert {expert_id}"
```

**Verify penalties are non-negative:**
```python
for expert_id, penalty in soft_penalties.items():
    assert penalty >= 0, \
        f"Negative penalty {penalty} for expert {expert_id} (should be subtracted, not added)"
```

---

## Testing Integration

### Test 1: Hard Block Applied
```python
def test_hard_block_routing():
    """Verify hard block masks expert to -inf."""
    layer = ChronoMoE(...)

    # Suppress expert 0 with cooldown
    layer.controller.suppress_expert(expert_id=0, duration_steps=100)

    # Forward pass
    x = torch.randn(1, 10, d_model)
    output = layer(x)

    # Verify expert 0 received no tokens
    utilization = layer.controller.last_observation.utilization
    assert utilization[0] == 0, "Hard blocked expert should receive no tokens"
```

### Test 2: Soft Penalty Applied
```python
def test_soft_penalty_routing():
    """Verify soft penalty reduces expert selection."""
    layer = ChronoMoE(...)

    # Get baseline utilization
    baseline_util = run_forward_passes(layer, n=100)

    # Apply soft penalty to expert 0
    layer.controller.suppress_expert(expert_id=0, duration_steps=0)  # No cooldown

    # Get utilization with penalty
    penalty_util = run_forward_passes(layer, n=100)

    # Verify expert 0 utilization dropped (but not zero)
    assert penalty_util[0] < baseline_util[0], "Soft penalty should reduce utilization"
    assert penalty_util[0] > 0, "Soft penalty should not block entirely"
```

### Test 3: No Content Leakage
```python
def test_no_content_access():
    """Verify controller never receives content."""
    layer = ChronoMoE(...)

    # Monkey-patch controller to detect content access
    original_get = layer.controller.get_routing_adjustments
    content_accessed = False

    def patched_get(current_step):
        # Should only receive step number, not content
        assert isinstance(current_step, int)
        return original_get(current_step)

    layer.controller.get_routing_adjustments = patched_get

    # Forward pass
    x = torch.randn(1, 10, d_model)
    output = layer(x)

    # Verify no content leaked
    assert not content_accessed, "Controller should not access content"
```

---

## Performance Considerations

### Overhead
- `get_routing_adjustments()` is O(num_suppressed_experts), typically 0-2 experts
- Hard block: O(num_hard_blocks) indexing operations
- Soft penalty: O(num_soft_penalties) subtraction operations
- **Total overhead:** Negligible (<1% of forward pass)

### Caching
- Adjustments are **step-level**, not **batch-level**
- Can cache within a step if multiple forward passes occur
- **DO NOT cache across steps** (penalties decay, cooldowns expire)

---

## Failure Modes and Mitigations

### Failure: Expert resurrection
- **Symptom:** Inactive expert receives tokens during suppression
- **Cause:** Adjustments applied before active_mask
- **Mitigation:** Apply adjustments after active_mask, assert expert is active

### Failure: Incorrect penalty sign
- **Symptom:** Penalty increases logit instead of decreasing
- **Cause:** Adding penalty instead of subtracting
- **Mitigation:** Always subtract: `logits -= penalty`, assert penalty >= 0

### Failure: Cooldown doesn't block
- **Symptom:** Hard blocked expert still receives tokens
- **Cause:** Hard block applied after softmax, or float('-inf') not used
- **Mitigation:** Apply before softmax, use exact `float('-inf')` value

---

## Diff Summary

**File:** `chronomoe_integration/chronomoe_layer.py`

**Location:** `ChronoMoE.forward()` method, after active_mask, before softmax

**Changes:**
```python
# BEFORE (current):
logits = self.router(x)
logits = logits + active_mask_adjustment
probs = F.softmax(logits, dim=-1)

# AFTER (with suppression):
logits = self.router(x)
logits = logits + active_mask_adjustment

# Apply suppression adjustments (Milestone F Phase 1.5)
if hasattr(self, 'controller') and self.controller is not None:
    hard_blocks, soft_penalties = self.controller.get_routing_adjustments(self.current_step)

    # Hard blocks (mask to -inf)
    for expert_id in hard_blocks:
        logits[:, :, expert_id] = float('-inf')

    # Soft penalties (subtract)
    for expert_id, penalty in soft_penalties.items():
        logits[:, :, expert_id] -= penalty

probs = F.softmax(logits, dim=-1)
```

**Lines added:** ~10
**Complexity:** O(num_suppressed_experts), typically O(1)

---

## Approval Gates

Before implementing router integration:
1. ✅ Two-channel API defined (hard_blocks vs soft_penalties)
2. ✅ Scale-aware penalty computation available
3. ✅ Suppression trial criteria documented
4. ⬜ Router integration reviewed and approved
5. ⬜ Test coverage for integration points
6. ⬜ Performance overhead measured

**Router integration proceeds only after explicit approval.**

---

## Summary

Router integration is a **pure, mechanical logit adjustment** with:
- No content access
- No controller queries beyond `get_routing_adjustments(step)`
- No side effects
- Two-channel application (hard blocks first, then soft penalties)
- Applied after active_mask, before softmax
- Audit logging for transparency
- Minimal overhead (<1% of forward pass)

This prevents shadow routing while enabling safe suppression trials.
