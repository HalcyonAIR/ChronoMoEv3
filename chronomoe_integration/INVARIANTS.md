# ChronoMoE Fixed-Width Routing: Invariants

**Contract version:** 1.0
**Last validated:** 2026-02-10

---

## The Five Laws

### 1. Router Dimension is Fixed at Initialization

**Rule:** Router outputs `max_experts` logits from step 0, not `moe_num_experts`.

**Enforcement:**
- `FixedWidthRouter.w_g.out_features == max_experts` (assert in __init__)
- Never dynamically resize router output dimension
- Test: `test_fixed_width_initialization()` verifies dimension

**Violation symptoms:**
- IndexError when accessing `logits[:, expert_id]` for spawned experts
- Router logits shape mismatch with active_mask length

---

### 2. Experts are Pre-Allocated

**Rule:** ModuleList contains `max_experts` slots from initialization.

**Enforcement:**
- `len(self.experts) == max_experts` (assert in __init__)
- Never append/remove from ModuleList after initialization
- Spawn activates pre-existing slot, prune deactivates it

**Violation symptoms:**
- ModuleList length changes during training
- Parameter count changes unexpectedly
- Optimizer state corruption

---

### 3. Lifecycle Operations Change Masks Only

**Rule:** Spawn and prune flip `active_mask` and `ExpertRegistry` state only, never tensor shapes.

**Enforcement:**
- Spawn: `active_mask[expert_id] = True`, registry state = PROBATION
- Prune: `active_mask[expert_id] = False`, registry state = ARCHIVED
- Assert ModuleList length unchanged after spawn/prune
- Test: `test_spawn()` verifies mask changes, not shape changes

**Violation symptoms:**
- Tensor shape mismatch errors during training
- Dynamic graph recompilation
- Checkpoint incompatibility

---

### 4. Inactive Experts are Hard-Masked

**Rule:** Inactive experts get -inf logits before softmax, resulting in exactly zero selection probability.

**Enforcement:**
- `router_logits[:, ~active_mask] = float('-inf')` applied before softmax
- Post-mask: `min(inactive_logits) == max(inactive_logits) == -inf`
- Post-softmax: `P(select inactive expert) == 0.0` (within float tolerance)
- Test: `test_hard_mask()` verifies -inf masking and zero utilization
- Test: `test_ghost_routing_assertion()` verifies zero probability

**Violation symptoms:**
- Pruned experts still receive tokens
- Non-zero gradients for archived expert parameters
- Ghost routing (inactive experts showing utilization > 0)

---

### 5. Probation Boost is Bounded and Decays

**Rule:** Probation boost starts at `initial_boost`, decays linearly to 0 over `duration_steps`, then stops.

**Enforcement:**
- Boost at spawn: `boost(step=spawn_step) == initial_boost`
- Boost during probation: `0 <= boost(step) <= initial_boost`
- Boost after duration: `boost(step >= spawn_step + duration_steps) == 0.0`
- Boost only applied to experts in PROBATION state
- Test: `test_probation_boost()` verifies decay schedule

**Violation symptoms:**
- Unbounded boost (experts permanently advantaged)
- No decay (probation never ends)
- Boost applied to ACTIVE experts (leak)

---

### 6. Calm Gates for Lifecycle Operations

**Rule:** Lifecycle operations (spawn/prune/graduate) respect stress bands and calm credit.

**Stress Bands:**
- COMFORT: stress < comfort_ceiling (default: 1.0)
- STRAIN: comfort_ceiling <= stress < strain_ceiling (default: 2.0)
- PANIC: stress >= strain_ceiling

**Calm Credit:** Time spent continuously in COMFORT band (resets when leaving COMFORT).

**Enforcement:**
- PANIC band: All lifecycle operations frozen
- STRAIN band: Prune frozen, spawn/graduate allowed with calm credit
- COMFORT band: All operations allowed with calm credit

**Calm Credit Requirements:**
- Spawn: `spawn_calm_steps` (default: 200 steps in COMFORT)
- Prune: `prune_calm_steps` (default: 500 steps in COMFORT)
- Graduate: `graduate_calm_steps` (default: 200 steps in COMFORT)

**Hysteresis:**
- Entering worse band: At threshold
- Exiting worse band: Below threshold - hysteresis margin (default: 5%)
- Prevents thrashing when stress hovers near boundaries

**Stress Signal:**
- EMA-smoothed to prevent spikes from triggering band changes
- Alpha (default: 0.05) controls smoothing rate

**Violation symptoms:**
- Spawn happens under high stress (panic/strain without calm)
- Prune happens under strain (removes capacity when needed)
- Graduation happens under stress (commits identity while unstable)
- Lifecycle operations cause cascading failures

**Test:** `test_stress_bands.py` verifies all calm gates enforced

---

## Enforcement Points

### Initialization
- Assert router dimension = max_experts
- Assert len(experts) = max_experts
- Assert active_mask.shape[0] = max_experts

### Forward Pass
- Assert router_logits.shape[1] = max_experts
- Assert inactive logits = -inf (before softmax)
- Assert inactive expert utilization = 0 (after dispatch)

### Spawn
- Assert capacity_remaining > 0
- Assert expert_id == next_expert_id (sequential allocation)
- Assert len(experts) unchanged
- Assert active_mask[expert_id] = True after spawn

### Prune
- Assert expert in registry
- Assert len(experts) unchanged
- Assert active_mask[expert_id] = False after prune

### Probation
- Assert 0 <= boost <= initial_boost
- Assert boost = 0 after duration_steps
- Assert boost only applied to PROBATION state experts

---

## Validation

Run validation suite to verify all invariants:
```bash
pytest swiss-ai-MoE/chronomoe_integration/tests/
```

Expected: 6/6 tests pass, all invariants enforced.

---

## Violation Recovery

If an invariant is violated:

1. **Stop training immediately** - Do not attempt to continue
2. **Check logs** - Identify which invariant failed
3. **Verify initialization** - Ensure max_experts, router dim, ModuleList length correct
4. **Check lifecycle operations** - Ensure spawn/prune only change masks
5. **Run validation suite** - Identify which test fails
6. **Report issue** - File bug with reproduction steps

---

## Contract Updates

Changes to these invariants require:
- Version bump in this file header
- Re-validation of all 6 tests
- Update to enforcement assertions
- Changelog entry explaining why invariant changed

Do not relax invariants without clear justification and full re-validation.
