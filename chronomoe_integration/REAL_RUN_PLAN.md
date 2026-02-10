# Real Training Run Plan

**Goal:** Prove lifecycle operations survive contact with real forward/backward loop in swiss-ai/MoE.

**Status:** Planned, not yet executed

---

## Success Criteria

This is NOT about convergence or performance. This is about proving governance:

1. **A spawn happens** - Evidence: expert ID N+1 appears in registry, active_mask updates
2. **Probation expert takes real load** - Evidence: utilization > 0 during probation period
3. **A prune makes expert unreachable** - Evidence: expert utilization goes to exactly 0 after prune
4. **Audit log stays clean** - Evidence: no exceptions, no NaN losses, no parameter corruption
5. **Calm gates respected** - Evidence: lifecycle operations only fire when allowed by stress bands

---

## Training Configuration

Keep it small and fast:

```python
# Model config
config = GPTConfig(
    vocab_size=256,           # Small vocab (character-level)
    block_size=128,           # Short sequences
    n_layer=2,                # Only 2 layers (reduce complexity)
    n_head=4,
    n_embd=128,
    moe_num_experts=4,        # Start with 4 experts
    moe_num_experts_per_tok=2,
    moe_softmax_order="softmax_topk",
    dropout=0.0,
)

# ChronoMoE config
max_experts = 8               # Allow doubling
probation = ProbationConfig.default()  # 30 steps, 1500 tokens, +1.0 boost

# Training config
steps = 500                   # Short run
batch_size = 4
learning_rate = 3e-4
```

---

## Lifecycle Triggers (Minimal)

**Spawn Trigger:**
- At step 100: Manually spawn 1 expert in layer 0
- Strategy: blank initialization
- Parent: expert 0

**Prune Trigger:**
- At step 300: Manually prune 1 expert in layer 0
- Target: expert with lowest utilization over last 50 steps

**Probation Check:**
- Every 10 steps: Check probation status, graduate or fail

---

## Data

Use toy character-level data (Shakespeare or random):

```python
# Option 1: Shakespeare (real language)
data = load_shakespeare()

# Option 2: Random tokens (faster, simpler)
data = torch.randint(0, 256, (10000,))
```

Either is fine - we're proving governance, not learning.

---

## Instrumentation

Log every lifecycle event:

```python
# Spawn
print(f"[{step:06d}] SPAWN: Layer {layer_id}, Expert {expert_id}")

# Probation update
print(f"[{step:06d}] PROBATION: Expert {expert_id}, tokens={tokens}, boost={boost:.3f}")

# Graduate
print(f"[{step:06d}] GRADUATE: Expert {expert_id}, tokens={tokens}")

# Fail
print(f"[{step:06d}] PROBATION_FAIL: Expert {expert_id}, tokens={tokens} < {min_tokens}")

# Prune
print(f"[{step:06d}] PRUNE: Layer {layer_id}, Expert {expert_id}")

# Utilization tracking
every_10_steps: log expert utilizations for all layers
```

---

## Validation Checks (During Training)

### After Spawn (Step 100)

Check:
- [ ] Expert 4 exists in registry
- [ ] Expert 4 state = PROBATION
- [ ] Expert 4 in active_mask
- [ ] Expert 4 utilization > 0 (probation boost working)
- [ ] ModuleList length unchanged (= 8)

### During Probation (Steps 100-130)

Check:
- [ ] Expert 4 receives tokens each step
- [ ] Probation boost decays over time
- [ ] Token count accumulates

### After Probation (Step 130)

Check:
- [ ] Expert 4 graduated to ACTIVE (if tokens >= 1500)
- [ ] OR Expert 4 pruned as ARCHIVED (if tokens < 1500)

### After Prune (Step 300)

Check:
- [ ] Pruned expert not in active_mask
- [ ] Pruned expert utilization = 0
- [ ] Pruned expert logits = -inf
- [ ] No gradients flowing to pruned expert

---

## Expected Output

```
[000000] Training started, 2 layers, 4 active experts each
...
[000100] SPAWN: Layer 0, Expert 4 (blank from expert 0)
[000100] PROBATION: Expert 4, tokens=0, boost=1.000
[000101] PROBATION: Expert 4, tokens=45, boost=0.967
[000102] PROBATION: Expert 4, tokens=89, boost=0.933
...
[000130] GRADUATE: Expert 4, tokens=1687
...
[000300] PRUNE: Layer 0, Expert 2 (low utilization)
[000301] Expert 2 utilization: 0 (pruned)
...
[000500] Training complete, no exceptions, audit clean
```

---

## Failure Modes to Watch For

### Spawn Failures

- IndexError when accessing router logits for expert 4
- ModuleList size changes unexpectedly
- Optimizer parameter group corruption
- Expert 4 gets 0 tokens (probation boost not working)

### Probation Failures

- Boost not applied (expert never selected)
- Boost doesn't decay (permanent advantage)
- Token count not tracking correctly
- Graduate/fail logic broken

### Prune Failures

- Pruned expert still receives tokens (ghost routing)
- Pruned expert shows non-zero gradients
- Active mask not updated
- Softmax probability > 0 for pruned expert

### Training Loop Failures

- NaN losses
- Parameter explosion
- Checkpoint corruption
- Graph recompilation (if using torch.compile)

---

## Implementation Script

Location: `swiss-ai-MoE/examples/real_run_validation.py`

Pseudocode:

```python
# Setup
model = create_gpt_with_chronomoe(config)
optimizer = AdamW(model.parameters())
data_loader = create_data_loader()

# Training loop
for step in range(500):
    # Forward + backward
    x, y = next(data_loader)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Lifecycle operations
    if step == 100:
        model.layers[0].spawn_expert(parent_id=0, strategy="blank")

    if step % 10 == 0:
        for layer in model.layers:
            layer.check_probation_graduations()

    if step == 300:
        target = find_lowest_utilization_expert(model.layers[0])
        model.layers[0].prune_expert(target)

    # Logging
    if step % 10 == 0:
        log_expert_utilizations(model.layers)
```

---

## Next Steps

1. Create `examples/real_run_validation.py` script
2. Run with logging enabled
3. Verify all 5 success criteria met
4. Document results in `REAL_RUN_RESULTS.md`
5. If failures occur, debug and fix before declaring integration-ready

---

## Timeline

- **Estimated time:** 2-4 hours (script creation + run + analysis)
- **Target completion:** Before moving to Issue #1 (stress bands)

---

## Notes

This run is deliberately simple. No multi-layer spawns, no clone-seeded, no complex triggers. Just prove the basic mechanics work in a real gradient flow environment.

Once this passes, we can add complexity (stress bands, multi-layer coordination, etc.).
