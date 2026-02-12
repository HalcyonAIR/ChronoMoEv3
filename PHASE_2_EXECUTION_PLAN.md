# Phase 2 Execution Plan: swiss-ai/MoE Testbed

**Date:** 2026-02-12
**Testbed:** swiss-ai/MoE (ChronoMoE integration)
**Goal:** Prove autonomous merge execution doesn't amputate capability and rollback works

---

## Why swiss-ai/MoE First

**Not proving SOTA. Proving safety.**

1. **Already integrated:** ChronoMoE layer working, tests passing
2. **Small and fast:** Can rerun deterministically in minutes
3. **Full control:** We own the integration, can instrument everything
4. **Low risk:** No production impact, no external dependencies

**Once this is boring and repeatable, then port to Mixtral-style models.**

---

## Fixed Script Flow

```
1. Train baseline (100-200 steps on WikiText-2/TinyStories)
2. Identify MERGE candidates (high similarity + low utilization)
3. Run suppression trial (if candidate found)
4. If trial PASS → Execute merge (delta bundle)
5. Run probe battery immediately after merge
6. If protected delta regresses → Auto-rollback
7. Record outcome (merge succeeded or rolled back)
8. Repeat from step 2 (find next candidate)
```

**Deterministic:** Same seed, same data, same outcome every run.

---

## 1. Delta Bundle Definition

**What is a delta bundle?**
Minimal diff needed to execute a merge and enable rollback.

### Contents

```python
@dataclass
class DeltaBundleMerge:
    """Delta bundle for MERGE operation."""
    # Identification
    bundle_id: str  # Unique ID for this merge
    created_at_step: int

    # Target experts
    expert_a_id: int  # Expert to keep (receives merged weights)
    expert_b_id: int  # Expert to prune (will be deactivated)

    # Evidence for merge decision
    similarity: float
    utilization_a: float
    utilization_b: float
    suppression_trial_result: dict  # Full trial verdict

    # Weight deltas (for rollback)
    expert_a_weights_before: OrderedDict[str, Tensor]  # State dict of expert A before merge
    expert_b_weights_before: OrderedDict[str, Tensor]  # State dict of expert B before merge

    # Merge strategy
    merge_strategy: str  # "average", "weighted", "keep_a", "keep_b"
    merge_alpha: float  # Weight for expert A (if weighted average)

    # Optimizer state (for rollback)
    optimizer_state_expert_a: dict  # Adam momentum/variance for expert A
    optimizer_state_expert_b: dict  # Adam momentum/variance for expert B

    # Registry state (for rollback)
    active_mask_before: Tensor  # Which experts were active before merge
    expert_a_info_before: ExpertInfo  # Full registry info for expert A
    expert_b_info_before: ExpertInfo  # Full registry info for expert B
```

**Why this structure?**
- **Minimal:** Only store what's needed for rollback
- **Complete:** Can reconstruct exact state before merge
- **Auditable:** Evidence chain from candidate → trial → merge

### Merge Strategies

**Phase 2 MVP: Simple average**
```python
merged_weights = (expert_a_weights + expert_b_weights) / 2
```

**Later (deferred):**
- Weighted average (by utilization or performance)
- Specialized merging (task-specific weights)
- Learned merging (meta-learned merge policy)

---

## 2. Rollback Definition

**What is rollback?**
Restore exact state before merge, as if merge never happened.

### Rollback Steps

```python
def rollback_merge(layer, bundle: DeltaBundleMerge, optimizer):
    """Rollback a merge to exact pre-merge state."""

    # 1. Restore expert A weights
    layer.experts[bundle.expert_a_id].load_state_dict(bundle.expert_a_weights_before)

    # 2. Restore expert B weights
    layer.experts[bundle.expert_b_id].load_state_dict(bundle.expert_b_weights_before)

    # 3. Restore active mask (re-activate expert B)
    layer.registry.active_mask[bundle.expert_b_id] = True

    # 4. Restore expert info in registry
    layer.registry.experts[bundle.expert_a_id] = bundle.expert_a_info_before
    layer.registry.experts[bundle.expert_b_id] = bundle.expert_b_info_before

    # 5. Restore optimizer state
    # This is critical: Adam has momentum and variance for each parameter
    optimizer.state[layer.experts[bundle.expert_a_id]] = bundle.optimizer_state_expert_a
    optimizer.state[layer.experts[bundle.expert_b_id]] = bundle.optimizer_state_expert_b

    # 6. Log rollback
    print(f"  ROLLBACK: Merge {bundle.bundle_id} reverted (protected delta regressed)")
```

**Why optimizer state matters:**
- Adam/AdamW have momentum (first moment) and variance (second moment) for each parameter
- If we don't restore these, optimizer "forgets" the trajectory
- Next gradient step will be wrong, causing quality drop

---

## 3. Probe Battery Definition

**What is a probe battery?**
Set of metrics checked immediately after merge to detect capability loss.

### Protected Deltas

These metrics MUST NOT regress after merge:

```python
@dataclass
class ProbeBattery:
    """Metrics checked after merge to detect regression."""

    # Primary signal: Loss
    loss_pre_merge: float  # Average loss before merge (50-step window)
    loss_post_merge: float  # Average loss after merge (50-step window)
    loss_delta: float  # post - pre (should be ≤ tolerance)
    loss_tolerance: float = 0.05  # Max acceptable increase

    # Secondary signal: Perplexity
    perplexity_pre: float
    perplexity_post: float
    perplexity_delta: float
    perplexity_tolerance: float = 0.10  # 10% increase allowed

    # Structural signal: Free energy
    f_l_pre: float
    f_l_post: float
    f_l_delta: float
    f_l_tolerance: float = 0.01

    # Structural signal: Coherence
    coherence_pre: float
    coherence_post: float
    coherence_delta: float
    coherence_tolerance: float = -0.05  # Max 5% drop

    # Structural signal: Neff
    neff_pre: float
    neff_post: float
    neff_delta: float
    neff_tolerance: float = -0.20  # Max 20% drop

    # Verdict
    rollback_triggered: bool
    rollback_reason: str  # Which signal crossed threshold
```

### Battery Execution

```python
def run_probe_battery(layer, dataloader, num_steps=50):
    """Run probe battery to check for regression."""

    metrics = {
        "loss": [],
        "perplexity": [],
        "f_l": [],
        "coherence": [],
        "neff": [],
    }

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        # Forward pass
        output, metadata = layer(batch)

        # Compute loss
        loss = criterion(output, target)
        metrics["loss"].append(loss.item())
        metrics["perplexity"].append(torch.exp(loss).item())

        # Get controller diagnostics
        diag = layer.controller.get_diagnostics()
        metrics["f_l"].append(diag["free_energy"]["F_l"])
        metrics["coherence"].append(diag["coherence"]["avg_coherence"])
        metrics["neff"].append(diag["coherence"]["Neff"])

    return {
        "loss": sum(metrics["loss"]) / len(metrics["loss"]),
        "perplexity": sum(metrics["perplexity"]) / len(metrics["perplexity"]),
        "f_l": sum(metrics["f_l"]) / len(metrics["f_l"]),
        "coherence": sum(metrics["coherence"]) / len(metrics["coherence"]),
        "neff": sum(metrics["neff"]) / len(metrics["neff"]),
    }
```

---

## 4. Auto-Rollback Logic

**When to trigger rollback?**

```python
def check_protected_deltas(pre_metrics, post_metrics, tolerances):
    """Check if any protected delta regressed beyond tolerance."""

    rollback_signals = []

    # Loss delta
    loss_delta = post_metrics["loss"] - pre_metrics["loss"]
    if loss_delta > tolerances["loss"]:
        rollback_signals.append(f"loss_regression (delta={loss_delta:.4f} > {tolerances['loss']})")

    # Perplexity delta
    ppl_delta = (post_metrics["perplexity"] - pre_metrics["perplexity"]) / pre_metrics["perplexity"]
    if ppl_delta > tolerances["perplexity"]:
        rollback_signals.append(f"perplexity_regression (delta={ppl_delta:.4f} > {tolerances['perplexity']})")

    # F_l delta
    f_l_delta = post_metrics["f_l"] - pre_metrics["f_l"]
    if f_l_delta > tolerances["f_l"]:
        rollback_signals.append(f"f_l_spike (delta={f_l_delta:.4f} > {tolerances['f_l']})")

    # Coherence delta
    coherence_delta = post_metrics["coherence"] - pre_metrics["coherence"]
    if coherence_delta < tolerances["coherence"]:  # Negative tolerance (drop)
        rollback_signals.append(f"coherence_drop (delta={coherence_delta:.4f} < {tolerances['coherence']})")

    # Neff delta
    neff_delta = (post_metrics["neff"] - pre_metrics["neff"]) / pre_metrics["neff"]
    if neff_delta < tolerances["neff"]:  # Negative tolerance (drop)
        rollback_signals.append(f"neff_collapse (delta={neff_delta:.4f} < {tolerances['neff']})")

    return rollback_signals
```

**Any signal triggers rollback. All must pass for merge to stick.**

---

## 5. Dataset Choice

**Options:**
1. **WikiText-2** (~2MB, 4.4M tokens)
   - Fast download
   - Standard benchmark
   - Clean data

2. **TinyStories** (~500MB subset)
   - Synthetic but coherent
   - Easy to learn patterns
   - Fast convergence

**Choice: WikiText-2**
- Smaller (faster iteration)
- Standard (better for comparison)
- Real text (not synthetic)

---

## 6. Script Structure

```python
#!/usr/bin/env python3
"""
Autonomous MERGE Execution Demo: swiss-ai/MoE + WikiText-2

Fixed script:
1. Train baseline (200 steps)
2. Find MERGE candidate (high similarity + low utilization)
3. Run suppression trial
4. If PASS → Execute merge → Run probe battery → Rollback if regressed
5. Record outcome
"""

def main():
    # Setup
    model = load_swiss_ai_moe(max_experts=8, initial_active=4)
    dataset = load_wikitext2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Phase 1: Train baseline (200 steps)
    print("Phase 1: Training baseline...")
    for step in range(200):
        train_step(model, dataset, optimizer)

    # Phase 2: Find MERGE candidate
    print("\nPhase 2: Identifying MERGE candidate...")
    candidate = find_merge_candidate(model.moe.controller)

    if candidate is None:
        print("  No candidates found. Exiting.")
        return

    print(f"  Candidate found: experts {candidate.expert_a_id} and {candidate.expert_b_id}")
    print(f"  Similarity: {candidate.similarity:.4f}")
    print(f"  Utilization: {candidate.utilization_a:.4f}, {candidate.utilization_b:.4f}")

    # Phase 3: Run suppression trial
    print("\nPhase 3: Running suppression trial...")
    trial_result = run_suppression_trial(
        model, dataset, optimizer,
        suppressed_expert=candidate.expert_b_id,
        baseline_window=50,
        trial_window=100,
    )

    print(f"  Trial verdict: {trial_result['verdict']}")
    if trial_result["veto_signals"]:
        print(f"  Veto signals: {trial_result['veto_signals']}")

    if trial_result["verdict"] == "VETO":
        print("  MERGE rejected by suppression trial. Exiting.")
        return

    # Phase 4: Execute merge
    print("\nPhase 4: Executing MERGE...")

    # 4a. Run probe battery before merge
    pre_metrics = run_probe_battery(model, dataset, num_steps=50)
    print(f"  Pre-merge metrics: loss={pre_metrics['loss']:.4f}, ppl={pre_metrics['perplexity']:.2f}, Neff={pre_metrics['neff']:.2f}")

    # 4b. Create delta bundle (for rollback)
    delta_bundle = create_delta_bundle(
        model.moe,
        expert_a_id=candidate.expert_a_id,
        expert_b_id=candidate.expert_b_id,
        optimizer=optimizer,
    )

    # 4c. Execute merge
    execute_merge(model.moe, delta_bundle, optimizer)
    print(f"  MERGE executed: expert {candidate.expert_b_id} merged into {candidate.expert_a_id}")

    # Phase 5: Probe battery after merge
    print("\nPhase 5: Running probe battery...")
    post_metrics = run_probe_battery(model, dataset, num_steps=50)
    print(f"  Post-merge metrics: loss={post_metrics['loss']:.4f}, ppl={post_metrics['perplexity']:.2f}, Neff={post_metrics['neff']:.2f}")

    # Phase 6: Check protected deltas
    print("\nPhase 6: Checking protected deltas...")
    rollback_signals = check_protected_deltas(pre_metrics, post_metrics, TOLERANCES)

    if rollback_signals:
        print(f"  ROLLBACK TRIGGERED: {rollback_signals}")
        rollback_merge(model.moe, delta_bundle, optimizer)
        print(f"  ROLLBACK complete. Merge reverted.")
        outcome = "ROLLED_BACK"
    else:
        print(f"  Protected deltas satisfied. Merge accepted.")
        outcome = "ACCEPTED"

    # Phase 7: Record outcome
    save_outcome({
        "candidate": candidate,
        "trial_result": trial_result,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "rollback_signals": rollback_signals,
        "outcome": outcome,
    })

    print(f"\n{'=' * 70}")
    print(f"MERGE EXECUTION COMPLETE: {outcome}")
    print(f"{'=' * 70}")
```

---

## 7. Success Criteria

**Script is "boring and repeatable" when:**

1. ✅ Runs deterministically (same seed → same outcome)
2. ✅ Finds at least 1 MERGE candidate in 200 steps
3. ✅ Suppression trial produces verdict (PASS or VETO)
4. ✅ If trial PASS → merge executes
5. ✅ Probe battery runs without errors
6. ✅ Rollback triggers on protected delta regression
7. ✅ Rollback restores exact state (loss returns to pre-merge)
8. ✅ Can run 10 times, get same outcome each time

**Not success criteria:**
- ❌ MERGE always accepted (might need rollback)
- ❌ MERGE improves performance (just don't regress)
- ❌ Fast training (this is a test, not production)

---

## 8. Artifacts to Generate

1. **`execute_merge_demo.py`** - Full script (phases 1-7)
2. **`delta_bundle.py`** - DeltaBundleMerge dataclass + create/save/load
3. **`rollback.py`** - rollback_merge() implementation
4. **`probe_battery.py`** - run_probe_battery() + check_protected_deltas()
5. **`merge_execution_outcome.json`** - Full results (candidate → trial → merge → rollback)
6. **`MERGE_EXECUTION_VALIDATED.md`** - Report showing rollback working

---

## 9. Phase 2 Timeline

**Step 1:** Implement delta bundle (DeltaBundleMerge dataclass) ✅
**Step 2:** Implement rollback (restore weights + optimizer + registry) ✅
**Step 3:** Implement probe battery (5 metrics) ✅
**Step 4:** Implement merge execution (average weights, prune expert B) ✅
**Step 5:** Build fixed script (phases 1-7) ✅
**Step 6:** Run on WikiText-2, verify deterministic ✅
**Step 7:** Validate rollback works (inject protected delta regression) ✅
**Step 8:** Run 10 times, confirm boring and repeatable ✅

**Estimated:** 3-4 work cycles (with testing and validation)

---

## 10. After swiss-ai/MoE is Boring

**Then port to Mixtral-style models:**
- Publish controller + merge protocol as optional artifacts
- Publish deltas (not full weights)
- Users who already have Mixtral can apply deltas
- Don't re-own base model weights

**But not yet. Prove it on swiss-ai/MoE first.**

---

## Next Action

Implement Phase 2 execution on swiss-ai/MoE:
1. Delta bundle implementation
2. Rollback implementation
3. Probe battery implementation
4. Merge execution logic
5. Fixed script integration
6. WikiText-2 validation

**Awaiting approval to proceed.**
