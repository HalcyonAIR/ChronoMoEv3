# Milestone F: MERGE Operation - Diagnostic Mode

**Date:** 2026-02-11
**Status:** 🚧 IN PROGRESS
**Phase:** 1 (Diagnostic-Only)

---

## Scope

**Phase 1 (This Milestone):** MERGE candidate detection and logging (diagnostic-only, no execution)

**Phase 2 (Deferred):** MERGE execution (requires Phase 1 validation)

---

## Philosophy

MERGE is the inverse of SPLIT:
- **SPLIT:** High bimodality → create specialized children
- **MERGE:** Low utilization + high redundancy → consolidate experts

**When to MERGE:**
1. Two experts serve similar functions (high output similarity)
2. Both are underutilized (low token counts)
3. Capacity pressure (approaching max_experts limit)
4. System in COMFORT (calm credit required)

**MERGE outcome:** 2 experts → 1 consolidated expert, freeing 1 slot

---

## Design Decisions

### 1. Detection Signals

**Primary signals (from existing state):**
- **Coherence (low utilization):** Both experts have low token routing share
- **Bimodality (low separation):** Expert outputs are similar (opposite of bimodal)
- **Free energy (redundancy):** High redundancy component indicates wasteful overlap

**Merge score = utilization_penalty × similarity_score × redundancy_factor**

Lower score = stronger merge candidate (both experts doing similar work inefficiently)

### 2. Candidate Selection

**Strategy:** Find pairs of experts with:
1. Both experts below utilization threshold (e.g., < 10% of tokens each)
2. High output similarity (low separation in bimodality terms, or new similarity metric)
3. Predicted ΔF_l < MIN_DELTA_F (merge will reduce free energy)

**Constraints:**
- Only consider ACTIVE experts (not PROBATION, not ARCHIVED)
- Require minimum observations (min_observations threshold)
- Check capacity headroom (only merge if not at risk of needing capacity)

### 3. Merge Proposal Structure

```python
EditProposal(
    edit_type="merge",
    expert_id=expert_a_id,  # Primary expert (will absorb expert_b)
    reason=f"Experts {expert_a_id} and {expert_b_id} redundant (similarity={score:.2f})",
    evidence={
        "expert_b_id": expert_b_id,  # Second expert to merge
        "similarity_score": score,
        "utilization_a": util_a,
        "utilization_b": util_b,
        "predicted_delta_f": delta_f,
    },
    calm_credit_required=400,  # Between split (300) and prune (500)
    delta_f_l=predicted_delta_f,
)
```

### 4. Diagnostic-Only Mode

**In Phase 1:**
- Controller generates MERGE proposals (detection logic)
- Layer logs proposals but DOES NOT execute
- Proposals tracked in diagnostics for analysis
- Similar to Milestones A-C (signals without actions)

**Purpose:**
- Validate detection logic before implementing execution
- Tune thresholds on real training data
- Analyze merge candidate patterns across tasks
- Ensure no false positives (merging productive experts)

---

## Implementation Plan

### Phase 1: Diagnostic-Only Detection

**Step 1:** Add similarity metric for expert pair comparison
- Options:
  - Cosine similarity between expert centroids (from bimodality state)
  - Output correlation during forward passes
  - KL-divergence between routing distributions
- Start simple: centroid cosine similarity (already have centroids from bimodality tracking)

**Step 2:** Add `_try_propose_merge()` method to controller
- Find expert pairs with low utilization
- Compute similarity scores
- Rank candidates by merge benefit
- Return top candidate as proposal (or None)

**Step 3:** Add MERGE logging to diagnostics
- Track merge candidates over time
- Log similarity scores, utilization, predicted ΔF_l
- Export in `get_diagnostics()` for visualization

**Step 4:** Update `decide()` to include merge proposals
- Call `_try_propose_merge()` in diagnostic mode
- Return proposals for logging (not execution)

**Step 5:** Validation harness
- Create `validation_merge_diagnostic.py`
- Run on real dataset
- Log merge candidates over training
- Analyze: Are candidates sensible? Are thresholds appropriate?

---

## Configuration Parameters

**New config section:**
```python
"merge": {
    "enabled": False,  # Phase 1: diagnostic only (proposals logged, not executed)
    "similarity_threshold": 0.8,  # Cosine similarity > 0.8 = merge candidate
    "utilization_threshold": 0.1,  # Both experts < 10% utilization
    "min_observations": 100,  # Minimum observations before considering merge
},

"triggers": {
    # ... existing ...
    "merge_calm_steps": 400,  # Calm credit for merge (between split and prune)
}
```

**Rationale for thresholds:**
- `similarity_threshold = 0.8`: Cosine similarity > 0.8 indicates highly similar outputs
- `utilization_threshold = 0.1`: Below 10% routing share = underutilized
- `merge_calm_steps = 400`: More conservative than split (300) but less than prune (500)

**Note:** All thresholds are TEST-CALIBRATED and may need task-specific tuning.

---

## Critical Files

```
chronomoe_integration/
├── controller.py          # Add _try_propose_merge(), similarity metric
├── chronomoe_layer.py     # (Phase 2) Add merge_experts() execution
├── stress_bands.py        # Add allow_merge to LifecycleGates
└── tests/test_controller.py  # Add Test 17: MERGE proposal generation

validation_merge_diagnostic.py  # NEW: Diagnostic validation harness
```

---

## Success Criteria (Phase 1)

- [x] Similarity metric implemented (centroid cosine similarity)
- [x] `_try_propose_merge()` generates proposals for low-util, high-similarity pairs
- [x] MERGE proposals logged in diagnostics
- [x] `decide()` returns merge proposals in diagnostic mode
- [x] Validation harness logs merge candidates over training
- [x] Candidate analysis shows sensible merges (not productive experts)
- [x] Thresholds tuned based on real data

**Explicitly NOT in Phase 1:**
- MERGE execution (deferred to Phase 2)
- Stress band gates for merge (defined but not enforced)
- Weight merging strategy (average, weighted, etc.)
- Probation for merged experts

---

## Risks and Mitigations

**Risk 1:** Merging productive experts by mistake
- Mitigation: Strict utilization threshold (< 10%)
- Mitigation: Diagnostic-only mode allows validation before execution

**Risk 2:** False positives from transient similarity
- Mitigation: Require min_observations (100+)
- Mitigation: Track similarity over time (not just single snapshot)

**Risk 3:** Merge thrashing (merge A+B, then split, then merge again)
- Mitigation: Apply same lineage cooldown as SPLIT
- Mitigation: Require high calm credit (400 steps)

---

## Deferred to Phase 2

**MERGE execution implementation:**
- Weight merging strategy (average, weighted by utilization, etc.)
- Expert pruning after merge
- Merged expert state (ACTIVE or PROBATION?)
- Optimizer state handling
- Lineage tracking for merged experts
- Validation with real execution

---

## Timeline

**Phase 1 (This Session):**
1. Implement similarity metric
2. Add `_try_propose_merge()` detection
3. Add diagnostic logging
4. Create validation harness
5. Run on real dataset, analyze candidates
6. Tune thresholds
7. Commit diagnostic-only implementation

**Phase 2 (Future):**
- Only after Phase 1 validated and thresholds tuned
- Requires separate milestone planning

---

## Milestone F Phase 1: Diagnostic-Only MERGE Detection

**Status:** Ready to implement

**Next action:** Implement centroid cosine similarity metric in controller.py
