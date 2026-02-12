# Suppression Trial Success/Failure Criteria

**Date:** 2026-02-12
**Status:** Specification (implementation pending approval)

---

## Purpose

Suppression trials are counterfactual probes that test whether apparent redundancy (high similarity + low utilization) is real before MERGE execution. Instead of immediately merging similar+underutilized experts, we temporarily suppress one expert and observe system behavior.

**Key insight:** If suppressing an expert causes no quality degradation, redundancy is real. If quality drops, experts were not actually redundant despite appearing similar.

---

## Trial Phases

### Phase 1: Hard Cooldown (100 steps)
- Expert is **hard blocked** (masked out entirely, logit → -inf)
- Router cannot select this expert at all
- System must adapt by routing to remaining experts
- **Purpose:** Force counterfactual without gradual transitions

### Phase 2: Soft Decay (~50 steps)
- Hard block lifted, **soft penalty** applied (logit -= penalty)
- Penalty decays exponentially (penalty *= 0.9 per step)
- Router gradually returns to normal weighting
- **Purpose:** Verify system can recover smoothly

### Phase 3: Recovery (~50 steps)
- Penalty below cleanup threshold (< 0.01), fully removed
- Expert returns to normal routing competition
- **Purpose:** Confirm system returned to baseline behavior

**Total trial duration:** ~200 steps

---

## Success Criteria (Merge-Eligible)

For a suppression trial to succeed, **ALL** of the following must hold:

### 1. Loss Stability
```
max(loss) - baseline_loss < loss_tolerance
```
- `baseline_loss`: Average loss over 50 steps before suppression
- `loss_tolerance`: Configurable threshold (default: +0.05)
- **Rationale:** Quality should not degrade significantly during suppression

### 2. Free Energy Stability
```
max(F_l) - baseline_F_l < f_l_tolerance
```
- `baseline_F_l`: Average F_l over 50 steps before suppression
- `f_l_tolerance`: Configurable threshold (default: +0.01)
- **Rationale:** Layer capacity pressure should not increase

### 3. Coherence Stability
```
min(coherence) > baseline_coherence - coherence_tolerance
```
- `baseline_coherence`: Average coherence over 50 steps before suppression
- `coherence_tolerance`: Configurable threshold (default: -0.05)
- **Rationale:** Expert specialization should not collapse

### 4. No Stress Band Degradation
```
stress_band ∈ {COMFORT} for entire trial
```
- If system enters STRAIN or PANIC during trial → **FAIL**
- **Rationale:** Suppression should not push system under stress

### 5. No Routing Collapse
```
min(Neff) > baseline_Neff * 0.8
```
- `baseline_Neff`: Average Neff over 50 steps before suppression
- **Rationale:** Router should not collapse to fewer experts (e.g., all traffic to one expert)

### 6. No Bimodality Spikes
```
max(bimodality_scores) < 0.7
```
- Check all remaining experts for bimodality during trial
- **Rationale:** Suppression should not force remaining experts to become bimodal

---

## Failure Criteria (Merge-Rejected)

If **ANY** of the following occur, trial fails immediately:

1. **Loss spike:** Loss increases beyond tolerance
2. **F_l spike:** Free energy rises significantly
3. **Coherence drop:** Specialization degrades
4. **Stress band change:** System enters STRAIN or PANIC
5. **Routing collapse:** Neff drops significantly
6. **Bimodality spike:** Any expert develops high bimodality

**Failure action:** Mark candidate as `merge_rejected`, do not retry for N steps (e.g., 1000).

---

## Outcome States

### Merge-Eligible
- Trial succeeded (all criteria met)
- Candidate is **eligible** for merge, not **approved**
- Human review or additional validation required before execution
- **Note:** "Eligible" ≠ "execute immediately"

### Merge-Rejected
- Trial failed (at least one criterion violated)
- Candidate blacklisted for cooldown period
- May retry after sufficient time (system may evolve)

### Trial-Aborted
- External condition interrupted trial (e.g., capacity change, expert pruned)
- Not counted as success or failure
- May retry when conditions stabilize

---

## Configuration

```python
"suppression_trial_criteria": {
    # Baseline window (steps before trial)
    "baseline_window": 50,

    # Tolerance thresholds
    "loss_tolerance": 0.05,
    "f_l_tolerance": 0.01,
    "coherence_tolerance": 0.05,
    "neff_drop_tolerance": 0.2,  # 20% drop allowed
    "bimodality_spike_threshold": 0.7,

    # Trial duration
    "cooldown_steps": 100,
    "decay_steps": 50,  # Estimated (penalty < 0.01)
    "recovery_steps": 50,

    # Failure handling
    "retry_cooldown": 1000,  # Steps before retrying rejected candidate
},
```

---

## Multi-Signal Philosophy

**Why not just use loss?**

Loss is the most obvious signal, but insufficient for safety:
- **Long-tail tasks:** Expert may handle rare cases loss doesn't capture
- **Safety filtering:** Expert may prevent harmful outputs
- **Specialization:** Expert may maintain routing topology even if loss is stable

**Multi-signal veto system:**
- Any signal can veto merge
- Ensures conservative, multi-dimensional validation
- Prevents merge when one signal looks good but others degrade

---

## Budgeting Trial Frequency

**Problem:** Trials are expensive (~200 steps each). Running too many wastes capacity.

**Solution:** Rate limit trials

```python
"suppression_trial_budget": {
    "max_concurrent_trials": 1,  # Only one trial at a time per layer
    "min_steps_between_trials": 300,  # Cooldown between trials
    "max_trials_per_1000_steps": 2,  # Sliding window budget
},
```

**Rationale:**
- One trial at a time prevents confounding (can't isolate which expert caused issues)
- Minimum gap ensures clean baseline periods
- Sliding window budget prevents trial spam

---

## Audit Log Format

Each trial should log:

```json
{
  "trial_id": "layer0_step5000_expert3",
  "layer_id": 0,
  "start_step": 5000,
  "suppressed_expert_id": 3,
  "suspected_redundant_with": 5,
  "baseline": {
    "loss": 2.45,
    "f_l": 0.032,
    "coherence": 0.87,
    "neff": 3.2,
    "stress_band": "COMFORT"
  },
  "trial_results": {
    "max_loss": 2.48,
    "max_f_l": 0.033,
    "min_coherence": 0.85,
    "min_neff": 2.9,
    "stress_band_entered": ["COMFORT"],
    "bimodality_spikes": []
  },
  "outcome": "merge_eligible",
  "veto_signals": [],
  "duration_steps": 203
}
```

---

## Implementation Notes

1. **Baseline collection** happens continuously (rolling window)
2. **Trial trigger** only when MERGE candidate detected + budget available
3. **Monitoring** runs every step during trial (eager veto on failure)
4. **Recovery verification** extends trial if metrics haven't stabilized

---

## Non-Goals

- **Not a merge execution plan:** This only defines trial success, not merge mechanics
- **Not a rollback plan:** Trials are read-only probes (no weights changed)
- **Not a universal solution:** Some redundancy may not be detectable via suppression

---

## Next Steps (Deferred)

1. Implement baseline collection (rolling window stats)
2. Implement trial state machine (trigger → monitor → verdict)
3. Add audit logging for trial results
4. Validate criteria on real dataset with injected redundancy
5. Tune thresholds based on evidence (same as MERGE detection thresholds)

**MERGE execution remains HARD DISABLED until suppression trials validated on real data.**

---

## Summary

Suppression trials are a **conservative, multi-signal, budgeted** probe system that tests redundancy before MERGE. Success requires:
- Stable loss
- Stable F_l
- Stable coherence
- No stress band change
- No routing collapse
- No bimodality spikes

Any signal can veto. Trials are rate-limited to prevent waste. This makes MERGE rare, safe, and evidence-based.
