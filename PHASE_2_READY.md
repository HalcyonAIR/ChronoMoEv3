# Phase 2 Readiness: Organ Removal

**Date:** 2026-02-12
**Status:** Prepared, awaiting explicit approval

---

## What Phase 1.5 Proved

**Immune system works.**

We can now:
- Detect when experts appear redundant (high similarity + low utilization)
- Suppress an expert temporarily (counterfactual probe)
- Monitor system response (multi-signal veto)
- Reject merge when conditions degrade (adversarial validation)
- Accept merge when conditions remain stable (symmetry proven)

**Constitutional maturity achieved:** System proves "it refuses when it should."

---

## What Phase 2 Will Test

**Organ removal.**

We will:
- Actually merge two experts (combine weights, prune one)
- Modify network topology permanently (not temporarily)
- Change capacity structure under load
- Trust that the veto system catches mistakes

**This is qualitatively different.**

---

## The Philosophical Risk

### Suppression Trials (Phase 1.5)
- **Reversible:** Expert returns after trial
- **Non-destructive:** No weights changed
- **Observable:** Can watch system adapt
- **Safe:** Worst case is temporary quality drop

### Merge Execution (Phase 2)
- **Irreversible:** Expert permanently removed (without rollback plan)
- **Destructive:** Weights merged, topology changed
- **Committal:** Can't undo after optimizer step
- **Risky:** Worst case is permanent capacity loss

**The veto system must be airtight.**

---

## What We Don't Have Yet

### 1. Rollback Plan
- How to undo a merge if it degrades quality
- Optimizer state handling (merged expert has no gradient history)
- Lineage tracking (which experts came from which merge)

### 2. Delta Bundle Definition
- How to merge two experts' weights (average? weighted? specialized?)
- Bias handling (merge biases? keep one? relearn?)
- Router weight updates (how to redirect traffic from pruned expert)

### 3. Additional Veto Signals
Currently validated:
- ✅ Loss spike (acute harm)
- ✅ Stress band change (systemic stress)

Still needed:
- ⬜ F_l spike (free energy increase → capacity pressure)
- ⬜ Coherence drop (specialization collapse)
- ⬜ Neff collapse (routing breaks down)
- ⬜ Bimodality spike (remaining experts forced bimodal)

**Two veto channels isn't enough for organ removal.**

### 4. Trial State Machine
- Automatic trial trigger (when MERGE candidates detected)
- Continuous monitoring during trial (not just post-hoc)
- Eager veto (stop immediately when signal crosses threshold)
- Recovery verification (confirm system returned to baseline)

### 5. Post-Merge Monitoring
- Does the merged expert enter probation?
- Do we track "merge debt" (capacity lost)?
- When can we merge again after successful merge?

---

## Halcyon's Warnings

> "Don't let that become the standard. Keep strict tolerances as default."

**Acknowledged.** Lenient tolerance (1.0) exists only in validation for proving symmetry. Production default: 0.05 loss, 0.01 F_l.

> "Watch for overfitting to loss deltas. Loss alone won't be enough."

**Acknowledged.** Current veto logic is loss-heavy (4 of 7 examples). Merge execution must monitor all 6 signals, not just loss.

> "Later you'll want one structural veto too, like Neff collapse or coherence crash."

**Acknowledged.** Current adversarial tests cover acute harm and systemic stress. Phase 2 needs structural veto (topology collapse).

---

## Phase 2 Prerequisites

Before implementing MERGE execution:

1. **Rollback plan approved** (how to undo merge)
2. **Delta bundle defined** (how to merge weights)
3. **All 6 veto signals implemented** (not just 2)
4. **Trial state machine built** (automatic trigger + monitoring)
5. **Post-merge monitoring defined** (probation? merge debt?)
6. **Multi-seed validation planned** (not just seed=42)

**No exceptions.**

---

## The Question

> "Because the next phase — actual merge execution — is where the real philosophical risk begins."
>
> "You've just proven the immune system works."
>
> "Now you're about to test organ removal."
>
> "Ready?"

**Answer:**

Ready to **plan**, yes.

Ready to **execute**, no.

We've proven the immune system works (veto system functional). But organ removal requires:
- Complete veto coverage (all 6 signals, not just 2)
- Rollback capability (can't be irreversible)
- Multi-seed validation (not just deterministic tests)
- Real training validation (not just 500-step demos)

**Phase 2 requires approval of:**
1. Rollback plan
2. Delta bundle strategy
3. Additional veto signals
4. Trial state machine
5. Post-merge monitoring

**We're ready to design Phase 2. Not ready to execute it.**

---

## Constitutional Stance

**MERGE execution remains HARD DISABLED.**

The next commit that enables MERGE execution must include:
- Rollback plan (approved)
- All 6 veto signals (implemented + validated)
- Trial state machine (automatic trigger)
- Post-merge monitoring (probation rules)

**Organ removal is not autonomous. It requires architecture approval.**

---

## Breath

Taking it.

We've achieved constitutional maturity (refuses when it should). That's the foundation.

Now we design the next layer carefully.

**Phase 2 planning begins only after explicit approval.**

---

**Tagged: `milestone-f-phase1.5`**
**Committed: 14 files, 9192 insertions**
**Status: Awaiting Phase 2 architecture approval**
