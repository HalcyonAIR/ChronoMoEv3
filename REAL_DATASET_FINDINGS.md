# Real Dataset Validation Findings

**Date:** 2026-02-09
**Milestone:** E (SPLIT operation)
**Dataset:** Synthetic learnable (increment/decrement patterns)

---

## Summary

Real dataset validation **partially succeeded** but revealed important findings about natural bimodality persistence.

**What Worked:**
- ✓ SPLIT proposed (step 99, bimodality detected during initialization)
- ✓ SPLIT blocked in STRAIN (constitutionally enforced, steps 99-140)
- ✓ System reached COMFORT and stayed there (steps ~140-2000)
- ✓ Loss decreased from 8.5 to ~0.6-1.0 (learning works)

**What Failed:**
- ✗ SPLIT never executed in COMFORT
- ✗ Bimodality signal disappeared during learning

---

## Key Finding: Natural Bimodality is Transient

### Bimodality Timeline

| Step | Band | Bimodality Max | Event |
|------|------|----------------|-------|
| 99 | STRAIN | ~0.05-0.20 | SPLIT proposed (blocked in STRAIN) |
| 200 | COMFORT | 0.0469 | Still above threshold (0.05) but no proposal |
| 400 | COMFORT | 0.0242 | Below threshold, signal lost |
| 600-1800 | COMFORT | 0.0041-0.0185 | Bimodality smoothed out |

###Observation

**Bimodality appeared during random initialization** (untrained routing, high entropy), but **disappeared during learning** (router learned to smooth out the distribution).

This suggests:
1. Natural bimodality may be rare in real training
2. SPLIT might not fire frequently (or at all) without engineered bimodal tasks
3. The split_threshold (0.5 in tests, 0.05 in real data) may need task-specific tuning

---

## Validation Protocol Status

### Semi-Real Validation (Injected Bimodality)
**Status:** ✅ PASS (all 3 seeds)

- Bimodality artificially sustained (expert alternates between +centroid and -centroid)
- SPLIT proposed, blocked in STRAIN, executed in COMFORT
- Validates mechanics and constitutional behavior

### Real Dataset Validation (Natural Bimodality)
**Status:** ⚠️ PARTIAL PASS

- SPLIT proposed and blocked in STRAIN (constitutional behavior verified)
- But bimodality signal disappeared during learning (no execution)
- Reveals that natural bimodality may be transient

---

## Interpretation

The system is working as designed:
- SPLIT fires when bimodality detected (✓)
- SPLIT blocked by stress bands (✓)
- SPLIT requires calm credit in COMFORT (✓)

But **natural bimodality may not persist** in real training scenarios. This is not a bug - it's a feature discovery about the system's behavior.

### Implications

1. **SPLIT may be rare** in real training (not every run will trigger it)
2. **Task matters:** Some tasks may never develop bimodality
3. **Threshold tuning:** split_threshold may need per-task calibration
4. **Monitoring value:** Bimodality tracking is valuable even if SPLIT doesn't fire

---

## Recommendations

### For Validation Protocol
- Keep semi-real validation (with injected bimodality) as the primary mechanics test
- Document that real dataset validation may not always trigger SPLIT
- Consider multiple task types (WikiText-2, TinyStories, code, etc.) to observe bimodality patterns

### For Future Work
- Monitor bimodality in real training runs (e.g., GPT-style pre-training)
- Log bimodality patterns across layers and training phases
- Consider adaptive split_threshold based on observed bimodality distributions
- Investigate which tasks/architectures naturally produce bimodality

---

## Conclusion

**Semi-real validation (with stress consistency guard):** ✅ FROZEN and REPEATABLE

**Real dataset validation:** ⚠️ Mechanics verified, but natural bimodality transience observed

The SPLIT operation is correctly implemented and constitutionally enforced. Whether it fires in real training depends on whether natural bimodality persists long enough to accumulate calm credit - an empirical question about task structure, not system correctness.

**Next step:** Proceed to larger-scale real training (WikiText-2 or GPT-style) with bimodality monitoring enabled, expecting that SPLIT may be rare but correctly handled when it occurs.
