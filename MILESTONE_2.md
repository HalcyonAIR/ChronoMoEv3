# Milestone 2: Geometry Logging + Manifold Analysis

Stable internal checkpoint. All claims below are empirically verified across 2 seeds (15, 22) with ablations.

## What We Built

Per-step geometry logging: router entropy, churn (Jaccard distance), Neff, flipflop EMA, scar hit, baseline loss, routing weights. ~160 bytes/line added to JSONL traces. Backward compatible.

Perturbation protocol: `--perturb-at-step N --perturb-duration M` toggles scar recording off/on mid-run. `--no-scars` disables scars entirely. Produces before/during/after phase annotations in JSONL.

Analysis tooling: `analyze_manifold.py` — PCA manifold comparison, hysteresis trajectory analysis, feature ablation via `--features` presets.

## Results

### Governance Regime Separation

Governance introduces a stable, low-dimensional regime variable that separates baseline and governed trajectories.

| Metric | Seed 15 | Seed 22 |
|--------|---------|---------|
| Silhouette (all features) | 0.54 | 0.54 |
| PC1+PC2 variance explained | 62% | 63% |

PC1 is dominated by medium_activation (0.52) and scar_debt (0.50). The primary separating axis is governance state, not routing geometry.

### Feature Ablation

| Feature set | Seed 15 | Seed 22 |
|-------------|---------|---------|
| All (9 features) | 0.54 | 0.54 |
| Governance only (medium_act, scar_debt, flipflop) | 0.79 | 0.80 |
| Routing only (entropy, churn, flipflop, neff) | 0.36 | 0.37 |

Routing geometry shifts detectably under governance, but the primary separating axis is governance state. The clocks drive separation; routing responds.

### Hysteresis

Medium-clock dynamics produce reproducible hysteresis loops under perturbation.

| Metric | Seed 15 | Seed 22 |
|--------|---------|---------|
| Centroid distance (before->after, scars) | 1.12 | 0.93 |
| Centroid distance (before->after, no scars) | 1.42 | 0.88 |
| Hysteresis confirmed | YES | YES |

Hysteresis persists across both perturb window positions (step 35 and step 60), indicating intrinsic system dynamics rather than phase artifacts.

### Scar Bounding

Scars consistently reduce loop area, acting as stabilizing constraints rather than sources of hysteresis.

| Condition | Seed 15 Loop Area | Seed 22 Loop Area |
|-----------|-------------------|-------------------|
| No scars | 15.69 | 15.15 |
| With scars | 2.66 | 7.44 |
| Reduction | 5.9x | 2.0x |

Unconstrained medium-clock dynamics have a characteristic excursion envelope (~15). Scars impose path-dependent boundary constraints that reduce amplitude. This is immediate bounding, not yet demonstrated as long-term consolidation.

## What This Does NOT Show

- Routing geometry is NOT the driver of manifold separation (governance state is)
- Scar bounding is NOT yet shown to progressively tighten over longer runs
- No conserved relationship identified between loop area and internal invariants
- Single model (OLMoE-1B-7B) only

## Open Questions for Future Work

1. Does loop area converge to a fixed value with increasing run length? (long-term scar consolidation)
2. Does routing-only separation increase as scars accumulate? (geometric deepening)
3. Is there a conserved relationship between loop area and an internal invariant? (entropy mass, Neff)
4. Does the same qualitative behavior hold on a different MoE model? (generality)

Any of these would elevate this from "architecture result" to "structural discovery."

## Files

| File | Action |
|------|--------|
| `MILESTONE_1.md` | Unchanged (baseline lock) |
| `MILESTONE_2.md` | New (this file) |
| `backends/adapter.py` | +1 field (mean_entropy on LayerSnapshot) |
| `backends/hf_adapter.py` | +8 lines (entropy in hook + passthrough) |
| `bob_core/medium_clock.py` | +2 lines (last_churn field + store) |
| `bob_core/telemetry.py` | +7 fields, +14 lines in to_dict |
| `bob_core/substrate.py` | +25 lines (compute geometry, pass to trace) |
| `bob_core/ledgers.py` | +6 lines (enabled flag on ScarLedger) |
| `olmoe_governed_experiment.py` | +30 lines (CLI flags + perturbation logic) |
| `analyze_manifold.py` | New (~300 lines) |
