# Bob Core Phase 1: Governor + Ledgers + Medium Clock

## What Phase 1 Proves

The governor mechanism changes downstream failure rates without degrading quality.

- `commit_then_violate_within_5` drops reliably when governor is enabled
- Loss never degrades (improves slightly in most seeds)
- Governor fires selectively, not uniformly

## Acceptance Criteria (validated 20 seeds, 8E top-2)

1. `commits_blocked > 0` in 19/20 seeds
2. CTV improved in 7/7 seeds with sufficient sample (>= 20 authorized commits)
3. Loss delta <= 0.05 in 20/20 seeds

## How to Run

### Single seed (quick check)

```bash
python bob_governed_experiment.py --experts 8 --seed 42
```

### 20-seed evaluation (the real test)

```bash
python bob_governed_experiment.py --seeds 20 --experts 8
```

### What "sufficient sample" means

CTV is only reported as meaningful when a seed has >= 20 authorized commits.
Below that, the denominator is too small and the metric is noise.

## Components

| File | Purpose |
|---|---|
| `bob_core/ledgers.py` | GovernanceCoords, RoutingVector, Commitment, Scar, CostSignal, BobCore |
| `bob_core/medium_clock.py` | Instability detection (churn, flipflop, outcome variance, escalation EMAs) |
| `bob_core/governor.py` | Commit authorization with minimum-commit-rate guardrail |
| `bob_core/identity.py` | Identity boundary weighting |
| `bob_core/promotion.py` | Per-context-class stability tracking |
| `bob_core/telemetry.py` | DecisionTrace with governor fields (backward compatible) |
| `bob_core/substrate.py` | Wired all components with None-guard backward compat |

## Known Limitations (toy model)

- Scar saturation ~92% because toy has no class-conditional routing
- Total debt pins at 1.0 for same reason
- 13/20 seeds don't reach sufficient sample for CTV
- Governor is validated as mechanism; selectivity requires a real MoE backend

## What's NOT Done

- `motifs.py` not refactored (frozen until post-OLMoE)
- OLMoE adapter upgrade
- Provisional commitments
- Geometric promotion
