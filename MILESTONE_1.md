# Milestone 1: Bias-Not-Forcing Works

Stable baseline for all future geometry and manifold work.

## Approach
Router bias + per-token entropy confidence replaces expert forcing.
Bias strength modulated by medium clock activation and promotion score:
`bias = base * (1 - medium_activation) * max(0.3, promotion_score)`

## Results (4 seeds)
- **CTV drops 0.17-0.30** across all seeds
- **Loss improves 0.04-0.08** (governed never degrades vs baseline)
- **Category scarring**: code 25-50%, math 15-25%, dialogue/factual/reasoning 0%
- Persists under prompt perturbation (offset=10, seed 42)

## Key Properties
- Governor blocks commits via medium clock + scar debt, not hard locks
- Scars are routing-region-specific with exponential decay (half_life=200)
- Cost accounting: cheap path uses `num_tokens * len(motif_experts)` per layer
- Loss at cheap-path commit: `< baseline * 1.2` = success, else scar

## Files
- `olmoe_governed_experiment.py` — main harness
- `bob_core/substrate.py` — decision loop
- `bob_core/ledgers.py` — commitment + scar + cost ledgers
- `bob_core/governor.py` — ALLOW/BLOCK/ESCALATE
- `backends/hf_adapter.py` — router bias hooks
