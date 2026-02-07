# ChronoMoEv3

**One Mechanism, Three Projections**

ChronoMoEv3 is a multi-timescale dynamical system where routing decisions are made while prior computation is still decaying, and only the parts that stay phase-aligned across longer time constants earn the right to keep influencing future routing.

Three clocks with their own expert pools, overlapping delayed responses that sustain context, and slow trimming of what fails to persist are three projections of the same architecture. v3 makes that unity implementable by defining a single state variable — **phase coherence** — tracked at three decay rates. Lifecycle actions (spawn, prune, split, merge) are what the slow clock does when coherence shows irreversible drift.

Phase coherence is estimated from the raw router mixture by comparing an expert's output direction to the mixture direction with its own contribution removed.

Fast and medium clocks can raise an emergency flag that expedites slow-clock review; irreversible edits still require slow confirmation (additive edits may be expedited; pruning waits).

> [!WARNING]
> **Experimental / Early Development**
>
> **Status:** Early development. Architecture design and core implementation in progress. API unstable.

## Testable Claim: Accumulated State Latent Until Constraint

**Precise claim**: Routing state (beta coefficients) accumulates from coherence feedback but only affects behavior when degrees of freedom collapse.

**Not claiming**: "Beta drives all routing"
**Claiming**: "Beta only matters under constraint"

**Empirical test** ([capacity whiplash experiments](experiments/)):
- Two systems, different routing histories (β divergence: 0.016 L1, earned via asymmetric environments)
- Under top-4 routing: behave similarly (plenty of acceptable paths)
- Under top-1 routing: choose different experts (forced choice reveals accumulated geometry)
- Result: **Deformation exists but is latent until constraint forces selection**

**Mechanism**: Router computes `z_biased = z_clean + β`. Under high-k routing, small β differences wash out. Under top-1, they become decisive. System lives near decision boundaries; history tips the balance.

**Generalization**: Applies to any system where:
1. State accumulates across episodes (β, attention weights, memory addressing)
2. State consulted when choice forced (resource allocation, single-path selection)
3. Not MoE-specific — testable in any accumulated-state routing system

**Status**: Empirically validated, not metaphor. See [experiments/README.md](experiments/README.md) for protocol, results, and defense against objections.

**Implication**: Testing under plenty (top-4) misses the effect. Testing under constraint (top-1) reveals it. This is not about everyday routing — it's about collapse under pressure.

## Architecture: Three-Timescale EMA on Phase Coherence

**Single state variable**: `φ_e = cos(y_e, y_mix)` — phase alignment between expert output and mixture

**Three decay rates** (same variable, different retention):

Each expert tracks φ at three exponential moving average timescales. Not three separate variables — one variable filtered at three rates.

| Clock | Retention | Half-life | Governs |
|-------|-----------|-----------|---------|
| Fast | γ ~ 0.9 | ~10 steps | Routing and token dispatch |
| Medium | γ ~ 0.99 | ~100 steps | Lens controller (v2 soft redistribution) |
| Slow | γ ~ 0.999 | ~1000 steps | Lifecycle decisions (structural changes) |

Half-lives illustrative; ratios matter.

**Lifecycle as slow-clock physics**:

Structural changes triggered when slow-window coherence shows irreversible drift:

| Action | Trigger | Interpretation |
|--------|---------|----------------|
| **Prune** | φ_slow < threshold, monotonic decline | Failed persistence test |
| **Split** | φ_fast oscillates, φ_slow stable | Serving incompatible basins |
| **Merge** | Two experts' φ_slow converge | Redundant substrates |
| **Spawn** | Layer-wide φ_slow drops, individuals healthy | Insufficient capacity |

Not external interventions. Slow-window dynamics determines structural actions.

**Bridge detector**: β can bias routing but cannot hallucinate experts. If `overlap_only = (p_biased - p_clean).clamp(min=0).sum() > threshold`, relevance → 0, β suppressed. Prevents "mass going nowhere."

## Lineage

| Version | Focus | What it does |
|---------|-------|--------------|
| [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) | Multi-clock architecture | Temporal separation, routing deliberation, safety arbitration |
| [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) | Telemetry and governance | Observe topology, compute debt, apply lens pressure (medium clock) |
| **ChronoMoEv3** | Unified lifecycle | Phase coherence as the single state variable; lifecycle as slow-clock physics |

v3 wraps v2's `Controller` into a unified `ChronoSystem`. From the outside, one update call. Inside, three clocks tick at their own rates.

## Repository Status

- Architecture design in progress — see [projectdesign.md](projectdesign.md) for the full specification
- API unstable and subject to change
- Not yet ready for production use
- See [firststeps.md](firststeps.md) for getting started

## Planned Package Structure

```
chronomoe_v3/
├── __init__.py          # Public API
├── coherence.py         # Phase coherence computation and EMA tracking
├── clocks.py            # Three-timescale retention rates and decay helpers
├── lifecycle_ops.py     # Coordinator for all structural edits
├── ops/
│   ├── spawn.py         # Expert spawning (layer starvation response)
│   ├── prune.py         # Expert pruning (irreversible decoherence)
│   ├── split.py         # Expert splitting (bimodal coherence)
│   └── merge.py         # Expert merging (convergent substrates)
├── registry.py          # Expert registry with simplified state model
├── basin.py             # Basin history for interpretability and merge detection
├── system.py            # ChronoSystem — unified wrapper around v2 Controller
├── decisions.py         # Lifecycle decision logging (JSONL)
└── config.py            # ChronoConfig dataclass
```

## Quick Orientation

```python
from chronomoe import Controller, ControlConfig
from chronomoe_v3 import ChronoSystem, ChronoConfig

system = ChronoSystem(
    model=model,
    controller=Controller(n_layers=4, n_experts_per_layer=[8,8,8,8], config=ControlConfig()),
    config=ChronoConfig(),
)

# One call. Three clocks.
result = system.step(snapshot, lenses)
# result.control_decisions  — lens adjustments (medium clock)
# result.lifecycle_actions  — spawn/prune/split/merge (slow clock)
# result.coherence_state    — phi vectors for all experts
```

## Documentation

- **[projectdesign.md](projectdesign.md)** — Full architectural design: the unified phase coherence model, formal definitions, lifecycle actions as slow-clock physics, configuration, decision logging format
- **[firststeps.md](firststeps.md)** — Getting started guide: prerequisites, installation, project layout, key concepts
- **[dataflow_mixtral.md](dataflow_mixtral.md)** — Mixtral MoE dataflow analysis: router → expert → mixture wiring, hook points for coherence measurement
- **[dataflow_switch_transformer.md](dataflow_switch_transformer.md)** — Switch Transformer dataflow analysis: capacity-based dispatch, einsum patterns, clean tensor access
- **[dataflow_comparison.md](dataflow_comparison.md)** — Comparison of MoE implementations and recommendations for coherence.py design
- **[coherence_hooks.md](coherence_hooks.md)** — Reference implementation: MoETrace interface, minimal hook patches for both patterns, ChronoMoEv3 canonical API

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgements

ChronoMoEv3 builds on [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) and [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2), and is informed by the broader work at [Halcyon AI Research](https://www.halcyon.ie) on temporal architectures, resonance-driven learning, and mixture-of-experts stability.
