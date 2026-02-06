# ChronoMoEv3

**One Mechanism, Three Projections**

ChronoMoEv3 is a multi-timescale dynamical system where routing decisions are made while prior computation is still decaying, and only the parts that stay phase-aligned across longer time constants earn the right to keep influencing future routing.

Three clocks with their own expert pools, overlapping delayed responses that sustain context, and slow trimming of what fails to persist are three projections of the same architecture. v3 makes that unity implementable by defining a single state variable — **phase coherence** — tracked at three decay rates. Lifecycle actions (spawn, prune, split, merge) are what the slow clock does when coherence shows irreversible drift.

Phase coherence is estimated from the router-induced mixture output: compare an expert's mean output direction to the mixture direction produced by the router on the same batch.

Fast and medium clocks can raise an emergency flag that expedites slow-clock review; irreversible edits still require slow confirmation.

> [!WARNING]
> **Experimental / Early Development**
>
> **Status:** Early development. Architecture design and core implementation in progress. API unstable.

## The Architecture in Brief

Each clock is a sliding window of unresolved influence, not a memory store. The past is present only insofar as it has not finished decaying. Fast trails a few steps — immediate continuity, the pressure of the last few turns still exerting influence on routing. Medium trails context — alignment across interruption, the reason a conversation has shape instead of being a bag of replies. Slow trails trajectory — what has proven important enough, repeatedly enough, under enough pressure, that letting it decay would break continuity.

None of these clocks look backward. They trail forward. They do not remember. They just have not let go yet. And because each window slides rather than fixes, alignment is maintained without freezing. The system can drift, but it cannot teleport. Discontinuous jumps through a decay window that will not allow them are what trigger structural responses.

For the formal treatment of pressure, hysteresis, and selector locus formation that motivates this architecture, see [Pressure, Hysteresis, and the Geometry of Becoming](https://halcyon.ie/blog/pressure-hysteresis-and-the-geometry-of-becoming/).

Each expert carries one state variable: `phi`, a measure of whether its output stays phase-aligned with the router-induced mixture direction. This scalar is smoothed at three timescales:

| Clock | Decay | Half-life | Governs |
|-------|-------|-----------|---------| 
| Fast | α ~ 0.9 | ~10 steps | Routing and token dispatch |
| Medium | α ~ 0.99 | ~100 steps | Lens controller (v2 soft redistribution) |
| Slow | α ~ 0.999 | ~1000 steps | Lifecycle decisions (structural changes) |

Half-lives are illustrative; only the ratios matter.

The slow clock is a persistence filter. Patterns that survive its decay window earn structural influence. Patterns that do not are removed by the same exponential that decays them. Lifecycle actions are not external interventions — they are slow-clock physics:

**Prune** — Coherence monotonically declining through the slow window. Irreversible decoherence. The expert failed the persistence test.

**Split** — Coherence oscillating at the fast timescale while stable at the slow. The expert is serving two phase-incompatible basins. It needs to become two.

**Merge** — Two experts' coherence traces converging. Redundant substrates for the same functional role.

**Spawn** — Layer-wide coherence dropping while individual experts remain healthy. Not enough representational capacity to cover the phase space.

Lifecycle actions are evaluated as candidate structural edits; the slow clock applies an edit only when evidence beats a complexity cost through persistence.

## Lineage

| Version | Focus | What it does |
|---------|-------|--------------|
| [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) | Multi-clock architecture | Temporal separation, routing deliberation, safety arbitration |
| [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) | Telemetry and governance | Observe topology, compute debt, apply lens pressure (medium clock) |
| **ChronoMoEv3** | Unified lifecycle | Phase coherence as the single state variable; lifecycle as slow-clock physics |

v3 wraps v2's `Controller` into a unified `ChronoSystem`. From the outside, one update call. Inside, three clocks tick at their own rates.

## Repository Status

> [!WARNING]
> **Experimental / Early Development**

- Architecture design in progress — see [projectdesign.md](projectdesign.md) for the full specification
- API unstable and subject to change
- Not yet ready for production use
- See [firststeps.md](firststeps.md) for getting started

## Planned Package Structure

```
chronomoe_v3/
├── __init__.py          # Public API
├── coherence.py         # Phase coherence computation and EMA tracking
├── clocks.py            # Three-timescale decay constants and update logic
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

## License

MIT — see [LICENSE](LICENSE) for details.

## Acknowledgements

ChronoMoEv3 builds on [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) and [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2), and is informed by the broader work at [Halcyon AI Research](https://www.halcyon.ie) on temporal architectures, resonance-driven learning, and mixture-of-experts stability.
