# ChronoMoEv3

**Expert Lifecycle Management for Mixture-of-Experts Models**

ChronoMoEv3 closes the loop on MoE topology control. Where [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) introduced multi-clock temporal separation and [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) added telemetry and governance (observe → pressure → lens), v3 adds the missing piece: **expert lifecycle management** — spawning, pruning, splitting, merging, and basin tracking for experts that live and die under real training pressure.

> **Status:** Early development. Architecture design and core implementation in progress. API unstable.
>
> ---
>
> ## Why Lifecycle?
>
> ChronoMoEv2 can detect collapse, measure topology debt, and apply low-rank lens interventions to redistribute routing. But it cannot create new experts when the topology is starved, remove dead experts cleanly, or split an overloaded expert into specialised children. The governance loop observes and pressures, but it cannot restructure.
>
> v3 adds structural adaptation: the ability to change the expert population itself in response to sustained topological signals.
>
> Without lifecycle management, MoE systems accumulate dead weight. Experts that collapse to zero utilisation still consume memory and compute. Overloaded experts become bottlenecks but have no mechanism to delegate. The topology ossifies into whatever shape emerged from initialisation, regardless of whether that shape serves the data.
>
> ## What v3 Adds
>
> **Expert Spawning** — When sustained routing pressure indicates capacity starvation (high entropy, saturated top-k), v3 can initialise new experts from the current topology. Spawning is gated by configurable thresholds and cooldown periods to prevent expert population explosion.
>
> **Expert Pruning** — Dead or near-dead experts (as defined by v2's strict telemetry: share == 0 over a sustained window) can be removed from the active pool. Pruning reclaims memory and reduces dispatch overhead without disrupting live routing.
>
> **Expert Splitting** — When a single expert consistently absorbs disproportionate routing share, it can be split into two children that inherit the parent's weights with controlled perturbation. This allows organic specialisation without manual architecture search.
>
> **Expert Merging** — When two experts converge to near-identical representations (measured by weight cosine similarity and routing overlap), they can be merged to reduce redundancy and free capacity for future spawning.
>
> **Basin Tracking** — Each expert maintains a lightweight history of its routing basin: which token distributions it attracts, how its share evolves over time, and where it sits in the topology graph. Basin histories inform all lifecycle decisions and provide post-hoc interpretability into how the expert population evolved during training.
>
> ## Relationship to Prior Versions
>
> | Version | Focus | Core Loop |
> |---------|-------|-----------|
> | [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) | Multi-clock architecture | Routing deliberation, temporal separation, safety arbitration |
> | [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) | Telemetry & governance | Observe → compute debt → pressure → lens warp |
> | **ChronoMoEv3** | Expert lifecycle | Observe → decide → spawn/prune/split/merge → re-observe |
>
> v3 depends on v2's telemetry primitives (`RoutingEvent`, `SystemSnapshot`, `ChronoLens`, `Controller`) and extends them with lifecycle actions. It does not replace v2 — it builds on top of it.
>
> ## Repository Status
>
> > ⚠️ **Experimental / Early Development**
> > >
> > >> - Architecture design in progress
> > >> - > - API unstable and subject to change
> > >>   > - > - Not yet ready for production use
> > >>   >   > - > - See [projectdesign.md](projectdesign.md) for the design document
> > >>   >   >   > - > - See [firststeps.md](firststeps.md) for getting started
> > >>   >   >   >   >
> > >>   >   >   >   > - ## Planned Package Structure
> > >>   >   >   >   >
> > >>   >   >   >   > - ```
> > >>   >   >   >   >   chronomoe_lifecycle/
> > >>   >   >   >   >   ├── __init__.py              # Public API
> > >>   >   >   >   >   ├── spawner.py               # Expert spawning logic and initialisation strategies
> > >>   >   >   >   >   ├── pruner.py                # Dead expert detection and removal
> > >>   >   >   >   >   ├── splitter.py              # Overloaded expert splitting with weight perturbation
> > >>   >   >   >   >   ├── merger.py                # Convergent expert merging
> > >>   >   >   >   >   ├── basin.py                 # Per-expert routing basin history and tracking
> > >>   >   >   >   >   ├── lifecycle.py             # Lifecycle coordinator (orchestrates spawn/prune/split/merge)
> > >>   >   >   >   >   ├── registry.py              # Expert registry — tracks active/inactive/pending experts
> > >>   >   >   >   >   ├── decisions.py             # Lifecycle decision logging (JSONL)
> > >>   >   >   >   >   ├── config.py                # LifecycleConfig dataclass
> > >>   >   >   >   >   └── hooks.py                 # Integration hooks for v2 Controller
> > >>   >   >   >   >   ```
> > >>   >   >   >   >
> > >>   >   >   >   > ## Getting Started
> > >>   >   >   >   >
> > >>   >   >   >   > See [firststeps.md](firststeps.md) for installation, dependencies, and initial setup.
> > >>   >   >   >   >
> > >>   >   >   >   > ## Design
> > >>   >   >   >   >
> > >>   >   >   >   > See [projectdesign.md](projectdesign.md) for the full architectural design, decision rationale, and lifecycle state machine specification.
> > >>   >   >   >   >
> > >>   >   >   >   > ## License
> > >>   >   >   >   >
> > >>   >   >   >   > MIT — see [LICENSE](LICENSE) for details.
> > >>   >   >   >   >
> > >>   >   >   >   > ## Acknowledgements
> > >>   >   >   >   >
> > >>   >   >   >   > ChronoMoEv3 builds on the foundation laid by [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) and [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2), and is informed by the broader work at [Halcyon AI Research](https://www.halcyon.ie) on temporal architectures, resonance-driven learning, and mixture-of-experts stability.
