# First Steps

A guide to getting started with ChronoMoEv3.

---

## Prerequisites

ChronoMoEv3 extends ChronoMoEv2. You should have a working understanding of:

- Python 3.10+
- - PyTorch (2.0+ recommended)
  - - Basic MoE concepts (expert routing, top-k dispatch, load balancing)
    - - ChronoMoEv2's telemetry primitives (`RoutingEvent`, `SystemSnapshot`, `ChronoLens`, `Controller`)
     
      - If you are new to the ChronoMoE project, read the [ChronoMoEv2 README](https://github.com/HalcyonAIR/ChronoMoEv2) first.
     
      - ## Read the Design Document First
     
      - Before touching code, read [projectdesign.md](projectdesign.md). The design has a specific framing that matters:
     
      - ChronoMoEv3 is **one mechanism, not three**. There is a single state variable — phase coherence (`phi`) — tracked at three decay rates. The fast clock drives routing, the medium clock drives the lens controller, and the slow clock drives lifecycle decisions (spawn, prune, split, merge). Lifecycle actions are not external interventions stacked on top. They are what the slow clock does when coherence shows irreversible drift.
     
      - If you find yourself thinking about lifecycle as a separate layer sitting on top of v2, re-read the design doc. The unified framing is not decoration — it determines the implementation.
     
      - ## Installation
     
      - ### From source (recommended during early development)
     
      - ```bash
        git clone https://github.com/HalcyonAIR/ChronoMoEv3.git
        cd ChronoMoEv3
        pip install -e ".[dev]"
        ```

        ### Dependencies

        Core dependencies (will be specified in `pyproject.toml`):

        ```
        torch >= 2.0
        chronomoe >= 0.2.0  # ChronoMoEv2 package
        numpy
        ```

        Development dependencies:

        ```
        pytest
        ruff
        mypy
        ```

        ## Project Layout

        ```
        ChronoMoEv3/
        ├── chronomoe_v3/            # Core package
        │   ├── __init__.py
        │   ├── coherence.py         # Phase coherence computation and EMA tracking
        │   ├── clocks.py            # Three-timescale decay constants and update logic
        │   ├── spawner.py           # Expert spawning (layer starvation response)
        │   ├── pruner.py            # Expert pruning (irreversible decoherence)
        │   ├── splitter.py          # Expert splitting (bimodal coherence)
        │   ├── merger.py            # Expert merging (convergent substrates)
        │   ├── registry.py          # Expert registry with simplified state model
        │   ├── basin.py             # Basin history for interpretability and merge detection
        │   ├── system.py            # ChronoSystem — unified wrapper around v2 Controller
        │   ├── decisions.py         # Lifecycle decision logging (JSONL)
        │   └── config.py            # ChronoConfig dataclass
        ├── tests/
        ├── examples/
        ├── docs/
        ├── firststeps.md            # This file
        ├── projectdesign.md         # Architecture design document
        ├── README.md
        ├── LICENSE
        ├── .gitignore
        └── pyproject.toml
        ```

        ## Key Concepts

        ### The One State Variable

        Every expert carries a coherence scalar `phi` — a running cosine similarity between what the router expects the expert to contribute and what it actually produces. This is smoothed at three timescales via exponential moving averages:

        ```
        phi_fast  (alpha ~ 0.9)    — what the expert is doing right now
        phi_mid   (alpha ~ 0.99)   — what the expert has been doing recently
        phi_slow  (alpha ~ 0.999)  — what the expert has been doing structurally
        ```

        The difference `delta = phi_fast - phi_slow` tells you about stability. If delta is negative, fast performance is below the structural baseline — degradation is in progress.

        ### The Three Clocks

        The clocks are not three separate systems. They are three decay constants applied to the same measurement:

        - **Fast clock** — Drives routing. Tokens go to experts with high fast coherence. This is where the raw physics of dispatch happens.
        - - **Medium clock** — Drives the v2 lens controller. When medium-timescale coherence shows drift, the lens applies a low-rank warp to redistribute routing pressure. Soft correction.
          - - **Slow clock** — Drives lifecycle. When slow-timescale coherence shows irreversible drift, structural changes fire: spawn new experts, prune dead ones, split overloaded ones, merge redundant ones. Hard correction.
           
            - ### The Persistence Filter
           
            - The slow clock is a persistence filter. Its job is not to remember — it is to test survivability. A pattern that enters through fast-clock dynamics and propagates to the slow clock has earned structural influence. A pattern that decays out of the slow window has failed the survivability test and loses influence.
           
            - This is why lifecycle actions do not need separate trigger mechanisms. Pruning is what happens when `phi_slow` drops below threshold. Spawning is what happens when layer-wide `Psi_slow` drops while individual experts are healthy. The persistence filter IS the decision mechanism.
           
            - ### Expert States (Simplified)
           
            - Only three states: **cooling** (just born, not yet evaluated by slow clock), **active** (participating, coherence tracked), and **archived** (removed, basin history preserved). No complex state machine. Transitions are determined entirely by the coherence state variables.
           
            - ## Running Tests
           
            - ```bash
              # Run the full test suite
              pytest tests/ -v

              # Run with coverage
              pytest tests/ --cov=chronomoe_v3 --cov-report=term-missing
              ```

              ## Contributing

              The project is in early development. If you want to contribute:

              1. Read [projectdesign.md](projectdesign.md) thoroughly — especially "The Key Insight (Restated for Implementers)"
              2. 2. Phase 1 (coherence computation) is the critical foundation. If `phi` does not track functional participation, everything else fails
                 3. 3. Write tests alongside any new code
                    4. 4. Keep commits focused — one logical change per commit
                       5. 5. Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
                         
                          6. ## What is Not Here Yet
                         
                          7. - The `chronomoe_v3/` package directory and source files are not yet implemented
                             - - `pyproject.toml` with build configuration is pending
                               - - The coherence estimator (the most important piece) needs to be prototyped and validated
                                 - - Integration tests against v2's Controller are pending
                                   - - Benchmark comparisons are planned but not started
                                    
                                     - The design documents describe the target architecture. The code will follow, starting with `coherence.py` — the single most important file in the project.
                                    
                                     - ## Links
                                    
                                     - - [ChronoMoEv3 README](README.md)
                                       - - [Project Design Document](projectdesign.md)
                                         - - [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) — telemetry and governance
                                           - - [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) — original multi-clock architecture
                                             - - [Halcyon AI Research](https://www.halcyon.ie)
