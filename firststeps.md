# First Steps

A guide to getting started with ChronoMoEv3, whether you're exploring the architecture, contributing code, or integrating lifecycle management into your own MoE training pipeline.

---

## Prerequisites

ChronoMoEv3 builds on top of ChronoMoEv2. You should have a working understanding of:

- Python 3.10+
- - PyTorch (2.0+ recommended)
  - - Basic MoE concepts (expert routing, top-k dispatch, load balancing)
    - - ChronoMoEv2's telemetry primitives (`RoutingEvent`, `SystemSnapshot`, `ChronoLens`, `Controller`)
     
      - If you're new to the ChronoMoE project, read the [ChronoMoEv2 README](https://github.com/HalcyonAIR/ChronoMoEv2) first to understand the telemetry and governance loop that v3 extends.
     
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

        Once the package structure is in place, the repository will look like this:

        ```
        ChronoMoEv3/
        ├── chronomoe_lifecycle/     # Core package
        │   ├── __init__.py
        │   ├── spawner.py           # Expert spawning
        │   ├── pruner.py            # Expert pruning
        │   ├── splitter.py          # Expert splitting
        │   ├── merger.py            # Expert merging
        │   ├── basin.py             # Basin tracking
        │   ├── lifecycle.py         # Lifecycle coordinator
        │   ├── registry.py          # Expert registry
        │   ├── decisions.py         # Decision logging
        │   ├── config.py            # Configuration
        │   └── hooks.py             # v2 Controller integration
        ├── tests/                   # Test suite
        ├── examples/                # Usage examples
        ├── docs/                    # Extended documentation
        ├── firststeps.md            # This file
        ├── projectdesign.md         # Architecture design document
        ├── README.md
        ├── LICENSE
        ├── .gitignore
        └── pyproject.toml
        ```

        ## Understanding the Architecture

        Before writing code, read [projectdesign.md](projectdesign.md) to understand:

        1. **The lifecycle state machine** — how experts transition between states (active, cooling, pending-spawn, pending-prune, etc.)
        2. 2. **Decision gating** — how lifecycle actions are triggered by sustained telemetry signals rather than instantaneous spikes
           3. 3. **The registry model** — how the expert population is tracked across layers and training steps
              4. 4. **Integration with v2** — how lifecycle hooks attach to the existing `Controller` update loop
                
                 5. ## Running Tests
                
                 6. ```bash
                    # Run the full test suite
                    pytest tests/ -v

                    # Run a specific test module
                    pytest tests/test_spawner.py -v

                    # Run with coverage
                    pytest tests/ --cov=chronomoe_lifecycle --cov-report=term-missing
                    ```

                    ## Key Concepts

                    ### The Lifecycle Loop

                    ChronoMoEv3 extends the v2 governance loop with structural actions:

                    ```
                    observe → compute debt → update pressure → gate lens → LIFECYCLE DECISION → observe
                                                                    │
                                                        spawn / prune / split / merge
                    ```

                    The lifecycle coordinator runs at a slower cadence than the lens controller. While the lens adjusts every eval checkpoint, lifecycle decisions happen only after sustained signals persist across multiple checkpoints. This prevents thrashing and ensures structural changes are justified by genuine topological need.

                    ### Expert States

                    Each expert in the registry has a state:

                    - **active** — participating in routing, receiving tokens
                    - - **dormant** — share has dropped to zero but not yet confirmed dead
                      - - **pending_prune** — confirmed dead, scheduled for removal
                        - - **pending_spawn** — new expert initialised but not yet integrated into routing
                          - - **cooling** — recently spawned or split, in a grace period before full evaluation
                            - - **archived** — removed from the active pool, basin history preserved for analysis
                             
                              - ### Basin Histories
                             
                              - Every expert accumulates a rolling history of its routing characteristics: average share, entropy contribution, token-type distribution, and co-activation patterns with other experts. Basin histories are the primary input to lifecycle decisions. They answer questions like: is this expert drifting toward another expert's basin? Is this expert absorbing too much routing share? Has this expert been dormant long enough to prune?
                             
                              - ## Contributing
                             
                              - The project is in early development. If you want to contribute:
                             
                              - 1. Read [projectdesign.md](projectdesign.md) thoroughly
                                2. 2. Check the issue tracker for open tasks
                                   3. 3. Write tests alongside any new code
                                      4. 4. Keep commits focused — one logical change per commit
                                         5. 5. Use conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)
                                           
                                            6. ## What's Not Here Yet
                                           
                                            7. This is an honest assessment of what doesn't exist yet and what to expect:
                                           
                                            8. - The `chronomoe_lifecycle/` package directory and source files are not yet implemented
                                               - - `pyproject.toml` with build configuration is pending
                                                 - - Integration tests against v2's Controller are pending
                                                   - - Example notebooks and scripts are pending
                                                     - - Benchmark comparisons (with/without lifecycle) are planned but not started
                                                      
                                                       - The README and design documents describe the target architecture. The code will follow.
                                                      
                                                       - ## Links
                                                      
                                                       - - [ChronoMoEv3 README](README.md)
                                                         - - [Project Design Document](projectdesign.md)
                                                           - - [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) — telemetry and governance
                                                             - - [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) — original multi-clock architecture
                                                               - - [Halcyon AI Research](https://www.halcyon.ie)
