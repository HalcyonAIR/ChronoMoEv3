# Project Design — ChronoMoEv3

**Expert Lifecycle Management for Mixture-of-Experts Models**

This document describes the architectural design of ChronoMoEv3. It is a living document that will evolve as implementation progresses.

---

## Design Philosophy

ChronoMoEv3 follows three principles inherited from the broader ChronoMoE project:

**Observe before acting.** No lifecycle decision is made on a single measurement. All actions require sustained telemetry signals that persist across multiple evaluation checkpoints. Instantaneous spikes in routing metrics are noise; persistent topological shifts are signal.

**Structural changes are expensive.** Spawning, pruning, splitting, and merging experts are disruptive operations. They change the dimensionality of the routing problem. The system treats them as last-resort interventions that only fire when the v2 lens controller cannot resolve the underlying topology debt through soft redistribution alone.

**Every decision is logged.** Lifecycle decisions are written to JSONL alongside v2's telemetry and control decisions. The full history of expert births, deaths, splits, and merges is recoverable for post-hoc analysis. Reproducibility and interpretability are not optional.

---

## System Context

ChronoMoEv3 sits on top of the v2 governance stack:

```
┌─────────────────────────────────────────────────┐
│                 Training Loop                    │
│  (your MoE model, optimizer, data pipeline)      │
├─────────────────────────────────────────────────┤
│           ChronoMoEv2 — Telemetry               │
│  RoutingEvent → SystemSnapshot → Alerts          │
├─────────────────────────────────────────────────┤
│           ChronoMoEv2 — Governance              │
│  Controller → topology debt → pressure → lens    │
├─────────────────────────────────────────────────┤
│         ChronoMoEv3 — Lifecycle (NEW)           │
│  Registry → basin tracking → lifecycle decisions │
│  Spawner / Pruner / Splitter / Merger            │
└─────────────────────────────────────────────────┘
```

v3 reads from v2's `SystemSnapshot` and `ControlDecision` streams. It does not modify v2's internal state directly. Instead, it issues lifecycle actions that change the expert population, which v2 then observes on the next telemetry cycle.

---

## Expert Registry

The registry is the central data structure in v3. It tracks every expert across all MoE layers.

### Registry Fields (per expert)

| Field | Type | Description |
|-------|------|-------------|
| `expert_id` | `str` | Unique identifier (e.g., `"L2_E5"` for layer 2, expert 5) |
| `layer_id` | `int` | Which MoE layer this expert belongs to |
| `state` | `ExpertState` | Current lifecycle state (see state machine below) |
| `born_step` | `int` | Training step when this expert was created |
| `parent_id` | `Optional[str]` | If spawned via split, the parent expert's ID |
| `basin` | `BasinHistory` | Rolling routing basin history |
| `weight_ref` | `WeakRef` | Reference to the expert's parameters in the model |

### Expert States

```
                    ┌──────────┐
          spawn     │ pending  │
       ┌───────────►│  spawn   │
       │            └────┬─────┘
       │                 │ integrate
       │                 ▼
       │            ┌──────────┐         ┌──────────┐
       │            │ cooling  │────────►│  active   │◄─────────┐
       │            └──────────┘ grace   └────┬─────┘          │
       │              period          │         │ recover  │
       │              expires              │         │          │
       │                              share=0 │         │          │
       │                                   ▼         │          │
       │                             ┌──────────┐    │          │
       │                             │ dormant  │────┘          │
       │                             └────┬─────┘              │
       │                                  │ confirmed           │
       │                                  │ dead                │
       │                                  ▼                     │
       │                             ┌──────────┐              │
       │                             │ pending  │              │
       │                             │  prune   │              │
       │                             └────┬─────┘              │
       │                                  │ removed             │
       │                                  ▼                     │
       │                             ┌──────────┐              │
       └─────────────────────────────│ archived │──────────────┘
              (basin history          └──────────┘   (rare: revive
               preserved)                            from archive)
```

State transitions are governed by the lifecycle coordinator based on sustained telemetry signals and configurable thresholds.

---

## Lifecycle Actions

### Spawning

**Trigger:** Sustained capacity starvation across a layer — high routing entropy (tokens spread too thin) or saturated top-k (all active experts at capacity) persisting for `spawn_patience` consecutive checkpoints.

**Mechanism:**
1. The lifecycle coordinator identifies the layer with the highest unresolved topology debt (after lens intervention).
2. 2. A new expert is initialised. Initialisation strategy is configurable: random init, clone-and-perturb from the most loaded expert, or mean-of-cluster from a group of co-activated experts.
   3. 3. The new expert enters `pending_spawn` state.
      4. 4. On the next forward pass, the expert is integrated into the routing table.
         5. 5. The expert transitions to `cooling` state with a grace period before its share is evaluated.
           
            6. **Constraints:**
            7. - Maximum experts per layer (`max_experts_per_layer`)
               - - Global spawn cooldown (`spawn_cooldown_steps`) — minimum steps between any two spawn events
                 - - Spawn budget per lifecycle cycle (`max_spawns_per_cycle`)
                  
                   - ### Pruning
                  
                   - **Trigger:** An expert has been in `dormant` state (share == 0) for `prune_patience` consecutive checkpoints, and no recent lens intervention has attempted to revive it.
                  
                   - **Mechanism:**
                   - 1. The expert transitions to `pending_prune`.
                     2. 2. Its weights are detached from the model's parameter groups and optimizer state.
                        3. 3. Its routing slot is removed from the dispatch table.
                           4. 4. The expert transitions to `archived`. Its basin history is preserved.
                              5. 5. Memory is reclaimed.
                                
                                 6. **Constraints:**
                                 7. - Minimum experts per layer (`min_experts_per_layer`) — pruning cannot reduce below this floor
                                    - - Prune budget per lifecycle cycle (`max_prunes_per_cycle`)
                                     
                                      - ### Splitting
                                     
                                      - **Trigger:** A single expert consistently absorbs disproportionate routing share (`share > split_share_threshold`) for `split_patience` consecutive checkpoints, and the layer has capacity for an additional expert.
                                     
                                      - **Mechanism:**
                                      - 1. The parent expert's weights are cloned.
                                        2. 2. Controlled perturbation is applied to the clone (scaled noise proportional to the weight norms, configurable via `split_perturbation_scale`).
                                           3. 3. The parent remains active; the child enters `pending_spawn` → `cooling`.
                                              4. 4. Both experts share the parent's routing basin initially. Over subsequent checkpoints, their basins should diverge as the router learns to distinguish them.
                                                
                                                 5. **Constraints:**
                                                 6. - Same capacity constraints as spawning
                                                    - - Split cooldown per expert (`split_cooldown_per_expert`) — an expert cannot be split again within this window
                                                      - - The parent's share must still exceed the threshold at the moment of execution (re-checked to avoid stale decisions)
                                                       
                                                        - ### Merging
                                                       
                                                        - **Trigger:** Two experts in the same layer have converged to near-identical representations. Convergence is measured by weight cosine similarity (`> merge_similarity_threshold`) AND routing basin overlap (`> merge_overlap_threshold`) sustained for `merge_patience` checkpoints.
                                                       
                                                        - **Mechanism:**
                                                        - 1. The two experts' weights are averaged (or weighted by their respective routing shares).
                                                          2. 2. One expert absorbs the merged weights; the other is pruned.
                                                             3. 3. The merged expert inherits the combined basin history.
                                                                4. 4. The merged expert enters a brief `cooling` period.
                                                                  
                                                                   5. **Constraints:**
                                                                   6. - Minimum experts per layer floor applies
                                                                      - - Merge budget per lifecycle cycle (`max_merges_per_cycle`)
                                                                       
                                                                        - ---

                                                                        ## Basin Tracking

                                                                        Each expert maintains a `BasinHistory` — a rolling window of routing characteristics computed at each evaluation checkpoint.

                                                                        ### Basin Metrics (per checkpoint)

                                                                        | Metric | Description |
                                                                        |--------|-------------|
                                                                        | `share` | Expert's utilisation share (fraction of routed tokens) |
                                                                        | `entropy_contribution` | How much this expert contributes to the layer's routing entropy |
                                                                        | `token_type_dist` | Distribution over token types/positions routed to this expert |
                                                                        | `co_activation` | Which other experts are frequently co-selected with this one (for top-k > 1) |
                                                                        | `weight_norm` | L2 norm of the expert's parameters (tracks drift) |
                                                                        | `weight_delta` | Change in parameters since last checkpoint (tracks learning rate) |

                                                                        ### Basin Window

                                                                        Basin histories use a configurable rolling window (`basin_window_size`, default 50 checkpoints). Older entries are discarded. This keeps memory bounded while providing enough history for lifecycle decisions.

                                                                        ### Basin Similarity

                                                                        For merge detection, basin similarity between two experts is computed as the weighted combination of weight cosine similarity and token distribution overlap (Jensen-Shannon divergence). The exact formula and weights are configurable.

                                                                        ---

                                                                        ## Configuration

                                                                        All lifecycle behaviour is controlled through a `LifecycleConfig` dataclass:

                                                                        ```python
                                                                        @dataclass
                                                                        class LifecycleConfig:
                                                                            # Spawning
                                                                            spawn_patience: int = 5          # Checkpoints of sustained starvation before spawn
                                                                            spawn_cooldown_steps: int = 500  # Minimum training steps between spawns
                                                                            max_spawns_per_cycle: int = 1    # Maximum spawns per lifecycle evaluation
                                                                            max_experts_per_layer: int = 16  # Hard cap on experts per layer
                                                                            spawn_init_strategy: str = "clone_and_perturb"  # "random", "clone_and_perturb", "mean_cluster"

                                                                            # Pruning
                                                                            prune_patience: int = 10         # Checkpoints of zero share before prune
                                                                            max_prunes_per_cycle: int = 1
                                                                            min_experts_per_layer: int = 2   # Floor — never prune below this

                                                                            # Splitting
                                                                            split_share_threshold: float = 0.4   # Share above which an expert is considered overloaded
                                                                            split_patience: int = 5
                                                                            split_perturbation_scale: float = 0.01
                                                                            split_cooldown_per_expert: int = 1000  # Steps before same expert can split again

                                                                            # Merging
                                                                            merge_similarity_threshold: float = 0.95  # Weight cosine similarity
                                                                            merge_overlap_threshold: float = 0.8      # Basin routing overlap
                                                                            merge_patience: int = 8

                                                                            # Basin tracking
                                                                            basin_window_size: int = 50

                                                                            # Lifecycle cadence
                                                                            lifecycle_eval_interval: int = 5  # Run lifecycle eval every N v2 controller updates
                                                                            cooling_period: int = 3           # Checkpoints in cooling state before full evaluation
                                                                        ```

                                                                        ---

                                                                        ## Decision Logging

                                                                        Every lifecycle action produces a `LifecycleDecision` record written to `lifecycle_decisions.jsonl`:

                                                                        ```json
                                                                        {
                                                                          "step": 15000,
                                                                          "run_id": "experiment_42",
                                                                          "action": "SPAWN",
                                                                          "layer_id": 2,
                                                                          "expert_id": "L2_E9",
                                                                          "parent_id": "L2_E3",
                                                                          "reason": "sustained_starvation",
                                                                          "metrics_at_decision": {
                                                                            "layer_entropy": 1.82,
                                                                            "layer_neff": 4.1,
                                                                            "starvation_duration": 6,
                                                                            "topology_debt": 0.73
                                                                          },
                                                                          "config_snapshot": {
                                                                            "spawn_patience": 5,
                                                                            "spawn_init_strategy": "clone_and_perturb"
                                                                          }
                                                                        }
                                                                        ```

                                                                        This format is designed to be compatible with v2's `control_decisions.jsonl` for unified post-hoc analysis.

                                                                        ---

                                                                        ## Integration with v2

                                                                        v3 integrates via a `LifecycleHook` that attaches to v2's `Controller`:

                                                                        ```python
                                                                        from chronomoe import Controller, ControlConfig
                                                                        from chronomoe_lifecycle import LifecycleCoordinator, LifecycleConfig

                                                                        # Initialise v2 controller
                                                                        controller = Controller(
                                                                            n_layers=4,
                                                                            n_experts_per_layer=[8, 8, 8, 8],
                                                                            config=ControlConfig(),
                                                                        )

                                                                        # Attach v3 lifecycle
                                                                        lifecycle = LifecycleCoordinator(
                                                                            controller=controller,
                                                                            model=model,  # Your MoE model
                                                                            config=LifecycleConfig(),
                                                                        )

                                                                        # At each eval checkpoint:
                                                                        decisions = controller.update(snapshot, lenses)
                                                                        lifecycle_actions = lifecycle.evaluate(snapshot, decisions)
                                                                        # lifecycle_actions contains any spawn/prune/split/merge operations performed
                                                                        ```

                                                                        The lifecycle coordinator does not run on every controller update. It runs every `lifecycle_eval_interval` controller updates, providing a slower cadence appropriate for structural changes.

                                                                        ---

                                                                        ## Open Design Questions

                                                                        These are unresolved questions that will be addressed during implementation:

                                                                        1. **Optimizer state handling on spawn/prune.** When a new expert is spawned, should its optimizer state (momentum, variance) be initialised from the parent or from scratch? Inheriting state may speed convergence; zeroing it may prevent carrying over stale gradients.
                                                                       
                                                                        2. 2. **Router weight adjustment on population change.** When the expert count changes, the router's output dimension changes. Should the router be patched in-place (add/remove columns) or re-initialised? Patching preserves learned routing preferences; re-initialisation forces re-exploration.
                                                                          
                                                                           3. 3. **Cross-layer lifecycle correlation.** Should lifecycle decisions in one layer influence decisions in another? For example, if layer 2 spawns an expert, should layer 3 be given a higher spawn priority to maintain depth balance?
                                                                             
                                                                              4. 4. **Checkpoint compatibility.** How should model checkpoints handle variable expert counts? The checkpoint format needs to record the current registry state so that training can resume with the correct expert population.
                                                                                
                                                                                 5. 5. **Distributed training.** In DDP or model-parallel settings, lifecycle actions need to be synchronised across all ranks. The coordination protocol for this is not yet designed.
                                                                                   
                                                                                    6. ---
                                                                                   
                                                                                    7. ## Roadmap
                                                                                   
                                                                                    8. | Phase | Focus | Status |
                                                                                    9. |-------|-------|--------|
                                                                                    10. | Phase 1 | Registry and basin tracking | Not started |
                                                                                    11. | Phase 2 | Pruning (simplest lifecycle action) | Not started |
                                                                                    12. | Phase 3 | Spawning with clone-and-perturb | Not started |
                                                                                    13. | Phase 4 | Splitting | Not started |
                                                                                    14. | Phase 5 | Merging | Not started |
                                                                                    15. | Phase 6 | Integration tests with v2 | Not started |
                                                                                    16. | Phase 7 | Benchmark suite (with/without lifecycle) | Not started |
                                                                                   
                                                                                    17. ---
                                                                                   
                                                                                    18. ## References
                                                                                   
                                                                                    19. - [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) — multi-clock architecture and temporal separation
                                                                                        - - [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) — telemetry, governance, and lens control
                                                                                          - - [nanoMoE](https://github.com/HalcyonAIR/nanoMoE) — minimal MoE training baseline
                                                                                            - - [Halcyon AI Research](https://www.halcyon.ie)
