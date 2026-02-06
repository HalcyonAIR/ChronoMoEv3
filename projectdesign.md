# Project Design — ChronoMoEv3

**One Mechanism, Three Projections**

---

## The Core Idea

ChronoMoEv3 is not three systems bolted together. It is one dynamical system viewed from three measurement planes.

The single mechanism:

> A multi-timescale dynamical system where routing decisions are made while prior computation is still decaying, and only the parts that stay phase-aligned across longer time constants earn the right to keep influencing future routing.
>
> Three clocks with their own expert pools, overlapping delayed responses that sustain context, and slow trimming of what fails to persist — these are not three features. They are three projections of the same architecture. The goal of v3 is to make that unity explicit and implementable.
>
> ---
>
> ## Why One Thing, Not Three
>
> Previous versions described ChronoMoE as having separable components: temporal clocks, memory overlap, and safety/trimming. That framing was useful for building intuition but misleading for implementation. It encouraged treating lifecycle management as a layer stacked on top of governance, which is how the first draft of this document was written.
>
> The corrected framing:
>
> **Three clocks with their own experts** exist because distinct time constants need distinct representational substrates. Without separation, fast computation parasitises the slow substrate, and stable slow structure never forms. Separate expert pools enforce separation of time constants. This is not a design choice layered on top — it is a consequence of wanting multiple decay rates to coexist without collapse.
>
> **Overlapping continuous delayed responses** exist because the slow clock is not "memory" — it is unresolved computation with decay. When slow responses are still arriving while the next prompt is being processed, context is not being reloaded. It is being sustained. The overlap is the actual substrate of continuity.
>
> **Trimming what is not deliverable** exists because the only stable way to keep multi-timescale overlap from becoming noise is selection under persistence. The slow clock is a persistence filter: if a pattern cannot survive compression and still matter after decay, it loses influence. Not because it is "forgotten", but because it failed the survivability test.
>
> The clean unification:
>
> > Three clocks are three persistence filters with different half-lives, each with its own representational substrate, and a router that learns under the constraint that influence must be re-earned across longer persistence windows.
> >
> > This makes "experts per clock", "overlap keeping the locus alive", and "slow trimming" not three ideas, but three projections of the same architecture.
> >
> > ---
> >
> > ## The Litmus Test
> >
> > If the architecture is truly unified, we should be able to define a single scalar (or small vector) of state variables that each clock updates with different decay constants, and that the router consumes, such that lifecycle actions (spawn, prune, split, merge) are just slow-clock interventions triggered when those state variables show irreversible drift.
> >
> > If we can do that, we are specifying physics, not storytelling.
> >
> > ---
> >
> > ## The Unifying State Variable: Phase Coherence
> >
> > We choose **phase coherence** as the unifying state variable over the alternative candidate (reachability).
> >
> > ### Why Not Reachability
> >
> > Reachability measures how much policy space remains available to an expert. It answers the counterfactual: "could this expert still become useful?" The problem is that reachability degrades gracefully and then collapses suddenly. An expert can look reachable — weights non-degenerate, routing logits unclamped — right up until it is functionally dead. Reachability measures potential rather than actual coupling, and potential lies to you until it does not.
> >
> > Reachability would give clean signals for prune (reachability bottoms out) and spawn (layer-wide reachability is low), but split and merge would require additional signals bolted on. It does not unify the lifecycle actions under one measurement.
> >
> > ### Why Phase Coherence
> >
> > Phase coherence measures how aligned an expert's active contribution is with what the faster clocks are resolving. It answers: "is this expert's output actually landing in sync with the ensemble?"
> >
> > This is a measurement of current functional participation, not hypothetical future participation. And critically, phase coherence has a smooth, interpretable relationship with every lifecycle action:
> >
> > **Prune signal** — Coherence monotonically declining through the slow window. The expert is irreversibly decoupling from the ensemble. It failed the persistence test.
> >
> > **Split signal** — Coherence is high on average but exhibits growing variance or bimodal oscillation. One expert is trying to serve two phase-incompatible basins simultaneously. It needs to become two experts, each serving one basin.
> >
> > **Merge signal** — Two experts' coherence traces are converging. Their phase relationships with the rest of the ensemble are becoming indistinguishable. They are redundant substrates for the same functional role.
> >
> > **Spawn signal** — Layer-wide coherence is dropping while individual experts remain internally stable. The layer lacks the representational capacity to cover the phase space the router is trying to address. New substrate is needed.
> >
> > All four lifecycle actions fall out of one state variable's trajectory through the slow persistence window.
> >
> > ### Practical Advantages
> >
> > Phase coherence is cheaper to approximate during training than reachability. It can be estimated from the relationship between routing weights and expert output projections across consecutive steps — a running cosine similarity between what the router expects an expert to contribute and what it actually contributes, smoothed at each clock's decay rate. Reachability would require something closer to a Jacobian or policy gradient computation.
> >
> > The vulnerability of phase coherence is that it can be temporarily destroyed by learning rate spikes or data distribution shifts, which could trigger false lifecycle actions. But this is exactly what the persistence filter handles — if the coherence loss does not survive the slow clock's decay window, no action fires. Transient decoherence gets absorbed. Only irreversible drift passes the survivability test.
> >
> > ---
> >
> > ## Formal Definition
> >
> > ### Per-Expert Phase Coherence
> >
> > For expert `e` in layer `l` at training step `t`:
> >
> > ```
> > phi_e(t) = cosine_similarity(
> >     router_expected_output(e, t),
> >     actual_expert_output(e, t)
> > )
> > ```
> >
> > Where:
> > - `router_expected_output(e, t)` is the direction the router's gating weights predict expert `e` should contribute (the router's learned representation of what this expert does)
> > - - `actual_expert_output(e, t)` is the mean output vector expert `e` actually produced over the tokens routed to it at step `t`
> >  
> >   - This raw coherence is then smoothed at each clock's timescale:
> >  
> >   - ```
> >     Phi_e^(k)(t) = alpha_k * Phi_e^(k)(t-1) + (1 - alpha_k) * phi_e(t)
> >     ```
> >
> > Where `alpha_k` is the decay constant for clock `k`:
> > - `alpha_fast` ~ 0.9 (short half-life, ~10 steps)
> > - - `alpha_mid` ~ 0.99 (medium half-life, ~100 steps)
> >   - - `alpha_slow` ~ 0.999 (long half-life, ~1000 steps)
> >    
> >     - Each expert thus carries a small vector of state: `[Phi_fast, Phi_mid, Phi_slow]`.
> >    
> >     - ### Layer-Level Coherence
> >    
> >     - For layer `l`, aggregate coherence across active experts:
> >
> > ```
> > Psi_l(t) = weighted_mean(Phi_e^(slow)(t) for e in active_experts(l))
> > ```
> >
> > Weighted by routing share, so experts that handle more tokens contribute more to the layer signal.
> >
> > ### Cross-Scale Coherence Gradient
> >
> > The relationship between fast and slow coherence tells us about stability:
> >
> > ```
> > Delta_e(t) = Phi_e^(fast)(t) - Phi_e^(slow)(t)
> > ```
> >
> > - `Delta > 0`: Fast coherence is higher than slow. Expert is currently performing well but slow track has not caught up. Recent improvement, possibly transient.
> > - - `Delta ~ 0`: Fast and slow are aligned. Expert is in stable equilibrium.
> >   - - `Delta < 0`: Fast coherence is lower than slow. Expert is currently underperforming relative to its historical baseline. Degradation in progress.
> >    
> >     - Sustained `Delta < 0` through the slow window is the primary signal for lifecycle intervention.
> >    
> >     - ---
> >
> > ## The Persistence Filter
> >
> > The slow clock is the persistence filter. Its job is not to "remember" — it is to test survivability.
> >
> > A pattern (expert contribution, routing preference, basin structure) enters the system through fast-clock dynamics. It propagates through the medium clock. If it reaches the slow clock and persists through the slow decay window, it has earned structural influence. If it does not persist, it decays out naturally.
> >
> > Lifecycle actions are what happen at the slow clock boundary when the filter's output changes:
> >
> > ```
> > ┌─────────────────────────────────────────────────────────────┐
> > │                     FAST CLOCK                               │
> > │  Raw routing, token dispatch, expert forward passes          │
> > │  phi_e(t) computed here                                      │
> > │  Decay: alpha_fast ~ 0.9                                     │
> > ├─────────────────────────────────────────────────────────────┤
> > │                    MEDIUM CLOCK                              │
> > │  v2 lens controller operates here                            │
> > │  Soft redistribution via low-rank warp                       │
> > │  Phi_e^(mid) updated here                                    │
> > │  Decay: alpha_mid ~ 0.99                                     │
> > ├─────────────────────────────────────────────────────────────┤
> > │                     SLOW CLOCK                               │
> > │  Lifecycle decisions happen here                             │
> > │  Phi_e^(slow) and Delta_e evaluated here                     │
> > │  Structural changes: spawn / prune / split / merge           │
> > │  Decay: alpha_slow ~ 0.999                                   │
> > │                                                              │
> > │  This is the persistence filter. Only irreversible drift     │
> > │  that survives the slow decay window triggers action.        │
> > └─────────────────────────────────────────────────────────────┘
> > ```
> >
> > The three clocks are not three separate systems. They are three decay constants applied to the same state variable, with different actions gated at each timescale.
> >
> > ---
> >
> > ## Lifecycle Actions as Slow-Clock Physics
> >
> > ### Pruning: Irreversible Decoherence
> >
> > ```
> > Condition: Phi_e^(slow)(t) < prune_threshold
> >             AND Delta_e(t) < 0 (still declining, not recovering)
> >             AND duration(Phi_e^(slow) < prune_threshold) > prune_patience
> > ```
> >
> > The expert's contribution has decoupled from the ensemble and the slow filter confirms the drift is irreversible. The expert is not "dead" in the sense of zero utilisation (though that is often correlated) — it is dead in the sense that even when tokens are routed to it, its output is not phase-aligned with what the router expects. It has become noise.
> >
> > **Action:** Remove from active pool. Detach weights from optimizer. Archive basin history. Reclaim memory.
> >
> > ### Spawning: Capacity Starvation
> >
> > ```
> > Condition: Psi_l^(slow)(t) < spawn_threshold (layer coherence is low)
> >             AND mean(Phi_e^(slow)(t) for e in active) > individual_health_floor
> >             AND duration(Psi_l^(slow) < spawn_threshold) > spawn_patience
> > ```
> >
> > The layer's aggregate coherence is low, but individual experts are healthy. This means the layer lacks enough representational substrate to cover the phase space the router needs. Each expert is doing its job, but there are not enough of them.
> >
> > **Action:** Initialise a new expert. Default strategy is clone-and-perturb from the highest-share expert (the one most likely serving multiple basins). New expert enters cooling period before slow-clock evaluation begins.
> >
> > ### Splitting: Bimodal Coherence
> >
> > ```
> > Condition: variance(phi_e(t) over recent fast window) > split_variance_threshold
> >             AND Phi_e^(slow)(t) > split_health_floor (expert is not dying)
> >             AND Phi_e^(mid)(t) oscillates (not monotonically declining)
> >             AND duration(high variance) > split_patience
> > ```
> >
> > The expert's fast coherence is oscillating — high on some batches, low on others — while its slow coherence remains reasonable. This means the expert is serving two or more phase-incompatible basins and alternating between them. It is healthy on average but unstable in practice.
> >
> > **Action:** Clone the expert. Apply controlled perturbation to the clone's weights. Both experts enter the routing table. Over subsequent steps, the router should learn to send each basin's tokens to the appropriate child. The parent's coherence should stabilise (less variance) and the child's should rise from its cooling baseline.
> >
> > ### Merging: Convergent Substrates
> >
> > ```
> > Condition: cosine_similarity(weights_e1, weights_e2) > merge_weight_threshold
> >             AND cosine_similarity(Phi_e1^(slow), Phi_e2^(slow)) > merge_coherence_threshold
> >             AND duration(both conditions) > merge_patience
> > ```
> >
> > Two experts have converged to serving the same functional role. Their weights are similar, their phase coherence traces are similar, and this has persisted through the slow window. They are redundant.
> >
> > **Action:** Average weights (weighted by routing share). Archive the absorbed expert. The surviving expert enters a brief cooling period. The freed capacity is available for future spawning.
> >
> > ---
> >
> > ## Expert Registry
> >
> > The registry tracks every expert. It is simpler than the previous design because state is now carried by the coherence vector rather than by a complex state machine.
> >
> > ### Per-Expert Record
> >
> > ```python
> > @dataclass
> > class ExpertRecord:
> >     expert_id: str              # e.g., "L2_E5"
> >     layer_id: int
> >     born_step: int
> >     parent_id: Optional[str]    # If from split/spawn
> >     active: bool                # In the routing table or not
> >     cooling_until: Optional[int]  # Step when cooling period ends
> >
> >     # The core state — three persistence filters
> >     phi_fast: float             # EMA with alpha_fast
> >     phi_mid: float              # EMA with alpha_mid
> >     phi_slow: float             # EMA with alpha_slow
> >     delta: float                # phi_fast - phi_slow
> >
> >     # Basin history (rolling window for interpretability and merge detection)
> >     basin: BasinHistory
> > ```
> >
> > ### Expert Lifecycle (Simplified)
> >
> > The previous design had six states (pending_spawn, cooling, active, dormant, pending_prune, archived). Under the unified framing, we need fewer:
> >
> > ```
> >     spawn/split
> >         │
> >         ▼
> >    ┌──────────┐       phi_slow healthy       ┌──────────┐
> >    │ cooling  │ ──────────────────────────► │  active   │
> >    └──────────┘                              └─────┬─────┘
> >         ▲                                          │
> >         │ merge (survivor)              phi_slow irreversible drift
> >         │                                          │
> >         │                                          ▼
> >         │                                    ┌──────────┐
> >         └────────────────────────────────────│ archived │
> >                                              └──────────┘
> > ```
> >
> > There are really only three states: **cooling** (just born, not yet evaluated by slow clock), **active** (participating, coherence tracked), and **archived** (removed, basin history preserved). The transitions between them are determined entirely by the coherence state variables. No separate "dormant" or "pending" states are needed — those conditions are now readable directly from `phi_slow` and `delta`.
> >
> > ---
> >
> > ## Configuration
> >
> > ```python
> > @dataclass
> > class ChronoConfig:
> >     # Clock decay constants
> >     alpha_fast: float = 0.9
> >     alpha_mid: float = 0.99
> >     alpha_slow: float = 0.999
> >
> >     # Pruning
> >     prune_threshold: float = 0.2        # Phi_slow below which expert is decoherent
> >     prune_patience: int = 10            # Slow-clock evaluations of sustained decoherence
> >     min_experts_per_layer: int = 2
> >
> >     # Spawning
> >     spawn_threshold: float = 0.5        # Layer Psi_slow below which capacity is starved
> >     individual_health_floor: float = 0.4  # Individual experts must be healthy for spawn (not prune)
> >     spawn_patience: int = 5
> >     spawn_cooldown_steps: int = 500
> >     max_experts_per_layer: int = 16
> >     spawn_strategy: str = "clone_and_perturb"
> >
> >     # Splitting
> >     split_variance_threshold: float = 0.15  # Fast-coherence variance indicating bimodal service
> >     split_health_floor: float = 0.4         # Expert must be healthy to split (not dying)
> >     split_patience: int = 5
> >     split_perturbation_scale: float = 0.01
> >
> >     # Merging
> >     merge_weight_threshold: float = 0.95    # Weight cosine similarity
> >     merge_coherence_threshold: float = 0.9  # Slow coherence trace similarity
> >     merge_patience: int = 8
> >
> >     # Timing
> >     cooling_period: int = 3                 # Slow-clock evals before coherence tracking begins
> >     lifecycle_eval_interval: int = 5        # Slow-clock evals per v2 controller update cycle
> >
> >     # Basin tracking
> >     basin_window_size: int = 50
> > ```
> >
> > Note that the thresholds are now expressed in terms of a single family of measurements (coherence at different timescales) rather than a grab-bag of different metrics (share, entropy, Neff, topology debt). The old metrics are still computed by v2 for telemetry, but lifecycle decisions consume only the coherence state variables.
> >
> > ---
> >
> > ## Basin Tracking
> >
> > Basin tracking remains important for two reasons: merge detection (requires comparing what two experts are doing, not just how coherent each is individually) and post-hoc interpretability (understanding how the expert population evolved).
> >
> > ### Per-Checkpoint Basin Snapshot
> >
> > | Metric | Source | Purpose |
> > |--------|--------|---------|
> > | `share` | v2 telemetry | Routing load |
> > | `phi_fast` | Coherence computation | Current alignment |
> > | `phi_mid` | Coherence computation | Medium-term trend |
> > | `phi_slow` | Coherence computation | Structural health |
> > | `delta` | phi_fast - phi_slow | Stability gradient |
> > | `weight_norm` | Expert parameters | Drift tracking |
> > | `co_activation` | Routing events | Which other experts co-fire (for merge detection) |
> > | `output_direction` | Expert forward pass | Mean output vector (for merge detection) |
> >
> > ### Basin Similarity (for Merge)
> >
> > Two experts are merge candidates when both their coherence traces AND their functional roles converge. Coherence convergence means they relate to the ensemble the same way. Functional convergence means they produce similar outputs. Both are needed — two experts could have similar coherence but serve completely different token populations.
> >
> > ```
> > basin_similarity(e1, e2) = w1 * weight_cosine(e1, e2)
> >                          + w2 * coherence_trace_cosine(e1, e2)
> >                          + w3 * output_direction_cosine(e1, e2)
> > ```
> >
> > Default weights: `w1 = 0.4, w2 = 0.3, w3 = 0.3`.
> >
> > ---
> >
> > ## Decision Logging
> >
> > Every lifecycle action produces a `LifecycleDecision` record. The format now reflects the unified state variable:
> >
> > ```json
> > {
> >   "step": 15000,
> >   "run_id": "experiment_42",
> >   "clock": "slow",
> >   "action": "PRUNE",
> >   "layer_id": 2,
> >   "expert_id": "L2_E5",
> >   "coherence_state": {
> >     "phi_fast": 0.12,
> >     "phi_mid": 0.18,
> >     "phi_slow": 0.15,
> >     "delta": -0.03,
> >     "decoherence_duration": 12
> >   },
> >   "layer_state": {
> >     "psi_slow": 0.61,
> >     "n_active_experts": 7,
> >     "mean_phi_slow": 0.58
> >   },
> >   "trigger": "irreversible_decoherence",
> >   "config_snapshot": {
> >     "prune_threshold": 0.2,
> >     "prune_patience": 10,
> >     "alpha_slow": 0.999
> >   }
> > }
> > ```
> >
> > ---
> >
> > ## Integration with v2
> >
> > v3 does not sit "on top of" v2 as a separate layer. It is the slow-clock expression of the same system v2 governs at the medium clock.
> >
> > ```python
> > from chronomoe import Controller, ControlConfig
> > from chronomoe_v3 import ChronoSystem, ChronoConfig
> >
> > # The unified system
> > system = ChronoSystem(
> >     model=model,
> >     controller=Controller(
> >         n_layers=4,
> >         n_experts_per_layer=[8, 8, 8, 8],
> >         config=ControlConfig(),
> >     ),
> >     config=ChronoConfig(),
> > )
> >
> > # At each eval checkpoint:
> > # 1. v2 telemetry fires (fast clock observation)
> > # 2. v2 controller updates pressure and lens (medium clock)
> > # 3. If lifecycle_eval_interval reached, slow clock evaluates coherence
> > #    and may trigger structural changes
> > result = system.step(snapshot, lenses)
> >
> > # result.control_decisions  — v2 lens adjustments (medium clock)
> > # result.lifecycle_actions  — spawn/prune/split/merge (slow clock)
> > # result.coherence_state    — current phi vectors for all experts
> > ```
> >
> > The `ChronoSystem` wraps v2's `Controller` and adds the slow-clock coherence tracking and lifecycle logic. From the outside, there is one update call. Inside, three clocks tick at their own rates.
> >
> > ---
> >
> > ## The Key Insight (Restated for Implementers)
> >
> > If you are reading this to write code, here is what matters:
> >
> > 1. There is one state variable per expert: a coherence scalar `phi`, smoothed at three decay rates to produce `[phi_fast, phi_mid, phi_slow]`.
> >
> > 2. 2. `phi` measures whether the expert's output is aligned with what the router expects. It is a running cosine similarity, exponentially decayed.
> >   
> >    3. 3. The fast clock (alpha ~ 0.9) drives routing. The medium clock (alpha ~ 0.99) drives the lens controller. The slow clock (alpha ~ 0.999) drives lifecycle decisions.
> >      
> >       4. 4. Lifecycle actions are not external interventions imposed on the system. They are what the slow clock does when `phi_slow` shows irreversible drift. Prune is decoherence. Split is bimodal oscillation. Merge is convergence. Spawn is layer starvation.
> >         
> >          5. 5. The persistence filter is not a separate mechanism. It is the slow decay constant itself. Patterns that survive the slow EMA have earned structural influence. Patterns that do not are removed by the same exponential that decays them.
> >            
> >             6. 6. Every decision is logged with the coherence state that triggered it. Reproducibility is a property of the state variable trajectory, not of a separate audit log.
> >               
> >                7. ---
> >               
> >                8. ## Open Design Questions
> >               
> >                9. 1. **Coherence estimation method.** The cosine similarity between router expectation and expert output is the proposed estimator, but there may be cheaper approximations (e.g., using routing logit magnitudes as a proxy for expected alignment, or tracking the gradient of the routing loss with respect to expert assignment). The right estimator should be cheap per-step, numerically stable, and monotonically related to functional participation.
> > 
2. **Decay constant selection.** The proposed alpha values (0.9, 0.99, 0.999) correspond to half-lives of roughly 7, 69, and 693 steps. These may need to be expressed relative to the training schedule (fraction of total steps, or fraction of epoch) rather than as absolute step counts. The ratio between clocks matters more than the absolute values.

3. 3. **Optimizer state on structural change.** When an expert is spawned or split, should its optimizer state (momentum, Adam variance) inherit from the parent or start fresh? The coherence framework suggests starting fresh: the new expert should earn its own coherence from scratch, and stale optimizer state would confuse the fast-clock signal.
  
   4. 4. **Router dimensionality.** When expert count changes per layer, the router's output dimension changes. The cleanest approach is probably to maintain a fixed maximum router width and use a mask, rather than dynamically resizing. This avoids optimizer state discontinuities and keeps the router's learned preferences for existing experts intact.
     
      5. 5. **Cross-layer coherence.** Should `Psi_l` (layer coherence) influence decisions in other layers? The current design treats layers independently. A cross-layer signal might help maintain depth balance, but adds coupling that could cause cascade effects.
        
         6. 6. **Distributed synchronisation.** In DDP or model-parallel settings, coherence state must be consistent across ranks. Since `phi` is a simple EMA scalar per expert, synchronising it is cheap (all-reduce after each step). Lifecycle actions need to be deterministic given the same coherence state, so they can be computed independently on each rank without communication.
           
            7. ---
           
            8. ## Roadmap
           
            9. | Phase | Focus | Status |
            10. |-------|-------|--------|
            11. | Phase 1 | Coherence computation and EMA tracking at three timescales | Not started |
            12. | Phase 2 | Expert registry with simplified state model | Not started |
            13. | Phase 3 | Pruning via irreversible decoherence | Not started |
            14. | Phase 4 | Spawning via layer starvation detection | Not started |
            15. | Phase 5 | Splitting via bimodal coherence detection | Not started |
            16. | Phase 6 | Merging via convergent substrates | Not started |
            17. | Phase 7 | Integration as `ChronoSystem` wrapping v2 Controller | Not started |
            18. | Phase 8 | Benchmark: with/without lifecycle, coherence vs. ad-hoc metrics | Not started |
           
            19. Phase 1 is the critical foundation. If the coherence estimator works — if `phi` actually tracks functional participation and its slow EMA actually filters transients — everything else follows from the same mechanism. If it does not, we need a different state variable, and better to find out before building the lifecycle logic on top.
           
            20. ---
           
            21. ## References
           
            22. - [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) — multi-clock architecture and temporal separation
                - - [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) — telemetry, governance, and lens control
                  - - [nanoMoE](https://github.com/HalcyonAIR/nanoMoE) — minimal MoE training baseline
                    - - [Halcyon AI Research](https://www.halcyon.ie)
