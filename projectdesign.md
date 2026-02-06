# Project Design — ChronoMoEv3

**One Machine. One Job.**

Maintain a stable decision-centre under pressure by letting fast dynamics explore, mid dynamics negotiate, and slow dynamics commit — where "commit" literally means removing degrees of freedom.

---

## The Machine

Three expert pools. Not three models, not three controllers. Three time constants attached to the same state variables. The "overlap" between prompts is not memory. It is delayed computation and delayed influence.

Every expert carries one compact state vector, updated every step, decayed at three rates:

- **r_e** — a role vector representing what this expert tends to output when it is functioning
- - **phi_e** — a coherence signal measuring whether the expert is still playing its role in the current routing regime
  - - **b_e** — a bimodality signature detecting "I am serving two basins"
   
    - Three exponential smoothers are applied to the same signals: fast, mid, slow. Not three different computations. Three EMAs on the same streams.
   
    - ---

    ## The Coherence Measurement

    This is the part that needs to be real.

    Routers do not predict output directions. They pick experts. The most honest "expectation" you can compute without inventing a ghost network is the mixture output the router actually induced.

    ### Mixture Output

    For a given layer at step `t`, for each token `x` routed to top-k experts with gates `g_i(x,t)` and expert outputs `y_i(x,t)`:

    ```
    y_mix(x, t) = sum_{i in topk} g_i(x, t) * y_i(x, t)
    ```

    ### Expert Mean Output

    For each expert `e`, its mean output direction over the tokens it saw that step:

    ```
    y_bar_e(t) = mean_{x -> e} [ y_e(x, t) ]
    ```

    ### Coherence

    Coherence is "is the expert's contribution directionally aligned with what the layer actually produced under the router's decision":

    ```
    phi_e(t) = cosine( y_bar_e(t), y_bar_mix(t) )
    ```

    Where `y_bar_mix(t)` is the mean of `y_mix(x,t)` over the same token subset or the whole layer, depending on how strict you want it.

    This is almost too simple. But it is the right kind of simple. It is measurable, cheap, and it has teeth. If an expert is being routed tokens but its output is off in its own universe, `phi` drops.

    ---

    ## Three Clocks as One Persistence Filter

    For each expert `e`, keep three EMAs of `phi`:

    ```
    Phi_e^(k)(t) = alpha_k * Phi_e^(k)(t-1) + (1 - alpha_k) * phi_e(t)
    ```

    Fast `k=f`, mid `k=m`, slow `k=s`. The only thing that differs is `alpha`.

    | Clock | alpha | Half-life | Role |
    |-------|-------|-----------|------|
    | Fast | ~0.9 | ~10 steps | Explore. Raw routing and dispatch. |
    | Mid | ~0.99 | ~100 steps | Negotiate. v2 lens controller. Reversible soft redistribution. |
    | Slow | ~0.999 | ~1000 steps | Commit. Structural change. Irreversible. |

    That is the "one thing". Everything else is gated by which smoother you consult.

    ---

    ## The Locus: Delayed Influence, Not Delayed Compute

    The cleanest way to make "locus alive across prompts" real without pretending we have literal asynchronous compute returning late from a previous forward pass.

    Introduce a slow, decaying **influence bias** on routing. Not stored memory. A decayed state derived from coherence.

    For each expert, maintain a slow bias term:

    ```
    beta_e(t) = clip(
        beta_e(t-1) + eta_beta * (Phi_e^(s)(t) - tau),
        beta_min,
        beta_max
    )
    ```

    Add it into the router logits as a small additive prior:

    ```
    l'_e(x, t) = l_e(x, t) + beta_e(t)
    ```

    This does two things:

    **Cross-prompt continuity without RAG.** What persists is the routing geometry, not the tokens. Experts that have been coherent for a long time get a small routing advantage. The locus is not a sentence memory. It is a constraint field over routing.

    **Automatic trimming.** If something fails to stay coherent, its `beta` decays and then turns negative. The slow clock does not "forget" — it actively removes influence. This is the guillotine, in math, without melodrama.

    ---

    ## Fast Explores, Mid Negotiates, Slow Commits

    **Fast clock** — raw routing, token dispatch, expert forward passes. `phi_e(t)` computed here.

    **Mid clock** — v2 lens control. It operates on routing geometry softly. In this unified machine, the mid clock is simply "the control policy that tries to raise coherence without changing structure." v2 is trying to maximise layer coherence by warping routing distributions, reducing topology debt. But it is always reversible. It is negotiation.

    **Slow clock** — where structure changes happen. But to make it holistic: not separate spawn/prune/split/merge triggers. Instead, candidate edits competing under one slow objective.

    ---

    ## The Slow Objective: Free Energy with a Complexity Tax

    For each layer `l`, define a slow score to minimise:

    ```
    F_l = (1 - Psi_l) + lambda * N_l + rho * R_l + kappa * I_l
    ```

    Where:

    **Psi_l** = layer coherence, weighted by routing share:
    ```
    Psi_l = sum_{e in l} w_e * Phi_e^(s)
    ```

    **N_l** = number of active experts in the layer (complexity tax).

    **R_l** = redundancy: how many near-duplicates exist, computed from output-direction similarity and co-activation overlap.

    **I_l** = instability: how many experts show persistent bimodality.

    One number per layer. The slow clock acts only when it can reduce `F_l` enough to justify the structural disruption.

    This single objective replaces the entire rule bag.

    ---

    ## How Edits Work

    At each slow evaluation, propose a handful of candidate edits and estimate the delta in `F_l`:

    **Spawn** — Add one expert. Predict improvement based on current misfit and saturation. The complexity tax (`lambda * N_l`) punishes spawning unless it is truly needed.

    **Prune** — Remove the worst-coherence expert, but only if doing so reduces redundancy or instability enough, or if misfit does not worsen beyond a tolerance. This avoids pruning something that is low coherence because the layer is actually starving.

    **Merge** — Merge the most redundant pair. Complexity tax rewards it. Redundancy term collapses. Misfit should barely change.

    **Split** — Split the most bimodal expert. Complexity tax punishes it, but instability reduction and misfit improvement can outweigh.

    Then pick the **single best edit** per layer (or globally) subject to budgets and cooldowns. **Do nothing if no edit beats a minimum improvement threshold.**

    The calmness this buys is enormous. The system stops being eager to fiddle.

    ---

    ## The Bimodality Detector

    Variance of `phi` is a hint, but it is not a detector. The robust cheap detector is to track two running centroids of the expert's output direction.

    Maintain two unit vectors `c_e1`, `c_e2` per expert. Each step, assign `y_bar_e(t)` to the closer centroid and update it with an EMA. Also track assignment balance.

    ```python
    # Assign to closer centroid
    sim1 = cosine(y_bar_e, c_e1)
    sim2 = cosine(y_bar_e, c_e2)
    if sim1 >= sim2:
        c_e1 = ema_update(c_e1, y_bar_e, alpha_centroid)
        count1 += 1
    else:
        c_e2 = ema_update(c_e2, y_bar_e, alpha_centroid)
        count2 += 1

    # Bimodality score
    separation = 1 - cosine(c_e1, c_e2)
    balance = min(count1, count2) / max(count1, count2)
    bimodality = separation * balance  # High when both centroids are separated and both used
    ```

    If both centroids remain separated and both receive significant mass over time, you have persistent bimodality. That is split, not prune.

    Still small-state. Still cheap. Catches "serving two basins" even if average coherence stays decent.

    ---

    ## What Lifecycle Means in This Machine

    **Prune** is not "share == 0". Prune is "even when I am used, I am decoherent, and this persists through slow." If an expert never fires, treat it as unmeasurable and prune via utilisation as a sampling fallback. But coherence is the primary ontology.

    **Spawn** is not "entropy high". Spawn is "layer misfit stays high while individual experts are coherent." That is starvation, not chaos.

    **Merge** is not "weights similar". Merge is "outputs and slow coherence traces are functionally redundant." Weight similarity can be a cheap prefilter, but it is not the truth.

    **Split** is not "one expert has too much share". Split is "one expert has two roles."

    All four are expressions of the same slow objective.

    ---

    ## Per-Expert State

    ```python
    @dataclass
    class ExpertState:
        expert_id: str              # e.g., "L2_E5"
        layer_id: int
        born_step: int
        parent_id: Optional[str]
        active: bool
        cooling_until: Optional[int]

        # Three persistence filters on the same signal
        phi_fast: float
        phi_mid: float
        phi_slow: float

        # Slow influence bias (the locus mechanism)
        beta: float

        # Bimodality detector
        centroid_1: Tensor          # Unit vector, d_model
        centroid_2: Tensor          # Unit vector, d_model
        centroid_balance: float     # min(c1,c2)/max(c1,c2)
        bimodality: float           # separation * balance

        # Role vector (what this expert tends to produce)
        role_vector: Tensor         # EMA of y_bar_e, d_model

        # Basin history for logging and merge detection
        basin: BasinHistory
    ```

    Three states: **cooling** (just born, not yet evaluated by slow clock), **active** (participating, coherence tracked), **archived** (removed, basin preserved).

    ---

    ## Router: Fixed Width with Masking

    Do not resize the router. Keep a fixed maximum width per layer and mask inactive experts. This makes optimizer state stable and distributed training sane.

    When you spawn, unmask a column and initialise its parameters and its `beta` in a controlled way. When you prune, mask it and optionally park weights for archive.

    ```python
    # Router output: [batch, seq, max_experts_per_layer]
    logits = router(x)                          # Always full width
    logits = logits + beta_vector               # Add slow influence bias
    logits = logits.masked_fill(~active_mask, -inf)  # Mask inactive
    gates = topk_softmax(logits, k)
    ```

    ---

    ## Optimizer State on Structural Change

    Resolved: **fresh for children, always.**

    If you inherit Adam moments, you inherit the parent's past, and you confuse the coherence dynamics because the child will move as if it had earned its role already. The whole point is that new substrate must earn influence through persistence.

    For merge, the survivor keeps its optimizer state. The absorbed expert is archived.

    ---

    ## Configuration

    ```python
    @dataclass
    class ChronoConfig:
        # Clock decay constants
        alpha_fast: float = 0.9
        alpha_mid: float = 0.99
        alpha_slow: float = 0.999

        # Slow bias (locus mechanism)
        eta_beta: float = 0.01      # Learning rate for beta updates
        tau: float = 0.5            # Coherence threshold for beta growth
        beta_min: float = -1.0
        beta_max: float = 1.0

        # Free energy weights
        lambda_complexity: float = 0.01   # Complexity tax per expert
        rho_redundancy: float = 0.1       # Redundancy penalty
        kappa_instability: float = 0.1    # Instability penalty

        # Edit selection
        min_improvement: float = 0.01     # Minimum delta-F to justify an edit
        max_edits_per_cycle: int = 1      # Maximum structural changes per slow eval
        cooldown_steps: int = 500         # Minimum steps between edits in same layer

        # Expert bounds
        min_experts_per_layer: int = 2
        max_experts_per_layer: int = 16

        # Spawning
        spawn_strategy: str = "clone_and_perturb"
        split_perturbation_scale: float = 0.01

        # Bimodality detector
        alpha_centroid: float = 0.95      # EMA rate for centroid updates

        # Timing
        cooling_period: int = 3
        lifecycle_eval_interval: int = 5

        # Basin tracking
        basin_window_size: int = 50
    ```

    ---

    ## Decision Logging

    Every lifecycle action logs the coherence state AND the free energy delta that justified it:

    ```json
    {
      "step": 15000,
      "run_id": "experiment_42",
      "clock": "slow",
      "action": "SPLIT",
      "layer_id": 2,
      "expert_id": "L2_E3",
      "child_id": "L2_E9",
      "coherence_state": {
        "phi_fast": 0.71,
        "phi_mid": 0.65,
        "phi_slow": 0.62,
        "beta": 0.34,
        "bimodality": 0.73
      },
      "free_energy": {
        "F_before": 0.82,
        "F_after_predicted": 0.68,
        "delta_F": -0.14,
        "components": {
          "misfit_delta": -0.08,
          "complexity_delta": 0.01,
          "redundancy_delta": 0.0,
          "instability_delta": -0.07
        }
      },
      "competing_edits": [
        {"action": "PRUNE", "target": "L2_E7", "delta_F": -0.02},
        {"action": "MERGE", "targets": ["L2_E1", "L2_E4"], "delta_F": -0.05},
        {"action": "SPAWN", "delta_F": -0.03}
      ],
      "trigger": "bimodal_expert_split_wins_objective"
    }
    ```

    Not "we pruned because share was low", but "we committed because the slow objective improved beyond threshold and the improvement persisted long enough to justify irreversible change." That is how you stop reviewers calling it vibes.

    ---

    ## Integration with v2

    ```python
    from chronomoe import Controller, ControlConfig
    from chronomoe_v3 import ChronoSystem, ChronoConfig

    system = ChronoSystem(
        model=model,
        controller=Controller(
            n_layers=4,
            n_experts_per_layer=[8, 8, 8, 8],
            config=ControlConfig(),
        ),
        config=ChronoConfig(),
    )

    # One call. Three clocks.
    result = system.step(snapshot, lenses)

    # Fast: routing happened, phi_e computed
    # Mid: v2 lens adjusted routing pressure (negotiation)
    # Slow (if interval reached): F_l evaluated, best edit selected or nothing
    ```

    ---

    ## What This Hits

    **Three clocks with separate experts** — there, but now explained as separation of substrates for separation of half-lives.

    **Overlapping delayed responses** — exists as persistent influence (`beta`) that decays. The operational core of "unresolved computation" without the engineering chaos of true async expert returns.

    **Trimming** — not a bolt-on. The same slow decay and the same slow objective penalising complexity. Things that do not persist simply lose influence and eventually lose existence.

    **Locus maintenance** — a measurable phenomenon: stable slow bias fields over routing that resist perturbation and only change when slow evidence beats the commitment threshold.

    **v3 lifecycle** — stops being a separate "manager". Becomes the slow-clock expression of the same state variables v2 is already trying to stabilise.

    ---

    ## The Key Question for Implementers

    When the slow clock says "do nothing" while the fast clock is screaming, do you trust the slow clock?

    The answer must be yes. If you override the slow clock because the fast dynamics look alarming, you have broken the persistence filter. You have told the system that commitment is negotiable. And it will learn that structural influence can be earned by being loud rather than by being durable.

    The slow clock's "do nothing" is not passivity. It is the system saying "this has not persisted long enough to justify irreversible change." That restraint is the entire mechanism. Without it, you are back to reactive heuristics.

    ---

    ## Roadmap

    | Phase | Focus | Status |
    |-------|-------|--------|
    | Phase 1 | Coherence computation: mixture-output cosine, EMA at three rates | Not started |
    | Phase 2 | Slow bias beta: influence on router logits, decay dynamics | Not started |
    | Phase 3 | Bimodality detector: two-centroid tracking | Not started |
    | Phase 4 | Free energy F_l: misfit + complexity + redundancy + instability | Not started |
    | Phase 5 | Edit proposal and selection under F_l | Not started |
    | Phase 6 | Expert registry with fixed-width router masking | Not started |
    | Phase 7 | Integration as ChronoSystem wrapping v2 Controller | Not started |
    | Phase 8 | Benchmarks: with/without lifecycle, F_l vs. ad-hoc triggers | Not started |

    Phase 1 remains critical. If phi does not track functional participation, nothing else matters. But now Phase 2 is equally critical — the slow bias is what makes the locus real.

    ---

    ## References

    - [ChronoMoE](https://github.com/HalcyonAIR/ChronoMoE) — multi-clock architecture and temporal separation
    - - [ChronoMoEv2](https://github.com/HalcyonAIR/ChronoMoEv2) — telemetry, governance, and lens control
      - - [nanoMoE](https://github.com/HalcyonAIR/nanoMoE) — minimal MoE training baseline
        - - [Halcyon AI Research](https://www.halcyon.ie)
