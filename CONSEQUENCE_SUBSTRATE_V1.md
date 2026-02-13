# ChronoMoE Consequence Substrate v1

**Systems Architecture Document**
**Status**: Canonical spec — repo-level reference
**Authors**: Jeff Colhoun (HalcyonAIR), with collaborative development from Halcyon (GPT) and Claude (Anthropic)
**Date**: February 2026

---

## 1. System Overview

### 1.1 Objective

Consequence accumulation reduces compute cost per decision without degrading quality. The system caches and reuses **decision structure** — the routing constraint that produced an answer — not the answer itself.

### 1.2 Core Invariant

**The Pareto curve (cost vs. quality) moves inward over experience.**

Cost is measured as expert invocations per decision. Quality is measured as task accuracy, constraint satisfaction, or validation loss. As experience accumulates, the system reaches acceptable decisions with fewer expert calls, fewer tokens processed, and fewer exploratory routing steps.

### 1.3 Prerequisite: Routing Commitment

**Consequence accumulation requires routing commitment.** If the router does not develop dominant motifs under the given task and architecture, the commitment cache remains mostly unused and cost will not fall. The compound gate will correctly refuse to shortcut, and the system will continue paying full routing cost indefinitely.

This is not a failure of the system — it is correct behaviour. The gate is doing its job. Cost reduction is earned, not assumed.

Observed architectural boundary: in 8-expert top-2 configurations, routing stability remained flat and the compound gate almost never opened. The action space was too sparse relative to expert count for dominant motifs to form. This establishes that consequence accumulation is not guaranteed by architecture alone. It depends on the interaction between expert count, top-k branching factor, and task complexity. See Section 7.6 and Section 10.1.

### 1.4 Mechanism (One Paragraph)

An external temporal controller monitors MoE routing dynamics across three timescales. As the system encounters repeated decision classes, stable routing patterns (motifs) crystallise and are stored in a commitment cache indexed by context class. On subsequent encounters, a compound structural gate evaluates whether accumulated evidence justifies executing the cached motif directly (cheap path) rather than full exploratory routing (expensive path). The gate is structural, not confidence-based: it requires routing stability, low debt, and motif survival evidence. Governance state determines whether cheap-path access is permitted at all.

---

## 2. Core Components

### 2.1 Three-Clock Architecture

The system operates across three temporal scales. Each clock has a dedicated keeper process that maintains constraint state.

| Clock | Timescale | Keeper Role | Irreversible Event |
|-------|-----------|-------------|-------------------|
| Fast (τ_f) | Token / micro-batch | Reactive routing, reflex detection | Reflex Crystallisation ("always this") |
| Medium (τ_m) | Session / conversation window | Pressure-temperature coupling, continuity arbitration | Epoch Boundary ("before & after") |
| Slow (τ_s) | Multi-session / deployment horizon | Scar accumulation, identity constraint | Scar Formation ("never again") |

Design principle: if something can happen instantly, it is not identity — it is a glitch. All irreversible events require evidence accumulation across their respective timescale.

### 2.2 Three-State Convergence Detector

The system classifies its current regime into one of three states. State classification in the validated implementation uses a **slope-plus-magnitude** regime on gradient norms and loss trajectory. Primary classifier signal: slope and magnitude of gradient norms over a trailing window determine transition direction and thresholds. Routing entropy is supplementary telemetry — logged, correlated, but not authoritative for state transitions.

| State | Condition | Routing Behaviour |
|-------|-----------|-------------------|
| **DRIFT** | Gradient norm slope positive and steep, loss trajectory unstable (positive second derivative) | Full exploratory routing mandatory. Cheap path disabled. |
| **EQUILIBRIUM** | Gradient norm slope near-zero, loss trajectory converging (negative second derivative sustained) | Cheap path permitted if compound gate passes. Motif library updated. |
| **SETTLEMENT** | Gradient norm slope near-zero sustained for extended window, loss trajectory flat at local minimum, motifs surviving drift tests | Conditional freeze evaluation enabled. Topology gating active. |

Transitions are hysteretic: entering SETTLEMENT requires sustained EQUILIBRIUM plus detectable baseline shift. Exiting SETTLEMENT back to DRIFT requires sustained routing instability, not a single perturbation.

### 2.3 Governance State Machine

The governance layer enforces constraints on what the routing system is permitted to do in each state.

```
DRIFT ──(entropy stabilises, loss converges)──→ EQUILIBRIUM
EQUILIBRIUM ──(sustained stability + baseline shift)──→ SETTLEMENT
SETTLEMENT ──(sustained instability)──→ DRIFT
SETTLEMENT ──(crystallisation gates pass)──→ FROZEN (conditional)
FROZEN ──(sustained DRIFT signal)──→ SETTLEMENT (reversible)
```

Rules:
- DRIFT disables cheap path entirely. All inputs pay full routing cost.
- EQUILIBRIUM allows cheap path subject to compound gate.
- SETTLEMENT enables conditional freeze and crystallisation candidate evaluation.
- FROZEN pathways are reversible: sustained DRIFT signal triggers re-evaluation via epoch review.

Epoch review asks: "If we erased this constraint today, would we recreate it under current conditions?" If yes, it persists. If no, it decays.

### 2.4 Commitment Cache

The commitment cache stores **routing motifs** — not answers, not expert weights, not output logits.

A motif is a validated sequence of expert activations for a given context class. Each entry contains:

```
motif_id:           unique identifier
context_class:      coarse classification of input (from router's own representations)
expert_sequence:    ordered list of (layer, expert_id, weight) tuples
creation_step:      when this motif was first observed
survival_count:     number of times executed successfully under drift
failure_count:      number of times execution produced quality degradation
last_success_step:  recency signal
commitment_strength: survival_count / (survival_count + failure_count)
scar_associations:  list of scar_ids that constrain this motif
```

The cache is **not** a memo table. Two inputs from the same context class may produce different outputs through the same motif, because the motif defines routing topology, not output content.

### 2.5 Context Class Conditioner

Context classes are derived from the router's own penultimate-layer activations via online clustering. The router is already implicitly computing context distinctions; the conditioner reads them off.

This creates a recursive property: the router's learned representations define the context classes over which its own routing diversity is measured.

Risk: if router representations degrade under pressure (manifold collapse), context classes blur, and the measurement apparatus degrades alongside the thing being measured. Mitigation: a secondary monitor tracks average inter-class distance in router embedding space. Contraction of inter-class distance while per-class metrics appear stable triggers a meta-stability alert.

### 2.6 Motif Library

The motif library is the persistent store of validated routing patterns, indexed by context class.

Operations:
- **Lookup**: Given context class, return top-k motifs ranked by commitment strength.
- **Update**: After successful cheap-path execution, increment survival count.
- **Scar**: After failed cheap-path execution, increment failure count, log as scar against motif-in-context.
- **Prune**: Motifs whose commitment strength falls below threshold are retired.
- **Birth**: New motifs are created when full-routing discovers a stable pattern not yet in the library.

Per-context reachability volume:

```
V_c(t) = exp(H(π(t) | context = c))
```

Scar formation triggers when V_c(t) collapses for any context class with sufficient task-relevant mass. Domain-specific collapse is sufficient; global collapse is not required.

### 2.7 Compound Structural Gate

The gate that permits cheap-path execution. It is **structural**, not confidence-based. Model confidence (softmax entropy) is explicitly excluded as a primary signal because it produces brittle early exits and hidden failures.

The gate requires **all three** of the following signals to be within threshold simultaneously:

| Signal | Measures | Threshold Semantics |
|--------|----------|-------------------|
| **Routing Stability** | Is the router picking the same motif consistently for this context class? Measured as cosine similarity of expert-usage vectors across recent inputs in this class. | High stability = low jitter. If the router is oscillating between motifs, cheap path is unsafe. |
| **Debt Level** | Are we in a high-pressure region where accumulated scars constrain flexibility? Measured as count of active scar constraints affecting this context class. | Low debt = freedom to commit. High debt = pay more compute to avoid catastrophic over-commitment. |
| **Motif Survival** | How often has this specific motif succeeded recently under drift? Measured as commitment_strength with recency weighting. | High survival = evidence of robustness. Low or decaying survival = motif may be stale. |

Single-sentence version: **Cheap decisions are only allowed when the system is phase-locked, low-debt, and the motif has survival evidence.**

### 2.8 Drift Suppression of Cheap Path

When the convergence detector classifies the regime as DRIFT:
- Compound gate is not evaluated.
- All inputs route through full exploratory path.
- Motif library entries accumulate failure evidence if previously cached motifs would have been selected.
- Cost increases. This is correct behaviour — the system is paying for uncertainty.

Re-entry to EQUILIBRIUM requires the drift signal to subside, not merely for loss to improve.

### 2.9 Conditional Freeze (Topology Gating)

**Status: Designed, not yet experimentally validated.** The five-gate protocol below is architecturally specified and implementation-ready, but has not been tested against live routing data. It should be treated as v1.1 planned capability until artifact evidence exists. The detector, commitment cache, and compound gate (Sections 2.2–2.7) are validated.

In SETTLEMENT state, routing pathways that meet the crystallisation criteria may be conditionally frozen. Crystallisation requires five gates, applied in order:

**Gate 1 — Relative Freedom**: The system must have utilised most available degrees of freedom during the qualifying window. Measured as effective rank of the geodesic Gram matrix of routing trajectories under the Fisher information metric, relative to available dimensions after current exclusion constraints.

**Gate 2 — Absolute Freedom**: Effective rank must exceed a minimum dimensionality threshold (d_min). Prevents crystallisation in a heavily constrained system that has explored a small remaining space.

**Gate 3 — Regional Dominance**: The candidate motif must dominate in at least k of K clustered trajectory regions. Prevents context-specific habit from being promoted to structural commitment.

**Gate 4 — Consequence**: Computational dimensionality (effective rank of expert load signature trajectory) must be proportional to routing dimensionality. If the router moved through many dimensions but load signatures cluster into few patterns, the exploration was cosmetic.

**Gate 5 — Stability**: All four preceding gates must be sustained across a two-step commit window with low variance.

**Gate 0 (Prerequisite) — Overlap Independence**: Before any gate evaluation begins, the candidate pathway must pass the counterfactual routing test (Section 8.8). If the pathway is overlap-dependent — meaning its routing changes materially when overlap state is suppressed — it is not eligible for crystallisation regardless of gates 1–5. This gate is numbered zero because it is a precondition, not a threshold.

Frozen pathways are reversible. Epoch review periodically suspends frozen constraints in simulation and measures outcome divergence.

---

## 3. Decision Flow

Step-by-step, every input:

```
1. Input arrives.
2. Context class assigned (router penultimate activations → online clustering).
3. Governance state read (DRIFT / EQUILIBRIUM / SETTLEMENT).
4. If DRIFT:
     → Force full routing. Skip to step 8.
5. If EQUILIBRIUM or SETTLEMENT:
     → Retrieve top motif for this context class from commitment cache.
     → Evaluate compound structural gate:
         a. Routing stability for this class ≥ θ_stability?
         b. Debt level for this class ≤ θ_debt?
         c. Motif survival ≥ θ_survival?
     → If ALL THREE pass:
         → Execute cached motif (cheap path).
         → Early exit. Log cost = motif expert count.
         → Skip to step 9.
     → If ANY fails:
         → Fall through to full routing.
6. Full exploratory routing executes.
     → All experts evaluated per standard MoE top-k.
     → Log cost = full expert invocation count.
7. If full routing succeeds:
     → Check if resulting pattern matches existing motif.
       → If yes: increment survival count.
       → If no: evaluate as new motif candidate.
     → If pattern is novel and stable: create new motif entry.
8. If full routing fails or degrades quality:
     → Log scar against any motif that would have been selected.
     → Increment debt for this context class.
9. Survival update:
     → All motifs in this context class decay slightly (recency weighting).
     → Executed motif gets survival/failure update based on outcome.
10. Telemetry log emitted:
     → (timestamp, context_class, governance_state, path_taken,
        expert_invocations, motif_id_if_used, gate_signals,
        outcome_quality, cost)
```

Every decision logged. No exceptions.

---

## 4. Cost Model

### 4.1 Primary Cost Unit

**Expert invocations per decision.**

One expert invocation = one expert receiving tokens and producing output at one layer. A 4-expert top-2 MoE processing one input across 2 layers = 4 expert invocations at baseline.

### 4.2 Secondary Cost Unit

**Sequence length processed.** Total tokens attended to, including KV cache reads. This captures attention bandwidth cost that expert-invocation count alone misses.

### 4.3 Rules

- No wall-clock cheating. Faster hardware does not count as cost reduction.
- All decisions logged with both cost units.
- Cheap-path cost = motif expert count (strictly less than full routing).
- Full-path cost = standard MoE invocation count.
- Overhead of gate evaluation and context classification is logged separately and must remain small relative to saved expert invocations.

### 4.4 Pareto Validation

Track the (cost, quality) pair for every decision. Over time, bin by context class and plot the Pareto frontier. The system's core claim is validated if and only if:

- Quality holds or improves.
- Cost decreases.
- The Pareto frontier moves inward (toward origin).

If quality degrades, the gate is too permissive. If cost doesn't decrease, the motif library isn't accumulating useful structure.

### 4.5 Operationalising "Experience"

"Experience" is measured in training/deployment steps on a controlled difficulty ladder, not in calendar time. The ladder must have:

- **Repeating structure**: Task classes recur so motifs can form and be reused.
- **Drifting surface form**: Inputs within each class vary so the system cannot succeed by memorising answers. Only structural reuse (motifs) will reduce cost.
- **Injected drift phases**: Periodic shifts in task distribution force the system through DRIFT → EQUILIBRIUM transitions, demonstrating recovery and re-accumulation.

The key figure is the Pareto frontier plotted over ladder steps. Inward movement must be demonstrated in-run, within a single experimental session. No hypothetical deployment horizon.

Diagnostic tools (Takens/FNN dimension estimates, routing entropy trajectories) explain when and why the curve bends. They are supporting evidence, not the headline.

### 4.6 Absolute Economics Baselines

The cost reduction claim must survive an absolute economics check. Three baselines, same tasks, same quality threshold:

| Configuration | What It Tests |
|--------------|---------------|
| **Top-2 vanilla** (no cache, no controller) | Industry default. Does consequence accumulation beat the cheap-and-simple baseline in absolute expert invocations? |
| **Top-4 vanilla** (no cache, no controller) | Controls for branching factor. If top-4+cache doesn't beat top-4 vanilla, the cache is paying for itself with the savings from higher per-decision cost — a wash. |
| **Top-4 + consequence substrate** | The system under test. Must beat top-2 vanilla in absolute cost at equal quality, AND must beat top-4 vanilla to prove the cache contribution. |

If top-4+cache beats top-4 vanilla but loses to top-2 vanilla in absolute cost, we have proven mechanism without economic relevance. The paper is weaker but still publishable as architecture contribution. If top-4+cache beats both, the economic argument lands.

---

## 5. Structural Gate Definition

### 5.1 Routing Stability

```
stability(c, t) = mean(cosine_sim(usage_vec(i), usage_vec(j)))
                  for all pairs (i, j) in recent inputs of class c
```

Where `usage_vec` is the vector of expert activation weights for a single input. High mean cosine similarity = the router is consistently selecting the same expert combination for this class.

Threshold: θ_stability. Tuned per deployment. Starting value: 0.85.

### 5.2 Debt Level

```
debt(c, t) = count(active_scars affecting context class c)
           + weighted_sum(recent_failure_events in class c)
```

Scars are append-only constraints. Recent failures carry higher weight via exponential decay.

Threshold: θ_debt. The gate blocks cheap path when debt exceeds threshold. Starting value: calibrated to 90th percentile of debt distribution during EQUILIBRIUM.

### 5.3 Motif Survival

```
survival(m, t) = Σ(success_i × recency_weight(t - t_i))
               / Σ((success_i + failure_i) × recency_weight(t - t_i))
```

Where recency_weight decays older observations. A motif that succeeded 100 times a month ago but failed 5 times today has lower survival than one that succeeded 20 times today.

Threshold: θ_survival. Starting value: 0.80.

### 5.4 Compound Requirement

Cheap path permitted if and only if:

```
stability(c, t) ≥ θ_stability  AND
debt(c, t) ≤ θ_debt            AND
survival(m, t) ≥ θ_survival
```

No single signal is sufficient. No weighted combination. All three must independently pass.

---

## 6. Governance Interaction Rules

| Governance State | Cheap Path | Motif Updates | Crystallisation | Freeze |
|-----------------|------------|---------------|-----------------|--------|
| DRIFT | Disabled | Failures logged, no new motifs created | Disabled | Disabled |
| EQUILIBRIUM | Enabled (gated) | Active (create, update, scar) | Disabled | Disabled |
| SETTLEMENT | Enabled (gated) | Active | Enabled (5-gate protocol) | Conditional |
| FROZEN | Pathway locked | Monitoring only | N/A | Active, reversible |

State transition triggers:

- **DRIFT → EQUILIBRIUM**: Routing entropy variance below threshold for N consecutive windows AND loss trajectory stabilising (negative second derivative).
- **EQUILIBRIUM → SETTLEMENT**: All EQUILIBRIUM conditions sustained for M windows AND detectable shift in baseline routing statistics (new normal, not just stability).
- **SETTLEMENT → DRIFT**: Routing entropy variance exceeds threshold for K consecutive windows. Crystallisation candidates suspended.
- **SETTLEMENT → FROZEN**: Specific pathway passes all five crystallisation gates. Only that pathway freezes; system may remain in SETTLEMENT globally.
- **FROZEN → SETTLEMENT**: Epoch review finds pathway would not be recreated under current conditions. Pathway unfreezes. Rare.

---

## 7. Experimental Validation

### 7.1 Pressure-Temperature Coupling (4-Expert Baseline)

Architecture: 2-layer, 8-expert MoE with top-2 routing. External P×T controller with geological temperature field.

**Initial result** (live geology vs frozen baseline):
- Turn separation: +13.0% (geology creates structured routing)
- T̄ variance: grew from 0.0 → 0.0009 (active differentiation)
- Loss penalty: +29.3% (too aggressive coupling)
- All 8/8 experts active in both conditions

**After coupling sweep** (η × P grid search):
- Sweet spot found at η=0.006, P=1.1
- Loss: −18.4% vs frozen baseline (Pareto-better)
- Turn separation: +5.8% (structure maintained)
- The geological controller acts as a beneficial regulariser

**Stable operating regime** (η=0.015, P=0.5):
- Seed robustness: 3/3 (100%)
- Δ Loss: −0.4% ± 0.3%
- Routing separation: +6.9% ± 2.0%
- T̄ variance: ≈ 2×10⁻³

### 7.2 Ablation: Why Coupling Matters

| Configuration | Pareto-better | Δ Loss | Δ Separation | Variance |
|--------------|---------------|--------|--------------|----------|
| Full P×T | 3/3 | −0.4% | +6.9% | Low |
| Pressure-only (η_T=0) | 1/3 | −0.4% | −4.1% | 7× higher |
| Temperature-only (P=0) | 1/3 | +0.1% | +11.2% | Moderate |

Pressure alone achieves similar mean loss but with 7× variance and degraded structure. Temperature alone improves separation but fails on loss. Only the coupled system is consistently Pareto-better with 100% seed robustness.

Conclusion: Pressure supplies direction. Temperature supplies memory. Neither is sufficient alone.

### 7.3 Scaling Stress Test (4-Layer, 16-Expert Mixtral)

2× depth, 2× breadth relative to baseline. Top-2 routing.

- Seed robustness: 3/3 (100%)
- Δ Loss: −15.28% ± 1.43% (38× larger improvement than 2L/8E)
- Separation: ~1400 (strong structure formation)
- T̄ variance: 0.000047 (lower than small model — dynamics differ at scale)
- Architecture integrates cleanly; no modifications needed

### 7.4 Topology Preservation Evidence

Same 4-expert architecture, different training objectives:

| Method | Dead Experts | Val Loss | Utilisation |
|--------|-------------|----------|-------------|
| Scratch-trained | 0 | 0.16 | [24.6%, 25.1%, 25.3%, 25.0%] |
| Pruned (8→4) | 0 | 1.89 (fine-tuned) | All active |
| Merged (8→4) | 0 | 1.92 (fine-tuned) | All active |
| Distilled (8→4) | 2 | 4.02 | [0%, 50%, 0%, 50%] |

Aux loss sweep on distilled model: even at aux_loss=1.0 (val loss craters to 35.6), 2 experts remain dead. The model produces garbage rather than activate those experts. Topology is load-bearing; output-matching cannot reconstruct it.

### 7.5 Architectural Boundary: Routing Curvature

P×T coupling requires routing curvature. Validated on top-k routing (k≥2). Switch Transformer (top-1) showed no structural effect — curvature requires at least two active experts per token for the temperature field to have geometry to operate on.

### 7.6 Architectural Boundary: Expert-to-Branch Ratio

In 8-expert top-2 configurations, the commitment cache remained largely unused. Routing stability never rose sufficiently for the compound gate to open. Expert usage stayed flat across context classes, and dominant motifs did not form.

Interpretation: no routing commitment emerged under top-2 with 8 experts. Routing stability remained near zero, so the compound gate correctly refused to open. The commitment cache accumulated entries but none reached survival thresholds.

Hypothesised cause: the action space is too sparse relative to expert count — each input touches only 2 of 8 experts, producing high combinatorial variance that prevents any single routing pattern from dominating. This remains hypothesis until tested.

Hypothesised mitigation: increase top-k (e.g., top-4 with 8 experts). Higher branching factor increases routing curvature and may allow motifs to form. Alternatively, increase task complexity so the router is forced to specialise. This is queued as next experimental priority.

### 7.7 Drift Recovery (Observed)

When geological coupling is active and input distribution shifts:
- Routing entropy spikes (DRIFT detected)
- Controller increases exploration (pressure rises)
- New motifs form over 50-200 steps
- System returns to EQUILIBRIUM with updated routing patterns
- T̄ variance increases during transition, then stabilises at new level

Recovery time correlates with accumulated pressure duration. Longer drift periods require more recovery steps, consistent with hysteresis predictions.

### 7.8 Pending: Controlled Difficulty Ladder

Not yet run. This is the experiment that validates the core claim.

Protocol:
- 3-5 task classes, each with repeating structure but drifting surface form.
- Ladder runs for N thousand steps. Task class recurs every ~200 steps with surface variation.
- Drift injection at defined intervals (shift task distribution, measure recovery).
- All three baselines run on same ladder (top-2 vanilla, top-4 vanilla, top-4+consequence substrate).
- Primary metric: Pareto frontier (cost, quality) plotted over ladder steps.
- Diagnostic: Takens FNN dimension estimate on routing trajectories at intervals.
- Success criterion: Pareto frontier moves inward over steps for consequence substrate while baselines remain flat or worsen.

This experiment is the paper. Everything prior is foundation.

---

## 8. Failure Modes & Safeguards

### 8.1 Survival Inflation

**Risk**: Motif survival count grows monotonically in stable environments, making the gate trivially permissive.

**Safeguard**: Recency weighting on survival. Old successes decay. A motif must continue succeeding to maintain gate access. Additionally, scheduled perturbation probes test motif robustness even during SETTLEMENT.

### 8.2 Premature Entropy Collapse

**Risk**: Router collapses to a small number of motifs too early, before the routing manifold has been adequately explored.

**Safeguard**: Crystallisation Gate 1 (Relative Freedom) and Gate 2 (Absolute Freedom) require evidence of broad exploration before any pathway is promoted. EQUILIBRIUM must be sustained, not merely observed once.

### 8.3 Cheap Path Over-Aggression

**Risk**: Gate thresholds are too loose, and the system takes the cheap path when it shouldn't. Quality degrades silently.

**Safeguard**: Every cheap-path execution is logged with outcome quality. The system tracks cheap-path failure rate per context class. If failure rate exceeds threshold, the class-level gate thresholds tighten automatically. Additionally, the debt signal accumulates failures and blocks cheap path when failures cluster.

### 8.4 Drift Blindness

**Risk**: Context class representations degrade under pressure (manifold collapse), making the system unable to detect that it should be in DRIFT state.

**Safeguard**: Secondary monitor on inter-class distance in router embedding space. If inter-class distance contracts while per-class metrics appear stable, meta-stability alert fires and forces DRIFT classification regardless of per-class signals.

### 8.5 Measurement Apparatus Degradation

**Risk**: The tools used to measure routing health are themselves affected by the conditions they're measuring. If the router's representations degrade, the context classifier, stability estimator, and survival tracker all degrade simultaneously.

**Safeguard**: Constitution Clock (slow-clock meta-process) independently monitors whether the measurement apparatus is producing consistent readings. Divergence between measurement sources triggers escalation. This is a known hard problem with no complete solution; the safeguard reduces but does not eliminate the risk.

### 8.6 Long-Session Degradation

Long sessions blur the boundary between signal and residue. Three failure modes emerge as session length increases:

**False credit assignment**: When a decision succeeds in a long session, the cause is ambiguous — the input may have been easy, the motif may have been correct, or overlap state may have carried the answer. Survival counts inflate for the wrong reason. When a decision fails, the source of failure is equally unclear. Long sessions produce "false stability" (survival credit assigned to momentum) and "false blame" (failure attributed to the wrong component).

**Stealth drift**: In short episodes, distribution shift is obvious because the system must re-derive from scratch. In long sessions, scratchpad continuity and medium-clock state can patch over drift for extended periods. This looks like robustness but is momentum. When it finally breaks, it breaks late, unpredictably, and in ways that don't map cleanly to the point where drift actually began.

**Measurement coupling**: The context class conditioner reads the router's own representations. If those representations slowly compress or rotate over a long session, the context classes drift with them. All downstream metrics can appear stable because the coordinate system itself is moving. This is a specific instance of the measurement apparatus degradation risk (Section 8.5), amplified by session length.

**Safeguard**: Session-length-aware telemetry. Track credit assignment confidence as a function of session length. If the ratio of overlap-explained variance to input-explained variance increases monotonically over session duration, the session enters **credit contamination**.

Credit contamination is a **weight on updates, not a binary flag**. Severity is computed as:

```
contamination(t) = clip(
  EMA(overlap_influence_score, window=W_contam) × monotonicity_penalty,
  0, 1
)

where:
  overlap_influence_score = var_explained_by_overlap / var_total  (from Section 8.8)
  monotonicity_penalty = 1.0 if overlap_influence has been non-decreasing
                         for M consecutive steps, else 0.5
  W_contam = trailing window length (starting value: 50 steps, calibration needed)
```

The monotonicity penalty doubles the severity when overlap influence is steadily growing — the signature of a system increasingly dependent on residue rather than input. Non-monotonic overlap (fluctuating influence) gets half weight because fluctuation indicates the system is at least partially input-responsive.

When a session is contaminated:
- Motif survival updates and new motif births still occur, but at a discounted learning rate: `lr_effective = lr_base × (1 - contamination)`. A session with contamination = 0.8 gets 20% learning rate on survival increments and motif creation.
- Motif retirement is **not** discounted. Retirement is a safety valve. If contaminated sessions keep dead motifs alive, the library accumulates junk. A motif that fails during a contaminated session still gets full failure credit.
- Scar promotion and crystallisation are locked out entirely until a clean counterfactual passes. No discount — binary block. Permanence requires clean evidence.
- This prevents a single long marathon session from rewriting the substrate while still allowing the system to accumulate weak signal from contaminated periods.

**Cooling window**: Periodically, independent of quench triggers, the system forces a low-overlap interval — a short run (K_cool steps) where medium-clock influence is deliberately reduced even if nothing looks wrong.

Operationally: during a cooling window, the medium-clock contribution weight to router logits is scaled down:

```
w_m ← α_cool × w_m    for K_cool steps
α_cool ∈ [0, 0.3]      (starting value: 0.1, calibration needed)
K_cool = 10 steps       (calibration needed)
```

This is a single lever: the medium keeper's additive bias to routing is multiplied by α_cool. The fast clock operates normally. The slow clock remains in its current activation state (Section 8.9). Only the medium-clock continuity signal is attenuated.

This provides a periodic truth sample of how much behaviour is input-driven versus inertia-driven. If performance drops materially during cooling (quality delta exceeds θ_cool), the system has been coasting on momentum. If performance holds, the current motifs are genuinely input-responsive.

The cooling window is not a reset. It is a probe. It answers: "Would we still make this decision without continuity support?" The answer goes into telemetry alongside the quench results.

Design rule: **Continuity is allowed to improve performance. Continuity is not allowed to certify truth.** Overlap can help the system act, but it cannot help the system earn scars or freezes. Anything that becomes a long-term commitment must survive at least one counterfactual run where overlap is suppressed (Section 8.8). Otherwise the system is caching momentum, not decision structure.

Short summary: **Long sessions are where systems learn superstition. Short sessions are where they learn structure.** The cooling window and credit contamination discount ensure that long-session experience is treated as provisional until independently verified.

### 8.7 Phase Contamination (Clock Overlap Leakage)

**Risk**: In a three-clock overlap architecture, fast-clock residue leaks into medium-clock decisions, and medium-clock residue biases slow-clock commitments. The scratchpad (lingering computational state from prior passes) acts as a coupling term between clocks. This can produce self-reinforcing loops that look like stability but are merely persistence. The system never fully "cools," so drift signals don't get a clean reset, and the system keeps believing the old manifold is valid.

**Safeguard: Deliberate quench points.** Not full resets — moments where each clock is forced to re-justify its influence.

- **Fast-clock quench**: Every N turns, force one decision with scratchpad influence set to zero. Compare routing divergence against normal operation. If divergence is large, the scratchpad is driving behaviour more than the input is. That is a brittleness warning.
- **Medium-clock quench**: At topic shifts or constraint changes, temporarily widen exploration and reduce motif reuse for a short window. A controlled "reacquire" phase.
- **Slow-clock quench**: Before any scar formation or freeze, require at least one cycle where the decision succeeds without help from fast or medium overlap. If it cannot stand alone, it does not get promoted.

The slow-clock quench is the most important. In an overlap architecture, "it works" can mean "it works because the previous pass is still doing the work." That is not a commitment worth scarring or freezing.

### 8.8 Overlap Dependence Testing

Two tests make the quench protocol concrete:

**Counterfactual routing test**: Run the same input with overlap state suppressed. If routing changes materially (cosine distance between usage vectors exceeds θ_cf), label the decision as **overlap-dependent**. Overlap-dependent decisions are not eligible for scar promotion or crystallisation. They must demonstrate standalone success before earning permanence. Starting value: θ_cf = 0.15 (calibration needed — this should be set relative to the natural routing jitter observed during EQUILIBRIUM on the target architecture).

**Overlap influence score**: Measure the fraction of router logit variance explained by overlap state versus current input embedding (e.g., via linear regression or ablation). If overlap explains more than θ_overlap of total variance, the system is at risk of self-locking. This metric is logged in telemetry alongside the compound gate signals.

Design rule: **Overlap must be testable, never silently causal.** The system may benefit from persistence, but every persistent influence must be independently verifiable.

### 8.8.1 Quench Schedule

The quench schedule is **event-triggered with a deterministic floor**.

Pure deterministic quenching (every N steps) wastes compute during stable periods and may miss critical moments between scheduled checks. Pure event-triggered quenching depends on the detection apparatus working correctly — but the detection apparatus is exactly what long-session degradation (Section 8.6) compromises. Neither alone is safe.

Hybrid schedule:

- **Deterministic floor**: Fast-clock quench fires at minimum every N_floor steps regardless of system state. This catches cases where the event triggers themselves are compromised by overlap coupling. N_floor is set to the expected length of a full context class cycle on the difficulty ladder. Starting value: N_floor = 100 steps (calibration needed).
- **Event triggers** (any one fires the relevant quench immediately):
  - Gate jitter spike: compound gate signals for a context class oscillate between pass and fail within a short window. Indicates the system is on the boundary of commitment — exactly where false stability is most dangerous.
  - Inter-class distance contraction: context class centroids moving closer together in router embedding space. Indicates measurement apparatus coupling (Section 8.5).
  - Topic/distribution shift detected: medium-clock registers a discontinuity in input statistics. Triggers medium-clock reacquire window (widen exploration, reduce motif reuse).
  - Overlap influence score exceeds θ_overlap: the system is becoming overlap-dependent. Triggers counterfactual test on next K decisions.

**Cost**: Each quench costs one additional forward pass per quenched decision (the counterfactual run). At the deterministic floor rate, this is ~1% overhead. Event-triggered quenches add variable cost concentrated at the moments that matter most. Cooling windows add K_cool steps of reduced-influence operation per cooling interval. Total overhead is bounded and predictable.

The deterministic floor is the safety net. The event triggers are the intelligence. The cooling window is the truth probe. All three are required.

### 8.8.2 Truth-Check Ladder

Three tests produce overlapping signals: counterfactual routing test, quench divergence, and cooling performance delta. They ladder in precedence:

| Test Result | Interpretation | Consequence |
|-------------|---------------|-------------|
| **Counterfactual fails** (overlap-dependent) | Routing structure depends on residue, not input. | Overrides all. Decision ineligible for scar/freeze regardless of other signals. Credit contamination severity increases. |
| **Counterfactual passes, cooling delta bad** | Routing is input-responsive, but quality depends on medium-clock continuity. The motif fires correctly but needs the medium clock to produce good output. | Medium-clock contribution is load-bearing. Safe to update motif survival (the routing pattern is real), but slow-clock scar promotion is blocked (the quality evidence is medium-clock-assisted). |
| **Counterfactual passes, cooling delta good** | Routing is input-responsive and quality holds without medium-clock support. | Full trust. All updates at full learning rate. Scar and crystallisation eligible. |
| **Counterfactual passes, quench divergence high** | Routing is input-responsive at this moment, but historically volatile under quench probes. | Survival updates allowed. Scar promotion requires sustained low quench divergence across multiple probes (not just the latest). |

The ladder ensures that the strongest signal (counterfactual) always wins, while weaker signals (cooling, quench history) modulate how much trust the system places in its own stability assessment.

### 8.9 Slow Clock Activation Policy

The slow clock is **not** always-on. Always-on slow overlap is where dogma happens.

The slow clock refreshes only when the convergence detector classifies the regime as EQUILIBRIUM or SETTLEMENT. During DRIFT, the slow clock's influence on routing is suspended — its state is preserved and continues to update (it is always learning), but its outputs are not applied to routing decisions. This prevents the slow clock from anchoring the system to a manifold that is actively shifting, while ensuring it does not miss observations during DRIFT that may be relevant later.

Rationale: the slow clock's job is to accumulate identity-level constraints (scars, crystallised reflexes, epoch boundaries). These are high-consequence, low-frequency events. If the slow clock is always contributing to routing, it biases every decision toward historical patterns, even when those patterns are no longer valid. Gating slow-clock influence to stable regimes ensures that slow commitments are formed from genuine equilibria, not from residual heat.

The medium clock bridges: it is always-on but its influence weight scales with governance state. In DRIFT, medium-clock influence is reduced (wider exploration). In EQUILIBRIUM, medium-clock influence operates at calibrated strength. In SETTLEMENT, medium-clock influence is at maximum, and the slow clock also contributes.

### 8.10 Recovery Guarantees

- FROZEN pathways are always reversible via epoch review.
- Scars are append-only but subject to scheduled relevance testing.
- No irreversible state change occurs without passing through multi-step commit with evidence thresholds.
- If network connectivity is lost, local constraint state remains intact (sovereignty axiom). Inference capability degrades gracefully; governance state does not.

---

## 9. Non-Goals

This system is explicitly **not**:

| Often Confused With | How We Differ |
|-------------------|---------------|
| **Answer caching / RAG** | We cache routing motifs (decision structure), not outputs. Two inputs through the same motif produce different outputs. |
| **Hardware expert caching** (ExpertFlow, MoE-Infinity, AdapMoE) | Those systems cache expert weights in GPU memory to avoid reloading from CPU/disk. We cache validated routing patterns as first-class reusable objects. |
| **Entropy-only adaptive-K** (Adaptive-K MoE) | Adaptive-K uses routing entropy to dynamically select fewer experts when the router is confident. It is stateless — no accumulation, no motif library, no context conditioning, no structural gate. |
| **Layer-level early exit** (DeeBERT, PABEE, LayerSkip) | Those systems skip depth (exit at intermediate layers). We reduce routing breadth (fewer expert invocations per layer). The gate is structural (compound), not confidence-based (softmax entropy). |
| **Speculative decoding** (MoE-SpeQ) | Speculative decoding uses a draft model to predict expert needs for prefetching. We use accumulated experience to bypass exploratory routing entirely. |
| **Static MoE optimisation** (pruning, merging, distillation) | Those are one-time compression steps. We are a runtime system that gets cheaper over deployment time. |

---

## 10. Open Questions

### 10.1 Scaling: Expert Count × Branching Factor

P×T coupling validated at 8 and 16 experts with top-2 routing. However, the 8-expert top-2 configuration failed to produce routing commitment (Section 7.6), while 16-expert top-2 succeeded. This suggests the relationship between expert count, top-k, and task complexity is non-trivial.

The immediate experimental priority is: 8 experts with top-4. If increased branching restores motif formation, then the architectural boundary is the expert-to-branch ratio, not the expert count. If top-4 still stays flat, then task complexity is the bottleneck, not architecture.

Behaviour at 64+ experts (DeepSeek-scale) is unknown. The low-dimensional attractor hypothesis (d≈2-4 regardless of expert count) needs validation at larger scale. If attractor dimension grows linearly with expert count, the motif library becomes intractable.

### 10.2 Real-World Task Transfer

All current validation is on conversational/sequential data. Transfer to production task domains (customer support, code generation, multi-step reasoning) is untested. The controlled difficulty ladders described in the cost reduction methodology (repeating structure, drifting surface form) need concrete instantiation for target domains.

### 10.3 Long-Horizon Scar Accumulation

Scars are append-only with periodic relevance review. Over months of deployment, scar accumulation may constrain the routing manifold beyond recovery. The epoch review mechanism (suspend and test) addresses this theoretically but has not been validated over long deployment horizons.

### 10.4 Interaction with Optimiser State

The P×T controller operates outside the gradient path. Interaction with Adam/AdamW momentum and adaptive learning rate state is indirect (through loss landscape changes). Whether the controller and optimiser can enter adversarial dynamics (controller stabilises what the optimiser is trying to change, or vice versa) is an open empirical question.

### 10.5 Context Class Granularity

Online clustering granularity affects everything downstream. Too coarse: motifs are applied to inputs they don't fit. Too fine: motif library explodes, survival evidence is sparse. Optimal granularity likely depends on task domain and may need adaptive adjustment.

### 10.6 Multi-Agent Governance

Current architecture assumes a single system. Extension to distributed or federated settings (multiple ChronoMoE instances sharing motif libraries or governance signals) is architecturally undefined.

---

## Appendix A: Terminology

| Term | Definition |
|------|-----------|
| Motif | A validated sequence of expert activations for a context class. |
| Commitment cache | The persistent store of motifs, indexed by context class. |
| Cheap path | Motif-driven execution with fewer expert invocations. |
| Expensive path | Full exploratory MoE routing. |
| Scar | An append-only constraint recording a routing failure. |
| Reflex | A routing pathway promoted to permanent status via crystallisation. |
| Epoch boundary | An irreversible declaration that a developmental phase has ended. |
| Locus | The current position in constraint space from which routing decisions originate. |
| Basin | A stable region in routing space toward which the system tends to settle. |
| Geological field | The temperature component of the P×T controller; evolves slowly, provides memory. |
| Pressure | The directional component of the P×T controller; provides routing bias. |
| Phase contamination | Leakage of residual routing state from one clock into another's decisions. |
| Quench point | A deliberate moment where a clock must re-justify its influence without overlap assistance. |
| Overlap-dependent | A decision whose routing changes materially when overlap state is suppressed. Not eligible for scar or freeze promotion. |
| Overlap influence score | Fraction of router logit variance explained by overlap state vs. current input embedding. |
| Quench schedule | Hybrid event-triggered + deterministic-floor protocol for forcing overlap independence tests. |
| Credit contamination | Session-level severity score (0–1) indicating degraded credit assignment confidence due to high overlap influence. Discounts motif learning rate proportionally; locks out scar promotion entirely. |
| Cooling window | Periodic forced low-overlap interval (K_cool steps) that probes how much behaviour is input-driven vs inertia-driven. Not a reset — a truth sample. |
| Truth-check ladder | Precedence ordering of counterfactual, cooling, and quench signals. Counterfactual overrides all; cooling modulates slow-clock trust; quench history modulates promotion eligibility. |
| Stealth drift | Distribution shift masked by scratchpad continuity and medium-clock momentum in long sessions. |

## Appendix B: Baseline Comparison Protocol

Any claim of cost reduction must be validated against three baselines on identical tasks and identical quality thresholds:

| Baseline | Purpose | What Failure Means |
|----------|---------|-------------------|
| **Top-2 vanilla MoE** (no cache, no controller) | Absolute economics. Industry default. | If we lose here, mechanism works but economics don't. |
| **Top-4 vanilla MoE** (no cache, no controller) | Branching factor control. Same per-decision cost ceiling as our system. | If we only match this, the cache isn't contributing. |
| **Answer cache (RAG/memoisation)** | Tests whether our gains are simply answer recall by another name. | If RAG matches our cost curve, we're not caching structure — we're caching outputs with extra steps. |

The system must beat top-2 vanilla on absolute cost at equal quality (economic relevance), beat top-4 vanilla on cost trajectory (cache contribution), AND beat RAG on robustness to input drift (structural, not answer, reuse). Partial wins are documented honestly.

## Appendix C: Telemetry Schema

Every decision emits:

```
{
  "timestamp": ISO-8601,
  "run_id": string,
  "step": int,
  "context_class": int,
  "governance_state": "DRIFT" | "EQUILIBRIUM" | "SETTLEMENT" | "FROZEN",
  "path_taken": "cheap" | "full",
  "expert_invocations": int,
  "sequence_length": int,
  "motif_id": string | null,
  "gate_signals": {
    "routing_stability": float,
    "debt_level": float,
    "motif_survival": float
  },
  "gate_passed": bool,
  "overlap_metrics": {
    "overlap_influence_score": float,
    "overlap_dependent": bool,
    "last_quench_step": int,
    "quench_divergence": float | null,
    "quench_trigger": "deterministic" | "gate_jitter" | "class_contraction" | "topic_shift" | "influence_threshold" | null
  },
  "session_metrics": {
    "session_step": int,
    "overlap_influence_ema": float,
    "monotonicity_penalty": float,
    "credit_contamination_severity": float,
    "motif_learning_rate_discount": float,
    "scar_promotion_locked": bool,
    "in_cooling_window": bool,
    "cooling_performance_delta": float | null,
    "truth_check_result": "full_trust" | "medium_assisted" | "overlap_dependent" | "volatile" | null
  },
  "outcome_quality": float,
  "loss": float,
  "layer_costs": [{"layer_id": int, "experts_fired": int, "tokens_processed": int}]
}
```

No exceptions. No sampling. Every decision.
