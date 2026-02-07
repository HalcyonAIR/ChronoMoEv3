# ChronoMoEv3 Implementation Progress

**Status as of 2026-02-07**

---

## Summary

- **Phase 1:** ✅ COMPLETE (Coherence computation core)
- **Architecture:** ✅ COMPLETE (Three subsystems, 6 questions answered)
- **Sovereign Router:** ✅ SPECIFIED (SC-004: Local/cloud split)
- **Phase 2:** ✅ COMPLETE (All 5 steps complete, validated)
- **Constraint Testing:** ✅ COMPLETE (Injected + earned divergence validated)
- **Phase 3:** ✅ COMPLETE (Bimodality detector closes false coherence loophole)
- **Next:** Phase 4 (Free Energy objective)

---

## ✅ Phase 1: Coherence Computation (COMPLETE)

**The foundation of v3. If phi_e doesn't track functional participation, nothing else matters.**

### Implemented

- ✅ **MoETrace dataclass** ([`chronomoe_v3/coherence.py`](chronomoe_v3/coherence.py))
  - Canonical interface between MoE forward pass and coherence tracking
  - Works with both Mixtral-style (sequential loop) and Switch-style (batch)
  - Clean separation of router state, expert outputs, and mixture

- ✅ **Coherence computation** (`phi_e = cosine(y_bar_e, y_bar_mix)`)
  - Per-expert coherence: Measures directional alignment with mixture
  - Layer-wide coherence: Weighted average (Psi_l)
  - Validated on perfect alignment (phi=1), opposite (phi=-1), orthogonal (phi=0)

- ✅ **CoherenceState tracking** ([`chronomoe_v3/coherence.py`](chronomoe_v3/coherence.py))
  - Per-expert state with three-timescale EMAs
  - `phi_fast`, `phi_mid`, `phi_slow` persistence filters
  - Role vector tracking (what expert typically outputs)
  - Degradation detection via `phi_delta` (fast - slow)

- ✅ **Three-clock system** ([`chronomoe_v3/clocks.py`](chronomoe_v3/clocks.py))
  - ClockConfig: alpha ↔ half_life conversion
  - ThreeClockEMA: Fast (~10 steps), Mid (~100 steps), Slow (~1000 steps)
  - Validated convergence and differential response rates

- ✅ **Configuration** ([`chronomoe_v3/config.py`](chronomoe_v3/config.py))
  - ChronoConfig dataclass with all hyperparameters
  - Clock decay constants, slow bias, free energy weights
  - Edit selection policies, expert bounds

- ✅ **Tests** ([`tests/`](tests/))
  - `test_coherence.py`: 11 tests covering trace, state, batch updates
  - `test_clocks.py`: 12 tests covering clock config, EMA, conversions
  - All passing ✅

- ✅ **Demo** ([`examples/coherence_demo.py`](examples/coherence_demo.py))
  - Simulates 300 steps of MoE forward passes
  - Demonstrates expert degradation detection
  - Shows pruning and lifecycle transitions

### Key Insights

**Coherence is cheap:** <1-2% overhead. No extra matmuls, just means and cosines.

**Fast clock detects problems early:**
- Step 100: Expert 3 degrades, fast drops to 0.08 while slow still 0.095
- Step 110: `phi_delta = -0.013` (degrading flag triggered)
- Step 200: Expert pruned (no longer updated)

**Slow clock resists noise:**
- Healthy experts maintain slow coherence >0.25 after 300 steps
- Degraded experts show persistent negative delta

**Online mean computation works:**
- Mixtral-style: Reduces storage from 32MB/layer to 128KB/layer
- Switch-style: Extract from existing tensors (near-zero cost)

---

## ✅ Architecture Phase (COMPLETE)

**Comprehensive architecture design and decision documentation completed.**

### Dataflow Analysis

- ✅ **Mixtral MoE wiring** ([dataflow_mixtral.md](dataflow_mixtral.md))
  - Router → expert → mixture dataflow mapped
  - Hook points identified for coherence measurement
  - Sequential loop + index_add pattern documented

- ✅ **Switch Transformer wiring** ([dataflow_switch_transformer.md](dataflow_switch_transformer.md))
  - Capacity-based dispatch analyzed
  - Einsum patterns documented
  - Batch-style expert computation advantages identified

- ✅ **Comparison & recommendations** ([dataflow_comparison.md](dataflow_comparison.md))
  - Side-by-side analysis
  - ChronoMoEv3 design recommendations

- ✅ **Reference patches** ([coherence_hooks.md](coherence_hooks.md))
  - MoETrace interface specification
  - Minimal hooks for both Mixtral and Switch patterns
  - Online mean computation (250× storage reduction)

### Architecture Decisions

- ✅ **7 Critical questions answered** ([ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md))
  1. Slow bias location: Pre-softmax additive per expert ✓
  2. Checkpoint state: ~3MB for 64 experts (deterministic recovery) ✓
  3. Clean/biased disagreement: Hybrid escalation at 0.2/0.5/0.7 ✓
  4. Split + beta interaction: Relaxation trial protocol ✓
  5. z_clean computation: Explicit return in v3, hook for external ✓
  6. Falsification criterion: Low phi + high impact would invalidate ✓

- ✅ **Architecture refinements** ([ARCHITECTURE_REFINEMENTS.md](ARCHITECTURE_REFINEMENTS.md))
  - Scale-free beta: k * logit_std (empirically validated) ✓
  - JS divergence vs top-1 flips (use both) ✓
  - Calibrated crisis thresholds (regime-adapted) ✓
  - Temperature interaction clarified ✓
  - Checkpoint ownership contracted ✓
  - Determinism guarantee softened (hysteresis added) ✓

### State Architecture

- ✅ **State separation** ([STATE_SEPARATION.md](STATE_SEPARATION.md))
  - Jeff's yellow sticky note: role_vector → lifecycle, not coherence
  - Clean boundary established

- ✅ **Three subsystems** ([STATE_ARCHITECTURE_V2.md](STATE_ARCHITECTURE_V2.md))
  - **CoherenceState:** "Am I aligned?" (40 bytes, pure measurement)
  - **RoleState:** "What do I do?" (48KB, decision support)
  - **RouterState:** "What biases exist?" (5KB, routing infrastructure)
  - Lifecycle: Reader only, no accumulated state
  - One sentence per field discipline
  - No dumping grounds

### Empirical Validation

- ✅ **Beta saturation analysis** ([experiments/beta_saturation_analysis.py](experiments/beta_saturation_analysis.py))
  - Safe range: |beta| ≤ 1.0 validated
  - At beta=1.0: 12% flip rate (moderate influence)
  - beta / logit_std = 0.35 (gentle prior)

- ✅ **Scale-free beta validation** ([experiments/scale_free_beta_validation.py](experiments/scale_free_beta_validation.py))
  - Flip rate consistency: std=0.0026 ✓
  - JS divergence consistency: std=0.0029 ✓
  - Portability across regimes proven

---

## 🏛️ Sovereign Router Architecture (SC-004)

**Foundational principle: Identity (local) vs Capability (cloud) split.**

### Specification

- ✅ **Sovereignty Axiom** ([SOVEREIGN_ROUTER_ARCHITECTURE.md](SOVEREIGN_ROUTER_ARCHITECTURE.md))
  - "Everything that participates in commitment must be sovereign."
  - "Everything that provides capability can be shared."
  - All three temporal clocks (τ_f, τ_m, τ_s) must be local
  - Experts can be cloud-based and stateless

- ✅ **Local Sovereign Core**
  - Router + 3 temporal keepers (one per clock)
  - Keepers = untrusted processes with limited kernel API access
  - Graceful degradation: "I'm still here but can't reach experts"

- ✅ **Three Irreversible Identity Events**
  - Scar Formation (τ_s): "Never again" - reactive constraint
  - Reflex Crystallisation (τ_f): "Always this" - proactive habit (5 gates)
  - Epoch Boundary (τ_m): "Before & after" - developmental shift

- ✅ **Kernel API**
  - Write budget enforcement (prevents runaway self-modification)
  - Two-step commit (proposals require persisting evidence)
  - Monotonicity (only identity-grade events are irreversible)
  - Tamper-evident audit log (locus drift must be debuggable)

- ✅ **Geometric Foundations**
  - Fisher metric for all geometric computations
  - Geodesic dimensionality (not volume) measures freedom
  - Five-gate crystallisation protocol with anti-gaming safeguards
  - Expert load signatures (content-blind outcome proxy)

### Implementation Phases

**Keeper specifications:** Phase 6-7 (after lifecycle working)
**Crystallisation gates:** Phase 5+ (require lifecycle decisions)
**Kernel API:** Phase 7 (system integration)

---

## 📋 Phase 2: Slow Bias (beta) (IN PROGRESS)

**The locus mechanism: persistent routing geometry without RAG.**

### Implemented (Steps 1-5)

- ✅ **RouterState** ([`chronomoe_v3/router.py`](chronomoe_v3/router.py))
  - Scale-free beta: beta_eff = k * logit_std
  - Disagreement metrics: JS divergence, flip rate, overlap-only
  - Crisis detection thresholds
  - Dual distribution: z_clean, z_biased

- ✅ **ChronoRouter** ([`chronomoe_v3/router.py`](chronomoe_v3/router.py))
  - Routes with biased distribution, logs disagreement with clean
  - Bridge detector with relevance modulation
  - Temperature support

- ✅ **CoherenceBuffer** ([`chronomoe_v3/coherence_gpu.py`](chronomoe_v3/coherence_gpu.py))
  - GPU-resident coherence tracking
  - Three-timescale EMAs (phi_fast, phi_mid, phi_slow)
  - Update every step on GPU, snapshot only on eval intervals
  - Memory: ~2KB per layer (vs ~48KB with role vectors)
  - Multi-layer wrapper for convenience

- ✅ **Beta Update** ([`chronomoe_v3/router.py`](chronomoe_v3/router.py))
  - PROMOTION prior: delta = η(φ_slow - τ)
  - GPU-optimized: update_beta_from_buffer
  - CPU snapshot: update_beta (compatible)
  - Scale-free clamping to [-k_max, k_max]

- ✅ **Bridge Detector** ([`chronomoe_v3/router.py`](chronomoe_v3/router.py))
  - Overlap-only mass: (p_biased - p_clean).clamp(min=0).sum()
  - Relevance modulation: r ∈ [0, 1] based on overlap
  - Prevents "Krypto from nowhere"

- ✅ **Lifecycle Coordinator** ([`chronomoe_v3/lifecycle.py`](chronomoe_v3/lifecycle.py))
  - Dry-run prune detection based on phi_slow
  - Starvation prevention (layer coherence guardrail)
  - Minimum observation threshold
  - Decision logging with full context
  - Neff and saturation metrics for routing collapse detection

- ✅ **Tests** ([`tests/`](tests/))
  - `test_router.py`: 9 tests for RouterState and dual distribution
  - `test_coherence_gpu.py`: 9 tests for GPU coherence buffer
  - `test_beta_update.py`: 7 tests for beta feedback loop
  - `test_bridge_detector.py`: 9 tests for relevance modulation
  - `test_lifecycle.py`: 10 tests for lifecycle coordinator
  - All passing ✅

- ✅ **Demos** ([`examples/`](examples/))
  - `step1_demo.py`: Dual distribution and disagreement metrics
  - `step2_demo.py`: GPU coherence performance (~46K updates/sec)
  - `step3_demo.py`: Beta convergence and closed loop
  - `step4_demo.py`: Bridge detector preventing hallucination
  - `step5_demo.py`: Lifecycle coordinator dry-run detection

### Key Results

**Closed loop verified:**
- Expert with phi_slow=0.80 → beta=+0.064 (promoted)
- Expert with phi_slow=0.30 → beta=-0.057 (demoted)
- Beta converges: early_Δ=0.030 → late_Δ=0.027
- Beta responds to coherence drops

**Bridge detector verified:**
- Overlap-only correctly measures hallucination
- Relevance modulates beta: overlap=0.20 → r=0.48
- Prevents Krypto: vetoes beta when overlap > 0.3

**Lifecycle coordinator verified:**
- Detects prune candidates: phi_slow < 0.3 → candidate
- Starvation prevention: layer_coh < 0.5 → no pruning
- Min tokens filter: 1000 token threshold working
- Decision logging: 6 decisions logged across 3 steps
- Neff metrics: uniform=8.0, concentrated=1.0, half=2.0
- Dry-run confirmed: detects but doesn't execute

**Performance:**
- GPU coherence: ~46K updates/sec on CPU
- Memory: 1.7KB for 64 experts across 4 layers
- No CPU sync bottleneck in training loop

### Implementation Plan

**Complete 5-step vertical slice specified** ([PHASE2_IMPLEMENTATION_PLAN.md](PHASE2_IMPLEMENTATION_PLAN.md))

**Step 1:** ✅ RouterState + beta application (COMPLETE)
- ✅ RouterState with beta_coeff, logit_std_ema
- ✅ Compute z_clean, z_biased
- ✅ Route with z_biased
- ✅ Disagreement metrics (JS divergence, flip rate, overlap-only)
- ✅ Tests: 9 tests in test_router.py
- ✅ Demo: step1_demo.py showing dual distribution

**Step 2:** ✅ Coherence on GPU with buffered state (COMPLETE)
- ✅ CoherenceBuffer: GPU-resident tensors
- ✅ Update every step (no CPU sync bottleneck)
- ✅ Snapshot to CPU only on eval intervals
- ✅ Memory efficient: ~2KB per layer
- ✅ Tests: 9 tests in test_coherence_gpu.py
- ✅ Demo: step2_demo.py showing performance (~46K updates/sec)

**Step 3:** ✅ Beta update function (COMPLETE)
- ✅ PROMOTION prior: delta = eta * (phi_slow - tau)
- ✅ GPU-optimized: update_beta_from_buffer (no CPU sync)
- ✅ Scale-free: clamp to [-k_max, k_max]
- ✅ Closed loop: coherence → beta → routing → coherence
- ✅ Tests: 7 tests in test_beta_update.py
- ✅ Demo: step3_demo.py showing convergence and response

**Step 4:** ✅ Bridge detector veto (COMPLETE)
- ✅ Overlap-only mass: direct hallucination measure
- ✅ Relevance modulation: beta_eff = r * beta_eff
- ✅ Prevents "Krypto from nowhere"
- ✅ Better than JS divergence for veto decisions
- ✅ Tests: 9 tests in test_bridge_detector.py
- ✅ Demo: step4_demo.py showing overlap vs JS comparison

**Step 5:** ✅ Lifecycle coordinator (dry-run) (COMPLETE)
- ✅ Detect prune candidates based on phi_slow
- ✅ Log decisions, don't execute yet
- ✅ Starvation guardrail (layer coherence, Neff, saturation)
- ✅ Min tokens threshold filtering
- ✅ Tests: 10 tests in test_lifecycle.py
- ✅ Demo: step5_demo.py showing detection and prevention

### Pre-Implementation Questions Answered

**All critical questions resolved** ([PHASE2_REFINEMENTS.md](PHASE2_REFINEMENTS.md))

1. ✅ Beta sign: PROMOTION prior (high coherence → beta increases)
2. ✅ JS divergence: Per-token with 10% sampling
3. ✅ File organization: coherence.py (API) + coherence/buffer.py (GPU)
4. ✅ Relevance metric: Overlap-only mass (not just JS)
5. ✅ Starvation signal: Neff + saturation proxy
6. ✅ Stability criterion: 4 explicit assertions

**Timeline:** 5 days (1 step per day)

**Testing harness:** experiments/phase2_vertical_slice.py

### Why This Matters

The slow clock doesn't just measure — it acts. Experts that persist through the slow window earn a routing advantage (`beta > 0`). Experts that fail to persist lose influence (`beta → negative`). This is the trimming mechanism, in math.

### Constraint Testing ✅ COMPLETE

**Hypothesis:** "Identity (constraint accumulation) shows up most clearly under constraint, not under plenty."

When the world is wide, many systems look similar. When options narrow to almost nothing, only the deepest accumulated constraints (scars, crystallized reflexes, beta) still exert force.

#### Capacity Whiplash: Injected Divergence ([experiments/capacity_whiplash_test.py](experiments/capacity_whiplash_test.py))

**Setup:**
- System A: β initialized to favor experts 0,1
- System B: β initialized to favor experts 2,3
- Phase 1: Both train with top-4 (identical environment)
- Phase 2: Both forced to top-1 (constraint)
- Phase 3: Both return to top-4 (hysteresis test)

**Results:**
- ✅ Divergence detected: A→expert 1, B→expert 3 under top-1
- β divergence: L1=0.4 (strong, seeded)
- Hysteresis: L1=0.027 (minimal trail formation)
- **Validates**: Persistent state variable affects routing under constraint

#### Capacity Whiplash: Earned Divergence ([experiments/capacity_whiplash_earned.py](experiments/capacity_whiplash_earned.py))

**Setup:**
- Both systems: β starts at 0.0 (NO seeding)
- System A: Low-frequency input bias
- System B: High-frequency input bias
- Phase 1: Asymmetric environments, β drifts naturally
- Phase 2: Same neutral environment, forced to top-1
- Phase 3: Same neutral environment, return to top-4

**Results:**
- ✅ Divergence detected: A→expert 7, B→expert 5 under top-1
- β divergence: L1=0.016 (tiny, earned through interaction)
- Hysteresis: L1=0.171 (strong trail formation)
- **Validates**: Trails emerge from interaction, not just seeding

**Key Finding:**
Even minimal earned divergence (0.016) causes different choices under constraint. The landscape is shaped by experience, not programmer intervention.

**Comparison:** [experiments/CAPACITY_WHIPLASH_COMPARISON.md](experiments/CAPACITY_WHIPLASH_COMPARISON.md)

---

## ✅ Phase 3: Bimodality Detector (COMPLETE)

**Detect "this expert is serving two basins."**

Closes the loophole: **High average coherence can hide pathology.**

### Implemented

- ✅ **BimodalityState** ([`chronomoe_v3/bimodality.py`](chronomoe_v3/bimodality.py))
  - Two-centroid tracking (running means with EMA updates)
  - Smart initialization: waits for distant point before initializing second centroid
  - Separation metric: cosine distance between centroids
  - Balance metric: min(p_A, p_B) / max(p_A, p_B) — balanced usage → 1.0
  - Bimodality score: separation × balance

- ✅ **BimodalityDetector** ([`chronomoe_v3/bimodality.py`](chronomoe_v3/bimodality.py))
  - Layer-wide bimodality tracking for all experts
  - Split candidate detection (score > threshold, min observations met)
  - Statistics reporting and snapshot functionality

- ✅ **Tests** ([`tests/test_bimodality.py`](tests/test_bimodality.py))
  - 10 comprehensive tests
  - Single mode: low score (0.004)
  - Two modes (balanced): high score (0.688)
  - Skewed modes: reduced score (0.076) — balance term working
  - False coherence scenario: avg_coherence=0.0, bimodality=2.0 (reveals pathology)
  - All passing ✅

- ✅ **Demo** ([`examples/bimodality_demo.py`](examples/bimodality_demo.py))
  - False coherence demonstration
  - Healthy vs pathological comparison
  - Split candidate detection
  - Lifecycle integration matrix

### Key Results

**False Coherence Detected:**
- Expert alternating between opposite basins: avg coherence = 0.0 (looks degraded)
- But bimodality score = 2.0 → reveals it's serving two incompatible modes
- Coherence alone would miss this pathology

**Clear Separation:**
- Healthy (unimodal): score = 0.004
- Pathological (bimodal): score = 1.011
- Skewed bimodal: score = 0.211 (balance term reduces it appropriately)

**Lifecycle Integration:**
| Coherence | Bimodality | Decision |
|-----------|------------|----------|
| High | Low | ✓ Keep (healthy) |
| High | High | ✗ SPLIT (false coherence) |
| Low | Low | ✗ PRUNE (decoherent) |
| Low | High | ✗ PRUNE (unstable bimodal) |

### Why This Matters

**Closes the loophole**: "High coherence can still be pathological"

An expert serving two phase-incompatible basins can maintain decent average coherence (averaging over the two modes). Without bimodality detection, lifecycle would keep this expert because coherence looks healthy.

**Framework distinguishes stability from health**:
- Stability: consistent output (low variance)
- Health: consistent output *in same direction* (low bimodality)

Control theorists will respect this distinction.

---

## 📋 Phase 4: Free Energy Objective (NOT STARTED)

**Single objective replacing the rule bag.**

### To Implement

- [ ] `F_l = (1 - Psi_l) + lambda*N_l + rho*R_l + kappa*I_l`
- [ ] Misfit term (1 - Psi_l)
- [ ] Complexity tax (N_l)
- [ ] Redundancy detection (R_l)
- [ ] Instability penalty (I_l from bimodality)

---

## 📋 Phase 5: Edit Proposal and Selection (NOT STARTED)

**Lifecycle as slow-clock physics.**

### To Implement

- [ ] Spawn: Add expert when layer starving
- [ ] Prune: Remove expert when irreversibly decoherent
- [ ] Split: Divide bimodal expert
- [ ] Merge: Combine redundant experts
- [ ] Candidate evaluation under F_l
- [ ] "Do nothing" threshold

---

## 📋 Phase 6: Expert Registry (NOT STARTED)

**Fixed-width router with masking.**

### To Implement

- [ ] ExpertRegistry managing active/cooling/archived states
- [ ] Fixed-width router (max_experts per layer)
- [ ] Active mask for spawn/prune
- [ ] Optimizer state management on structural changes

---

## 📋 Phase 7: ChronoSystem Integration (NOT STARTED)

**Wrap v2 Controller into unified system.**

### To Implement

- [ ] ChronoSystem class
- [ ] Single `step()` call for all three clocks
- [ ] Integration with ChronoMoEv2 Controller (mid clock)
- [ ] Decision logging (JSONL)

---

## 📋 Phase 8: Benchmarks (NOT STARTED)

**Validate that this works.**

### To Implement

- [ ] Toy model: 8 experts, 2 layers, Shakespeare
- [ ] With vs without lifecycle
- [ ] F_l vs ad-hoc triggers
- [ ] Targeting correlation (like nanoMoE/Halcyon validation)

---

## Documentation Completed

- ✅ [dataflow_mixtral.md](dataflow_mixtral.md) — Mixtral wiring facts
- ✅ [dataflow_switch_transformer.md](dataflow_switch_transformer.md) — Switch wiring facts
- ✅ [dataflow_comparison.md](dataflow_comparison.md) — Comparison and recommendations
- ✅ [coherence_hooks.md](coherence_hooks.md) — Reference patches and MoETrace interface
- ✅ [projectdesign.md](projectdesign.md) — Full architectural specification
- ✅ [firststeps.md](firststeps.md) — Getting started guide

---

## Critical Path

**Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5**

Phase 1 (coherence) is the foundation. Everything else depends on `phi_e` tracking functional participation.

Phase 2 (slow bias) makes the locus real: persistent routing geometry that survives across prompts.

Phase 3 (bimodality) prevents misidentifying "serving two basins" as "low coherence."

Phase 4 (free energy) unifies spawn/prune/split/merge under one objective.

Phase 5 (lifecycle) implements the objective as slow-clock physics.

---

## Next Session

**Implement Phase 2: Slow Bias**

1. Create `SlowBias` class
2. Add `beta` parameter to router
3. Implement update rule
4. Validate persistence across "prompts" (batch boundaries)
5. Test that high-phi experts gain routing advantage

Then move to Phase 3 (bimodality detector).

---

**Status:** Phase 1 complete. v3 has a working heart. 🎯
