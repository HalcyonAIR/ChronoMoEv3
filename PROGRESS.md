# ChronoMoEv3 Implementation Progress

**Status as of 2026-02-06**

---

## Summary

- **Phase 1:** ✅ COMPLETE (Coherence computation core)
- **Architecture:** ✅ COMPLETE (Three subsystems, 6 questions answered)
- **Phase 2 Plan:** ✅ READY (5-step vertical slice specified)
- **Next:** Begin Step 1 implementation (RouterState + beta application)

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

## 📋 Phase 2: Slow Bias (beta) (READY TO START)

**The locus mechanism: persistent routing geometry without RAG.**

### Implementation Plan

**Complete 5-step vertical slice specified** ([PHASE2_IMPLEMENTATION_PLAN.md](PHASE2_IMPLEMENTATION_PLAN.md))

**Step 1:** RouterState + beta application (one layer)
- Add RouterState with beta_coeff, logit_std_ema
- Compute z_clean, z_biased
- Route with z_biased
- Log disagreement metrics (JS divergence, flip rate)

**Step 2:** Coherence on GPU with buffered state
- CoherenceBuffer: GPU-resident tensors
- Update every step (no CPU sync bottleneck)
- Snapshot to CPU only on eval intervals

**Step 3:** Beta update function
- Simple rule: phi_slow < tau → reduce beta, > tau → increase
- Scale-free: normalize by logit_std_ema
- Clamp to [-k_max, k_max]

**Step 4:** Bridge detector veto
- Compute relevance scalar from overlap-only mass
- Modulate beta strength: beta_eff = r * beta_eff
- Prevent "Krypto from nowhere"

**Step 5:** Lifecycle coordinator (decisions only, dry-run)
- Detect prune candidates
- Log decisions, don't execute yet
- Starvation guardrail (Neff + saturation)

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

---

## 📋 Phase 3: Bimodality Detector (NOT STARTED)

**Detect "this expert is serving two basins."**

### To Implement

- [ ] Two-centroid tracking per expert
- [ ] Assignment and update logic (EMA on centroids)
- [ ] Separation × balance metric
- [ ] Integration with coherence state

### Why This Matters

High coherence doesn't mean healthy. An expert serving two incompatible basins can have decent average coherence but should split, not prune.

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
