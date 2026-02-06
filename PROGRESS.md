# ChronoMoEv3 Implementation Progress

**Status as of 2026-02-06**

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

## 📋 Phase 2: Slow Bias (beta) (NOT STARTED)

**The locus mechanism: persistent routing geometry without RAG.**

### To Implement

- [ ] `SlowBias` class tracking `beta_e` per expert
- [ ] Router integration: `z_biased = z_clean + beta`
- [ ] Beta update rule: `beta(t) = clip(beta(t-1) + eta*(phi_slow - tau), beta_min, beta_max)`
- [ ] Clean vs biased logit separation (for bridge detection later)
- [ ] Cross-prompt persistence validation

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
