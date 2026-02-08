# Paper 1: Diagnostic Substrate for MoE Lifecycle

## Working Title

**"Detecting Pathological Routing Dynamics in Mixture-of-Experts: A Diagnostic Framework"**

Or:

**"Three Diagnostics for MoE Lifecycle: Coherence Collapse, Latent Path Dependence, and False Stability"**

---

## Positioning

**Venue**: Skeptical ML conference (ICML, NeurIPS, ICLR)

**Framing**: Mechanistic ML result, not philosophy

**Claim**: We can empirically detect three pathologies in MoE routing that existing methods miss:
1. Loss of functional coherence (expert outputs misaligned with mixture)
2. Latent path dependence (history matters only under constraint)
3. False stability (bimodality hiding behind average coherence)

**Not claiming**: Identity, agency, sovereignty, or philosophical stance on decision-making

---

## Structure

### 1. Introduction

**Problem**: MoE systems suffer from routing pathologies that are invisible to standard metrics:
- Load balancing looks good but experts are decoherent
- Average coherence looks healthy but expert serves incompatible modes
- Routing appears stable under high-k but collapses differently under constraint

**Gap**: Existing work focuses on capacity, load balancing, sparse routing optimization. No unified diagnostic framework for *functional* pathologies.

**Contribution**: Three diagnostics with empirical validation:
1. Phase coherence (φ) at three timescales
2. Accumulated routing state under constraint (capacity whiplash)
3. Bimodality detection (separation × balance)

**Result**: Can detect pathologies early, justify lifecycle interventions (prune/split) with evidence, not heuristics.

---

### 2. Background

**MoE Routing Basics**: Top-k selection, capacity factors, load balancing

**Prior Work**:
- Expert specialization (task-level, not functional alignment)
- Load balancing objectives (auxiliary losses)
- Pruning heuristics (magnitude, gradient-based)

**Gap**: No work on *functional coherence* (directional alignment) or *constraint-dependent* routing state.

---

### 3. Phase Coherence: Detecting Loss of Functional Alignment

**Definition**: φ_e = cos(ȳ_e, ȳ_mix \ e)
Expert output direction vs mixture direction (with expert's contribution removed)

**Three timescales**:
- Fast (α=0.9, ~10 steps): Immediate degradation
- Medium (α=0.99, ~100 steps): Context-level stability
- Slow (α=0.999, ~1000 steps): Persistent trends

**Why this matters**:
- Standard metrics: accuracy, perplexity (task-level, slow feedback)
- Load balancing: counts tokens, ignores functional contribution
- Ours: Direct measure of "is this expert helping?"

**Validation**:
- Toy model: 8 experts, inject degradation
- φ_fast drops within 10 steps, φ_slow confirms after 100
- Degraded expert shows φ_slow < 0.3 persistently

**Result**: Can detect decoherence before task metrics degrade.

---

### 4. Latent Path Dependence Under Constraint

**Observation**: Routing state (β coefficients) accumulates from coherence feedback but only affects behavior when degrees of freedom collapse.

**Setup**: Two routers, different histories
- Injected: β seeded differently (mechanism test)
- Earned: β drifts from asymmetric environments (emergence test)

**Protocol**:
1. Phase 1 (top-4): Systems behave similarly
2. Phase 2 (top-1): Systems choose different experts
3. Phase 3 (top-4): Test hysteresis

**Results**:
- Injected: β divergence 0.4 L1 → different choices under top-1
- Earned: β divergence **0.016 L1** (tiny!) → still causes divergence
- Hysteresis: Earned shows 6× stronger persistence (0.171 vs 0.027)

**Interpretation**:
- Not claiming: "β drives all routing"
- Claiming: "β latent until constraint forces choice"
- Systems near decision boundaries; history tips the balance

**Mechanism**: z_biased = z_clean + β. Under top-4, small β washes out. Under top-1, decisive.

**Generalization**: Any accumulated-state routing system under constraint, not MoE-specific.

---

### 5. Bimodality: Detecting False Stability

**Problem**: Expert alternating between opposite basins can have decent average coherence (averaging over modes).

**Solution**: Two-centroid tracking
- Separation: cosine distance (scale-invariant, directional)
- Balance: min(p_A, p_B) / max(p_A, p_B)
- Bimodality score: separation × balance

**Why balance term matters**: Prevents flagging rare excursions. Skewed bimodal gets reduced score.

**Results**:
- Single mode: score = 0.004 (healthy)
- Balanced bimodal: score = 0.688 (split candidate)
- Skewed bimodal: score = 0.076 (balance reduces it)
- False coherence: avg_coherence = 0.0, bimodality = 2.0 (reveals pathology)

**Lifecycle matrix**:
| φ_slow | Bimodality | Decision |
|--------|------------|----------|
| High   | Low        | Keep     |
| High   | High       | Split    |
| Low    | Low        | Prune    |
| Low    | High       | Prune    |

**Distinction**: Stability (low variance) ≠ Health (low bimodality)

---

### 6. Experimental Validation

**Setup**: 8-expert MoE, 2 layers, toy task

**Three experiments**:
1. **Coherence tracking**: Inject degradation, measure detection latency
2. **Capacity whiplash**: Test constraint reveals history
3. **Bimodality**: Compare healthy vs pathological experts

**Metrics**:
- Detection latency (steps until φ_fast < threshold)
- Divergence under constraint (different top-1 choices)
- False positive rate (healthy experts flagged as bimodal)

**Results**: [Tables/Figures]

---

### 7. Discussion

**Contributions**:
1. Phase coherence: Functional alignment metric, three timescales
2. Constraint testing: Reveals latent routing state
3. Bimodality: Distinguishes stability from health

**Limitations**:
- Two centroids (not tri-modal or manifold experts)
- Toy tasks (need scaling validation)
- Lifecycle execution not implemented (diagnostic only)

**Future work**:
- Scale to larger models
- Higher-order multimodality
- Control objective (Phase 4)

**Related to**: Expert pruning, neural architecture search, meta-learning

---

### 8. Conclusion

We presented three diagnostics for MoE routing pathologies:
- φ detects loss of functional coherence
- Constraint testing reveals latent path dependence
- Bimodality exposes false stability

These form a diagnostic substrate for lifecycle decisions. No belief in identity/agency required. Purely mechanistic.

---

## What This Paper Does NOT Claim

- Not claiming: Free energy objective (Phase 4)
- Not claiming: Sovereign router architecture (SC-004)
- Not claiming: Identity, agency, or philosophical stance
- Not claiming: This replaces existing MoE methods

**Claiming**: Three empirically validated diagnostics that existing methods miss.

---

## Rhetorical Strategy

**For skeptical ML reviewers**:
- Lead with coherence (easy to understand)
- Capacity whiplash is "just" path dependence (familiar concept)
- Bimodality is "just" two-centroid clustering (standard technique)
- Frame as "better diagnostics" not "new paradigm"

**Deflect attacks**:
- "Why cosine?" → Scale-invariance, aligns with coherence
- "Why two centroids?" → First non-trivial pathology, higher-order is future work
- "Where's the application?" → Diagnostic layer, execution is Phase 4+
- "Why not just use accuracy?" → Accuracy is task-level, slow. Ours is expert-level, fast.

**Key message**: "I don't know if I agree with where they're going, but the machinery they've built is real."

---

## Title Candidates

1. "Detecting Pathological Routing Dynamics in Mixture-of-Experts"
2. "Three Diagnostics for MoE Lifecycle: Coherence, Constraint, and Bimodality"
3. "Functional Coherence in Mixture-of-Experts: A Diagnostic Framework"
4. "Beyond Load Balancing: Detecting Functional Pathologies in MoE Routing"
5. "Diagnosing MoE Routing: When High Coherence Lies"

---

## Estimated Length

- 8 pages main text (ICLR/NeurIPS format)
- 2 pages appendix (proofs, additional experiments)
- 10 figures/tables

---

## Timeline (If Pursuing)

1. **Week 1-2**: Scale experiments (larger models, real tasks)
2. **Week 3**: Write draft (intro, methods, results)
3. **Week 4**: Revisions, figures, polish
4. **Week 5**: Internal review, submit

---

## Why This Paper First

**Strategic**: Establishes diagnostic machinery without asking reviewers to buy into:
- Sovereignty
- Identity
- Agency
- Philosophical framing

**Tactical**: Makes Phase 4 (control objective) easier to publish later:
- "We showed diagnostics work (Paper 1)"
- "Now we unify them under one objective (Paper 2)"

**Credibility**: Shows the machinery is real, not just philosophy hunting for data.

---

## After This Paper

**Paper 2 options**:
1. Free Energy objective (Phase 4+5): Unify lifecycle under one metric
2. Sovereign Router (SC-004): Identity architecture
3. Scaling study: Apply diagnostics to real models (GPT-scale)

**Each is publishable separately**. Don't collapse too soon.
