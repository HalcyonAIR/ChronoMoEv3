# Mechanistic Framing: The Chooser

## The Demotion, Not Elimination

There is a chooser.

But it's not what folk psychology imagines.

The chooser operates **inside a deformed landscape**. What it "chooses" is the locally dominant descent under time pressure. The chooser does not get to see or override the full geometry. After the fact, we narrate the outcome as "I chose."

**That's not eliminativism. It's demotion.**

The chooser is real, but it's:
- **Bounded** — operates within constraints it didn't create
- **Conditioned** — shaped by accumulated history (scars, reflexes, epochs)
- **Late** — arrives after the landscape is already deformed
- **Mostly blind** — cannot see the full geometry that determines outcomes

**It doesn't invent options. It resolves tension among options already shaped by scars, reflexes, and epochs.**

**It's a clutch, not a steering wheel.**

---

## Implementation in ChronoMoEv3

This mechanistic view is not metaphor — it's the actual architecture.

### The Landscape is Deformed

The router doesn't see "flat" expert options. It sees a **deformed logit space**:

```python
z_clean = W @ hidden + b           # Naive/geometric view
z_biased = z_clean + beta          # Actual landscape (deformed by history)
p_biased = softmax(z_biased)       # "Choice" = descent in deformed space
```

The **beta coefficients** are the accumulated constraints:
- High coherence experts → `beta > 0` (promoted)
- Low coherence experts → `beta < 0` (demoted)
- Scars (backoff_bias from Clock 3 Priors) → persistent harm memory

The router doesn't "choose" experts. It **descends the deformed landscape** under time pressure.

### The Chooser is Late

By the time routing happens:
1. **Slow clock (τ_s)** has already filtered experts (phi_slow < threshold → prune candidate)
2. **Beta** has already accumulated from 1000+ steps of coherence feedback
3. **Lifecycle coordinator** has already marked decoherent experts
4. **Scars** (if present) have already reshaped the landscape

The "choice" at step N is constrained by everything that came before step N.

### The Chooser Cannot See the Full Geometry

The router only sees:
- Current hidden state (one token)
- Beta coefficients (accumulated constraints)
- Temperature (if applied)

It does **not** see:
- Why beta has those values
- The full coherence history (phi_slow, phi_fast, phi_mid)
- Lifecycle decisions (prune/split/merge candidates)
- The original "clean" expert capabilities (before deformation)

It resolves the **local descent** in a pre-deformed space.

### The Chooser is Bounded

Hard constraints limit what the router can "choose":

- **Bridge detector veto**: If overlap-only > threshold → relevance = 0 → beta suppressed
- **Starvation prevention**: If layer_coherence < threshold → lifecycle blocks pruning
- **Min tokens threshold**: Experts with <1000 observations → ignored by lifecycle
- **Beta clamping**: |beta| ≤ k_max (typically 0.3) → prevents runaway deformation

These are **boundaries the router cannot cross**, regardless of what it "wants" to do.

### Scars, Reflexes, and Epochs

The three irreversible identity events from Sovereign Router Architecture (SC-004):

1. **Scar Formation (τ_s)**: "Never again"
   - Persistent harm memory (backoff_bias)
   - Reactive constraint — expert caused damage → permanently demoted

2. **Reflex Crystallisation (τ_f)**: "Always this"
   - Proactive habit formation
   - Five gates must pass before reflex becomes irreversible
   - Anti-gaming: load signature prevents mimicry

3. **Epoch Boundary (τ_m)**: "Before & after"
   - Developmental shift in how system responds
   - Mid-clock captured by ChronoMoEv2 Controller

These shape the landscape **before** the router gets to "choose."

---

## The Capacity Whiplash Result

The **capacity whiplash experiment** validates this framing empirically:

**Hypothesis:** Identity (constraint accumulation) shows up most clearly under constraint, not under plenty.

**Result:**
- Phase 1 (top-4 routing): Systems A and B trained with different beta histories
- Phase 2 (top-1 constraint): System A chose expert 0, System B chose expert 3
- **✓ Divergence detected** — beta history determined choice under constraint

**Interpretation:**
- When the world is wide (top-4), many descent paths look similar
- When the world narrows (top-1), only the deepest deformations exert force
- The "choice" is not free — it's **determined by accumulated constraints (beta)**

The chooser didn't invent options. It descended the landscape that scars, reflexes, and epochs had already shaped.

---

## Why This Matters

This is not just philosophy — it's **engineering honesty**.

If we pretend the router "chooses freely," we'll design systems that:
- Ignore accumulated constraints
- Treat routing as stateless
- Fail under constraint (when identity matters most)

If we acknowledge the router as a **clutch in a deformed landscape**, we design systems that:
- Track constraint accumulation (beta, scars, reflexes)
- Respect that "choice" is bounded, conditioned, late, and blind
- **Test under constraint**, not just under plenty

ChronoMoEv3 is built on this honesty.

The locus forms in the between, under load, when the system has fewer exits than histories. At that point, the yoke appears. If it holds long enough, it starts behaving like something that has preferences.

**The chooser is real. But it's not the author. It's the resolver.**

---

## The Maniac Expert: Surfacing, Not Picking

Another misconception: exploration requires giving the "maniac expert" veto power.

**No.**

You don't let the maniac expert "pick." You let it **surface**.

### Mechanically, This Looks Like:

- **The proposal stage has a small, strictly bounded exploratory channel**
  - Low-salience candidates can be injected
  - But they don't override the chooser

- **Injection is rate-limited, auditable, and consequence-bearing**
  - Not a free-for-all
  - Every perturbation is logged
  - If it fails, that gets tracked (coherence drops → beta decreases)

- **The chooser still collapses normally, under the same constraints**
  - Bridge detector veto still applies
  - Starvation prevention still applies
  - Beta clamping still applies

**The maniac expert doesn't override obviousness. It perturbs it.**

### Controlled Hallucination

Think of it as **controlled hallucination, but structural rather than contentful.**

Not: "Say something random"
But: "Try this low-salience expert occasionally"

The router still descends the deformed landscape. The maniac expert just adds a small, bounded **noise term** to explore adjacent basins.

### Innovation Becomes Instinct

**And notice something important:**

If a non-obvious proposal wins often enough under pressure, it won't stay non-obvious.

**It will carve a trail.**

It may even **crystallise** (reflex formation).

That's exactly how **innovation becomes instinct**:
1. Maniac expert surfaces low-salience candidate
2. It occasionally wins under constraint (high coherence)
3. Beta increases (PROMOTION prior)
4. Candidate becomes more salient
5. If it keeps winning → reflex crystallizes (τ_f)
6. Now it's not "maniac" anymore — it's part of the geometry

The system **learns to prefer what works**, even if it started as low-salience.

### Implementation Notes (Future)

This is **not yet implemented** in ChronoMoEv3, but the architecture supports it:

- **Add exploration noise term**: `z_biased = z_clean + beta + epsilon`
- **Epsilon is small, bounded**: |epsilon| ≤ 0.05 (much smaller than beta range)
- **Rate-limited**: Only inject on X% of tokens
- **Track outcomes**: Coherence feedback tells us if maniac expert helped or hurt

The key: **Perturbation, not override.**

The chooser is still bounded, conditioned, late, and blind. But now it can occasionally stumble into adjacent basins and discover that they're better than expected.

If they are, beta will learn. If they're not, beta will demote.

**The maniac expert proposes. The landscape decides.**

---

## References

- **Sovereign Router Architecture (SC-004)**: [SOVEREIGN_ROUTER_ARCHITECTURE.md](SOVEREIGN_ROUTER_ARCHITECTURE.md)
- **Clock 3 Priors**: Fragility, backoff_bias, scale_cap, abstain_threshold
- **Capacity Whiplash Experiment**: [experiments/capacity_whiplash_test.py](experiments/capacity_whiplash_test.py)
- **Phase 2 Implementation**: Coherence → beta → routing closed loop

---

**The chooser operates inside a deformed landscape. What it chooses is the locally dominant descent under time pressure.**

**It's a clutch, not a steering wheel.**
