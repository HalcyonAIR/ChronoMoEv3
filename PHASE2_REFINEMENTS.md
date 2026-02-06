# Phase 2 Implementation Refinements

**Halcyon's pre-implementation questions answered.**

---

## Question 1: Beta Sign and Meaning ⚠️ CRITICAL

### The Issue

Current formula:
```python
delta = eta * (phi_slow - tau)
beta_coeff += delta
```

This means:
- `phi_slow > tau` → delta positive → beta increases
- `phi_slow < tau` → delta negative → beta decreases

Combined with: `z_biased = z_clean + beta_coeff * logit_std`

**Higher beta → higher logits → more routing mass**

### The Question

**Is beta a "promotion" prior or "penalty" prior?**

### The Answer

**Beta is a PROMOTION prior.**

- High coherence (`phi_slow > tau`) → beta increases → expert gets routing advantage
- Low coherence (`phi_slow < tau`) → beta decreases → expert loses routing advantage
- Negative beta → active suppression (expert worse than random)

This matches the "locus" concept: experts that persist through the slow window **earn** routing influence.

### Naming Convention

```python
@dataclass
class RouterState:
    # PROMOTION prior (not penalty)
    beta_coeff: Tensor  # Positive = routing advantage, negative = suppression

    # Alternative names considered:
    # beta_prior ✓ (explicit about promotion)
    # beta_trust ✓ (emphasizes earned influence)
    # beta_penalty ✗ (wrong semantics)
```

**Decision:** Keep `beta_coeff` but document explicitly as promotion prior.

### Sign Verification

```python
# Expert with high coherence
phi_slow = 0.8, tau = 0.5
delta = 0.01 * (0.8 - 0.5) = +0.003
beta_coeff += 0.003  # Increases (promotion) ✓

# Expert with low coherence
phi_slow = 0.2, tau = 0.5
delta = 0.01 * (0.2 - 0.5) = -0.003
beta_coeff -= 0.003  # Decreases (suppression) ✓

# Expert far below threshold
phi_slow = 0.1, tau = 0.5
After many steps: beta_coeff → negative
z_biased = z_clean + (negative) → active suppression ✓
```

**The sign is correct.** No change needed.

---

## Question 2: JS Divergence Computation Detail

### The Issue

Plan says "compute JS divergence" but doesn't specify:
- Per-token then average?
- Batch aggregate distribution?

### Options

**Option A: Per-token JS**
```python
js_per_token = compute_js_divergence_per_token(p_clean, p_biased)  # [B×S]
js = js_per_token.mean().item()
```
- Pro: Accurate, catches token-level spikes
- Con: Expensive (B×S divergence computations)

**Option B: Batch aggregate JS**
```python
p_clean_agg = p_clean.mean(dim=0)  # [E]
p_biased_agg = p_biased.mean(dim=0)  # [E]
js = compute_js_divergence(p_clean_agg, p_biased_agg)
```
- Pro: Cheap (one divergence computation)
- Con: Hides "Krypto" spikes (single bad token masked by batch)

**Option C: Per-token on routed experts only (Halcyon's suggestion)**
```python
# Only compute JS over experts that got routed
top_k_mask = create_mask_from_top_k(top_k_indices)  # [B×S, E]
p_clean_routed = p_clean * top_k_mask
p_biased_routed = p_biased * top_k_mask
js = compute_js_per_token(p_clean_routed, p_biased_routed).mean()
```
- Pro: More accurate than aggregate, cheaper than full per-token
- Con: Still O(B×S×k)

### Decision

**Use Option A (per-token JS) with sampling for efficiency.**

```python
def compute_disagreement_js(p_clean: Tensor, p_biased: Tensor, sample_rate: float = 0.1) -> float:
    """
    Compute JS divergence per token, then average.

    Args:
        p_clean: [B×S, E]
        p_biased: [B×S, E]
        sample_rate: Fraction of tokens to sample (default 0.1 = 10%)

    Returns:
        js: Scalar JS divergence
    """
    B_S, E = p_clean.shape

    # Sample tokens for efficiency
    num_samples = max(1, int(B_S * sample_rate))
    sample_indices = torch.randperm(B_S)[:num_samples]

    p_clean_sampled = p_clean[sample_indices]  # [num_samples, E]
    p_biased_sampled = p_biased[sample_indices]

    # JS divergence per token
    p_clean_sampled = p_clean_sampled + 1e-9
    p_biased_sampled = p_biased_sampled + 1e-9

    m = 0.5 * (p_clean_sampled + p_biased_sampled)

    kl_pm = (p_clean_sampled * (p_clean_sampled / m).log()).sum(dim=-1)
    kl_qm = (p_biased_sampled * (p_biased_sampled / m).log()).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    return js.mean().item()
```

**Rationale:** Per-token is more accurate. 10% sampling is cheap. Can adjust sample_rate if needed.

---

## Question 3: File Organization Clarity

### The Issue

README mentions `coherence.py`, plan mentions `coherence_gpu.py`. Risk: two "coherence engines."

### Decision

**Clear separation with single entry point:**

```
chronomoe_v3/
├── coherence.py          # Public API (MoETrace, CoherenceState, compute_coherence)
├── coherence/
│   ├── __init__.py
│   ├── buffer.py         # CoherenceBuffer (GPU implementation)
│   └── trace.py          # MoETrace construction helpers
├── router.py             # RouterState, ChronoRouter
└── lifecycle.py          # LifecycleCoordinator
```

**Public imports:**
```python
from chronomoe_v3 import MoETrace, CoherenceState, compute_coherence
from chronomoe_v3.coherence.buffer import CoherenceBuffer  # Explicit opt-in for GPU
```

**No ambiguity.** One coherence API. Buffer is explicitly a performance optimization.

---

## Question 4: Relevance from Overlap-Only Mass vs JS

### The Issue

Plan uses JS divergence to compute relevance. But the real risk is **overlap-only mass**: routing mass going to experts that clean router wouldn't select.

### JS vs Overlap-Only

**JS divergence:** Measures distribution shift (KL-based)
- High even when beta is doing something useful but sharp
- Doesn't distinguish "clean expert boosted" vs "hallucinated expert"

**Overlap-only mass:** Measures hallucination risk
```python
overlap_only = (p_biased - p_clean).clamp(min=0).sum(dim=-1).mean()
```
- Mass given to experts that wouldn't have been selected
- Direct measure of "Krypto from nowhere"

### Decision

**Use overlap-only mass for relevance. Keep JS as dashboard metric.**

```python
def compute_relevance(
    p_clean: Tensor,
    p_biased: Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute routing relevance from overlap-only mass.

    Overlap-only = mass given to experts that wouldn't have been selected under clean routing.
    High overlap-only → beta is hallucinating experts → low relevance.

    Args:
        p_clean: [B×S, E]
        p_biased: [B×S, E]
        threshold: Overlap-only threshold for full relevance

    Returns:
        r: Relevance scalar in [0, 1]
    """
    # Positive part: mass added by beta
    overlap_only = (p_biased - p_clean).clamp(min=0).sum(dim=-1).mean().item()

    if overlap_only < threshold:
        return 1.0  # Full relevance (beta is aligned)
    elif overlap_only < threshold * 3:
        # Linear decay
        return 1.0 - (overlap_only - threshold) / (threshold * 2)
    else:
        return 0.0  # No relevance (beta is hallucinating)

def compute_disagreement_metrics(p_clean: Tensor, p_biased: Tensor) -> dict:
    """
    Compute both JS and overlap-only for logging.

    Returns:
        {
            'js': float,  # Dashboard metric
            'overlap_only': float,  # Used for relevance
            'flip_rate': float,  # Interpretability metric
        }
    """
    return {
        'js': compute_js_divergence(p_clean, p_biased),
        'overlap_only': (p_biased - p_clean).clamp(min=0).sum(dim=-1).mean().item(),
        'flip_rate': (p_clean.argmax(-1) != p_biased.argmax(-1)).float().mean().item(),
    }
```

**Rationale:** Overlap-only is closer to the risk we care about (hallucination). JS still useful for dashboard.

---

## Question 5: Starvation Signal Without v2 Telemetry

### The Issue

Plan says "prune if low observability AND not starving." But v2 lens telemetry isn't wired in Phase 2.

**Without starvation signal, lifecycle can't responsibly propose prune.**

### Simple Proxy Metrics

**Neff (Effective number of experts):**
```python
def compute_neff(p_biased: Tensor) -> float:
    """
    Effective number of experts being used.

    Neff = exp(entropy(p_biased))

    Low Neff → few experts doing all the work → starvation
    """
    p_avg = p_biased.mean(dim=0)  # [E]
    entropy = -(p_avg * (p_avg + 1e-9).log()).sum()
    neff = torch.exp(entropy).item()
    return neff
```

**Saturation (Max routing mass):**
```python
def compute_saturation(p_biased: Tensor) -> float:
    """
    Maximum routing mass to any single expert.

    High saturation → one expert dominating → starvation of others
    """
    p_avg = p_biased.mean(dim=0)  # [E]
    return p_avg.max().item()
```

### Decision

**Add simple starvation proxy to RouterState.**

```python
@dataclass
class RouterState:
    # ... existing fields ...

    # Starvation proxy metrics
    neff: float = 0.0  # Effective number of experts
    saturation: float = 0.0  # Max routing mass

    def update_starvation_metrics(self, p_biased: Tensor):
        """Compute Neff and saturation."""
        p_avg = p_biased.mean(dim=0)  # [E]

        # Neff
        entropy = -(p_avg * (p_avg + 1e-9).log()).sum()
        self.neff = torch.exp(entropy).item()

        # Saturation
        self.saturation = p_avg.max().item()

    def is_starving(self, num_experts: int) -> bool:
        """
        Is this layer starving for capacity?

        Starving if:
        - Neff < 2 (only 1-2 experts doing all work)
        - Saturation > 0.7 (one expert has >70% of mass)
        """
        return self.neff < 2.0 or self.saturation > 0.7
```

**Lifecycle uses this:**
```python
def evaluate(self, coherence_snapshot, router_state, step):
    # Check starvation
    if router_state.is_starving(num_experts=8):
        print(f"[WARNING] Layer starving (Neff={router_state.neff:.2f})")
        return []  # Don't propose any prunes

    # Proceed with prune detection
    # ...
```

**Rationale:** Simple, cheap, good enough for Phase 2. v2 telemetry can replace later.

---

## Question 6: Stability Criterion for Tests

### The Issue

Harness expected output says "Loop stable ✓" but doesn't define stability.

### Measurable Stability Criteria

```python
def assert_stability(router_state: RouterState, step: int, warmup_steps: int = 500):
    """
    Assert that the beta update loop is stable.

    Stability means:
    1. Beta doesn't blow up (bounded magnitude)
    2. Beta doesn't diverge (bounded variance)
    3. Disagreement stays under control (no crisis)
    4. Relevance doesn't collapse (beta still useful)
    """
    # Only check after warmup
    if step < warmup_steps:
        return

    beta_mean = router_state.beta_coeff.abs().mean().item()
    beta_std = router_state.beta_coeff.std().item()
    js = router_state.disagreement_js
    overlap_only = router_state.overlap_only

    # Compute relevance
    r = compute_relevance_from_overlap(overlap_only, threshold=0.1)

    # Assertions
    assert beta_mean < 0.5, f"Beta blowing up: mean={beta_mean:.3f}"
    assert beta_std < 0.3, f"Beta diverging: std={beta_std:.3f}"
    assert js < 0.5, f"Disagreement crisis: JS={js:.3f}"
    assert r > 0.3, f"Relevance collapsed: r={r:.3f}"

    print(f"✓ Stability check passed at step {step}")
```

### Integration

```python
# In test harness

for step in range(1000):
    # ... forward, backward, beta update ...

    # Stability check every 100 steps
    if step % 100 == 0 and step > 0:
        try:
            assert_stability(router_state, step, warmup_steps=500)
        except AssertionError as e:
            print(f"✗ Stability check FAILED: {e}")
            break

print("✓ Loop stable for 1000 steps")
```

**Rationale:** Explicit, measurable, catches regressions immediately.

---

## Minor: Valid Count Sync

### The Issue

Current plan syncs per-expert for valid_count. With GPU buffer, this becomes inefficient.

### Decision

**Use masked tensor operations, avoid sync.**

```python
# Instead of:
for expert_id in active_experts:
    count = len(token_indices[expert_id])
    total_tokens_seen[expert_id] += count  # Sync per expert

# Do:
active_mask = torch.zeros(num_experts, dtype=torch.bool, device='cuda')
active_mask[active_expert_ids] = True

num_tokens = torch.tensor([len(idx) for idx in token_row_indices], device='cuda')
num_tokens_full = torch.zeros(num_experts, device='cuda')
num_tokens_full[active_expert_ids] = num_tokens

total_tokens_seen += num_tokens_full  # Single GPU operation
```

**No per-expert syncs. Stays on GPU.**

---

## Summary of Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Beta sign** | Promotion prior (correct as-is) | High coherence → beta increases → routing advantage |
| **JS computation** | Per-token with 10% sampling | Accurate, catches spikes, efficient |
| **File organization** | `coherence.py` (API) + `coherence/buffer.py` (GPU) | Single entry point, no ambiguity |
| **Relevance metric** | Overlap-only mass (not JS) | Measures hallucination risk directly |
| **Starvation signal** | Neff + saturation proxy | Simple, cheap, good enough for Phase 2 |
| **Stability criterion** | 4 assertions (beta bounded, JS < 0.5, r > 0.3) | Explicit, measurable, catches regressions |

---

## Updated Step 3: Beta Update (Corrected)

```python
def update_beta(
    router_state: RouterState,
    coherence_snapshot: Dict[int, CoherenceState],
    eta: float = 0.01,
    tau: float = 0.5,
):
    """
    Update beta coefficients (PROMOTION prior).

    High coherence → beta increases → routing advantage
    Low coherence → beta decreases → suppression

    Args:
        router_state: RouterState to update
        coherence_snapshot: {expert_id: CoherenceState}
        eta: Learning rate (default 0.01)
        tau: Coherence threshold / target (default 0.5)
    """
    for expert_id, coh_state in coherence_snapshot.items():
        # Skip if not observed
        if not coh_state.is_being_observed:
            continue

        # PROMOTION prior: high phi → increase beta
        delta = eta * (coh_state.phi_slow - tau)

        # Update (scale-free coefficient)
        router_state.beta_coeff[expert_id] += delta

    # Clamp to prevent saturation
    router_state.beta_coeff.clamp_(-router_state.k_max, router_state.k_max)
```

**Sign verified:** High `phi_slow` increases beta (promotion). Correct. ✓

---

## Updated Step 4: Relevance from Overlap-Only

```python
def compute_relevance(
    p_clean: Tensor,
    p_biased: Tensor,
    threshold: float = 0.1,
) -> float:
    """
    Compute relevance from overlap-only mass (hallucination risk).

    Returns:
        r: Relevance scalar in [0, 1]
    """
    overlap_only = (p_biased - p_clean).clamp(min=0).sum(dim=-1).mean().item()

    if overlap_only < threshold:
        return 1.0
    elif overlap_only < threshold * 3:
        return 1.0 - (overlap_only - threshold) / (threshold * 2)
    else:
        return 0.0

# In router forward():
r = compute_relevance(p_clean, p_biased, threshold=0.1)
beta_eff_modulated = r * beta_eff
z_biased = z_clean + beta_eff_modulated.unsqueeze(0)
```

**Overlap-only is the discriminant. JS is dashboard metric.**

---

## Updated Step 5: Prune with Starvation Guard

```python
def evaluate(self, coherence_snapshot, router_state, step):
    """Evaluate prune candidates with starvation guard."""

    # Check starvation first
    if router_state.is_starving(num_experts=8):
        print(f"[WARNING] Layer starving, skipping prune evaluation")
        return []

    # Proceed with prune detection
    candidates = []

    for expert_id, coh_state in coherence_snapshot.items():
        # Low observability
        if coh_state.total_tokens_seen < 100 and step > 1000:
            candidates.append(PruneDecision(
                expert_id=coh_state.expert_id,
                reason="low_observability",
                phi_slow=coh_state.phi_slow,
            ))

        # Low coherence
        if coh_state.phi_slow < 0.3 and coh_state.total_tokens_seen > 1000:
            candidates.append(PruneDecision(
                expert_id=coh_state.expert_id,
                reason="low_coherence",
                phi_slow=coh_state.phi_slow,
            ))

    return candidates
```

**Safe prune detection with starvation guardrail.**

---

**Status:** All pre-implementation questions answered. Ready for Step 1.
