# MoE Dataflow Comparison: Mixtral vs Switch Transformer

**Summary for ChronoMoEv3 coherence.py implementation**

---

## Quick Reference

| Aspect | Mixtral (Modern) | Switch Transformer (Classic) | Winner for Coherence |
|--------|------------------|------------------------------|---------------------|
| **Top-k** | k=2 | k=1 | — |
| **Pre-softmax logits** | ✗ Overwritten | ✓ Returned for z-loss | Switch |
| **Expert outputs tensor** | ✗ Scattered via index_add | ✓ Clean [E, capacity, d] | Switch |
| **Dispatch structure** | One-hot mask, sequential loop | Capacity-based, einsum | — |
| **Mixture formula** | Literal Σ gᵢ yᵢ via index_add | Σ gᵢ yᵢ via einsum | — |
| **Hook complexity** | Needs 2 hooks inside loop | Single hook after stack | Switch |
| **Coherence cost** | ~1-2% (caching overhead) | <1% (read existing tensor) | Switch |
| **Deployment** | Widely used (Mixtral, Deepseek) | Older (T5-based models) | Mixtral |

---

## Critical Differences

### 1. Router Logits Availability

**Mixtral:**
```python
router_logits = F.linear(hidden_states, self.weight)
router_logits = F.softmax(router_logits.float(), dim=-1)  # ← DESTROYS raw logits
```
**Problem:** Cannot compute z-loss or analyze pre-softmax distribution.

**Switch Transformer:**
```python
router_logits = self.classifier(hidden_states)
router_probs = F.softmax(router_logits, dim=-1)
# Both returned separately
```
**Advantage:** Full access to routing decisions at all stages.

---

### 2. Expert Outputs Storage

**Mixtral (scatter-add pattern):**
```python
for expert_idx in experts:
    expert_out = expert[expert_idx](hidden_states[token_idx])
    weighted = expert_out * gate_weights[token_idx]
    final_hidden.index_add_(0, token_idx, weighted)  # Consumed immediately
```
**Problem:** `expert_out` exists only transiently. Must cache inside loop.

**Switch Transformer (batch pattern):**
```python
expert_outputs = torch.stack([
    expert[i](dispatched_input[i]) for i in range(num_experts)
], dim=0)  # [E, capacity, d_model]
# ← Clean tensor persists until einsum combine
```
**Advantage:** All expert outputs in single tensor. Natural interception point.

---

### 3. Token → Expert Mapping

**Mixtral:**
```python
top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
# Variable-length lists per expert
# token_idx: absolute positions in [B×S]
```
- Direct indexing
- Irregular (different number of tokens per expert)

**Switch Transformer:**
```python
dispatch_mask[expert, capacity, batch×seq]  # Sparse boolean tensor
# Each expert has fixed capacity, some slots empty
```
- Capacity-based allocation
- Regular (fixed capacity per expert, padding for unused slots)

---

### 4. Implementation Complexity for Coherence

**Mixtral — Requires 2 hooks:**
```python
# HOOK 1: Inside expert loop
for expert_idx in experts:
    expert_out = expert[expert_idx](current_state)
    expert_cache[expert_idx] = expert_out.detach()  # ← Must cache here
    weighted = expert_out * gate_weights
    final_hidden.index_add_(0, token_idx, weighted)

# HOOK 2: After loop
mixture = final_hidden.detach()
```

**Switch Transformer — Single hook:**
```python
# Compute all experts
expert_outputs = torch.stack([...])  # [E, capacity, d]
# ← Hook here, tensor already exists

# Use for mixture
combined = einsum('sec,ecm->sm', combine_mask, expert_outputs)
```

---

## Coherence Computation Strategy

### What phi_e Needs

```python
# Per expert
y_bar_e = mean(expert_out[token_idx])  # [d_model]

# Per layer
y_bar_mix = mean(final_mixture)  # [d_model]

# Coherence
phi_e = cosine(y_bar_e, y_bar_mix)  # scalar per expert
```

### Cost Analysis

| Operation | Mixtral | Switch Transformer | Notes |
|-----------|---------|-------------------|-------|
| Cache expert outputs | O(E × n̄ × d) memory | 0 (already exists) | n̄ = avg tokens per expert |
| Compute y_bar_e | O(E × n̄ × d) | O(E × capacity × d) | Same asymptotic cost |
| Compute y_bar_mix | O(B×S×d) | O(B×S×d) | Same |
| Cosine similarities | O(E × d) | O(E × d) | Same |
| **Total overhead** | 1-2% | <1% | Mixtral pays caching cost |

---

## Recommendations for ChronoMoEv3

### Coherence.py Hook Design

**Option 1: Mixtral-style (index_add) support**
```python
class CoherenceTracker:
    def __init__(self):
        self.expert_outputs = {}
        self.token_indices = {}
        self.gate_weights = {}

    def hook_expert_forward(self, expert_idx, token_idx, expert_out, gate):
        """Called inside expert loop"""
        self.expert_outputs[expert_idx] = expert_out.detach()
        self.token_indices[expert_idx] = token_idx
        self.gate_weights[expert_idx] = gate

    def hook_mixture(self, final_mixture):
        """Called after loop"""
        self.mixture = final_mixture.detach()

    def compute_coherence(self):
        phi = {}
        y_bar_mix = self.mixture.mean(dim=0)
        for expert_idx, expert_out in self.expert_outputs.items():
            y_bar_e = expert_out.mean(dim=0)
            phi[expert_idx] = F.cosine_similarity(
                y_bar_e.unsqueeze(0),
                y_bar_mix.unsqueeze(0)
            ).item()
        return phi
```

**Option 2: Switch-style (batch) support**
```python
class CoherenceTracker:
    def compute_from_batch(self, expert_outputs, dispatch_mask, mixture):
        """
        expert_outputs: [E, capacity, d_model]
        dispatch_mask: [E, capacity, B×S]
        mixture: [B×S, d_model]
        """
        phi = torch.zeros(expert_outputs.shape[0])
        y_bar_mix = mixture.mean(dim=0)

        for e in range(expert_outputs.shape[0]):
            # Filter padding slots
            valid_mask = dispatch_mask[e].any(dim=-1)  # [capacity]
            if valid_mask.any():
                valid_outputs = expert_outputs[e][valid_mask]  # [n_e, d]
                y_bar_e = valid_outputs.mean(dim=0)
                phi[e] = F.cosine_similarity(
                    y_bar_e.unsqueeze(0),
                    y_bar_mix.unsqueeze(0)
                )
        return phi
```

### Unified Interface

```python
@dataclass
class MoECoherenceState:
    """Captures everything needed for phi_e computation"""
    layer_id: int
    step: int

    # Router outputs
    router_logits: Optional[Tensor]  # [B×S, E] (if available)
    gate_weights: Tensor             # [B×S, k]
    expert_indices: Tensor           # [B×S, k]

    # Expert outputs (one of two formats)
    expert_outputs_batch: Optional[Tensor] = None      # [E, capacity, d] (Switch-style)
    expert_outputs_dict: Optional[Dict] = None         # {e: (out, idx, gate)} (Mixtral-style)

    # Mixture
    mixture: Tensor                  # [B×S, d]

    def compute_coherence(self) -> Tensor:
        """Compute phi_e for all experts. Returns [E]"""
        if self.expert_outputs_batch is not None:
            return self._compute_from_batch()
        else:
            return self._compute_from_dict()
```

---

## Wiring Recommendations

### For Mixtral-based models
1. **Modify expert loop** to cache unweighted outputs
2. **Hook before return** to capture mixture
3. **Use PyTorch hooks** or modify `forward()` signature
4. **Budget 1-2% overhead** for caching

### For Switch-based models
1. **Hook after expert stack** (expert_outputs already exists)
2. **Filter padding** using dispatch_mask
3. **Single hook point**, minimal modification
4. **Budget <1% overhead**

### For ChronoMoEv3's own MoE
**Design for coherence from the start:**
```python
class ChronoMoELayer(nn.Module):
    def forward(self, hidden_states, return_coherence_state=False):
        router_out = self.router(hidden_states)

        # Use Switch-style batch computation for clean tensors
        expert_outputs = self._compute_experts_batch(...)
        mixture = self._combine_experts(expert_outputs, router_out)

        if return_coherence_state:
            coherence_state = MoECoherenceState(
                layer_id=self.layer_idx,
                router_logits=router_out.logits,
                gate_weights=router_out.weights,
                expert_indices=router_out.indices,
                expert_outputs_batch=expert_outputs,
                mixture=mixture
            )
            return mixture, coherence_state

        return mixture
```

**Advantage:** Zero-cost abstraction. Coherence monitoring is pure observation.

---

## Key Insight

**Switch Transformer's einsum pattern is coherence-friendly by design.**

The difference isn't just "clean tensor exists" — it's that the batch-oriented computation naturally separates:
1. **Gather** (dispatch)
2. **Compute** (experts in parallel)
3. **Scatter** (combine)

Whereas Mixtral's loop fuses **compute** and **scatter**, destroying intermediate tensors.

**For ChronoMoEv3:** If efficiency permits, prefer Switch-style batch computation. If memory is tight, accept Mixtral-style caching overhead (~1-2%).

---

## Files

- [`dataflow_mixtral.md`](dataflow_mixtral.md) — Full Mixtral analysis
- [`dataflow_switch_transformer.md`](dataflow_switch_transformer.md) — Full Switch Transformer analysis
- This file — Comparison and recommendations
