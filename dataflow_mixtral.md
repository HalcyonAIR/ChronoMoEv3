# Mixtral MoE Dataflow Analysis

**Modern MoE Implementation (HuggingFace Transformers)**

---

## Dataflow Diagram

```
Input: hidden_states [B×S, d_model]
         │
         v
  ┌──────────────┐
  │ Router       │
  │ Linear(d,E)  │
  │ + Softmax    │
  └──────┬───────┘
         │
         ├─────────────────────────────────┐
         │                                 │
         v                                 v
  router_logits [B×S, E]         router_scores [B×S, k=2]
  (full softmax, NOT logits!)    router_indices [B×S, k=2]
         │                                 │
         │    ┌────────────────────────────┘
         │    │
         v    v
  ┌─────────────────┐
  │ One-hot Mask    │
  │ expert_mask     │
  │ [E, k, B×S]     │
  └────────┬────────┘
           │
           v
  ╔═══════════════════════════════════════╗
  ║ EXPERT LOOP (sequential over E)      ║
  ║                                       ║
  ║ for expert_idx in active_experts:    ║
  ║   1. token_idx ← where(mask[e])      ║
  ║      [variable length per expert]    ║
  ║                                       ║
  ║   2. current_state ← hidden[token_idx]║
  ║      [n_e, d_model]                  ║
  ║                                       ║
  ║   3. expert_out ← FFN_e(current)     ║  ← HOOK A: raw expert output
  ║      [n_e, d_model]                  ║
  ║                                       ║
  ║   4. weighted ← expert_out *         ║
  ║      gate_weights[token_idx]         ║
  ║                                       ║
  ║   5. final_hidden.index_add_(        ║
  ║      0, token_idx, weighted)         ║
  ╚═══════════════════════════════════════╝
           │
           v
    final_hidden_states [B×S, d_model]     ← HOOK B: raw mixture (Σ gᵢ yᵢ)
           │
           v
    [Reshape to (B, S, d_model)]
           │
           v
    [Add residual in parent layer]
           │
           v
    Output
```

---

## Tensor Table

| Variable | Shape | Content | Lifecycle | Available |
|----------|-------|---------|-----------|-----------|
| `hidden_states` (input) | `[B×S, d]` | Flattened tokens | Persistent | ✓ |
| `router_logits` | `[B×S, E]` | **Full softmax** (mislabeled!) | Returned | ✓ |
| `router_scores` | `[B×S, k]` | Renormalized top-k weights | Returned | ✓ |
| `router_indices` | `[B×S, k]` | Top-k expert IDs | Returned | ✓ |
| `expert_mask` | `[E, k, B×S]` | One-hot dispatch mask | Transient | ✓ |
| `token_idx` | `[n_e]` | Token positions for expert e | Per-expert | ✓ |
| `top_k_pos` | `[n_e]` | Slot index (0 or 1 for k=2) | Per-expert | ✓ |
| `current_state` | `[n_e, d]` | Gathered tokens | Per-expert | ✓ |
| `expert_out` (unweighted) | `[n_e, d]` | **Raw expert output** | Per-expert | ✗ Consumed immediately |
| `weighted` | `[n_e, d]` | After gate multiply | Per-expert | ✗ |
| `final_hidden_states` | `[B×S, d]` | **Accumulated mixture** | Persistent | ✗ Not returned separately |

**Legend:** B=batch, S=seq_len, E=num_experts, k=top_k (2 for Mixtral), d=d_model, n_e=tokens routed to expert e

---

## 1. Router Outputs

**What tensors?**
```python
router_logits, router_scores, router_indices = self.gate(hidden_states)
```

**Shapes:**
- `router_logits`: `[B×S, E]` — **MISLABELED: This is full softmax, not raw logits!**
- `router_scores`: `[B×S, k=2]` — Renormalized top-k weights (sum to 1 per token)
- `router_indices`: `[B×S, k=2]` — Top-k expert indices

**Where softmax happens:**
```python
# In MixtralTopKRouter.forward()
router_logits = F.linear(hidden_states, self.weight)  # Raw logits
router_logits = F.softmax(router_logits.float(), dim=-1)  # ← OVERWRITES variable!
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)  # Renormalize
```

**Are logits reused?**
- **Pre-softmax logits: LOST** — overwritten by softmax operation
- **Softmax probs: YES** — returned as `router_logits` (misleading name)
- **Top-k weights: YES** — used for weighting expert outputs

**Critical finding:** True pre-softmax logits are destroyed. If needed for coherence (e.g., z-loss), must modify router to return separately.

---

## 2. Dispatch Mechanics

**How tokens are grouped:**
```python
# Create one-hot dispatch mask
expert_mask = F.one_hot(router_indices, num_classes=num_experts)  # [B×S, k, E]
expert_mask = expert_mask.permute(2, 1, 0)                        # [E, k, B×S]

# For each expert
for expert_idx in active_experts:
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    # top_k_pos: which slot (0 or 1) this expert occupies
    # token_idx: absolute position in [B×S] sequence
```

**Data structure mapping:**
- **Token → Expert**: `router_indices[token_id, :]` gives up to k experts
- **Expert → Token**: `torch.where(expert_mask[expert_id])` gives assigned tokens
- **Row indices**: `token_idx` used for `gather` and `index_add_`

**Reassembly:**
```python
final_hidden = torch.zeros_like(hidden_states)  # [B×S, d]
for expert_idx in experts:
    expert_out = expert[expert_idx](hidden_states[token_idx])  # [n_e, d]
    weighted = expert_out * router_scores[token_idx, top_k_pos, None]
    final_hidden.index_add_(0, token_idx, weighted)  # Scatter-add
```

**Is mixture literally Σ gᵢ yᵢ?**
**YES.** No decoration. Direct accumulation via `index_add_`. Multiple experts contribute to same token sequentially.

---

## 3. Mixture Availability

**Clean point where raw mixture exists:**
```python
# End of MixtralExperts.forward()
return final_hidden_states  # ← Pure Σ gᵢ yᵢ, no residual yet
```

**Before residuals/projections/normalization?**
**YES.** Residual is added in parent `MixtralDecoderLayer`:
```python
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)  # ← MoE returns pure mixture
hidden_states = residual + hidden_states  # ← Residual added here
```

**Can we intercept without changing semantics?**
**YES.** Hook at end of `MixtralExperts.forward()` before return. Zero semantic impact (just observation).

**Problem:** Individual expert outputs are NOT stored—consumed immediately via `index_add_`. Must hook inside loop to capture unweighted `expert_out`.

---

## 4. Minimal Hooks for Coherence

**What coherence.py needs:**
- `y_bar_e`: Mean expert output (per expert)
- `y_bar_mix`: Mean mixture output (per layer)
- `gate_weights`: For optional weighted averaging

**Where to grab:**

```python
# HOOK 1: Router outputs (already available)
router_logits, router_scores, router_indices = self.gate(hidden_states)
# → Store router_scores [B×S, k] and router_indices [B×S, k]

# HOOK 2: Inside expert loop (needs modification)
for expert_idx in active_experts:
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    current_state = hidden_states[token_idx]
    expert_out = self.experts[expert_idx](current_state)  # ← RAW OUTPUT

    # ADD: Store for coherence
    expert_outputs[expert_idx] = {
        'output': expert_out.detach(),         # [n_e, d]
        'token_idx': token_idx,                # [n_e]
        'gate_weights': router_scores[token_idx, top_k_pos]  # [n_e]
    }

    # Continue normal flow
    weighted = expert_out * router_scores[token_idx, top_k_pos, None]
    final_hidden.index_add_(0, token_idx, weighted)

# HOOK 3: Final mixture (needs modification)
mixture = final_hidden.detach()  # ← RAW MIXTURE
return final_hidden, (expert_outputs, mixture)  # Optional return for hooks
```

**What's already there:**
- ✓ `router_scores`, `router_indices`
- ✓ `token_idx` per expert
- ✓ Final mixture (just needs capture)

**What needs adding:**
- ✗ Unweighted `expert_out` (consumed immediately, must cache)
- ✗ Return path for coherence data (or use PyTorch hooks)

---

## 5. Cost Model

**Extra matmuls needed:**
**ZERO.** Coherence uses existing tensors.

**Extra operations:**
- Detach expert outputs: O(1) per expert (pointer copy)
- Mean computation: `expert_out.mean(dim=0)` → O(n_e × d)
- Cosine similarity: O(d) per expert

**Can coherence run without touching expert forward?**
**Partial.** Can analyze router behavior (gate distribution, entropy) without expert access. Cannot compute `phi_e = cosine(y_bar_e, y_bar_mix)` without expert outputs.

**What can be batched?**
- All mean computations: `torch.stack([y_bar_e for e in experts]).mean(dim=1)`
- All cosine similarities: `F.cosine_similarity(expert_means, mixture_mean.unsqueeze(0), dim=-1)`
- Result: Single vectorized operation for all experts

**Total overhead estimate:**
~1-2% forward pass time. Dominated by mean/cosine, not memory operations.

---

## Where coherence.py Would Hook In

**Two hook points in `MixtralExperts.forward()`:**

1. **Inside expert loop** (after line where expert computes `current_hidden_states`):
   - Store unweighted expert output: `expert_outputs[e] = current_hidden_states.detach()`
   - Store metadata: `token_idx`, `gate_weights[token_idx, top_k_pos]`
   - Cost: Minimal (detach is pointer copy, not data copy)

2. **Before return** (after expert loop completes):
   - Capture raw mixture: `mixture = final_hidden_states.detach()`
   - Cost: Negligible

**Implementation strategy:**
- Use PyTorch `register_forward_hook` on `MixtralExperts` module
- Modify forward to return coherence data as auxiliary output
- Compute `phi_e = cosine(y_bar_e, y_bar_mix)` post-forward
- Store in external registry: `coherence_tracker[(layer_id, expert_id, step)] = phi_e`

**No semantic changes.** Forward pass returns identical `final_hidden_states`. Coherence is pure observation. Can be disabled at inference with zero cost (don't register hooks).

---

## Code Reference

**Source:** [HuggingFace Transformers - Mixtral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)

**Key sections:**
- `MixtralTopKRouter.forward()` (lines ~810): Router computation
- `MixtralExperts.forward()` (lines ~730-760): Expert dispatch and mixture
- `MixtralDecoderLayer.forward()` (lines ~920): Residual addition

**Critical line:**
```python
# Line ~810 in MixtralTopKRouter
router_logits = F.softmax(router_logits.float(), dim=-1)  # ← Pre-softmax logits lost
```
