# Switch Transformer MoE Dataflow Analysis

**Classic Top-1 MoE Implementation (HuggingFace Transformers)**

---

## Dataflow Diagram

```
Input: hidden_states [B, S, d_model]
         │
         v
  ┌──────────────┐
  │ LayerNorm    │
  └──────┬───────┘
         │
         v
  ┌──────────────┐
  │ Router       │
  │ Linear(d,E)  │
  │ + Softmax    │
  └──────┬───────┘
         │
         ├───────────────────────────────┐
         │                               │
         v                               v
  router_logits [B, S, E]        router_probs [B, S, 1]
  (kept for aux loss)            selected_experts [B, S, 1, E] (one-hot)
         │                               │
         │                               │
         v                               v
  ╔═══════════════════════════════════════╗
  ║ CAPACITY-BASED DISPATCH               ║
  ║                                       ║
  ║ 1. expert_capacity = tokens_per_batch ║
  ║    / num_experts * capacity_factor    ║
  ║                                       ║
  ║ 2. position_in_expert = cumsum       ║
  ║    (selected_experts, dim=1)          ║
  ║                                       ║
  ║ 3. tokens exceeding capacity →        ║
  ║    dropped or sent to overflow        ║
  ║                                       ║
  ║ 4. dispatch_mask [E, capacity, B×S]  ║
  ║    (sparse boolean mask)              ║
  ╚═══════════════════════════════════════╝
         │
         v
  ╔═══════════════════════════════════════╗
  ║ EXPERT COMPUTATION (parallel)         ║
  ║                                       ║
  ║ dispatched_input = einsum(           ║
  ║   'sec,sm->ecm',                     ║
  ║   dispatch_mask, hidden_states)       ║
  ║   → [E, capacity, d_model]           ║
  ║                                       ║
  ║ expert_outputs = FFN_e(dispatched)   ║  ← HOOK A: raw outputs [E, capacity, d]
  ║   → [E, capacity, d_model]           ║
  ║                                       ║
  ║ combined = einsum(                    ║
  ║   'sec,ecm->sm',                     ║
  ║   combine_mask, expert_outputs)      ║
  ║   → [B×S, d_model]                   ║
  ╚═══════════════════════════════════════╝
         │
         v
    combined_output [B, S, d_model]        ← HOOK B: raw mixture
         │
         v
    combined * router_probs                ← Weight by gate probability
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
| `hidden_states` | `[B, S, d]` | Input tokens | Persistent | ✓ |
| `router_logits` | `[B, S, E]` | Pre-softmax logits | Returned | ✓ (for z-loss) |
| `router_probs` | `[B, S, 1]` | Top-1 gate probability | Returned | ✓ |
| `selected_experts` | `[B, S, 1, E]` | One-hot expert selection | Returned | ✓ |
| `expert_capacity` | scalar | Max tokens per expert | Config-derived | ✓ |
| `dispatch_mask` | `[E, capacity, B×S]` | Sparse dispatch tensor | Transient | ✓ |
| `combine_mask` | `[E, capacity, B×S]` | Weighted combine tensor | Transient | ✓ |
| `dispatched_input` | `[E, capacity, d]` | Tokens gathered per expert | Transient | ✓ |
| `expert_outputs` | `[E, capacity, d]` | **Raw expert outputs** | Transient | ✓ Clean tensor! |
| `combined_output` | `[B×S, d]` | Weighted mixture | Persistent | ✓ |

**Legend:** B=batch, S=seq_len, E=num_experts, d=d_model, capacity=expert capacity

---

## 1. Router Outputs

**What tensors?**
```python
router_probs, selected_experts, router_logits = self.router(hidden_states)
```

**Shapes:**
- `router_logits`: `[B, S, E]` — **Pre-softmax logits** (kept for auxiliary load balancing loss)
- `router_probs`: `[B, S, 1]` — Top-1 softmax probability (gate weight)
- `selected_experts`: `[B, S, 1, E]` — One-hot encoding of selected expert

**Where softmax happens:**
```python
# In SwitchTransformersTop1Router.forward()
router_logits = self.classifier(hidden_states)  # [B, S, E]
router_probs = F.softmax(router_logits, dim=-1)  # [B, S, E]
router_probs, selected_experts = torch.max(router_probs, dim=-1)  # Top-1
router_probs = router_probs.unsqueeze(-1)  # [B, S, 1]
selected_experts = F.one_hot(selected_experts, num_classes=num_experts)  # [B, S, E]
```

**Are logits reused?**
- **Pre-softmax logits: YES** — explicitly returned for z-loss computation
- **Softmax probs: YES** — used for weighting final mixture
- **Selected experts: YES** — used for dispatch mask construction

**Key difference from Mixtral:** Switch Transformer preserves raw logits. Mixtral overwrites them.

---

## 2. Dispatch Mechanics

**How tokens are grouped:**

**Capacity-based allocation:**
```python
expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
# capacity_factor typically 1.0-2.0 to handle imbalance
```

**Position tracking:**
```python
# Assign each token a position within its chosen expert
position_in_expert = torch.cumsum(selected_experts, dim=-2) - 1  # [B, S, E]
# Tokens exceeding capacity are masked out
capacity_mask = position_in_expert < expert_capacity
```

**Dispatch mask construction:**
```python
# Create sparse mask [expert, capacity, batch×seq]
dispatch_mask = F.one_hot(position_in_expert, num_classes=expert_capacity)
dispatch_mask = dispatch_mask * selected_experts.unsqueeze(-2)  # Zero out non-selected
dispatch_mask = dispatch_mask * capacity_mask.unsqueeze(-2)      # Zero out overflow
# Reshape to [E, capacity, B×S]
```

**Data structure mapping:**
- **Sparse boolean tensor** `dispatch_mask[e, c, t]`:
  - `True` if token `t` is routed to expert `e` at capacity slot `c`
  - Most entries are `False` (sparse)
- No explicit `token_idx` list per expert (unlike Mixtral)

**Reassembly via einsum:**
```python
# Dispatch: gather tokens for each expert
dispatched_input = torch.einsum(
    'sec,sm->ecm',
    dispatch_mask.float(),  # [seq, expert, capacity]
    hidden_states           # [seq, model_dim]
)  # → [expert, capacity, model_dim]

# Expert computation (can be parallelized)
expert_outputs = torch.stack([
    expert[i](dispatched_input[i]) for i in range(num_experts)
], dim=0)  # [E, capacity, d_model]

# Combine: scatter expert outputs back to tokens
combined_output = torch.einsum(
    'sec,ecm->sm',
    combine_mask,      # [seq, expert, capacity] (weighted by router_probs)
    expert_outputs     # [expert, capacity, model_dim]
)  # → [seq, model_dim]
```

**Is mixture literally Σ gᵢ yᵢ?**
**YES**, but structured differently:
- Mixture is `einsum('sec,ecm->sm', combine_mask, expert_outputs)`
- `combine_mask = dispatch_mask * router_probs` embeds gate weights
- Result: Weighted sum, same as Mixtral but via einsum instead of index_add

---

## 3. Mixture Availability

**Clean point where raw mixture exists:**

**YES — Two clean interception points:**

1. **Expert outputs tensor:**
   ```python
   expert_outputs = torch.stack([expert[i](dispatched_input[i]) for i in range(E)])
   # Shape: [E, capacity, d_model]
   # Contains ALL expert outputs in single clean tensor
   ```

2. **Combined mixture (before residual):**
   ```python
   combined_output = torch.einsum('sec,ecm->sm', combine_mask, expert_outputs)
   # Shape: [B×S, d_model]
   # Pure weighted mixture, no residual yet
   ```

**Before residuals/projections/normalization?**
**YES.** Residual added in parent `SwitchTransformersBlock`:
```python
hidden_states = layer_norm(hidden_states)
hidden_states = moe_layer(hidden_states)  # ← Returns pure mixture
hidden_states = residual + hidden_states  # ← Residual added here
```

**Can we intercept without changing semantics?**
**YES.** Both `expert_outputs` and `combined_output` are clean intermediate tensors. Hook placement:
- After `expert_outputs` computation (before einsum combine)
- After `combined_output` computation (before return)

**Key advantage over Mixtral:** Expert outputs exist as a clean `[E, capacity, d]` tensor, not scattered via `index_add_`.

---

## 4. Minimal Hooks for Coherence

**What coherence.py needs:**
- `y_bar_e`: Mean expert output (per expert)
- `y_bar_mix`: Mean mixture output (per layer)
- Which tokens actually reached each expert (capacity accounting)

**Where to grab:**

```python
# HOOK 1: Router outputs (already available)
router_probs, selected_experts, router_logits = self.router(hidden_states)
# → Store router_probs [B, S, 1], selected_experts [B, S, 1, E], router_logits [B, S, E]

# HOOK 2: After expert computation (clean tensor exists!)
expert_outputs = torch.stack([
    self.experts[i](dispatched_input[i]) for i in range(self.num_experts)
], dim=0)  # [E, capacity, d_model]

# COHERENCE COMPUTATION (can happen inline):
for expert_idx in range(num_experts):
    # Get valid (non-padding) outputs for this expert
    valid_mask = dispatch_mask[expert_idx].any(dim=-1)  # [capacity]
    valid_outputs = expert_outputs[expert_idx][valid_mask]  # [n_e, d]
    y_bar_e = valid_outputs.mean(dim=0)  # [d_model]
    # Store for coherence tracking

# HOOK 3: After mixture combination
combined_output = torch.einsum('sec,ecm->sm', combine_mask, expert_outputs)
y_bar_mix = combined_output.mean(dim=0)  # [d_model]

# Compute phi_e = cosine(y_bar_e, y_bar_mix) for all experts
```

**What's already there:**
- ✓ `router_logits` (pre-softmax)
- ✓ `router_probs` (gate weights)
- ✓ `expert_outputs` as clean tensor `[E, capacity, d]`
- ✓ `dispatch_mask` (which tokens go where)
- ✓ `combined_output` (final mixture)

**What needs adding:**
- ✗ Nothing! All tensors needed for coherence already exist cleanly.
- Only need to extract means and compute cosine similarities.

---

## 5. Cost Model

**Extra matmuls needed:**
**ZERO.** All needed tensors already exist.

**Extra operations:**
- Mask valid outputs: `expert_outputs[e][valid_mask]` → O(capacity)
- Mean per expert: `valid_outputs.mean(dim=0)` → O(n_e × d)
- Mean mixture: `combined_output.mean(dim=0)` → O(B×S×d)
- Cosine similarities: O(E × d)

**Can coherence run without touching expert forward?**
**YES, partially:**
- Router analysis (entropy, balance, load): Fully independent
- Coherence `phi_e`: Requires `expert_outputs`, but that tensor exists—just read it

**What can be batched?**
- Extract all valid expert outputs: `[expert_outputs[e][valid_mask[e]] for e in range(E)]`
- Stack means: `torch.stack([y_bar_e for e in range(E)])` → `[E, d]`
- Vectorized cosine: `F.cosine_similarity(expert_means, y_bar_mix.unsqueeze(0), dim=-1)` → `[E]`

**Total overhead estimate:**
<1% forward pass time. No extra matmuls. All operations are cheap (means, cosines).

**Key advantage:** Switch Transformer's batch-style expert computation naturally produces clean tensors. Coherence monitoring is nearly free.

---

## Where coherence.py Would Hook In

**Single hook point in `SwitchTransformersSparseMLP.forward()`:**

```python
# After expert outputs are computed
expert_outputs = torch.stack([
    self.experts[f"expert_{i}"](dispatched_input[i])
    for i in range(self.num_experts)
], dim=0)  # [E, capacity, d_model]

# COHERENCE COMPUTATION (inline, negligible cost)
if coherence_monitor is not None:
    coherence_state = compute_coherence(
        expert_outputs=expert_outputs,      # [E, capacity, d]
        dispatch_mask=dispatch_mask,        # [E, capacity, B×S]
        combine_mask=combine_mask,          # [E, capacity, B×S]
        router_logits=router_logits,        # [B, S, E]
        layer_id=self.layer_idx
    )
    coherence_monitor.log(layer_id, coherence_state)

# Continue normal flow
combined_output = torch.einsum('sec,ecm->sm', combine_mask, expert_outputs)
return combined_output
```

**One paragraph on integration:**

Switch Transformer's einsum-based dispatch naturally produces a clean `expert_outputs` tensor `[E, capacity, d_model]` containing all expert computations in a single structure, making coherence monitoring trivial: after the stack/loop that computes expert outputs, simply extract valid (non-padding) outputs per expert using the `dispatch_mask`, compute means `y_bar_e`, and calculate cosine similarities with the mixture mean `y_bar_mix`. Unlike Mixtral's `index_add_` pattern that immediately consumes expert outputs, Switch Transformer's batch-oriented approach means coherence computation can happen inline with near-zero cost—no extra hooks inside expert loops, no caching scattered outputs, just read the existing tensor. The only consideration is filtering padding slots in the capacity dimension where no token was routed (check `dispatch_mask.any(dim=-1)`). Total overhead <1% forward time, fully vectorizable, and can be toggled via a single boolean flag.

---

## Code Reference

**Source:** [HuggingFace Transformers - Switch Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/switch_transformers/modeling_switch_transformers.py)

**Key sections:**
- `SwitchTransformersTop1Router.forward()`: Router with capacity-based dispatch
- `SwitchTransformersSparseMLP.forward()`: Einsum dispatch, expert computation, einsum combine
- `SwitchTransformersBlock.forward()`: Residual addition

**Critical advantage:**
```python
# Expert outputs exist as clean tensor
expert_outputs = torch.stack([expert[i](dispatched_input[i]) for i in range(E)])
# → [E, capacity, d_model]
# Perfect for coherence monitoring!
```
