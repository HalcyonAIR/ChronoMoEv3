# Coherence Hooks: Reference Implementation

**Minimal patches for extracting MoETrace from Mixtral and Switch patterns**

---

## The Canonical Interface

```python
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor

@dataclass
class MoETrace:
    """
    Coherence-ready trace from one MoE layer forward pass.

    This is the stable interface between router/dispatch and coherence.py,
    independent of which MoE backend (Mixtral-style loop or Switch-style batch).
    """

    # Layer-level outputs
    mixture: Tensor  # [B×S, d_model] - raw weighted mixture, pre-residual

    # Router state (optional, for bridge detection and routing analysis)
    router_logits_clean: Optional[Tensor] = None  # [B×S, E] - pre-softmax if available
    router_probs: Optional[Tensor] = None         # [B×S, E] - full softmax distribution

    # Per-expert state (aligned by expert_id)
    active_expert_ids: List[int] = None           # [num_active_experts]
    expert_mean_outputs: List[Tensor] = None      # List of [d_model] - mean direction per expert
    token_row_indices: List[Tensor] = None        # List of [n_e] - which tokens went to expert e
    gate_weights: List[Tensor] = None             # List of [n_e] - gate weights per token for expert e

    def compute_coherence(self) -> Tensor:
        """
        Compute phi_e = cosine(y_bar_e, y_bar_mix) for all active experts.

        Returns:
            phi: [num_active_experts] - coherence score per expert
        """
        import torch.nn.functional as F

        # Mixture mean direction
        y_bar_mix = self.mixture.mean(dim=0)  # [d_model]

        # Per-expert coherence
        phi = torch.zeros(len(self.active_expert_ids))
        for i, expert_id in enumerate(self.active_expert_ids):
            y_bar_e = self.expert_mean_outputs[i]  # [d_model]
            phi[i] = F.cosine_similarity(
                y_bar_e.unsqueeze(0),
                y_bar_mix.unsqueeze(0),
                dim=-1
            ).item()

        return phi

    @property
    def num_active_experts(self) -> int:
        return len(self.active_expert_ids)

    def get_expert_utilization(self) -> Tensor:
        """Return number of tokens routed to each active expert."""
        return torch.tensor([len(idx) for idx in self.token_row_indices])
```

---

## Deliverable A: Mixtral-Style Hook Patch

**Problem:** Sequential expert loop with `index_add_` consumes outputs immediately.

**Solution:** Compute per-expert mean direction **online** during the loop. No full caching.

### Patch Location

File: `transformers/models/mixtral/modeling_mixtral.py`
Class: `MixtralExperts`
Method: `forward()`

### Minimal Modification

```python
class MixtralExperts(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_trace: bool = False  # ← ADD: flag for coherence monitoring
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, MoETrace]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router
        router_logits, router_probs, selected_experts = self.gate(hidden_states)
        # Note: router_logits is actually softmax (mislabeled), pre-softmax is lost

        # Normalize top-k weights
        router_probs /= router_probs.sum(dim=-1, keepdim=True)

        # Dispatch mask
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)  # [num_experts, top_k, seq]

        # Initialize accumulator
        final_hidden_states = torch.zeros_like(hidden_states)

        # ====== COHERENCE HOOK SETUP ======
        if return_trace:
            active_experts = []
            expert_mean_outputs = []
            token_indices_list = []
            gate_weights_list = []
        # ==================================

        # Expert loop
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue  # No tokens routed to this expert

            # ====== COHERENCE HOOK: Track active expert ======
            if return_trace:
                active_experts.append(expert_idx)
            # ==================================================

            # Gather tokens for this expert
            current_state = hidden_states[top_x]  # [n_e, d_model]

            # Expert forward pass
            current_hidden_states = self.experts[expert_idx](current_state)

            # ====== COHERENCE HOOK: Compute mean direction online ======
            if return_trace:
                expert_mean_out = current_hidden_states.mean(dim=0).detach()  # [d_model]
                expert_mean_outputs.append(expert_mean_out)
                token_indices_list.append(top_x.detach())
                gate_weights_list.append(router_probs[top_x, idx].detach())
            # ============================================================

            # Weight and accumulate
            current_hidden_states = current_hidden_states * router_probs[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Reshape back
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        # ====== COHERENCE HOOK: Package trace ======
        if return_trace:
            trace = MoETrace(
                mixture=final_hidden_states.view(-1, hidden_dim).detach(),
                router_logits_clean=None,  # Lost in Mixtral (overwritten by softmax)
                router_probs=router_logits.detach(),  # Actually softmax probs
                active_expert_ids=active_experts,
                expert_mean_outputs=expert_mean_outputs,
                token_row_indices=token_indices_list,
                gate_weights=gate_weights_list,
            )
            return final_hidden_states, trace
        # ===========================================

        return final_hidden_states
```

### Storage Cost

**Without optimization:** Would need `O(Σ n_e × d)` to cache all expert outputs.

**With online mean:** Only `O(E × d)` for mean directions.

**Reduction:** ~k×(B×S)/E down to E. For Mixtral (8 experts, k=2, B×S=2048):
- Full cache: `2048 × 4096 × 4 bytes ≈ 32 MB` per layer
- Mean only: `8 × 4096 × 4 bytes ≈ 128 KB` per layer

**Speedup:** Computing mean is one extra reduction per expert (~1% overhead). No memory bottleneck.

---

## Deliverable B: Switch Transformer Hook Patch

**Advantage:** Expert outputs exist as clean `[E, capacity, d]` tensor.

**Solution:** Extract from existing tensors using dispatch mask.

### Patch Location

File: `transformers/models/switch_transformers/modeling_switch_transformers.py`
Class: `SwitchTransformersSparseMLP`
Method: `forward()`

### Minimal Modification

```python
class SwitchTransformersSparseMLP(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_trace: bool = False  # ← ADD
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, MoETrace]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Router with capacity dispatch
        router_probs, selected_experts, router_logits = self.router(
            hidden_states, self.expert_capacity
        )
        # router_logits: [B, S, E] - pre-softmax (available!)
        # router_probs: [B, S] - top-1 probability
        # selected_experts: [S, E, capacity] - dispatch mask

        # Flatten for expert computation
        hidden_states = hidden_states.reshape(-1, hidden_dim)  # [B×S, d]

        # Dispatch via einsum
        dispatch_mask = selected_experts.permute(1, 2, 0).float()  # [E, capacity, B×S]
        expert_inputs = torch.einsum(
            "sec,sm->ecm",
            dispatch_mask,
            hidden_states
        )  # [E, capacity, d_model]

        # Compute all experts (batch-style)
        expert_outputs = torch.stack([
            self.experts[f"expert_{i}"](expert_inputs[i])
            for i in range(self.num_experts)
        ], dim=0)  # [E, capacity, d_model]

        # ====== COHERENCE HOOK: Extract from clean tensor ======
        if return_trace:
            active_experts = []
            expert_mean_outputs = []
            token_indices_list = []
            gate_weights_list = []

            for expert_idx in range(self.num_experts):
                # Check which capacity slots have actual tokens (not padding)
                valid_mask = dispatch_mask[expert_idx].any(dim=-1)  # [capacity]
                if not valid_mask.any():
                    continue  # Expert not used

                active_experts.append(expert_idx)

                # Extract valid outputs (filter padding)
                valid_outputs = expert_outputs[expert_idx][valid_mask]  # [n_e, d]
                expert_mean_out = valid_outputs.mean(dim=0).detach()
                expert_mean_outputs.append(expert_mean_out)

                # Token indices: which positions in [B×S] were routed here
                token_mask = dispatch_mask[expert_idx][valid_mask]  # [n_e, B×S]
                token_idx = token_mask.argmax(dim=-1)  # [n_e] - row indices
                token_indices_list.append(token_idx.detach())

                # Gate weights (same for all tokens to this expert in top-1)
                gate_weights = router_probs.view(-1)[token_idx]  # [n_e]
                gate_weights_list.append(gate_weights.detach())
        # ========================================================

        # Combine via einsum
        combine_mask = dispatch_mask * router_probs.view(-1, 1, 1)  # Weight by gate
        combined_output = torch.einsum(
            "sec,ecm->sm",
            combine_mask,
            expert_outputs
        )  # [B×S, d_model]

        # Reshape back
        combined_output = combined_output.reshape(batch_size, sequence_length, hidden_dim)

        # ====== COHERENCE HOOK: Package trace ======
        if return_trace:
            trace = MoETrace(
                mixture=combined_output.view(-1, hidden_dim).detach(),
                router_logits_clean=router_logits.reshape(-1, self.num_experts).detach(),
                router_probs=F.softmax(router_logits, dim=-1).reshape(-1, self.num_experts).detach(),
                active_expert_ids=active_experts,
                expert_mean_outputs=expert_mean_outputs,
                token_row_indices=token_indices_list,
                gate_weights=gate_weights_list,
            )
            return combined_output, trace
        # ===========================================

        return combined_output
```

### Storage Cost

**Expert outputs tensor:** Already exists, `[E, capacity, d_model]` ≈ `8 × 256 × 4096 × 4 bytes ≈ 32 MB`

**Coherence extraction:** Only computes means, `O(E × d)` ≈ 128 KB

**Overhead:** <1%. Just filtering and reductions, no extra forward passes.

---

## Deliverable C: ChronoMoEv3 Canonical Pattern

**Decision:** Use **Switch-style batch computation** as the v3 reference implementation.

**Rationale:**
1. Coherence measurement is **core to the architecture**, not optional telemetry
2. Clean tensors make lifecycle decisions (spawn, prune, split, merge) observable
3. Bimodality detector needs expert output history — easier with batch tensors
4. Slow clock relies on persistence of coherence signal — need reliable measurement

**Mixtral-style is a compatibility target** for integrating with existing models.

### Proposed v3 MoE Layer Signature

```python
class ChronoMoELayer(nn.Module):
    """
    ChronoMoEv3 reference MoE layer.

    Uses Switch-style batch computation for measurement-friendly plumbing.
    Returns MoETrace by default (not optional).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        expert_capacity: Optional[int] = None,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.router = ChronoRouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            ChronoExpert(d_model) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        self.capacity_factor = capacity_factor

    def forward(
        self,
        hidden_states: Tensor,  # [B, S, d]
    ) -> Tuple[Tensor, MoETrace]:
        """
        Returns:
            output: [B, S, d] - mixture output
            trace: MoETrace - coherence-ready state
        """
        B, S, d = hidden_states.shape
        hidden_flat = hidden_states.view(-1, d)  # [B×S, d]

        # Router (keeps pre-softmax logits)
        router_logits = self.router(hidden_flat)  # [B×S, E]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k dispatch
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Capacity-based dispatch mask
        capacity = self._compute_capacity(B * S)
        dispatch_mask = self._create_dispatch_mask(
            top_k_indices, capacity
        )  # [E, capacity, B×S]

        # Dispatch: gather tokens per expert
        expert_inputs = torch.einsum(
            'ecb,bd->ecd',
            dispatch_mask,
            hidden_flat
        )  # [E, capacity, d]

        # Compute experts (parallel)
        expert_outputs = torch.stack([
            self.experts[i](expert_inputs[i])
            for i in range(self.num_experts)
        ], dim=0)  # [E, capacity, d]

        # Extract coherence trace BEFORE combining
        trace = self._extract_trace(
            expert_outputs=expert_outputs,
            dispatch_mask=dispatch_mask,
            router_logits=router_logits,
            router_probs=router_probs,
            top_k_probs=top_k_probs,
            top_k_indices=top_k_indices,
        )

        # Combine: weighted scatter
        combine_weights = self._create_combine_weights(
            dispatch_mask, top_k_probs, top_k_indices
        )  # [E, capacity, B×S]

        mixture = torch.einsum(
            'ecb,ecd->bd',
            combine_weights,
            expert_outputs
        )  # [B×S, d]

        output = mixture.view(B, S, d)
        trace.mixture = mixture.detach()  # Update with final mixture

        return output, trace

    def _extract_trace(
        self,
        expert_outputs: Tensor,
        dispatch_mask: Tensor,
        router_logits: Tensor,
        router_probs: Tensor,
        top_k_probs: Tensor,
        top_k_indices: Tensor,
    ) -> MoETrace:
        """Extract MoETrace from clean tensors."""
        active_experts = []
        expert_mean_outputs = []
        token_indices_list = []
        gate_weights_list = []

        for expert_idx in range(self.num_experts):
            valid_mask = dispatch_mask[expert_idx].any(dim=-1)  # [capacity]
            if not valid_mask.any():
                continue

            active_experts.append(expert_idx)

            # Mean expert output (over valid tokens)
            valid_outputs = expert_outputs[expert_idx][valid_mask]  # [n_e, d]
            expert_mean_outputs.append(valid_outputs.mean(dim=0).detach())

            # Token indices
            token_mask = dispatch_mask[expert_idx][valid_mask]  # [n_e, B×S]
            token_idx = token_mask.nonzero()[:, -1]  # Last dim is token position
            token_indices_list.append(token_idx.detach())

            # Gate weights for these tokens
            gate_weights = top_k_probs[token_idx].max(dim=-1).values  # Top-k max
            gate_weights_list.append(gate_weights.detach())

        return MoETrace(
            mixture=None,  # Will be filled after combine
            router_logits_clean=router_logits.detach(),
            router_probs=router_probs.detach(),
            active_expert_ids=active_experts,
            expert_mean_outputs=expert_mean_outputs,
            token_row_indices=token_indices_list,
            gate_weights=gate_weights_list,
        )

    def _compute_capacity(self, num_tokens: int) -> int:
        """Compute expert capacity with buffer for load imbalance."""
        if self.expert_capacity is not None:
            return self.expert_capacity
        return int(num_tokens / self.num_experts * self.capacity_factor)

    def _create_dispatch_mask(
        self,
        top_k_indices: Tensor,  # [B×S, k]
        capacity: int,
    ) -> Tensor:
        """Create dispatch mask [E, capacity, B×S]."""
        # Implementation details omitted for brevity
        # See Switch Transformer for capacity-based dispatch logic
        pass

    def _create_combine_weights(
        self,
        dispatch_mask: Tensor,
        top_k_probs: Tensor,
        top_k_indices: Tensor,
    ) -> Tensor:
        """Create combine weights [E, capacity, B×S] weighted by gate probs."""
        # Implementation details omitted for brevity
        pass
```

---

## Router Logits: Clean vs Biased

**For bridge detection later**, we need access to both:

1. **Clean router logits** (`z_clean`): Pre-softmax, no bias
2. **Biased router logits** (`z_biased`): After adding slow influence bias `beta_e`

### Hook Points

**In ChronoRouter:**
```python
class ChronoRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.beta = nn.Parameter(torch.zeros(num_experts))  # Slow influence bias

    def forward(
        self,
        hidden_states: Tensor,  # [B×S, d]
        return_clean_logits: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Returns:
            logits_biased: [B×S, E] - with beta added (used for routing)
            logits_clean: [B×S, E] - without beta (for coherence measurement)
        """
        z_clean = self.gate(hidden_states)  # [B×S, E]
        z_biased = z_clean + self.beta.unsqueeze(0)  # Add slow influence

        if return_clean_logits:
            return z_biased, z_clean
        return z_biased
```

**Why this matters:**

The **bridge detector** (from projectdesign.md) needs to measure:
```python
overlap_bias = beta_e - beta_e'  # Routing advantage between expert pairs
```

If we only have `z_biased`, we can't disentangle:
- "Router selected expert e because input favored it" (clean logits)
- "Router selected expert e because beta_e is high" (slow bias)

**For lifecycle decisions**, we want to know:
- Is this expert coherent because it's functionally aligned? (clean)
- Or because the slow clock gave it a persistent routing boost? (biased)

Split/merge should operate on functional redundancy (clean), not on slow bias artifacts.

### Cost

**Zero.** Just don't add `beta` to get `z_clean`. One subtraction if you only have `z_biased`:
```python
z_clean = z_biased - beta.unsqueeze(0)
```

### Where to Store

Add to `MoETrace`:
```python
@dataclass
class MoETrace:
    # ... existing fields ...

    router_logits_clean: Optional[Tensor] = None   # [B×S, E] - pre-softmax, no beta
    router_logits_biased: Optional[Tensor] = None  # [B×S, E] - pre-softmax, with beta
    slow_bias: Optional[Tensor] = None             # [E] - beta vector for this layer
```

This enables downstream analysis:
- **Coherence** uses `mixture` and `expert_mean_outputs`
- **Bridge detection** uses `router_logits_clean` and `slow_bias`
- **Routing analysis** uses `router_logits_biased` (what actually happened)

---

## Summary: Three Deliverables

| Deliverable | Pattern | Key Technique | Overhead | Status |
|-------------|---------|---------------|----------|--------|
| **A: Mixtral patch** | Sequential loop | Online mean computation | ~1% | Reference patch above |
| **B: Switch patch** | Batch einsum | Extract from existing tensors | <1% | Reference patch above |
| **C: ChronoMoEv3 API** | Switch-style (canonical) | MoETrace interface | <1% | Proposed above |

**Decision:** ChronoMoEv3 uses **Switch-style batch computation** as the internal reference. Mixtral-style loops are a compatibility target for external model integration.

**Rationale:** Coherence is not optional telemetry in v3. It's the state variable that drives the entire lifecycle. Measurement-friendly plumbing is architecture, not optimization.

---

## Next Steps

1. **Implement MoETrace dataclass** in `chronomoe_v3/coherence.py`
2. **Prototype ChronoMoELayer** with built-in trace extraction
3. **Test coherence computation** on toy model (8 experts, 2 layers, Shakespeare)
4. **Validate phi_e sensitivity** to expert dysfunction (prune one expert, watch phi drop)
5. **Add slow bias beta** to router and verify clean/biased logit separation

**Critical path:** Phase 1 (coherence) must work before Phase 2 (slow bias). If `phi_e` doesn't track functional participation, the locus mechanism has nothing to latch onto.
