# ChronoMoEv3 Integration Review
**Date**: 2026-02-09
**Context**: Pre-Phase 6 validation before swiss-ai/MoE integration

## Executive Summary

ChronoMoEv3's lifecycle system is **architecturally sound but has one critical gap**: optimizer state management. The constitutional layer (gates, proposals, audit trail) is robust. Structural edit operations are properly abstracted. But we need per-expert output hooks and optimizer state handling before real integration.

**Verdict**: Phase 6 is **mandatory**, not optional. Proceed with implementation.

---

## Finding 1: Signal Extraction Requirements ⚠️

### What We Need (MoETrace interface)
- ✅ `mixture`: [B×S, d_model] - Layer output
- ✅ `router_logits`: [B×S, E] - Pre-softmax logits
- ✅ `router_probs`: [B×S, E] - Softmax distribution
- ✅ `active_expert_ids`: List of experts that processed tokens
- ⚠️ `expert_mean_outputs`: [E, d_model] - **Per-expert outputs**
- ✅ `token_row_indices`: Which tokens → which expert
- ✅ `gate_weights`: Router weights per token

### What swiss-ai/MoE Provides
- ✅ Router logits/probs: Available from `MoE.forward()` return dict
- ✅ Selected experts: Available as `selected_experts` tensor
- ⚠️ Per-expert outputs: **Hidden inside expert loop** - need hook

### Gap Analysis
**Expert outputs are computed but not exposed:**
```python
for i in range(num_experts):
    batch_idx = torch.where(selected_experts == i)
    output, _ = expert(inputs_squashed[batch_idx])  # ← Need to capture this
    # ... accumulate into mixture ...
```

**Solution**: Wrap MoE layer with hooks to capture `output` on each expert call.

**Impact**: Critical. Without per-expert outputs, we cannot compute:
- Coherence (phi_e)
- Redundancy term (role vectors)
- Bimodality tracking (centroid updates)
- **Entire lifecycle system is blocked**

---

## Finding 2: Dependency Chain Validation ✅

Traced signal requirements through full stack:

```
Per-Expert Outputs
  ↓
MoETrace.expert_mean_outputs
  ↓
compute_coherence() → phi_e
  ↓
FreeEnergyState.misfit (layer coherence term)
  ↓
EditEvidence.f_l_before, f_l_predicted
  ↓
All structural edit proposals
```

**Also**:
```
Per-Expert Outputs
  ↓
BimodalityDetector.update(y_expert_mean)
  ↓
bimodality_scores
  ↓
FreeEnergyState.instability
  ↓
EditEvidence (split trigger)
```

**Also**:
```
Per-Expert Outputs → role_vectors (mean directions)
  ↓
FreeEnergyState.redundancy
  ↓
EditEvidence (merge trigger)
```

**Conclusion**: Per-expert outputs are the **single critical signal**. Everything else flows from them or from standard router outputs (which are available).

---

## Finding 3: Structural Edit Abstraction ✅

**Excellent separation of concerns:**

| Operation | Input | Output | PyTorch Awareness |
|-----------|-------|--------|-------------------|
| SPAWN | `parent_params: Dict[str, Tensor]` | `child_params: Dict[str, Tensor]` | ❌ No |
| PRUNE | `proposal_id: str` | `expert_id: int` | ❌ No |
| SPLIT | `source_params: Dict[str, Tensor]` | `(child_a_params, child_b_params)` | ❌ No |
| MERGE | `(source_a_params, source_b_params)` | `merged_params: Dict[str, Tensor]` | ❌ No |

**EditExecutor doesn't know about**:
- nn.ModuleList
- Optimizer state
- Checkpoint formats
- Router architecture

**Caller's responsibility**:
1. Extract params from model
2. Call EditExecutor
3. Create/remove/modify nn.Module
4. Update router mask/indices
5. **Manage optimizer state** ← Missing piece

This is **correct design**. Integration layer handles PyTorch-specific surgery.

---

## Finding 4: Router Architecture Compatibility ✅

**swiss-ai/MoE router structure:**
```python
self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
# ...
all_probs = F.softmax(router_logits, dim=1)
weights, selected_experts = torch.topk(all_probs, self.top_k)
```

**Issues**:
1. **Output dimension = num_experts** (fixed at init)
2. Adding/removing experts changes dimension
3. Requires router reinitialization or retraining

**Phase 6 Solution**: Fixed-width router with masking
- Router dimension = `max_experts` (e.g., 32)
- Mask inactive experts: `router_logits[:, active_mask]` before topk
- Structural edits modify mask, not router weights
- No retraining required

**Compatibility**: ✅ Clean. Swiss-ai/MoE uses standard topk selection. Masking can be injected before topk call without breaking forward pass.

---

## Finding 5: Optimizer State Management 🚨 CRITICAL GAP

**Current state**: Completely unimplemented. Only a comment in demo:
```python
print(f"    - Update optimizer state")  # ← Not real code
```

**Problem scenarios:**

### SPAWN (new expert)
```python
# After SPAWN, optimizer sees new parameters
optimizer.param_groups[0]['params'].append(new_expert_params)
# ← Adam has no momentum/variance for these params yet
# ← First gradient step will initialize them (okay, but suboptimal)
```

### PRUNE (remove expert)
```python
# After PRUNE, expert params deleted from model
# But optimizer.state still has buffers for deleted params
# ← Memory leak (buffers persist but unused)
# ← Need to clean up: del optimizer.state[param_id]
```

### MERGE (combine two experts)
```python
# Two experts → one merged expert
# Expert A has optimizer state (momentum_A, variance_A)
# Expert B has optimizer state (momentum_B, variance_B)
# Merged expert has new params
# ← Which optimizer state to use?
#    Option 1: Reset (zero momentum/variance)
#    Option 2: Average (momentum_merged = (momentum_A + momentum_B) / 2)
#    Option 3: Keep dominant (use higher utilization expert's state)
#    Option 4: Interpolate by utilization ratio
```

**Why this matters:**
- Adam momentum buffers are ~2x parameter memory
- Memory leaks accumulate (especially with frequent PRUNE)
- Training dynamics change if optimizer state handling is wrong
- Checkpoints break if param count != optimizer state count

**Phase 6 MUST implement**:
```python
class ExpertRegistry:
    def spawn_expert(self, ...):
        # 1. Add params to model
        # 2. Register in optimizer param group
        # 3. Initialize optimizer state (optional: warm start from parent)

    def prune_expert(self, ...):
        # 1. Remove params from model
        # 2. Remove from optimizer param group
        # 3. Clean up optimizer.state dict

    def merge_experts(self, ...):
        # 1. Create merged params
        # 2. Handle two optimizer states → one
        # 3. Remove old states, register merged state
```

**This is not optional.** Real training will fail/leak without this.

---

## Finding 6: Gradient Flow (Sanity Check) ✅

**Question**: Do structural edits break backprop?

**Answer**: No. As long as:
1. New experts are in `nn.ModuleList` (autograd tracks them)
2. Removed experts are deleted from `nn.ModuleList` (stop tracking)
3. Merged experts replace sources in `nn.ModuleList`

PyTorch autograd is dynamic. Adding/removing modules between forward passes is safe.

**One caution**: If you modify an expert's parameters **during** training (mid-forward or mid-backward), autograd will break. But our design is safe:
- Structural edits happen **between** training steps
- Forward/backward complete before edit execution
- Next forward pass sees updated model

---

## Finding 7: Checkpoint Compatibility 🤔 Needs Design

**Scenario**: Save checkpoint after SPAWN/PRUNE/MERGE. Load later.

**Issues**:
1. Expert count may differ between save and load
2. Router mask needs to be saved/loaded
3. Optimizer state must match model params

**Phase 6 needs**:
```python
def save_checkpoint(model, optimizer, registry):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'expert_registry': registry.to_dict(),  # Active mask, expert IDs
        'audit_log': registry.audit_log,
    }, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    # Reconstruct model with correct expert count
    # Rebuild optimizer with correct param groups
    # Restore expert registry (mask, IDs)
```

**Decision needed**: Fixed-width padding (always save 32 experts, mask inactive) vs dynamic (save only active)?

- Fixed-width: Simpler, larger checkpoints
- Dynamic: Complex, smaller checkpoints

Recommend **fixed-width** for Phase 6 MVP. Optimize later if needed.

---

## Phase 6 Requirements (Derived from Review)

### Must Implement

1. **ExpertRegistry class**
   - Tracks active/cooling/archived expert states
   - Manages fixed-width router mask (e.g., max 32 experts/layer)
   - Handles optimizer state on structural edits

2. **Signal extraction wrapper**
   - Hook swiss-ai/MoE expert loop to capture per-expert outputs
   - Convert to MoETrace format
   - No architectural changes to swiss-ai/MoE

3. **Optimizer state management**
   - SPAWN: Register new params in optimizer, optionally warm-start from parent
   - PRUNE: Clean up deleted params from optimizer.state
   - MERGE: Handle two optimizer states → one (strategy TBD)

4. **Masked router interface**
   - Pre-filter router logits by active mask before topk
   - Structural edits modify mask, not router weights
   - Router dimension fixed at max_experts

### Nice to Have (Defer to Phase 7)

- Checkpoint save/load with expert registry
- Optimizer state interpolation strategies for MERGE
- Dynamic expert capacity (grow max_experts at runtime)
- Expert cooling/warming mechanisms (soft enable/disable)

---

## Integration Path

### Minimal viable integration (~2-3 days):

1. **Implement ExpertRegistry** (~1 day)
   - Fixed-width router mask per layer
   - Active expert tracking
   - Basic optimizer state add/remove

2. **Wrap swiss-ai/MoE layer** (~0.5 day)
   - Hook expert loop for per-expert outputs
   - Inject mask before topk selection
   - Test: identical forward pass with/without wrapper

3. **Connect to ChronoMoEv3** (~0.5 day)
   - Construct MoETrace from wrapper signals
   - Run SPAWN with real gradients
   - Verify audit log captures real router distributions

4. **Validation** (~1 day)
   - Train small model (few steps)
   - Trigger SPAWN/PRUNE/SPLIT/MERGE
   - Verify no crashes, memory leaks, or gradient errors
   - Confirm optimizer state is clean after edits

### Blockers if skipped:
- Every structural edit requires retraining router (unusable)
- Memory leaks from orphaned optimizer state
- No per-expert outputs → lifecycle system inoperable

---

## Recommendations

1. ✅ **Proceed with Phase 6** - Not optional, mandatory for integration
2. ✅ **Fixed-width router with masking** - Correct architecture choice
3. ⚠️ **Optimizer state is priority #1** - Memory leaks and training instability without this
4. ✅ **EditExecutor abstraction is solid** - No changes needed
5. ✅ **Constitutional layer is robust** - Gates, proposals, audit trail all verified
6. 🤔 **Decide MERGE optimizer state strategy** - Average? Reset? Keep dominant?

**Overall verdict**: ChronoMoEv3's design is sound. One critical implementation gap (optimizer state) and one infrastructure piece (signal hooks) remain. Phase 6 addresses both.

**Confidence level**: High. No architectural red flags found. Integration feasible with modest effort.

---

## Appendix: Signal Availability Matrix

| Signal | Required By | Swiss-AI/MoE Status | Integration Complexity |
|--------|-------------|---------------------|------------------------|
| Router logits | Coherence, bridges | ✅ Available | Low (already exposed) |
| Router probs | Utilization | ✅ Available | Low (softmax of logits) |
| Selected experts | Utilization, masks | ✅ Available | Low (already exposed) |
| Per-expert outputs | Coherence, redundancy, bimodality | ⚠️ Hidden | Medium (need hook in loop) |
| Gate weights | Coherence weighting | ✅ Available | Low (topk weights) |
| Mixture output | Layer coherence | ✅ Available | Low (standard MoE output) |

**Critical path**: Per-expert outputs. Everything else is straightforward.
