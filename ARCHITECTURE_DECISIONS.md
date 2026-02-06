# ChronoMoEv3 Architecture Decisions

**Critical design questions that determine stability, checkpointing, and failure modes.**

---

## Question 1: Where Does Slow Bias Live in Router Math?

### The Options

**Option A: Pre-softmax additive bias (per expert)**
```python
z_clean = W_gate @ h  # [B×S, E]
z_biased = z_clean + beta  # [E] broadcast
probs = softmax(z_biased / temperature)
```

**Option B: Post-softmax multiplicative bias**
```python
z = W_gate @ h
probs_clean = softmax(z / temperature)
probs_biased = probs_clean * (1 + beta)
probs_biased = probs_biased / probs_biased.sum(dim=-1, keepdim=True)  # Renormalize
```

**Option C: Low-rank adjustment to routing projection**
```python
# Learn beta as a rank-1 perturbation
beta_proj = beta_vec.unsqueeze(0)  # [1, E]
W_biased = W_gate + (v @ beta_proj)  # v: [d, 1], learned direction
z_biased = W_biased @ h
```

### Analysis

**Option A (Recommended): Pre-softmax additive**

**Pros:**
- Simple, interpretable: beta directly shifts expert preference
- Temperature acts uniformly on both clean and bias terms
- Cheap: One vector addition, no extra parameters
- Clean/biased separation trivial: `z_clean = z_biased - beta`

**Cons:**
- Large beta can dominate clean logits (saturation risk)
- No inherent temperature adaptation

**Mathematical properties:**
```
z_biased = z_clean + beta
softmax(z/T) where z = z_clean + beta

When beta >> z_clean for expert e:
  softmax saturates → expert e gets all routing regardless of input
  This is bad (ignores input content)

When beta << z_clean:
  Beta acts as gentle prior
  This is good (biases toward persistent experts)
```

**Stability condition:** `|beta_max| << max(z_clean)`. In practice:
- Typical logit range: [-5, 5]
- Safe beta range: [-1, 1] (projectdesign.md defaults)
- Ratio: beta / logit_range ≈ 0.1 to 0.2 (gentle influence)

**Temperature interaction:**
- Higher T → softer routing → beta has less relative impact
- Lower T → sharper routing → beta has more relative impact
- This is actually desirable: as model becomes confident (low T), slow bias matters more

**Verdict:** Option A (pre-softmax additive) is the right choice for v3 reference implementation.

---

**Option B: Post-softmax multiplicative**

**Pros:**
- Cannot saturate (renormalization prevents collapse)
- Beta interpreted as "boost factor" rather than logit shift

**Cons:**
- Loses clean/biased separation (can't compute z_clean easily)
- Renormalization makes beta effect dependent on all other experts
- More expensive (multiply + sum + divide)
- Bridge detection becomes ambiguous (no clean logits)

**Verdict:** Reject. Breaks clean/biased separation needed for Phase 3+.

---

**Option C: Low-rank routing adjustment**

**Pros:**
- Learns optimal "slow direction" in input space
- Can model correlations (e.g., "beta increases for experts 2,4,6 jointly")
- More expressive than scalar per expert

**Cons:**
- Adds parameters (v vector, [d_model] size)
- Loses interpretability (what is v learning?)
- Complicates checkpoint (now have beta_vec and v)
- Overkill for v3 (can be Phase 9 extension)

**Verdict:** Defer to future work. v3 uses scalar beta per expert.

---

### Decision: Pre-Softmax Additive Bias

```python
class ChronoRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.W_gate = nn.Linear(d_model, num_experts, bias=False)
        self.beta = nn.Parameter(torch.zeros(num_experts))  # Slow bias
        self.temperature = 1.0  # Can be learned later

    def forward(self, h, return_clean=True):
        z_clean = self.W_gate(h)  # [B×S, E]
        z_biased = z_clean + self.beta.unsqueeze(0)  # [B×S, E]

        probs = F.softmax(z_biased / self.temperature, dim=-1)

        if return_clean:
            return probs, z_clean, z_biased
        return probs
```

**Constraints to enforce:**
- `beta.data.clamp_(-1.0, 1.0)` after each update
- Monitor `beta / z_clean.std()` → should stay < 0.5 to avoid saturation
- If ratio exceeds threshold, trigger global beta decay: `beta *= 0.9`

---

## Question 2: Minimal Checkpoint State for Deterministic Recovery

### What Must Be Saved

For deterministic lifecycle decisions after restart, we need:

**Per-expert state (required):**
```python
@dataclass
class ExpertCheckpoint:
    expert_id: str
    layer_id: int

    # Three-clock coherence
    phi_fast: float
    phi_mid: float
    phi_slow: float

    # Slow bias (the locus)
    beta: float

    # Role vector (what expert typically outputs)
    role_vector: Tensor  # [d_model]

    # Bimodality detector state
    centroid_1: Tensor  # [d_model]
    centroid_2: Tensor  # [d_model]
    centroid_balance: float

    # Tracking
    born_step: int
    last_update_step: int
    total_tokens_seen: int
    cooling_until: Optional[int]
    active: bool
```

**Per-layer state (required):**
```python
@dataclass
class LayerCheckpoint:
    layer_id: int

    # Layer-wide coherence at three timescales
    Psi_fast: float
    Psi_mid: float
    Psi_slow: float

    # Free energy components (for lifecycle)
    F_last: float  # Last computed free energy
    misfit_last: float
    complexity_last: float
    redundancy_last: float
    instability_last: float

    # Lifecycle cooldown
    last_edit_step: int
    cooling_until: int
```

**Global state (required):**
```python
@dataclass
class ChronoCheckpoint:
    step: int
    config: ChronoConfig
    expert_states: Dict[str, ExpertCheckpoint]
    layer_states: Dict[int, LayerCheckpoint]
```

### What Can Be Reconstructed (Not Saved)

- **Basin history:** Rebuild from scratch (50-step window, fine to lose)
- **Router logits:** Recomputed on next forward pass
- **Mixture outputs:** Transient, not persistent state
- **Token indices:** Forward-pass dependent, not state

### Checkpoint Size Estimate

**Per expert:**
- 3 floats (phi_fast, phi_mid, phi_slow): 12 bytes
- 1 float (beta): 4 bytes
- 1 tensor (role_vector, d_model=4096): 16 KB
- 2 tensors (centroids, 2×4096): 32 KB
- Metadata: ~100 bytes
- **Total: ~48 KB per expert**

**For 8-layer, 8-expert-per-layer model:**
- 64 experts × 48 KB = **3 MB**
- 8 layers × 200 bytes = 1.6 KB
- **Total checkpoint: ~3 MB** (negligible vs model weights ~7 GB)

### Determinism Guarantee

Given checkpoint at step T:
1. Load expert_states, layer_states
2. Resume training from step T+1
3. All lifecycle decisions (spawn/prune/split/merge) will be identical to non-interrupted run

**Why:** Lifecycle decisions depend only on:
- phi_slow (saved)
- beta (saved)
- F_l computed from Psi_slow and counts (both saved or reconstructible)
- Cooldown timers (saved)

**What breaks determinism:**
- Different random seed (splits/spawns use noise)
- Different batch order (affects which tokens route where)
- Optimizer state (Adam moments) — separate concern

**Solution:** Save random state and optimizer state separately (standard practice).

---

## Question 3: Failure Mode: Widespread Clean/Biased Disagreement

### Scenario

Clean and biased routers disagree on most experts for most tokens:

```python
# Token x
z_clean[x] = [-2, 3, 1, -1, 0, 2, -3, 1]  # Expert 1 wins
beta =       [ 4, -2, 0, 1, 2, -1, 3, 0]
z_biased[x] = [ 2, 1, 1, 0, 2, 1, 0, 1]  # Experts 0,4 tie

# Top-k selection flips completely
clean: selects experts [1, 5]
biased: selects experts [0, 4]
```

### When This Happens

**Root cause:** Beta has drifted far from reality.

**Triggers:**
1. **Distribution shift:** Model trained on A, deployed on B. Old experts no longer functional but have high beta from past success.
2. **Catastrophic forgetting:** Fast clock collapsed on new task, but slow clock still thinks old experts are good.
3. **Runaway bias:** Beta grew unchecked, now dominates clean logits.

### Detection Signal

**Measure disagreement:**
```python
# For each token, which expert would win under clean vs biased?
top1_clean = z_clean.argmax(dim=-1)  # [B×S]
top1_biased = z_biased.argmax(dim=-1)  # [B×S]

disagreement_rate = (top1_clean != top1_biased).float().mean()
```

**Thresholds:**
- disagreement_rate < 0.2: Healthy (beta is gentle prior)
- 0.2 < disagreement_rate < 0.5: Moderate (beta still informative but drifting)
- disagreement_rate > 0.5: **Crisis** (beta is fighting input)

### Response Strategy

**Option A: Global beta decay (recommended)**
```python
if disagreement_rate > 0.5:
    # Decay all betas toward zero
    beta *= 0.5
    logger.warning(f"High disagreement ({disagreement_rate:.2f}), decaying beta globally")
```

**Pros:**
- Simple, no hyperparameters
- Preserves relative ordering (high-beta experts still favored, just less)
- Self-correcting: If disagreement persists, beta decays to zero → clean routing restored

**Cons:**
- Loses locus information abruptly
- All layers affected even if only one layer has problem

---

**Option B: Per-layer beta decay**
```python
for layer_id in layers:
    disagreement_rate_l = compute_disagreement(layer_id)
    if disagreement_rate_l > 0.5:
        beta[layer_id] *= 0.5
```

**Pros:**
- Surgical (only problematic layer loses locus)
- Preserves good layers

**Cons:**
- More complex bookkeeping
- Requires tracking disagreement per layer

---

**Option C: Adaptive temperature increase**
```python
if disagreement_rate > 0.5:
    temperature *= 1.5  # Soften routing
```

**Rationale:** High temperature makes softmax less sensitive to beta, effectively reducing its impact without destroying locus.

**Pros:**
- Preserves beta (locus intact)
- Reversible (can lower temperature later)

**Cons:**
- Affects clean routing too (not just bias)
- May hurt performance even after recovery

---

**Option D: Relevance collapse (shrink retention windows)**
```python
if disagreement_rate > 0.5:
    # Shorten slow clock retention → faster forgetting
    alpha_slow *= 0.99  # e.g., 0.999 → 0.98901
    # Equivalent to halving half-life
```

**Rationale:** If slow clock is wrong, make it less persistent. Let fast clock re-establish ground truth faster.

**Pros:**
- Targets root cause (slow clock out of sync with reality)
- Automatic recovery as new coherence signal overwrites old

**Cons:**
- Changes clock semantics (no longer "slow" if half-life drops to 100 steps)
- Hard to decide when to restore original alpha_slow

---

### Decision: Hybrid Strategy

**Level 1 (disagreement 0.2-0.5): Warning only**
```python
if 0.2 < disagreement_rate < 0.5:
    logger.warning(f"Moderate disagreement: {disagreement_rate:.2f}")
    # No action, just monitor
```

**Level 2 (disagreement 0.5-0.7): Per-layer beta decay**
```python
if 0.5 < disagreement_rate < 0.7:
    for layer_id in layers:
        if layer_disagreement[layer_id] > 0.5:
            beta[layer_id] *= 0.8
            logger.warning(f"Layer {layer_id} beta decayed")
```

**Level 3 (disagreement > 0.7): Global beta reset**
```python
if disagreement_rate > 0.7:
    beta[:] = 0.0  # Full reset
    logger.error(f"Critical disagreement ({disagreement_rate:.2f}), resetting beta globally")
```

**Rationale:** Gradual escalation. Try surgical fix first, nuclear option only if crisis.

---

## Question 4: Split Detection + Slow Bias Interaction

### Scenario

Expert e has:
- High slow coherence: `phi_slow = 0.85` (expert is useful on average)
- High fast variance: `var(phi_fast over 50 steps) = 0.3` (oscillating between good and bad)
- Bimodality score: `separation * balance = 0.7` (serving two distinct basins)

**Hypothesis:** Beta is pulling expert toward routing basin A and B, but expert's weights are optimized for A only. When tokens from basin B arrive, coherence drops.

### The Question

Do we:
1. **Split immediately** (expert needs to become two)
2. **Relax beta first** (reduce beta_e → 0, wait N steps, see if variance collapses)

### Analysis

**Case 1: Beta is the cause (bias-induced bimodality)**

Expert is functionally unimodal, but beta pulls it into basin B where it doesn't belong.

**Evidence:**
- Bimodality appears only after beta > 0.5
- When beta = 0 (clean routing), expert coherence is stable

**Correct action:** Relax beta. Don't split.

**Outcome:** Variance collapses, expert remains coherent on basin A only.

---

**Case 2: Expert is inherently bimodal (bias is irrelevant)**

Expert's weights have learned to serve both A and B (e.g., via gradient pressure).

**Evidence:**
- Bimodality present even when beta ≈ 0
- Centroid separation persists across long timescales (>500 steps)

**Correct action:** Split. Beta is not the problem.

**Outcome:** Two child experts, each coherent on one basin.

---

### Decision Protocol

**Step 1: Detect bimodality**
```python
if bimodality_score > threshold (e.g., 0.6):
    # Potential split candidate
    pass
```

**Step 2: Check beta magnitude**
```python
if abs(beta_e) > 0.5:
    # Beta is significant, might be the cause
    # Enter "relaxation trial"
    relax_beta_trial(expert_e)
else:
    # Beta is small, not the cause
    # Proceed to split immediately
    propose_split(expert_e)
```

**Step 3: Relaxation trial (if beta high)**
```python
def relax_beta_trial(expert_e):
    # Temporarily zero out beta
    beta_e_saved = beta[e]
    beta[e] = 0.0

    # Wait N=100 steps, measure bimodality
    # (This would be checked in slow clock evaluation)
    trial_start_step = current_step
    expert_e.in_relaxation_trial = True
    expert_e.trial_start = trial_start_step
    expert_e.beta_saved = beta_e_saved
```

**Step 4: Evaluate trial outcome**
```python
# At step trial_start + 100
if bimodality_score_now < threshold * 0.5:
    # Variance collapsed! Beta was the cause.
    # Keep beta=0 for this expert
    logger.info(f"Expert {e} relaxation successful, beta suppressed")
    beta[e] = 0.0  # Permanent
    expert_e.in_relaxation_trial = False
else:
    # Variance persists. Expert is inherently bimodal.
    # Restore beta, proceed to split
    logger.info(f"Expert {e} relaxation failed, proceeding to split")
    beta[e] = expert_e.beta_saved
    propose_split(expert_e)
```

**Cooldown:** After relaxation trial (success or failure), expert cannot enter another trial for 500 steps.

### Cost of This Approach

- 100 steps of "wasted" slow-clock evaluation per high-beta bimodal expert
- Acceptable because:
  - Bimodality detection happens slowly anyway (slow clock eval every ~100 steps)
  - Avoids false positive splits (expert doesn't need splitting, just needs beta relief)
  - If wrong (beta wasn't the cause), we proceed to split anyway

### When to Skip Relaxation Trial

- Expert is new (born < 500 steps ago) → hasn't had time to accumulate beta
- Beta is already near zero (`|beta| < 0.1`)
- Layer is at max_experts (no room to split even if needed) → prune instead

---

## Question 5: Cheapest Place to Compute z_clean in Real Models

### Problem

In some implementations, router forward pass is fused or destructive:

**Bad example (Mixtral-style):**
```python
def forward(self, h):
    logits = self.gate(h)  # [B×S, E]
    logits = F.softmax(logits, dim=-1)  # ← Overwrites logits!
    top_k_probs, top_k_idx = torch.topk(logits, k)
    return logits, top_k_probs, top_k_idx
```

Pre-softmax logits are lost. Cannot compute z_clean.

### Solution Hierarchy

**Option 1: Modify router to return both (cleanest)**
```python
def forward(self, h):
    z_clean = self.gate(h)  # [B×S, E]
    probs = F.softmax(z_clean, dim=-1)
    # ... rest of logic
    return probs, z_clean  # ← Return raw logits
```

**Cost:** Zero. Just don't overwrite the variable.

**When to use:** v3 reference implementation, or when you control the model code.

---

**Option 2: Hook the Linear layer directly (model-agnostic)**
```python
# Hook on self.gate (the Linear layer)
def hook_fn(module, input, output):
    z_clean = output  # [B×S, E]
    # Store in external cache
    coherence_tracker.store_z_clean(layer_id, z_clean.detach())

router.gate.register_forward_hook(hook_fn)
```

**Cost:** One `.detach()` per forward pass. Negligible.

**When to use:** Integrating with external models (Mixtral, Switch) where you can't modify source.

---

**Option 3: Recompute from weights (if necessary)**
```python
# If logits are truly lost and no hook access
z_clean = F.linear(h, router.gate.weight, router.gate.bias)
```

**Cost:** One extra matmul. ~1-2% overhead.

**When to use:** Last resort (e.g., tracing through compiled model).

---

**Option 4: Subtract beta from z_biased (if bias is additive)**
```python
# If you only have z_biased = z_clean + beta
z_clean = z_biased - beta.unsqueeze(0)
```

**Cost:** Zero (one subtraction).

**When to use:** When router only returns biased logits but you have beta.

---

### Decision for v3 Reference

**Use Option 1 (explicit return):**
```python
class ChronoRouter(nn.Module):
    def forward(self, h, return_clean=True):
        z_clean = self.gate(h)
        z_biased = z_clean + self.beta.unsqueeze(0)
        probs = F.softmax(z_biased / self.temperature, dim=-1)

        if return_clean:
            return probs, z_clean, z_biased, self.beta
        return probs
```

**For external models (Mixtral, Switch), use Option 2 (hook on gate Linear).**

---

## Question 6: Falsification Criterion

### The Claim

**Coherence (phi_e) predicts expert degradation before traditional metrics.**

Specifically:
- `phi_slow` dropping below threshold → expert should be pruned
- `delta = phi_fast - phi_slow` negative → expert is degrading
- This happens before:
  - Expert utilization drops to zero
  - Loss starts increasing
  - Manual inspection reveals failure

### How to Falsify

**Experiment: Controlled Expert Damage**

1. Train 8-layer MoE on language modeling (e.g., Shakespeare, WikiText)
2. After convergence, intentionally damage one expert:
   - Zero out its weights: `expert_3.weight.data.zero_()`
   - Or: Add large noise: `expert_3.weight.data += torch.randn_like(expert_3.weight) * 10.0`
   - Or: Freeze it and continue training (stale expert)

3. Continue training for 1000 steps, tracking:
   - `phi_fast`, `phi_mid`, `phi_slow` for all experts
   - Expert utilization (tokens routed)
   - Layer-level loss
   - Per-token perplexity

4. **Expected outcome (if coherence is correct):**
   - Expert 3's `phi_fast` drops within 10-50 steps
   - Expert 3's `phi_slow` drops within 500-1000 steps
   - Utilization starts dropping ~100-200 steps later (router learns to avoid it)
   - Loss increase is small if other experts compensate

5. **Falsification outcome (if coherence is wrong):**
   - Expert 3's `phi_slow` remains high (>0.7) even after 1000 steps
   - But: Utilization drops to near-zero
   - Or: Loss increases sharply
   - **This would mean phi_slow is not tracking functional health**

### Alternative Metrics to Compare

**Metric A: Expert utilization**
```python
utilization_e = (num_tokens_routed_to_e) / (total_tokens)
```

**Prediction:** Should drop after expert is damaged, but with delay (router needs to learn).

---

**Metric B: Per-expert loss contribution**
```python
# Compute loss with and without expert e
loss_full = model(batch).loss
loss_without_e = model(batch, mask_expert=[e]).loss

contribution_e = loss_full - loss_without_e  # Negative if expert is harmful
```

**Prediction:** Should turn negative after expert is damaged.

**Problem:** Expensive (requires two forward passes per expert).

---

**Metric C: Router entropy for expert e**
```python
# For tokens routed to expert e, what is routing entropy?
p = router_probs[tokens_to_e, e]  # [n_e]
entropy_e = -(p * p.log()).mean()
```

**Prediction:** Should increase (router becomes uncertain about expert e).

**Problem:** Noisy, doesn't directly measure output quality.

---

**Metric D: Output magnitude**
```python
# Mean L2 norm of expert output
magnitude_e = expert_e(inputs).norm(dim=-1).mean()
```

**Prediction:** Might collapse if expert is damaged.

**Problem:** Doesn't measure directional alignment with mixture.

---

### What Would Actually Falsify Coherence?

**Scenario 1: Phi_slow stays high for broken expert**
- Expert 3 is zeroed out
- Coherence should be ~0 (random output uncorrelated with mixture)
- If `phi_slow > 0.7` after 1000 steps → **coherence is broken**
- Likely cause: Mixture direction is dominated by expert 3 historically, so even broken expert has residual correlation

**Scenario 2: Phi_slow drops for healthy expert**
- Expert 5 is working fine (high utilization, good loss contribution)
- But `phi_slow < 0.3` persistently
- This would mean coherence is measuring something other than functional health

**Scenario 3: Split/merge decisions guided by phi_slow perform worse than random**
- Use phi_slow to decide when to split/merge
- Compare to random splits/merges
- If random does better → coherence is not informative

---

### The Honest Answer

**What would make me distrust coherence:**

If an expert has:
- Low coherence (`phi_slow < 0.3`)
- High utilization (router confidently selects it)
- Low loss (removing it hurts performance)

**This would mean:**
- Expert is doing something useful
- But not aligned with mixture direction
- Possible explanation: Expert is a "corrective" expert that produces orthogonal corrections, not aligned outputs

**How to detect this:**
- Track loss delta: `loss_without_e - loss_with_e`
- If low-coherence expert has negative loss_delta → it's useful despite low phi
- This would require redefining lifecycle: prune only if (low phi AND low impact)

**But:** This is actually fine! It means coherence measures alignment, not usefulness. We'd just add loss_delta as a second signal.

---

## Summary of Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **1. Slow bias location** | Pre-softmax additive per expert | Simple, interpretable, cheap, preserves clean/biased separation |
| **2. Checkpoint state** | phi_{fast,mid,slow}, beta, role_vector, centroids, metadata (~3MB for 64 experts) | Sufficient for deterministic lifecycle recovery |
| **3. Clean/biased disagreement** | Hybrid: monitor→per-layer decay→global reset at 0.2/0.5/0.7 thresholds | Gradual escalation, surgical before nuclear |
| **4. Split + beta interaction** | Relax beta trial (100 steps) if beta>0.5, then split if bimodality persists | Avoids false positive splits from bias-induced variance |
| **5. z_clean computation** | Explicit return in v3, hook on gate Linear for external models | Zero cost for v3, minimal cost for integration |
| **6. Falsification** | Low phi + high utilization + low loss → coherence is wrong | Would require adding loss_delta as second lifecycle signal |

---

**Status:** Architecture decisions documented. Ready for Phase 2 implementation (slow bias).
