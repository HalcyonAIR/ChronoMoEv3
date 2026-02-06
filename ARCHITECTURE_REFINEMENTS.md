# Architecture Refinements: Response to Halcyon

**Engineering tightening before Phase 2 hardens into concrete.**

---

## 1. Scale-Free Beta Definition

### The Problem

Current approach uses absolute constraint: `|beta| ≤ 1.0`

**Issue:** This is regime-specific. Logit scales vary by:
- Layer depth (early layers often have smaller logits)
- Training phase (logits grow during training)
- Model architecture (routers with bias vs without)
- Dataset (perplexity affects routing confidence)

A fixed beta cap is not portable.

### The Solution: Normalize by Logit Scale

**Define beta as fraction of current logit standard deviation:**

```python
class ChronoRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Beta as scale-free coefficient, not absolute value
        self.beta_coeff = nn.Parameter(torch.zeros(num_experts))  # k in [-k_max, k_max]
        self.k_max = 0.3  # Max fraction of logit std

        # Running estimate of logit scale
        self.register_buffer('logit_std_ema', torch.tensor(1.0))
        self.logit_std_alpha = 0.99

    def forward(self, h, update_stats=True):
        z_clean = self.gate(h)  # [B×S, E]

        # Update logit scale estimate
        if update_stats:
            current_std = z_clean.std().item()
            self.logit_std_ema = (
                self.logit_std_alpha * self.logit_std_ema
                + (1 - self.logit_std_alpha) * current_std
            )

        # Clamp coefficients, not absolute values
        k = self.beta_coeff.clamp(-self.k_max, self.k_max)

        # Scale-free bias
        beta_eff = k * self.logit_std_ema  # [E]
        z_biased = z_clean + beta_eff.unsqueeze(0)

        probs = F.softmax(z_biased, dim=-1)

        return probs, z_clean, z_biased, beta_eff
```

### Canonical Update Rule

```python
def update_beta(expert_id, phi_slow, tau=0.5, eta=0.01):
    """
    Update scale-free beta coefficient.

    beta_coeff(t+1) = clip(
        beta_coeff(t) + eta * (phi_slow - tau),
        -k_max, k_max
    )

    Effective bias scales with current logit_std automatically.
    """
    delta = eta * (phi_slow - tau)
    beta_coeff[expert_id] += delta
    beta_coeff[expert_id] = beta_coeff[expert_id].clamp(-k_max, k_max)
```

### Invariant Properties

1. **Portable across layers:** k=0.3 means "30% of typical logit deviation" regardless of absolute scale
2. **Adapts to training:** As logits grow, beta_eff grows proportionally
3. **Self-regulating:** If logit_std shrinks (e.g., overconfident router), beta_eff shrinks too

### Recommended Default

`k_max = 0.3` based on saturation analysis showing `beta / logit_std = 0.35` is gentle prior.

### Config Update

```python
@dataclass
class ChronoConfig:
    # OLD: beta_min, beta_max (absolute)
    # NEW: k_max (scale-free coefficient)
    k_max: float = 0.3  # Maximum beta as fraction of logit std

    # Update rule unchanged
    eta_beta: float = 0.01
    tau: float = 0.5
```

---

## 2. Temperature vs Beta Interaction (Clarification)

### The Claim (Original)

> "Temperature acts uniformly on both clean and bias terms"

### The Reality

**Temperature and beta interact via softmax nonlinearity.** They are not independent.

**Algebraically separable:**
```python
z_biased = z_clean + beta
probs = softmax(z_biased / T) = softmax((z_clean + beta) / T)
```

Clean and biased terms can be isolated: `z_clean = z_biased - beta` (separation preserved ✓)

**But nonlinear interaction:**
```python
softmax(z/T)[i] = exp(z[i]/T) / sum_j exp(z[j]/T)

d/dT softmax(z/T) ≠ 0  # Temperature changes prob distribution shape

Effect of beta depends on temperature:
- High T (soft routing): beta has small relative impact
- Low T (sharp routing): beta has large relative impact
```

### Corrected Statement

**Temperature and beta are algebraically separable (can compute z_clean), but they interact through softmax nonlinearity.**

**Why this is actually desirable:**

When model is confident (low temperature), it respects slow bias more strongly. When model is uncertain (high temperature), it relies more on input-driven routing. This is the right dynamic: locus influence should be conditional on routing confidence.

**Practical implication:**

If you change temperature during training (e.g., temperature annealing), beta_eff magnitude needs monitoring. A fixed `k_max` is stable across temperature changes, but the *behavioral impact* varies.

### Documentation Fix

Replace:
> "Temperature acts uniformly (no competition with beta)"

With:
> "Temperature and beta are algebraically separable (z_clean = z_biased - beta), but interact via softmax nonlinearity. Higher T reduces beta's relative routing impact; lower T amplifies it. This is desirable: confident models respect locus more."

---

## 3. Crisis Threshold Calibration

### The Problem

Thresholds `[0.2, 0.5, 0.7]` for disagreement escalation are plausible but arbitrary.

**They should be relative to expected baseline variance.**

### Calibration Procedure

**Step 1: Baseline measurement (beta=0)**

```python
def calibrate_disagreement_thresholds(model, calibration_batches=100):
    """
    Measure baseline flip rate with beta=0, then with validated beta.
    Set thresholds relative to these distributions.
    """
    # Collect disagreement rates with beta=0 (baseline noise)
    flip_rates_baseline = []

    with torch.no_grad():
        for batch in calibration_batches:
            z_clean = router(batch.hidden_states)

            # Simulate small random perturbation (measurement noise)
            z_perturbed = z_clean + torch.randn_like(z_clean) * 0.1

            top1_clean = z_clean.argmax(dim=-1)
            top1_perturbed = z_perturbed.argmax(dim=-1)

            flip_rate = (top1_clean != top1_perturbed).float().mean().item()
            flip_rates_baseline.append(flip_rate)

    # Baseline: typical flip rate due to noise (~5-10%)
    baseline_flip = np.mean(flip_rates_baseline)
    baseline_std = np.std(flip_rates_baseline)

    return baseline_flip, baseline_std
```

**Step 2: Expected flip rate under validated beta**

```python
def measure_validated_beta_impact(model, calibration_batches=100, k=0.3):
    """
    Measure flip rate when beta = k * logit_std (validated safe range).
    """
    flip_rates_biased = []

    with torch.no_grad():
        for batch in calibration_batches:
            z_clean = router(batch.hidden_states)
            logit_std = z_clean.std().item()

            # Apply validated bias
            beta = torch.randn(num_experts) * k * logit_std  # Random beta within range
            z_biased = z_clean + beta.unsqueeze(0)

            top1_clean = z_clean.argmax(dim=-1)
            top1_biased = z_biased.argmax(dim=-1)

            flip_rate = (top1_clean != top1_biased).float().mean().item()
            flip_rates_biased.append(flip_rate)

    validated_flip = np.mean(flip_rates_biased)
    validated_std = np.std(flip_rates_biased)

    return validated_flip, validated_std
```

**Step 3: Set thresholds relative to baseline**

```python
def compute_crisis_thresholds(baseline_flip, validated_flip):
    """
    Set escalation thresholds as quantiles of expected distribution.

    Logic:
    - Warning: Above validated beta impact (healthy bias exceeded)
    - Serious: 2x validated impact (bias clearly wrong)
    - Crisis: 3x validated impact (catastrophic disagreement)
    """
    threshold_warning = max(baseline_flip + 2*baseline_std, validated_flip)
    threshold_serious = 2 * validated_flip
    threshold_crisis = 3 * validated_flip

    return threshold_warning, threshold_serious, threshold_crisis
```

### Example Calibration

```
Baseline flip rate (beta=0, noise only): 0.08 ± 0.03
Validated flip rate (beta=0.3*std): 0.15 ± 0.05

Computed thresholds:
- Warning: 0.15 (validated impact)
- Serious: 0.30 (2x validated)
- Crisis: 0.45 (3x validated)
```

These are regime-specific and adapt to model/data properties.

### Config Integration

```python
@dataclass
class ChronoConfig:
    # ... existing fields ...

    # Crisis thresholds (will be calibrated, these are fallbacks)
    disagreement_threshold_warning: float = 0.2
    disagreement_threshold_serious: float = 0.5
    disagreement_threshold_crisis: float = 0.7

    # Enable auto-calibration
    auto_calibrate_thresholds: bool = True
    calibration_steps: int = 100
```

---

## 4. Top-k Disagreement (Beyond Top-1 Flips)

### The Problem

Current metric: `(top1_clean != top1_biased).float().mean()`

**Issues:**
1. **Ignores top-2, top-3:** In top-k routing, secondary experts matter
2. **Binary:** Small prob shift (0.51→0.49) treated same as large shift (0.9→0.1)
3. **Misses overlap changes:** If top-1 stays same but top-2 flips, overlap dynamics change

### Better Metrics

**Option A: JS Divergence (Recommended)**

```python
def compute_disagreement_js(p_clean, p_biased):
    """
    Jensen-Shannon divergence between clean and biased routing distributions.

    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Range: [0, log(2)] for binary, [0, log(num_experts)] generally
    Symmetric, well-behaved, captures full distribution shift.
    """
    # Avoid log(0)
    p_clean = p_clean + 1e-9
    p_biased = p_biased + 1e-9

    # Middle distribution
    m = 0.5 * (p_clean + p_biased)

    # KL divergences
    kl_pm = (p_clean * (p_clean / m).log()).sum(dim=-1)
    kl_qm = (p_biased * (p_biased / m).log()).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    return js.mean().item()
```

**Interpretation:**
- js ≈ 0.01: Negligible disagreement
- js ≈ 0.1: Moderate shift
- js ≈ 0.5: Strong disagreement
- js → log(E): Distributions nearly disjoint

**Advantage:** Captures full distribution, symmetric, principled.

---

**Option B: Overlap Mass Difference**

```python
def compute_disagreement_overlap(p_clean, p_biased, k=2):
    """
    Measure change in top-k routing mass.

    For each token, compute:
    - overlap = sum of min(p_clean[i], p_biased[i]) for i in top-k union
    - disagreement = 1 - overlap
    """
    # Get top-k indices for both
    topk_clean = p_clean.topk(k, dim=-1).indices
    topk_biased = p_biased.topk(k, dim=-1).indices

    # Union of top-k sets
    union_mask = torch.zeros_like(p_clean, dtype=torch.bool)
    union_mask.scatter_(-1, topk_clean, True)
    union_mask.scatter_(-1, topk_biased, True)

    # Overlap mass
    overlap = torch.where(
        union_mask,
        torch.min(p_clean, p_biased),
        torch.zeros_like(p_clean)
    ).sum(dim=-1)

    disagreement = 1 - overlap

    return disagreement.mean().item()
```

**Interpretation:**
- overlap ≈ 1.0: Same experts, same weights
- overlap ≈ 0.5: Half the mass shifted
- overlap ≈ 0.0: Completely disjoint experts

**Advantage:** Intuitive, directly measures routing mass shift.

---

**Option C: Weighted Top-k Flip Rate**

```python
def compute_disagreement_weighted_flip(p_clean, p_biased, k=2):
    """
    Flip rate weighted by probability mass.

    Accounts for: "How much routing mass changed experts?"
    """
    topk_clean_probs, topk_clean_idx = p_clean.topk(k, dim=-1)
    topk_biased_probs, topk_biased_idx = p_biased.topk(k, dim=-1)

    # For each clean top-k expert, is it still in biased top-k?
    flipped_mask = ~torch.isin(topk_clean_idx, topk_biased_idx)

    # Weight by probability mass
    flipped_mass = (topk_clean_probs * flipped_mask.float()).sum(dim=-1)
    total_mass = topk_clean_probs.sum(dim=-1)

    disagreement = (flipped_mass / total_mass).mean().item()

    return disagreement
```

**Advantage:** Accounts for mass, not just counts.

---

### Recommendation

**Use JS divergence as primary metric** (principled, captures full distribution).

**Use top-1 flip rate as secondary** (interpretable headline metric).

**Report both:**
```python
disagreement_js = compute_disagreement_js(p_clean, p_biased)
disagreement_flip = (p_clean.argmax(-1) != p_biased.argmax(-1)).float().mean()

if disagreement_js > threshold_js_serious:
    # Escalate based on full distribution shift
    ...
```

### Calibrated Thresholds for JS

After calibration:
```python
# Baseline JS (beta=0): ~0.01-0.02
# Validated JS (beta=0.3*std): ~0.05-0.1

threshold_js_warning = 0.1
threshold_js_serious = 0.2
threshold_js_crisis = 0.4
```

---

## 5. Beta Relaxation Trial: Isolating the Effect

### The Problem

Zeroing beta for 100 steps changes routing → changes gradients → changes learning.

**Confound:** Variance might collapse because:
1. Beta was the cause (good diagnosis)
2. Training dynamics shifted (false positive)
3. Expert weights adapted to new routing (false negative)

### Solutions

**Option A: Freeze Expert Weights During Trial (Cleanest)**

```python
def start_relaxation_trial(expert_id):
    """
    Freeze expert weights to isolate beta effect.
    """
    expert = model.experts[expert_id]

    # Save requires_grad state
    saved_grad_state = {
        name: param.requires_grad
        for name, param in expert.named_parameters()
    }

    # Freeze
    for param in expert.parameters():
        param.requires_grad = False

    # Also freeze router row for this expert
    model.router.gate.weight.data[expert_id].requires_grad = False

    trial_state = {
        'expert_id': expert_id,
        'beta_saved': beta[expert_id].item(),
        'grad_state_saved': saved_grad_state,
        'start_step': current_step,
    }

    beta[expert_id] = 0.0

    return trial_state

def end_relaxation_trial(trial_state):
    """
    Restore weights, evaluate outcome.
    """
    expert_id = trial_state['expert_id']
    expert = model.experts[expert_id]

    # Restore requires_grad
    for name, param in expert.named_parameters():
        param.requires_grad = trial_state['grad_state_saved'][name]

    model.router.gate.weight.data[expert_id].requires_grad = True

    # Measure bimodality
    bimodality_now = get_bimodality_score(expert_id)
    bimodality_before = trial_state.get('bimodality_before', 0.7)

    if bimodality_now < bimodality_before * 0.5:
        # Success: beta was the cause
        beta[expert_id] = 0.0  # Keep suppressed
        logger.info(f"Relaxation trial succeeded for {expert_id}")
    else:
        # Failure: expert is inherently bimodal
        beta[expert_id] = trial_state['beta_saved']  # Restore
        propose_split(expert_id)
        logger.info(f"Relaxation trial failed for {expert_id}, proceeding to split")
```

**Pros:**
- Clean isolation of beta effect
- No gradient confounds

**Cons:**
- Expert can't adapt during trial (might be artificially rigid)
- Complicates training loop

---

**Option B: Partial Freeze (Router Only)**

```python
def start_relaxation_trial_light(expert_id):
    """
    Freeze only router, let expert weights train.

    Rationale: Routing is the primary beta effect. Expert can still learn.
    """
    model.router.gate.weight.data[expert_id].requires_grad = False
    beta[expert_id] = 0.0

    # Log routing entropy and loss for control
    trial_state = {
        'expert_id': expert_id,
        'beta_saved': beta[expert_id].item(),
        'entropy_before': compute_routing_entropy(expert_id),
        'loss_before': current_loss,
        'start_step': current_step,
    }

    return trial_state
```

**Pros:**
- Less invasive
- Expert can still learn

**Cons:**
- Expert adaptation might confound diagnosis

---

**Option C: No Freeze, Just Log Control Variables**

```python
def start_relaxation_trial_observational(expert_id):
    """
    Zero beta but don't freeze anything.

    Log:
    - Routing entropy change
    - Loss delta
    - Expert weight norm delta

    Use these as controls in interpretation.
    """
    beta[expert_id] = 0.0

    trial_state = {
        'expert_id': expert_id,
        'beta_saved': beta[expert_id].item(),
        'entropy_before': compute_routing_entropy(expert_id),
        'loss_before': current_loss,
        'weight_norm_before': expert_weight_norm(expert_id),
        'start_step': current_step,
    }

    return trial_state

def end_relaxation_trial_observational(trial_state):
    """
    Evaluate with control checks.
    """
    expert_id = trial_state['expert_id']

    # Measure outcome
    bimodality_now = get_bimodality_score(expert_id)
    entropy_now = compute_routing_entropy(expert_id)
    loss_now = current_loss
    weight_norm_now = expert_weight_norm(expert_id)

    # Control checks
    entropy_changed = abs(entropy_now - trial_state['entropy_before']) > 0.1
    loss_changed = abs(loss_now - trial_state['loss_before']) > 0.05
    weights_changed = abs(weight_norm_now - trial_state['weight_norm_before']) > 0.1

    if any([entropy_changed, loss_changed, weights_changed]):
        logger.warning(
            f"Relaxation trial for {expert_id} confounded: "
            f"entropy Δ={entropy_changed}, loss Δ={loss_changed}, weights Δ={weights_changed}"
        )
        # Proceed with caution or extend trial

    # Rest of evaluation logic...
```

**Pros:**
- No intervention in training
- Collects evidence for post-hoc analysis

**Cons:**
- Weaker causal claim
- Requires careful interpretation

---

### Recommendation

**Use Option B (freeze router only) as default.**

**Upgrade to Option A (freeze expert) if:**
- Trial results are ambiguous
- Expert is high-stakes (large routing share)
- Debugging a specific failure mode

**Always log control variables** (entropy, loss, weight norms) regardless of freeze strategy.

---

## 6. Checkpoint State: Ownership and Contracts

### The Question

> `role_vector` and `centroid_1`, `centroid_2` weren't in minimal v3 spec. What are they, and who owns them?

### Definitions

**`role_vector: Tensor[d_model]`**

**What:** EMA of expert's mean output direction.
```python
role_vector(t) = alpha_role * role_vector(t-1) + (1-alpha_role) * y_bar_e(t)
```

**Purpose:**
- Tracks "what this expert typically outputs"
- Used for redundancy detection (merge candidates have similar role_vectors)
- Used for spawn initialization (new expert clones parent's role_vector + noise)

**Owner:** `CoherenceState` (per-expert state)

**Lifecycle:** Born with expert, updated every step, persists until expert is archived.

---

**`centroid_1, centroid_2: Tensor[d_model]`**

**What:** Two-centroid tracker for bimodality detection.
```python
# Assign current output to closer centroid
if cosine(y_bar_e, centroid_1) > cosine(y_bar_e, centroid_2):
    centroid_1 = alpha_c * centroid_1 + (1-alpha_c) * y_bar_e
else:
    centroid_2 = alpha_c * centroid_2 + (1-alpha_c) * y_bar_e

# Bimodality = separation * balance
separation = 1 - cosine(centroid_1, centroid_2)
balance = min(count_1, count_2) / max(count_1, count_2)
bimodality = separation * balance
```

**Purpose:**
- Detects "expert serving two incompatible basins"
- Triggers split when bimodality > threshold and persists

**Owner:** `CoherenceState` (per-expert state)

**Lifecycle:** Born with expert (initialized to role_vector), updated every step, persists until expert is archived or split.

---

### Ownership Contract

**`CoherenceState` owns:**
- `phi_{fast, mid, slow}` (three-clock coherence)
- `role_vector` (typical output direction)
- `centroid_1, centroid_2, centroid_balance` (bimodality detector state)
- Metadata: `born_step`, `last_update_step`, `total_tokens_seen`

**Separate systems:**
- **Router:** Owns `beta_coeff` (slow bias coefficient)
- **BasinHistory:** Owns token-level routing history (50-step window, not checkpointed)
- **LifecycleCoordinator:** Owns decision log, cooldown timers, edit proposals

**Checkpoint includes:**
- `CoherenceState` for all experts (complete)
- Router `beta_coeff` and `logit_std_ema`
- Layer-level `Psi_{fast,mid,slow}` and `F_l` components
- Cooldown timers

**Checkpoint excludes:**
- `BasinHistory` (reconstructed from scratch)
- Transient forward pass data (mixture, logits, etc.)

---

### Refined Checkpoint Schema

```python
@dataclass
class ExpertCheckpoint:
    """Per-expert persistent state (48 KB per expert)."""

    # Identity
    expert_id: str
    layer_id: int

    # Coherence (12 bytes)
    phi_fast: float
    phi_mid: float
    phi_slow: float

    # Role (16 KB)
    role_vector: Tensor  # [d_model] - what expert typically outputs

    # Bimodality detector (32 KB + 8 bytes)
    centroid_1: Tensor  # [d_model]
    centroid_2: Tensor  # [d_model]
    centroid_balance: float

    # Tracking (24 bytes)
    born_step: int
    last_update_step: int
    total_tokens_seen: int
    active: bool
    cooling_until: Optional[int]

@dataclass
class RouterCheckpoint:
    """Per-layer router state (small)."""

    layer_id: int

    # Scale-free bias coefficients
    beta_coeff: Tensor  # [num_experts]
    logit_std_ema: float

    # Capacity info (for fixed-width router)
    max_experts: int
    active_mask: Tensor  # [max_experts] boolean

@dataclass
class LayerCheckpoint:
    """Per-layer aggregate state (200 bytes)."""

    layer_id: int

    # Layer coherence
    Psi_fast: float
    Psi_mid: float
    Psi_slow: float

    # Free energy
    F_last: float
    misfit_last: float
    complexity_last: float
    redundancy_last: float
    instability_last: float

    # Lifecycle
    last_edit_step: int
    cooling_until: int

@dataclass
class ChronoCheckpoint:
    """Complete system checkpoint (~3MB for 64 experts)."""

    version: str = "v3.0"
    step: int
    config: ChronoConfig

    expert_states: Dict[str, ExpertCheckpoint]  # ~48KB each
    router_states: Dict[int, RouterCheckpoint]  # ~1KB per layer
    layer_states: Dict[int, LayerCheckpoint]    # ~200B per layer
```

---

## 7. Determinism Guarantee (Softened)

### The Problem

Original claim:
> "Identical lifecycle decisions after restart"

**Too strong.** Requires:
- Exact floating point determinism (no CUDA nondeterminism)
- Exact batch order
- Exact random seed state
- Exact distributed collectives order
- No optimizer state drift

### Softened Guarantee

**"Functionally deterministic within tolerance"**

```python
@dataclass
class ChronoConfig:
    # ... existing fields ...

    # Determinism tolerance
    lifecycle_threshold_hysteresis: float = 0.02
    """
    Hysteresis for lifecycle decisions.

    Example: Prune threshold = 0.3
    - Enter prune zone: phi_slow < 0.3 - hysteresis = 0.28
    - Exit prune zone: phi_slow > 0.3 + hysteresis = 0.32

    Prevents float drift from toggling decisions.
    """
```

### Hysteresis Implementation

```python
class LifecycleDecisionTracker:
    """Track expert states with hysteresis."""

    def __init__(self, config):
        self.config = config
        self.in_prune_zone = {}  # expert_id -> bool
        self.in_split_zone = {}

    def should_prune(self, expert_id, phi_slow):
        """
        Prune decision with hysteresis.

        Enter zone: phi_slow < threshold - h
        Exit zone: phi_slow > threshold + h

        Once in zone, stay until clearly out.
        """
        threshold = self.config.prune_phi_threshold  # e.g., 0.3
        h = self.config.lifecycle_threshold_hysteresis  # e.g., 0.02

        currently_in_zone = self.in_prune_zone.get(expert_id, False)

        if currently_in_zone:
            # Exit if phi recovers above upper bound
            if phi_slow > threshold + h:
                self.in_prune_zone[expert_id] = False
                return False
            return True  # Stay in zone
        else:
            # Enter if phi drops below lower bound
            if phi_slow < threshold - h:
                self.in_prune_zone[expert_id] = True
                return True
            return False

    def should_split(self, expert_id, bimodality):
        """Similar hysteresis for split decisions."""
        threshold = self.config.split_bimodality_threshold
        h = self.config.lifecycle_threshold_hysteresis

        currently_in_zone = self.in_split_zone.get(expert_id, False)

        if currently_in_zone:
            if bimodality < threshold - h:
                self.in_split_zone[expert_id] = False
                return False
            return True
        else:
            if bimodality > threshold + h:
                self.in_split_zone[expert_id] = True
                return True
            return False
```

### Revised Guarantee

**Checkpoint determinism:**
1. **Same state → same lifecycle trajectory** within tolerance `h`
2. **Float drift < h/2 → no spurious toggles** (hysteresis prevents)
3. **Different batch order → same decisions eventually** (slow clock averages)

**What can still differ:**
- Exact step when split/prune happens (±10 steps acceptable)
- Which expert is merged first among near-ties (stable sort helps)
- Optimizer state convergence speed

**What is guaranteed:**
- Same experts exist at step T ± small window
- Same expert IDs and parent/child relationships
- Same layer topologies (within merge/split noise)

### Testing Determinism

```python
def test_checkpoint_determinism():
    """
    Verify checkpoint recovery is functionally equivalent.

    Run for 1000 steps, checkpoint at 500, restart, verify:
    - Expert count at step 1000: ±1
    - Layer coherence Psi_slow at step 1000: ±0.05
    - Number of lifecycle actions: ±2
    """
    # Original run
    model1 = train(steps=500)
    ckpt = save_checkpoint(model1)
    model1 = train(model1, steps=500)  # Total 1000

    # Restart from checkpoint
    model2 = load_checkpoint(ckpt)
    model2 = train(model2, steps=500)  # Total 1000

    # Compare final states
    assert abs(model1.num_experts - model2.num_experts) <= 1
    assert abs(model1.Psi_slow - model2.Psi_slow) < 0.05
    assert abs(model1.lifecycle_action_count - model2.lifecycle_action_count) <= 2
```

---

## Bonus: Useful Coherence (Future Extension)

### The Idea

Instead of adding `loss_delta` as separate signal, define:

```python
phi_useful = phi_consensus * impact_weight
```

**Where:**
- `phi_consensus = phi_slow` (alignment with mixture)
- `impact_weight`: Slow EMA of marginal contribution

**Marginal contribution estimation:**

```python
def estimate_impact_weight(expert_id, interval=1000):
    """
    Every N steps, measure expert's contribution via cheap proxy.

    Proxy: Routing mass * gradient norm
    (Experts with high mass AND high gradient are doing useful work)
    """
    utilization = get_utilization(expert_id)  # Fraction of tokens
    grad_norm = expert.weight.grad.norm().item()

    contribution = utilization * grad_norm

    # Slow EMA (alpha=0.999)
    impact_weight_ema = 0.999 * impact_weight_ema + 0.001 * contribution

    return impact_weight_ema
```

**Lifecycle decisions use `phi_useful`:**

```python
if phi_useful < prune_threshold:
    propose_prune(expert_id)
```

This way:
- Low phi, low impact → prune (decoherent and useless)
- Low phi, high impact → keep (useful corrective expert)
- High phi, low impact → keep but watch (redundant?)
- High phi, high impact → healthy expert

**Don't implement yet.** First see if low-phi + high-impact experts exist in practice.

---

## Summary of Refinements

| Question | Refinement | Status |
|----------|-----------|--------|
| 1. Scale-free beta | Use `k * logit_std` with k clamped, not beta directly | **Implement in Phase 2** |
| 2. Temperature interaction | Clarify: algebraically separable, nonlinearly interactive | **Documentation fix** |
| 3. Calibrated thresholds | Measure baseline/validated flip rates, set thresholds relative | **Implement in Phase 2** |
| 4. Top-k disagreement | Use JS divergence + top-1 flip (both reported) | **Implement in Phase 2** |
| 5. Relaxation trial isolation | Freeze router during trial, log control variables | **Implement in Phase 3** |
| 6. Checkpoint ownership | CoherenceState owns role_vector + centroids, contract clarified | **Documentation complete** |
| 7. Determinism guarantee | Soften to "functional determinism", add hysteresis | **Implement in Phase 2** |

**Bonus:** Useful coherence (`phi * impact`) deferred to post-validation.

---

**Status:** Architecture tightened. Phase 2 can proceed with these corrections baked in.
