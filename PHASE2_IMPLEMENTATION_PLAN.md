# Phase 2 Implementation Plan: Vertical Slice

**Goal:** Smallest loop that exercises CoherenceState → RouterState → beta → routing → coherence, with lifecycle as read-only observer + decision logger.

**Principle:** Nothing accumulates secret state. Every step is explicit and observable.

---

## Step 1: RouterState + Beta Application (One Layer)

**Goal:** Prove the dual distribution (clean/biased) exists and is logged.

### 1.1 Add RouterState

```python
# chronomoe_v3/router.py

@dataclass
class RouterState:
    """
    Routing infrastructure for one layer.

    Manages beta (slow bias) and routing health metrics.
    """
    layer_id: int
    num_experts: int

    # Scale-free bias coefficients
    beta_coeff: Tensor  # [num_experts], device='cuda'
    k_max: float = 0.3

    # Logit scale tracking
    logit_std_ema: float = 1.0
    logit_std_alpha: float = 0.99

    # Disagreement metrics (logged each step)
    disagreement_js: float = 0.0
    disagreement_flip: float = 0.0

    def compute_beta_eff(self) -> Tensor:
        """Effective bias: beta_coeff * logit_std."""
        k = self.beta_coeff.clamp(-self.k_max, self.k_max)
        return k * self.logit_std_ema

    def update_logit_std(self, z_clean: Tensor):
        """Update logit scale EMA."""
        current_std = z_clean.std().item()
        self.logit_std_ema = (
            self.logit_std_alpha * self.logit_std_ema
            + (1 - self.logit_std_alpha) * current_std
        )
```

### 1.2 Modify Router Forward Pass

**Target:** One MoE layer in nanoMoE (or simplest harness)

```python
# In router forward() for one layer

class ChronoRouter(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.router_state = RouterState(
            layer_id=self.layer_id,
            num_experts=num_experts,
            beta_coeff=torch.zeros(num_experts, device='cuda'),
        )

    def forward(self, hidden_states: Tensor, top_k: int = 2):
        """
        Router with dual distribution tracking.

        Returns both clean and biased routing decisions.
        """
        # 1. Compute clean logits
        z_clean = self.gate(hidden_states)  # [B×S, E]

        # 2. Update logit scale
        self.router_state.update_logit_std(z_clean)

        # 3. Compute biased logits
        beta_eff = self.router_state.compute_beta_eff()  # [E]
        z_biased = z_clean + beta_eff.unsqueeze(0)  # [B×S, E]

        # 4. Compute both distributions
        p_clean = F.softmax(z_clean, dim=-1)
        p_biased = F.softmax(z_biased, dim=-1)

        # 5. Compute disagreement (BEFORE top-k)
        self.router_state.disagreement_js = compute_js_divergence(p_clean, p_biased)
        self.router_state.disagreement_flip = compute_flip_rate(p_clean, p_biased, top_k)

        # 6. Route using biased distribution
        top_k_probs, top_k_indices = p_biased.topk(top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_probs, top_k_indices, z_clean, z_biased
```

### 1.3 Add Disagreement Metrics

```python
# chronomoe_v3/router.py

def compute_js_divergence(p: Tensor, q: Tensor) -> float:
    """Jensen-Shannon divergence."""
    p = p + 1e-9
    q = q + 1e-9
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean().item()

def compute_flip_rate(p: Tensor, q: Tensor, top_k: int) -> float:
    """Top-k flip rate."""
    top_k_p = p.topk(top_k, dim=-1).indices
    top_k_q = q.topk(top_k, dim=-1).indices
    # Check if top-1 changed
    flip = (top_k_p[:, 0] != top_k_q[:, 0]).float().mean().item()
    return flip
```

### 1.4 Logging

```python
# Every step, log to console or tensorboard
if step % 10 == 0:
    print(f"Step {step}: "
          f"beta_mean={router_state.beta_coeff.mean():.3f}, "
          f"JS={router_state.disagreement_js:.4f}, "
          f"flip={router_state.disagreement_flip:.3f}")
```

**Deliverable 1:** One layer routes with beta. Disagreement metrics logged. No lifecycle yet.

---

## Step 2: Coherence on GPU with Buffered State

**Goal:** Coherence updates every step on GPU. Sync to CPU only on eval intervals.

### 2.1 GPU-Resident Coherence Tensors

```python
# chronomoe_v3/coherence_gpu.py

class CoherenceBuffer:
    """
    GPU-resident coherence state for fast updates.

    Syncs to CPU CoherenceState only on eval intervals.
    """

    def __init__(self, num_experts: int, device='cuda'):
        self.num_experts = num_experts
        self.device = device

        # Three-timescale EMAs (on GPU)
        self.phi_fast = torch.zeros(num_experts, device=device)
        self.phi_mid = torch.zeros(num_experts, device=device)
        self.phi_slow = torch.zeros(num_experts, device=device)

        # Observation tracking (on GPU)
        self.total_tokens_seen = torch.zeros(num_experts, dtype=torch.long, device=device)
        self.last_update_step = torch.zeros(num_experts, dtype=torch.long, device=device)

        # Clock config
        self.alpha_fast = 0.9
        self.alpha_mid = 0.99
        self.alpha_slow = 0.999

    def update(self, phi_raw: Tensor, active_expert_ids: Tensor, step: int, num_tokens: Tensor):
        """
        Update coherence EMAs on GPU.

        Args:
            phi_raw: [num_active_experts] - raw coherence this step
            active_expert_ids: [num_active_experts] - which experts
            step: current training step
            num_tokens: [num_active_experts] - tokens per expert
        """
        # Three-clock EMA update
        self.phi_fast[active_expert_ids] = (
            self.alpha_fast * self.phi_fast[active_expert_ids]
            + (1 - self.alpha_fast) * phi_raw
        )

        self.phi_mid[active_expert_ids] = (
            self.alpha_mid * self.phi_mid[active_expert_ids]
            + (1 - self.alpha_mid) * phi_raw
        )

        self.phi_slow[active_expert_ids] = (
            self.alpha_slow * self.phi_slow[active_expert_ids]
            + (1 - self.alpha_slow) * phi_raw
        )

        # Observation tracking
        self.total_tokens_seen[active_expert_ids] += num_tokens
        self.last_update_step[active_expert_ids] = step

    def snapshot(self, step: int) -> Dict[int, CoherenceState]:
        """
        Sync to CPU and export CoherenceState snapshot.

        Called every N steps for lifecycle evaluation.
        """
        states = {}
        for expert_id in range(self.num_experts):
            states[expert_id] = CoherenceState(
                expert_id=f"L{self.layer_id}_E{expert_id}",
                layer_id=self.layer_id,
                phi_fast=self.phi_fast[expert_id].item(),
                phi_mid=self.phi_mid[expert_id].item(),
                phi_slow=self.phi_slow[expert_id].item(),
                last_update_step=self.last_update_step[expert_id].item(),
                total_tokens_seen=self.total_tokens_seen[expert_id].item(),
            )
        return states
```

### 2.2 Integration with Training Loop

```python
# In training loop

coherence_buffer = CoherenceBuffer(num_experts=8, device='cuda')

for step in range(num_steps):
    # Forward pass (MoE layer with beta)
    hidden_out, trace = moe_layer(hidden_in, return_trace=True)

    # Compute coherence on GPU
    phi_raw = trace.compute_coherence()  # [num_active_experts]
    active_ids = torch.tensor(trace.active_expert_ids, device='cuda')
    num_tokens = torch.tensor([len(idx) for idx in trace.token_row_indices], device='cuda')

    # Update coherence buffer (GPU)
    coherence_buffer.update(phi_raw, active_ids, step, num_tokens)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Lifecycle evaluation (every 100 steps)
    if step % 100 == 0:
        snapshot = coherence_buffer.snapshot(step)
        lifecycle_decisions = lifecycle_coordinator.evaluate(snapshot, router_state)
        # Log decisions, don't execute yet
```

**Deliverable 2:** Coherence updates every step on GPU. CPU snapshot only every 100 steps. No performance bottleneck.

---

## Step 3: Beta Update Function

**Goal:** Close the loop. Beta responds to coherence.

### 3.1 Simple Update Rule

```python
# chronomoe_v3/router.py

def update_beta(
    router_state: RouterState,
    coherence_snapshot: Dict[int, CoherenceState],
    eta: float = 0.01,
    tau: float = 0.5,
):
    """
    Update beta coefficients based on coherence feedback.

    Simple rule:
    - If phi_slow < tau: reduce beta (expert not earning routing advantage)
    - If phi_slow > tau: increase beta (expert earning routing advantage)

    Args:
        router_state: RouterState to update
        coherence_snapshot: {expert_id: CoherenceState}
        eta: Learning rate
        tau: Coherence threshold (target)
    """
    for expert_id, coh_state in coherence_snapshot.items():
        # Skip if not observed recently
        if not coh_state.is_being_observed:
            continue

        # Compute delta
        delta = eta * (coh_state.phi_slow - tau)

        # Update beta_coeff (scale-free)
        router_state.beta_coeff[expert_id] += delta

    # Clamp to prevent saturation
    router_state.beta_coeff.clamp_(-router_state.k_max, router_state.k_max)
```

### 3.2 Integration

```python
# In training loop, every 100 steps

if step % 100 == 0:
    # Snapshot coherence
    snapshot = coherence_buffer.snapshot(step)

    # Update beta based on coherence
    update_beta(
        router_state=router_state,
        coherence_snapshot=snapshot,
        eta=0.01,
        tau=0.5,
    )

    # Log beta state
    print(f"Beta update: mean={router_state.beta_coeff.mean():.3f}, "
          f"std={router_state.beta_coeff.std():.3f}, "
          f"max={router_state.beta_coeff.max():.3f}")
```

**Deliverable 3:** Beta updates based on phi_slow. Loop is closed. Observe if stable.

---

## Step 4: Bridge Detector Veto

**Goal:** Use JS divergence to modulate beta strength. Prevent Krypto-from-nowhere.

### 4.1 Compute Relevance Scalar

```python
# chronomoe_v3/router.py

def compute_relevance(router_state: RouterState, threshold: float = 0.2) -> float:
    """
    Compute routing relevance based on clean/biased agreement.

    r = 1.0 if disagreement is low (beta is aligned with input)
    r → 0.0 if disagreement is high (beta is fighting input)

    Args:
        router_state: RouterState with disagreement metrics
        threshold: JS divergence threshold for full relevance

    Returns:
        r: Relevance scalar in [0, 1]
    """
    js = router_state.disagreement_js

    if js < threshold:
        return 1.0  # Full relevance
    elif js < threshold * 3:
        # Linear decay
        return 1.0 - (js - threshold) / (threshold * 2)
    else:
        return 0.0  # No relevance (crisis)
```

### 4.2 Modulate Beta Strength

```python
# In router forward(), after computing beta_eff

# Compute relevance
r = compute_relevance(self.router_state, threshold=0.2)

# Modulate beta strength
beta_eff = self.router_state.compute_beta_eff()  # [E]
beta_eff_modulated = r * beta_eff  # Scale by relevance

# Apply modulated bias
z_biased = z_clean + beta_eff_modulated.unsqueeze(0)
```

### 4.3 Logging

```python
if step % 10 == 0:
    r = compute_relevance(router_state)
    print(f"Relevance: {r:.3f} (JS={router_state.disagreement_js:.4f})")
```

**Deliverable 4:** Beta is modulated by routing relevance. High disagreement → beta suppressed automatically.

---

## Step 5: Lifecycle Coordinator (Decisions Only, Dry-Run)

**Goal:** Detect prune candidates. Log decisions. Don't execute yet.

### 5.1 Lifecycle Coordinator

```python
# chronomoe_v3/lifecycle.py

@dataclass
class PruneDecision:
    """Proposed prune action."""
    expert_id: str
    reason: str  # "low_coherence" or "low_observability"
    phi_slow: float
    total_tokens_seen: int
    step: int

class LifecycleCoordinator:
    """
    Reads CoherenceState and RouterState.
    Proposes lifecycle decisions (prune only for now).
    Does NOT execute actions yet (dry-run).
    """

    def __init__(self, config: ChronoConfig):
        self.config = config
        self.decisions: List[PruneDecision] = []

    def evaluate(
        self,
        coherence_snapshot: Dict[int, CoherenceState],
        router_state: RouterState,
        step: int,
    ) -> List[PruneDecision]:
        """
        Evaluate prune candidates.

        Prune if:
        - (Low observability for long) AND (layer not starving)
        OR
        - (Low coherence) AND (observed enough)
        """
        candidates = []

        # Check layer starvation
        layer_coherence = self._compute_layer_coherence(coherence_snapshot)
        layer_is_starving = layer_coherence < 0.5

        for expert_id, coh_state in coherence_snapshot.items():
            # Criterion 1: Low observability (not being used)
            if coh_state.total_tokens_seen < 100 and step > 1000:
                if not layer_is_starving:  # Don't prune if layer needs capacity
                    candidates.append(PruneDecision(
                        expert_id=coh_state.expert_id,
                        reason="low_observability",
                        phi_slow=coh_state.phi_slow,
                        total_tokens_seen=coh_state.total_tokens_seen,
                        step=step,
                    ))

            # Criterion 2: Low coherence (decoherent)
            if coh_state.phi_slow < 0.3 and coh_state.total_tokens_seen > 1000:
                candidates.append(PruneDecision(
                    expert_id=coh_state.expert_id,
                    reason="low_coherence",
                    phi_slow=coh_state.phi_slow,
                    total_tokens_seen=coh_state.total_tokens_seen,
                    step=step,
                ))

        # Log decisions (dry-run)
        for decision in candidates:
            self.decisions.append(decision)
            print(f"[DRY-RUN] Step {step}: Prune {decision.expert_id} "
                  f"(reason={decision.reason}, phi_slow={decision.phi_slow:.3f})")

        return candidates

    def _compute_layer_coherence(self, snapshot: Dict[int, CoherenceState]) -> float:
        """Weighted average of phi_slow across experts."""
        phi_values = torch.tensor([s.phi_slow for s in snapshot.values()])
        weights = torch.tensor([s.total_tokens_seen for s in snapshot.values()], dtype=torch.float32)
        weights = weights / weights.sum()
        return (phi_values * weights).sum().item()
```

### 5.2 Integration

```python
# In training loop

lifecycle_coordinator = LifecycleCoordinator(config)

for step in range(num_steps):
    # ... forward, backward ...

    # Lifecycle evaluation (every 100 steps)
    if step % 100 == 0:
        snapshot = coherence_buffer.snapshot(step)

        # Update beta
        update_beta(router_state, snapshot)

        # Evaluate lifecycle (dry-run only)
        prune_decisions = lifecycle_coordinator.evaluate(snapshot, router_state, step)

        # Log but don't execute
        if prune_decisions:
            print(f"Step {step}: {len(prune_decisions)} prune candidates")
```

**Deliverable 5:** Lifecycle detects prune candidates. Logs decisions. Does not execute.

---

## Testing the Vertical Slice

### Minimal Harness

```python
# experiments/phase2_vertical_slice.py

import torch
import torch.nn as nn
from chronomoe_v3 import CoherenceBuffer, RouterState, LifecycleCoordinator
from chronomoe_v3.router import ChronoRouter, update_beta, compute_relevance

# Simple MoE layer
class SimpleMoELayer(nn.Module):
    def __init__(self, d_model=128, num_experts=8, top_k=2):
        super().__init__()
        self.router = ChronoRouter(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x, return_trace=False):
        # Router (with beta)
        top_k_probs, top_k_indices, z_clean, z_biased = self.router(x, self.top_k)

        # Dispatch and compute (simplified)
        # ... expert forward passes ...

        if return_trace:
            trace = self._build_trace(...)
            return output, trace
        return output

# Training loop
def test_vertical_slice():
    model = SimpleMoELayer(d_model=128, num_experts=8)
    coherence_buffer = CoherenceBuffer(num_experts=8)
    lifecycle = LifecycleCoordinator(config)

    for step in range(1000):
        x = torch.randn(32, 128, device='cuda')

        # Forward
        output, trace = model(x, return_trace=True)

        # Update coherence
        phi_raw = trace.compute_coherence()
        coherence_buffer.update(phi_raw, ...)

        # Loss and backward
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Lifecycle (every 100 steps)
        if step % 100 == 0:
            snapshot = coherence_buffer.snapshot(step)
            update_beta(model.router.router_state, snapshot)
            decisions = lifecycle.evaluate(snapshot, model.router.router_state, step)

        # Log
        if step % 10 == 0:
            print(f"Step {step}: JS={model.router.router_state.disagreement_js:.4f}")

if __name__ == "__main__":
    test_vertical_slice()
```

---

## Success Criteria

After implementing all 5 steps, we should observe:

1. **RouterState exists** ✓ Beta applied, disagreement logged
2. **Coherence updates on GPU** ✓ Fast, no CPU bottleneck
3. **Beta responds to coherence** ✓ Low phi_slow → beta decreases
4. **Bridge detector works** ✓ High disagreement → beta suppressed
5. **Lifecycle detects prune candidates** ✓ Logs decisions, doesn't execute

**If all 5 work:** The loop is closed and stable. Ready for Phase 3 (RoleState, split/merge).

**If any fail:** Debug that step in isolation before proceeding.

---

## Implementation Order

1. **Day 1:** Step 1 (RouterState + beta application)
2. **Day 2:** Step 2 (Coherence on GPU)
3. **Day 3:** Step 3 (Beta update function)
4. **Day 4:** Step 4 (Bridge detector)
5. **Day 5:** Step 5 (Lifecycle dry-run)

**Deliverable:** Working vertical slice in `experiments/phase2_vertical_slice.py`

---

**Status:** Implementation plan complete. Ready to start coding.
