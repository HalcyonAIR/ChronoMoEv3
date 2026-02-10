# ChronoMoE Integration for swiss-ai/MoE

Lifecycle-aware MoE layers with fixed-width routing architecture.

## Quick Start

```python
from chronomoe_integration import ChronoMoE, ProbationConfig
from gpt import GPTConfig  # Your swiss-ai config
from your_mlp import MLP   # Your MLP implementation

# Create config
config = GPTConfig(
    n_embd=256,
    moe_num_experts=4,
    moe_num_experts_per_tok=2,
    moe_softmax_order="softmax_topk",
)

# Create lifecycle-aware MoE layer
layer = ChronoMoE(
    config=config,
    mlp=MLP,
    layer_id=0,
    max_experts=8,  # Allow doubling
    probation_config=ProbationConfig(
        duration_steps=30,
        min_tokens=1500,
        initial_boost=1.0,
    ),
)

# Forward pass (update current_step before each forward)
layer.current_step = step
output, metadata = layer(inputs)

# Spawn new expert (defaults to blank strategy)
new_id = layer.spawn_expert(parent_id=0, optimizer=optimizer)  # Uses blank by default

# Explicit clone spawning (opt-in only, logs warning)
new_id = layer.spawn_expert(parent_id=0, strategy="clone", optimizer=optimizer)

# Check probation graduations
layer.check_probation_graduations(in_comfort_band=True)

# Prune expert
layer.prune_expert(expert_id=2)
```

## Architecture

ChronoMoE integration follows a **two-layer design** with strict boundary enforcement:

### Layer 1: Lifecycle Mechanics (✅ Complete)

Infrastructure for structural changes without retraining the router.

**Components:**
- **Fixed-width routing:** Router outputs `max_experts` logits from step 0
- **Pre-allocated experts:** All expert slots created upfront (mask-based activation)
- **Probation mechanism:** Temporary boost for spawned experts (prevents cold-start death)
- **Stress bands + calm gates:** Non-bypassable lifecycle gating (COMFORT/STRAIN/PANIC)
- **Blank spawn default:** Random initialization + probation (clone as opt-in)

**Status:** 18/18 tests passing. Ready for production use.

### Layer 2: Decision Intelligence (🟡 Incremental Integration)

Autonomous decision-making driven by coherence, free energy, and evidence.

**Boundary:** `chronomoe_integration/controller.py` is the ONLY interface to ChronoMoEv3 logic.

**Integration milestones:**
- **Milestone A (Issue #3A):** Coherence tracking (Phase 1) - logging only
- **Milestone B (Issue #3B):** Bimodality detection (Phase 3) - logging only
- **Milestone C (Issue #3C):** Free energy sensor (Phase 4) - logging only
- **Milestone D (Issue #3D):** Autonomous SPAWN/PRUNE triggers
- **Milestone E (Issue #3E):** SPLIT/MERGE operations (FUTURE)

**Current status:** Milestones A-E proposed. Layer 1 complete. Layer 2 being integrated incrementally.

**Philosophy:** Thin interface layer, incremental signal ports, prove each milestone before next.

See tracking issues: [ChronoMoEv3/TRACKING_ISSUES.md](../../ChronoMoEv3/TRACKING_ISSUES.md)

---

## Layer 1: Lifecycle Mechanics (COMPLETE)

### Fixed-Width Routing

**Key Principle:** Router action space is fixed-width from step 0. Lifecycle operations change masks, not tensor shapes.

**Components:**
1. **FixedWidthRouter:** Outputs `max_experts` logits (not `moe_num_experts`)
2. **Pre-allocated Experts:** `ModuleList` with `max_experts` slots from initialization
3. **Active Mask:** Boolean mask indicating selectable experts
4. **ExpertRegistry:** Tracks lifecycle states and probation

### Lifecycle States

- **PROBATION:** Newly spawned, receiving temporary router boost
- **ACTIVE:** Graduated, normal routing
- **ARCHIVED:** Pruned, hard-masked (unreachable)

### Spawn Strategies

**Default: Blank Initialization** (Recommended)
- New expert uses random initialization (pre-allocated weights)
- Combined with probation boost to ensure learning signal
- This is the standard, proven approach

**Opt-In: Clone from Parent**
- New expert copies parent weights
- Requires explicit `strategy="clone"` parameter
- Logs warning when used (to prevent silent defaults)
- Can be disabled entirely with `ProbationConfig.blank_only()`

**Enforcement:**
```python
# Default behavior (recommended)
layer.spawn_expert(parent_id=0)  # Uses blank + probation

# Explicit clone (opt-in, logs warning)
layer.spawn_expert(parent_id=0, strategy="clone")

# Enforce blank-only (no clone allowed)
layer = ChronoMoE(..., probation_config=ProbationConfig.blank_only())
layer.spawn_expert(parent_id=0, strategy="clone")  # Raises ValueError
```

### Probation Mechanism

Prevents cold-start death spiral for newly spawned experts (both blank and clone):

1. Spawn creates expert in PROBATION state
2. Router applies decaying logit boost
3. Token accumulation tracked
4. After N steps: graduate if tokens >= threshold, else prune

**Validated Parameters:**
- Duration: 30 steps
- Min tokens: 1500
- Initial boost: +1.0 logits
- Decay: Linear to 0

## API Reference

### ChronoMoE

```python
class ChronoMoE(nn.Module):
    def __init__(
        self,
        config,              # swiss-ai GPTConfig
        mlp,                 # MLP class (not instance)
        layer_id: int,       # Layer index
        max_experts: int,    # Max expert capacity (default: 2x initial)
        probation_config: ProbationConfig,
    )

    def forward(inputs: Tensor) -> Tuple[Tensor, Dict]:
        """
        Returns:
            (output, metadata) where metadata contains:
            - router_logits: [B*T, max_experts]
            - selected_experts: [B*T, top_k]
            - expert_utilization: [max_experts]
        """

    def spawn_expert(
        parent_id: int,
        strategy: str,       # "blank" or "clone"
        optimizer: Optional[Optimizer],
    ) -> int:
        """Spawn new expert, returns expert ID"""

    def prune_expert(expert_id: int) -> None:
        """Prune expert (deactivate in mask)"""

    def check_probation_graduations(in_comfort_band: bool) -> None:
        """Check probation experts for graduation/failure"""
```

### ProbationConfig

```python
@dataclass
class ProbationConfig:
    enabled: bool = True
    duration_steps: int = 30
    min_tokens: int = 1500
    initial_boost: float = 1.0
    decay_type: str = "linear"
    share_cap: float = 0.05          # Not yet enforced
    require_comfort_band: bool = True
```

### ExpertRegistry

```python
class ExpertRegistry:
    @property
    def num_active(self) -> int:
        """Count of active experts"""

    @property
    def capacity_remaining(self) -> int:
        """Remaining spawn capacity"""

    def status_summary(self) -> str:
        """Human-readable status"""

    def get_probation_boost(expert_id: int, current_step: int) -> float:
        """Get decaying boost for expert"""
```

## Training Example

```python
import torch
from chronomoe_integration import ChronoMoE, ProbationConfig

# Setup
model = GPT(config)  # Replace MoE layers with ChronoMoE
optimizer = torch.optim.AdamW(model.parameters())

# Training loop
for step in range(max_steps):
    # Update current_step in all layers BEFORE forward
    for layer in model.get_chrono_layers():
        layer.current_step = step

    # Forward pass
    logits, loss, metadata = model(x, y)

    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Lifecycle operations (every N steps)
    if step % 10 == 0:
        for layer in model.get_chrono_layers():
            # Check probation graduations
            layer.check_probation_graduations(in_comfort_band=True)

            # Example spawn trigger (implement your logic)
            if should_spawn(layer):
                layer.spawn_expert(parent_id=0, strategy="blank", optimizer=optimizer)

            # Example prune trigger (implement your logic)
            for expert_id in get_prunable_experts(layer):
                layer.prune_expert(expert_id)
```

## Validation

Run tests to verify integration:

```bash
python3 test_chronomoe.py
```

Expected output:
```
✓ TEST 1 PASSED: Fixed-width initialization correct
✓ TEST 2 PASSED: Forward pass correct
✓ TEST 3 PASSED: Spawn activates slot correctly
✓ TEST 4 PASSED: Probation boost working
✓ TEST 5 PASSED: Hard mask working (no ghost routing)
✓ TEST 6 PASSED: Probation graduation mechanism works

✓ ALL TESTS PASSED
```

## Design Principles

### 1. Fixed-Width from Step 0

Router outputs `max_experts` logits from initialization, not `moe_num_experts`.

**Why:** Prevents IndexError when spawning expert IDs beyond initial count.

### 2. Pre-Allocated Expert Slots

All expert slots created upfront, spawn activates existing slots.

**Why:** No tensor shape changes mid-flight, stable optimizer state.

### 3. Hard Mask Sovereignty

Inactive experts get -inf logits before softmax → zero selection probability.

**Why:** Pruned experts truly unreachable, no ghost routing.

### 4. Probation Boosts Applied Before Masking

Order: compute logits → apply boost → hard mask → softmax

**Why:** Probation experts must be selectable, boost must affect softmax.

## Troubleshooting

### Spawned expert gets 0 tokens

**Check:**
1. Router outputs `max_experts` logits (not `moe_num_experts`)
2. Probation boost applied in forward pass
3. Boost applied BEFORE hard mask
4. `current_step` updated before forward pass

### IndexError on spawn

**Check:**
1. `max_experts > moe_num_experts`
2. Expert slots pre-allocated to `max_experts`
3. Router dimension = `max_experts`

### All probation experts graduate

**Tune parameters:**
- Reduce `initial_boost` (e.g., 1.0 → 0.5)
- Reduce `duration_steps` (e.g., 30 → 20)
- Increase `min_tokens` (e.g., 1500 → 2000)

### All probation experts fail

**Tune parameters:**
- Increase `initial_boost` (e.g., 1.0 → 2.0)
- Increase `duration_steps` (e.g., 30 → 50)
- Reduce `min_tokens` (e.g., 1500 → 1000)

## References

- **Graduation Report:** `CHRONOMOE_GRADUATION_REPORT.md`
- **nanoMoE Validation:** `ChronoMoEv3/FIXED_WIDTH_SUCCESS.md`
- **Original Implementation:** `ChronoMoEv3/integration/nanomoe_integration.py`

## License

Same as swiss-ai/MoE (see repository LICENSE).

## Next Steps

See open tracking issues for post-graduation work:

**Issue #1:** [Port stress bands + calm gating](../../ChronoMoEv3/TRACKING_ISSUES.md#issue-1-port-stress-bands--calm-gating-into-swiss-aimoe-integration)
- Integrate comfort/strain/panic bands
- Enforce calm gates for lifecycle operations
- Add validation tests

**Issue #2:** [Blank spawn as default](../../ChronoMoEv3/TRACKING_ISSUES.md#issue-2-implement-blank-spawn-with-probation-as-default-keep-clone-seeded-as-optional)
- Make blank+probation the standard spawn behavior
- Keep clone-seeded as optional fallback
- Add config to enforce blank-only

## Contact

Questions? See ChronoMoEv3 repository or swiss-ai/MoE maintainers.
