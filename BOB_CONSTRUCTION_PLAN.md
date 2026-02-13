# Bob: Construction, Testing & Development Plan

**Status**: Active development plan
**Authors**: Jeff Colhoun (HalcyonAIR), Halcyon (GPT), Claude (Anthropic)
**Date**: February 2026
**Canonical spec**: CONSEQUENCE_SUBSTRATE_V1.md

---

## 1. What Bob Is

Bob is a **control plane** around Mixtral. Not a model fork. Not a product wrapper.

Bob intercepts MoE routing decisions, accumulates consequence, and learns when to take the cheap path. The model does the thinking. Bob decides how much thinking is necessary.

```
                    Bob (control plane)
                    ┌─────────────────────────────────┐
                    │                                 │
Input ──→ Context ──→ Commitment Cache ──→ Gate ──→ Decision
  │       Classifier   (motif library)    (compound)    │
  │                                                     │
  │       ┌─────────┐                          ┌────────┴────────┐
  │       │ Clocks  │                          │                 │
  │       │ f/m/s   │◄── Telemetry ◄──────────┤  Cheap Path     │
  │       └────┬────┘                          │  (1 expert)     │
  │            │                               │       OR        │
  │            ▼                               │  Expensive Path │
  │       Governance                           │  (full top-k)   │
  │       State Machine                        └────────┬────────┘
  │                                                     │
  └─────────────── Mixtral Backend ─────────────────────┘
                   (frozen weights, exposed router)
```

Bob owns:
- When to shortcut (compound gate)
- What to remember (motif library + scars)
- When to trust itself (governance state)
- When to doubt itself (quench protocol)

Bob does NOT own:
- Model weights
- Training loop
- Gradient computation
- Token generation

## 2. Backend Selection

### Primary Backend: Mixtral

| Property | Value |
|----------|-------|
| Architecture | 8 experts per layer, top-2 routing |
| Total params | 46.7B |
| Active params/token | 12.9B |
| MoE layers | 32 (all transformer layers) |
| Router | Linear projection, softmax, top-k selection |
| License | Apache 2.0 |
| Shared experts | None (clean routing, no bypass) |
| Router logits | Fully exposed via HuggingFace |

**Why Mixtral:**
- Clean MoE routing. No hidden controller magic. No shared experts.
- Open weights. Apache 2.0. Community tooling.
- Already validated architecture: our 16E top-2 scaling stress test (Section 7.3 of V1.md) used Mixtral-style routing and showed -15.28% loss improvement with 100% seed robustness.
- The swiss-ai/MoE codebase references Mistral source directly. Same family.
- 8 experts top-2 is the configuration where our commitment cache stayed flat (Section 7.6). This is a feature: if Bob can make it work on the hard case, it works anywhere.

**Development backend**: swiss-ai/MoE (our existing GPT-2 style MoE, ~1-10M params). Same routing mechanics as Mixtral, instant iteration. All logic developed here first, then validated on real Mixtral.

**Runtime**: Mixtral-8x7B quantized (4-bit AWQ/GPTQ, ~25GB) on single GPU. Or cloud inference (Together, Fireworks, etc.) during early development when we only need router logits, not weight surgery.

### The 8E Top-2 Problem

Our experiments showed 8 experts top-2 doesn't form routing commitments (Section 7.6). This is the exact Mixtral configuration. Three paths to address this:

1. **Top-4 routing override**: Force top-4 during Bob's observation window. Higher branching factor → more routing curvature → motif formation. Fall back to top-2 for production inference.
2. **Task complexity**: Our synthetic tasks (shift-1/3/7, double) may be too simple. Real language tasks have richer structure that may force specialisation even with top-2.
3. **Layer-selective commitment**: Not all 32 layers need to commit. Some layers may naturally specialise (early = syntactic, late = semantic). Bob tracks per-layer commitment independently.

This is the first experiment Bob needs to run. Not deferred — it's Phase 1 validation gate.

---

## 3. Architecture

### 3.1 Package Structure

Two packages. Boring names. Doing their job.

```
bob_core/                              # Control plane (backend-agnostic)
├── __init__.py                        # Public API
├── substrate.py                       # Main orchestrator (logic: decide, gate, update)
│
├── store/                             # Substrate store (persistence, separate from logic)
│   ├── __init__.py
│   ├── motifs.py                      # MotifEntry, MotifLibrary (what we remember about patterns)
│   ├── scars.py                       # ScarRegistry (what we remember about failures)
│   ├── epochs.py                      # EpochStore (what we remember about phases)
│   └── serialisation.py               # Save/load/migrate (file-backed initially)
│
├── cache/                             # Consequence accumulation (logic over the store)
│   ├── commitment.py                  # CommitmentCache v2 (reads/writes store)
│   └── context.py                     # ContextClassifier (online clustering)
│
├── clocks/                            # Temporal processing
│   ├── fast.py                        # τ_f: token/micro-batch (reflex)
│   ├── medium.py                      # τ_m: session/window (pressure-temperature)
│   └── slow.py                        # τ_s: multi-session (scar accumulation)
│
├── gates/                             # Decision gates
│   ├── compound.py                    # Structural gate (stability × debt × survival)
│   ├── crystallisation.py             # 5-gate freeze protocol
│   └── quench.py                      # Overlap independence testing
│
├── governance/                        # State machine
│   ├── machine.py                     # GovernanceStateMachine (adapted)
│   ├── detector.py                    # Three-state convergence detector (adapted)
│   └── contamination.py              # Credit contamination + cooling windows
│
├── telemetry/                         # Logging & measurement
│   ├── traces.py                      # DecisionTrace v2 — THE unit of truth
│   ├── pareto.py                      # Pareto frontier tracking
│   └── pipeline.py                    # Telemetry pipeline (every decision, no exceptions)
│
├── ladders/                           # Controlled experiments
│   ├── base.py                        # DifficultyLadder protocol
│   ├── synthetic.py                   # Synthetic task classes
│   ├── wikitext.py                    # WikiText-2 natural ladder
│   └── runner.py                      # Experiment runner + baselines
│
└── tests/                             # Test suite
    ├── test_store.py                  # Store persistence + migration
    ├── test_cache.py
    ├── test_clocks.py
    ├── test_gates.py
    ├── test_governance.py
    ├── test_telemetry.py
    ├── test_ladders.py
    ├── test_substrate.py              # Integration tests
    └── conftest.py                    # Shared fixtures

backends/                              # Backend adapters (SEPARATE from bob_core)
├── __init__.py
├── adapter.py                         # BackendAdapter protocol (THE contract)
├── types.py                           # LayerSnapshot, ForwardResult, MotifSpec, OverlapKind
├── swiss_adapter.py                   # swiss-ai/MoE (development)
├── mixtral_adapter.py                 # Mixtral-8x7B (primary validation)
└── tests/
    ├── test_adapter_contract.py       # Contract compliance tests (run against ANY adapter)
    ├── test_swiss.py
    └── test_mixtral.py
```

**Three-layer separation** (Halcyon's correction):

| Layer | Package | Responsibility |
|-------|---------|---------------|
| **Adapter** | `backends/` | "What happened in the model" — raw observations |
| **Bob core** | `bob_core/` (minus `store/`) | "What to do about it" — gates, governance, clocks, decisions |
| **Substrate store** | `bob_core/store/` | "What we remember" — motifs, scars, epochs, persistence |

Logic and persistence never tangle. Every new metric is a new field in the store, not a migration drama. The store is file-backed initially, with its own tests and its own serialisation module.

**Rule**: Everything in `bob_core/` talks ONLY to `BackendAdapter`. No Mixtral imports. No swiss-ai/MoE imports. No exceptions. If you catch yourself writing `from transformers import MixtralForCausalLM` inside `bob_core/`, you've already lost.

Future backends (DeepSeek, OLMoE, whatever comes next) are files in `backends/`. Bob never knows they exist.

### 3.2 Backend Contract

The contract is what Bob needs from any MoE backend. Minimal and ruthless.

**Per forward pass, per layer, Bob gets a snapshot:**

```python
@dataclass
class LayerSnapshot:
    """Raw observation from one layer of one forward pass."""
    layer_id: int                          # Stable identifier
    router_scores: Tensor                  # Pre-top-k scores/probabilities [B*T, num_experts]
    selected_experts: Tensor               # Expert indices chosen [B*T, top_k]
    routing_weights: Tensor                # Weights assigned to selected experts [B*T, top_k]
    expert_usage: Tensor                   # Per-expert usage vector [num_experts] (token counts or weight sums)
```

**Per forward pass, per decision, Bob gets a context embedding:**

```python
def get_context_embedding(self) -> Tensor:
    """One canonical context embedding per decision. Shape: [B*T, D].

    Adapter defines semantics: could be penultimate router state aggregated
    across layers, first MoE block state, final router state — whatever
    makes sense for this backend. Bob doesn't care as long as it's consistent
    within a backend and versioned.
    """
    ...

@property
def adapter_version(self) -> str:
    """Version string. If context embedding semantics change, version bumps."""
    ...
```

This decouples the context-class conditioner from Mixtral's internal naming. Two backends will disagree on what "router embedding" is. Bob doesn't care — it gets a consistent vector and clusters on it.

**Multi-layer motif replay:**

```python
@dataclass
class MotifSpec:
    """What to execute for cheap path. Can span multiple layers."""
    layers: Dict[int, LayerMotif]  # layer_id → (expert_ids, weights)

@dataclass
class LayerMotif:
    expert_ids: Tuple[int, ...]
    weights: Tuple[float, ...]
```

The cheap path is "replay a decision structure across layers", not "nudge one layer". Single-layer is a convenience case where `MotifSpec.layers` has one entry.

**All-or-nothing execution**: Cheap-path is atomic. The motif fires across ALL specified layers or NONE. There is no layer-progressive design where Bob fires layer 1's motif, checks confidence, then decides whether to continue. This is deliberate: a layer-progressive design would change the cost model (partial motif execution is a third cost tier between cheap and expensive) and the gate semantics (per-layer confidence thresholds would need calibration). The compound gate decides once; the motif executes fully or not at all. This matches V1.md Section 3 decision flow.

**Overlap semantics per backend:**

```python
class OverlapKind(Enum):
    NONE = "none"              # No overlap state available
    KV_CACHE = "kv_cache"      # KV cache is the overlap
    CONTROLLER_STATE = "controller_state"  # External controller state
    BOTH = "both"              # KV cache + controller state

@property
def supports_overlap(self) -> bool: ...

@property
def overlap_kind(self) -> OverlapKind: ...
```

If `overlap_kind == NONE`, Bob can still run, but freeze and scar promotion hard-disable. That's the sovereignty rule made executable: you can't earn permanence without being able to test independence.

**Full contract:**

```python
class BackendAdapter(Protocol):
    """What Bob needs from any MoE backend. No more, no less."""

    # --- Identity ---
    @property
    def num_experts(self) -> int: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def top_k(self) -> int: ...
    @property
    def adapter_version(self) -> str: ...

    # --- Overlap capability ---
    @property
    def supports_overlap(self) -> bool: ...
    @property
    def overlap_kind(self) -> OverlapKind: ...

    # --- Observation ---
    def forward(self, inputs: Tensor, **kwargs) -> ForwardResult:
        """Standard forward pass. Populates layer snapshots."""
        ...

    def get_layer_snapshot(self, layer_id: int) -> LayerSnapshot:
        """Retrieve snapshot for a specific layer from the last forward pass."""
        ...

    def get_all_snapshots(self) -> List[LayerSnapshot]:
        """All layer snapshots from the last forward pass."""
        ...

    def get_context_embedding(self) -> Tensor:
        """One canonical context embedding per decision. [B*T, D].
        Adapter-defined semantics. Consistent within backend. Versioned."""
        ...

    # --- Intervention ---
    def forward_with_motif(self, inputs: Tensor,
                           motif: MotifSpec) -> ForwardResult:
        """Force specific expert routing across specified layers (cheap path).
        Unspecified layers route normally."""
        ...

    def forward_counterfactual(self, inputs: Tensor) -> ForwardResult:
        """Run same input with overlap state suppressed (quench test).
        What 'suppressed' means depends on overlap_kind.
        Returns full snapshots for comparison.
        Raises NotImplementedError if supports_overlap is False."""
        ...

    # --- Cost ---
    def get_expert_invocations(self) -> int:
        """Total expert invocations from last forward pass."""
        ...

    def get_tokens_processed(self) -> int:
        """Total tokens processed from last forward pass."""
        ...
```

**DecisionTrace is Bob's unit of truth, not the adapter's.**

The adapter provides raw ingredients (snapshots, context embedding, cost counts). Bob constructs the `DecisionTrace` in `bob_core/telemetry/traces.py`. The trace is the atomic artifact — the thing you replay, diff, and audit. Every decision produces one trace. No exceptions.

```python
# In bob_core — NOT in backends
trace = DecisionTrace.from_decision(
    snapshots=adapter.get_all_snapshots(),
    context_embedding=adapter.get_context_embedding(),
    gate_signals=compound_gate.last_signals,
    governance_state=governance.current_state,
    path_taken="cheap" or "full",
    expert_invocations=adapter.get_expert_invocations(),
    tokens_processed=adapter.get_tokens_processed(),
    motif_id=motif.id if cheap else None,
    outcome_quality=loss,
    ...  # Full V1.md Appendix C schema
)
```

**Contract compliance test**: Any new adapter must pass `test_adapter_contract.py` which verifies all shapes, all methods, all edge cases, all overlap semantics. If the contract test doesn't pass, the adapter doesn't ship.

**Backend roadmap:**
- **Now**: `swiss_adapter.py` (development iteration — build and test here)
- **Next**: `mixtral_adapter.py` (primary validation target)
- **Later**: `deepseek_adapter.py` (when needed)
- **Never**: GPT adapter (can't expose routing, and that's fine)

### 3.3 What Gets Reused from chronomoe_integration

| Component | Source | Reuse Strategy |
|-----------|--------|----------------|
| CommitmentCache | `commitment_cache.py` | Evolve into `bob_core/cache/commitment.py`: add recency weighting, scar associations, per-signal thresholds |
| GovernanceStateMachine | `convergence.py:779` | Adapt into `bob_core/governance/machine.py`, same logic |
| Three-state detector | `convergence.py` | Adapt into `bob_core/governance/detector.py` |
| MotifEntry, DecisionTrace | `commitment_cache.py` | Evolve: extend fields per V1.md schema |
| routing_geometry tools | `routing_geometry.py` | Import directly for diagnostics |
| MoE class | `moe.py` | Wrap via `backends/swiss_adapter.py` |
| GPTBase | `gpt.py` | Use as-is for development model |
| All existing tests | `tests/` | Keep passing. Green invariant. |

**Isolation rule**: `bob_core/` never imports from `chronomoe_integration/` directly. Code is extracted and evolved into Bob's own modules. The original `chronomoe_integration/` remains untouched as the reference implementation.

### 3.4 What Gets Built New

| Component | Why New |
|-----------|---------|
| Three-clock keepers (fast/medium/slow) | Not in current codebase. V1.md Section 2.1. |
| Quench protocol | Not implemented. V1.md Section 8.7-8.8. |
| Credit contamination | Not implemented. V1.md Section 8.6. |
| Cooling windows | Not implemented. V1.md Section 8.6. |
| Scar registry + epoch review | Scars exist conceptually but no persistent store. V1.md Section 2.6. |
| Crystallisation gates (5-gate) | Designed, not implemented. V1.md Section 2.9. |
| Context class conditioner | Current: known classes. Need: online clustering. V1.md Section 2.5. |
| MixtralAdapter | New backend integration. |
| Difficulty ladder framework | Current: ad-hoc experiments. Need: structured protocol. V1.md Section 4.5. |
| Pareto evaluation framework | Current: manual analysis. Need: automated tracking. V1.md Section 4.4. |
| Full telemetry pipeline | Current: trace lists. Need: streaming, per-decision, no exceptions. V1.md Appendix C. |

---

## 4. Development Phases

### Phase 0: Foundation (Inference-First)

**Goal**: `bob_core` and `backends` packages exist. BackendAdapter contract is locked. swiss_adapter passes contract test. Bob runs in **pure inference mode** — no training loop required.

**Why inference-first**: Bob is a control plane. The commitment cache, compound gate, and governance can all work at inference time. Training integration (detector, pressure coupling) comes in Phase 2 when clocks are introduced. This means:
- Phase 0 is clean: forward pass → observe routing → build motif library → gate decisions
- Bob works with frozen models (the primary use case for Mixtral)
- The adapter contract doesn't need gradient access
- Telemetry works immediately

**Deliverables**:
1. `bob_core/` package with `__init__.py`
2. `bob_core/store/` — substrate store skeleton (MotifLibrary, ScarRegistry stubs, file-backed serialisation)
3. `backends/` package with `adapter.py` (BackendAdapter protocol + LayerSnapshot + MotifSpec + OverlapKind)
4. `backends/types.py` — shared types used by both Bob and adapters
5. `backends/swiss_adapter.py` wrapping `moe.py` MoE class
6. `backends/tests/test_adapter_contract.py` — contract compliance test suite (runs against any adapter)
7. `backends/tests/test_swiss.py` — swiss-specific smoke test
8. `bob_core/substrate.py` skeleton: `BobSubstrate` class with `observe()` and `decide()` methods
9. Smoke test: create adapter → forward pass → extract LayerSnapshot → verify all fields → construct DecisionTrace

**Test gate**:
```
- swiss_adapter passes full contract compliance test
- LayerSnapshot.router_scores has shape [B*T, num_experts]
- LayerSnapshot.selected_experts matches actual routing decisions
- LayerSnapshot.expert_usage sums correctly
- get_context_embedding() returns [B*T, D] with consistent semantics
- adapter_version is a non-empty string
- supports_overlap and overlap_kind are correctly reported
- forward_with_motif(MotifSpec) executes only specified experts at specified layers
- forward_counterfactual() works if supports_overlap, raises if not
- get_expert_invocations() returns correct count for cheap vs expensive path
- DecisionTrace constructed from adapter ingredients has all V1.md Appendix C fields
- Store serialisation round-trips correctly (save → load → identical)
- All existing chronomoe_integration tests still pass
```

**Estimated scope**: ~600 lines code, ~400 lines tests

---

### Phase 1: Core Decision Loop
**Goal**: Bob makes cheap/expensive decisions on swiss-ai/MoE with known context classes. Inference-mode only (no training loop, no gradient coupling).

This is the commitment cache experiment rebuilt as proper Bob infrastructure. The commitment cache reads and writes the substrate store. The compound gate reads the store and decides. The trace is constructed in bob_core, not the adapter.

**Deliverables**:
1. `bob_core/store/motifs.py` — MotifEntry v2, MotifLibrary with recency-weighted survival
   - Fix survival inflation (V1.md 8.1): old successes decay
   - Scar association tracking (which scars constrain which motifs)
   - MotifSpec generation for multi-layer replay
2. `bob_core/cache/commitment.py` — CommitmentCache v2
   - Reads/writes the substrate store (not its own state)
   - Per-context-class motif lookup and update
3. `bob_core/gates/compound.py` — Compound structural gate
   - Routing stability (cosine similarity of usage vectors, V1.md 5.1)
   - Debt level (scar count + recent failures, V1.md 5.2)
   - Motif survival (recency-weighted, V1.md 5.3)
   - All three must independently pass (V1.md 5.4) — not just multiplicative
4. `bob_core/governance/machine.py` — GovernanceStateMachine adapted from `convergence.py`
5. `bob_core/governance/detector.py` — Three-state detector adapted from `convergence.py`
6. `bob_core/telemetry/traces.py` — DecisionTrace v2 (full V1.md Appendix C schema)
   - Constructed in bob_core from adapter raw ingredients
   - The atomic artifact: replay, diff, audit
7. `bob_core/substrate.py` — Full `observe() → decide() → update()` loop

**Test gate**:
```
- Compound gate passes only when ALL THREE signals pass independently
- Gate correctly refuses under DRIFT governance
- Gate threshold lowers 0.7× under SETTLEMENT
- Recency weighting causes old successes to decay
- Motif survival inflation does NOT occur over 1000+ steps
- Decision traces match V1.md Appendix C schema exactly
- Store round-trips: save motif library → load → identical state
- DecisionTrace constructed from adapter ingredients, not adapter-provided
- End-to-end: 4-class synthetic task, cheap path activates, cost drops
- Replication: commitment_cache_experiment.py results reproduced through Bob
```

**Estimated scope**: ~900 lines code, ~600 lines tests

**Validation experiment**: Rerun commitment cache experiment through Bob. Same 4 classes, same drift, same baselines. Results must match within 5%: ~37% cost reduction at 4E top-2, ~0% at 8E top-2. If they don't match, Bob has a bug.

---

### Phase 2: Three Clocks
**Goal**: Bob processes signals at three timescales. Fast clock handles reflexes. Medium clock tracks session pressure. Slow clock accumulates scars.

**Deliverables**:
1. `bob_core/clocks/fast.py` — FastClock (τ_f)
   - Token/micro-batch timescale
   - Tracks: routing jitter, per-decision gate signals, reflex candidates
   - EMA half-life: ~10 steps
   - Output: reflex crystallisation candidates (V1.md 2.1)
2. `bob_core/clocks/medium.py` — MediumClock (τ_m)
   - Session/window timescale
   - Tracks: routing entropy trajectory, loss trajectory, pressure-temperature coupling
   - EMA half-life: ~100 steps
   - Output: governance state transitions, motif library updates
   - Influence weight scales with governance state (V1.md 8.9)
3. `bob_core/clocks/slow.py` — SlowClock (τ_s)
   - Multi-session/deployment timescale
   - Tracks: scar accumulation, epoch boundaries, identity constraints
   - EMA half-life: ~1000 steps
   - Output: scar formation events, epoch boundary declarations
   - Only active during EQUILIBRIUM/SETTLEMENT (V1.md 8.9)
4. `bob_core/substrate.py` updated: clocks integrated into `observe()` loop

**Test gate**:
```
- Fast clock EMA responds to individual decisions
- Medium clock EMA smooths over session-level noise
- Slow clock EMA only updates during EQUILIBRIUM/SETTLEMENT
- Slow clock influence suspended during DRIFT (state preserved, outputs not applied)
- Medium clock influence weight scales with governance state
- Clock signals are independent (no leakage between timescales without deliberate coupling)
- Integration: all three clocks run simultaneously, substrate aggregates correctly
```

**Estimated scope**: ~600 lines code, ~400 lines tests

---

### Phase 3: Quench & Contamination
**Goal**: Bob can test its own trustworthiness. Cheap decisions are not allowed to certify truth.

This is where Bob learns to doubt itself.

**Deliverables**:
1. `bob_core/gates/quench.py` — Quench protocol
   - Fast-clock quench: force one decision with scratchpad=0 (V1.md 8.7 — quench points within Phase Contamination safeguard)
   - Medium-clock quench: widen exploration at topic shifts (V1.md 8.7)
   - Slow-clock quench: require standalone success before scar/freeze (V1.md 8.7)
   - Counterfactual routing test: calls `adapter.forward_counterfactual()` (V1.md 8.8)
   - Counterfactual semantics per `OverlapKind`:
     - `KV_CACHE`: clear KV cache, rerun from scratch (no prefix context carry-over)
     - `CONTROLLER_STATE`: zero Bob's medium-clock contribution to routing; fast and slow clocks remain
     - `BOTH`: clear KV cache AND zero medium-clock contribution
     - `NONE`: counterfactual unavailable; quench testing skipped, freeze/scar promotion hard-disabled
   - Overlap influence score: variance explained by overlap vs input (V1.md 8.8)
   - Hybrid schedule: deterministic floor (N_floor=100 steps) + event triggers (V1.md 8.8.1)
   - Truth-check ladder with four cases (V1.md 8.8.2):
     - Counterfactual fails → overrides all, decision ineligible for scar/freeze, contamination increases
     - Counterfactual passes + cooling delta bad → survival updates allowed, slow-clock scar promotion blocked (medium-clock-assisted quality)
     - Counterfactual passes + cooling delta good → full trust, all updates at full learning rate
     - Counterfactual passes + quench divergence high → survival updates allowed, scar promotion requires sustained low divergence across multiple probes
2. `bob_core/governance/contamination.py` — Credit contamination
   - Contamination severity: EMA of overlap influence × monotonicity penalty (V1.md 8.6)
   - Starting values from V1.md: W_contam=50 steps, monotonicity_penalty=1.0 if non-decreasing for M consecutive steps else 0.5
   - Discounted learning rate: `lr_effective = lr_base × (1 - contamination)`
   - Retirement NOT discounted (safety valve)
   - Scar/crystallisation locked when contaminated (binary block, not discounted)
3. `bob_core/governance/cooling.py` — Cooling windows
   - Periodic forced low-overlap intervals on LIVE decisions (not a side-channel probe)
   - During K_cool steps (starting value: 10), Bob makes real cheap/expensive choices with reduced medium-clock influence
   - Medium-clock contribution scaled by α_cool (starting value: 0.1, range [0, 0.3])
   - Performance delta tracked: quality drop during cooling = system was coasting on momentum
   - Cooling window fires in the normal decision flow, not as a separate test pass

**Test gate**:
```
- Quench fires at deterministic floor interval regardless of system state
- Event triggers fire immediately on: gate jitter, class contraction, topic shift, influence threshold
- Counterfactual semantics correct per OverlapKind (KV_CACHE clears cache, CONTROLLER_STATE zeros medium-clock, BOTH does both, NONE raises)
- Counterfactual routing test: overlap-dependent decisions blocked from scar/freeze
- Truth-check ladder: all four cases tested independently:
  - CF fails → block all
  - CF passes + cooling bad → survival yes, scar no
  - CF passes + cooling good → full trust
  - CF passes + high quench divergence → survival yes, scar needs sustained low divergence
- Credit contamination scales motif learning rate proportionally
- Credit contamination does NOT discount retirement
- Scar promotion locked at any contamination > 0
- Cooling window runs on live decisions (not a side-channel)
- Cooling window reduces medium-clock influence by α_cool=0.1
- Cooling performance delta correctly measures input-driven vs inertia-driven behaviour
- Default parameters match V1.md: W_contam=50, K_cool=10, α_cool=0.1, N_floor=100
```

**Estimated scope**: ~800 lines code, ~600 lines tests

---

### Phase 4: Context Classification
**Goal**: Bob derives context classes from the router's own representations, not from known labels.

**Deliverables**:
1. `bob_core/cache/context.py` — ContextClassifier
   - Online clustering from router penultimate-layer activations (V1.md 2.5)
   - Recursive property: router representations define classes over which routing diversity is measured
   - Inter-class distance monitoring (drift blindness safeguard, V1.md 8.4)
   - Meta-stability alert when inter-class distance contracts while per-class metrics appear stable
2. Updated `bob_core/substrate.py`: context class derived from activations, not passed in

**Test gate**:
```
- Classifier produces stable clusters on synthetic data with known structure
- Classifier correctly separates 4 task classes without labels
- Inter-class distance monitor fires alert on manifold collapse
- Classifier adapts when task distribution shifts (new clusters form)
- Performance comparable to known-class baseline (within 10% cost reduction)
```

**Estimated scope**: ~400 lines code, ~300 lines tests

**Risk**: Online clustering adds noise to context assignment. Phase 4 validation must show that classifier-derived classes produce comparable commitment formation to known classes. If not, the classifier needs work before proceeding.

---

### Phase 5: Evaluation Framework
**Goal**: Bob has a proper experiment runner with difficulty ladders, baselines, and Pareto tracking.

**Deliverables**:
1. `bob_core/telemetry/pareto.py` — Pareto frontier tracker
   - Per-decision (cost, quality) pairs
   - Windowed frontier computation (50-step bins)
   - Inward movement detection (the core claim)
   - Per-context-class breakdown
2. `bob_core/ladders/base.py` — DifficultyLadder protocol
   - Repeating structure with drifting surface form (V1.md 4.5)
   - Injected drift phases at defined intervals
   - Configurable task classes, recurrence period, drift magnitude
3. `bob_core/ladders/synthetic.py` — Synthetic ladder (our 4-class tasks, evolved)
4. `bob_core/ladders/wikitext.py` — WikiText-2 natural ladder
   - Topic-based context classes (derived from document boundaries)
   - Natural surface drift (different documents, same topic)
5. `bob_core/ladders/runner.py` — Experiment runner
   - Runs all three baselines (V1.md Appendix B):
     - Top-2 vanilla MoE (no cache, no controller)
     - Top-4 vanilla MoE (no cache, no controller)
     - Answer cache: hash-based lookup table on input token sequences. On cache hit, skip MoE entirely (cost=0). On miss, full routing (cost=top-k). Tests whether our gains are simply answer recall by another name. Should fail under surface drift (different tokens, same structure).
   - Plus Bob (top-k + consequence substrate)
   - Produces Pareto comparison, governance timeline, per-class breakdown
   - JSON output for reproducibility

**Test gate**:
```
- Pareto tracker correctly identifies frontier movement
- Synthetic ladder produces repeating classes with drifting surface
- Drift injection causes measurable routing entropy spike
- All three baselines run identically (same data, same quality threshold)
- Runner produces complete JSON output matching V1.md telemetry schema
- Pareto curve moves inward for Bob on synthetic ladder (replication of existing result)
```

**Estimated scope**: ~800 lines code, ~400 lines tests

---

### Phase 6: Mixtral Integration
**Goal**: Bob runs on real Mixtral-8x7B. The 8E top-2 problem is confronted.

**Deliverables**:
1. `backends/mixtral_adapter.py` — MixtralAdapter
   - Wraps HuggingFace `MixtralForCausalLM`
   - Hooks into `MixtralSparseMoeBlock` for router logit extraction
   - `forward_with_motif()` implementation: intercept expert dispatch, execute subset
   - `forward_counterfactual()`: run with overlap suppressed for quench testing
   - Quantization support (AWQ/GPTQ 4-bit for single-GPU operation)
   - Layer-selective observation (not all 32 layers need Bob oversight)
   - Must pass `backends/tests/test_adapter_contract.py`
2. Mixtral-specific difficulty ladder using WikiText-2
3. Validation run: same protocol as swiss-ai/MoE, real model

**Test gate**:
```
- mixtral_adapter passes full contract compliance test (test_adapter_contract.py)
- LayerSnapshot.router_scores extracted from all 32 layers
- forward_with_motif() produces correct output for subset experts
- forward_counterfactual() produces different snapshots when overlap suppressed
- Router score shapes match expected [B*T, 8] per layer
- Quantized model produces same routing decisions as full-precision (within tolerance)
- Full experiment run completes without OOM on single 24GB GPU
```

**Estimated scope**: ~500 lines code, ~300 lines tests

**The 8E Top-2 Experiment** (critical validation):

Run three configurations on same WikiText-2 ladder:
```
A: Mixtral 8E top-2 (native) + Bob       → Does commitment form on real data?
B: Mixtral 8E top-4 (override) + Bob     → Does higher branching restore motifs?
C: Mixtral 8E top-2 (native), no Bob     → Absolute economics baseline
```

If A works: the synthetic task result was misleading. Real language has enough structure.
If B works but A doesn't: branching factor is the bottleneck. Bob should override to top-4.
If neither works: Mixtral's routing is too diffuse. Need different model or different strategy.

This experiment determines whether Bob's primary backend is viable. It runs before any further development.

---

### Phase 7: Scars & Crystallisation
**Goal**: Bob accumulates permanent consequences (scars) and can freeze validated pathways.

**Deliverables**:
1. `bob_core/cache/scars.py` — ScarRegistry
   - Append-only constraint store (V1.md 2.6)
   - Per-context-class scar tracking
   - Epoch review: periodic relevance testing (suspend + measure divergence)
   - Scar decay via scheduled relevance review (V1.md 8.10)
2. `bob_core/gates/crystallisation.py` — CrystallisationGates
   - Gate 0: Overlap independence (prerequisite, V1.md 2.9)
   - Gate 1: Relative freedom (geodesic Gram matrix effective rank)
   - Gate 2: Absolute freedom (minimum dimensionality threshold)
   - Gate 3: Regional dominance (k of K clustered regions)
   - Gate 4: Consequence (computational dimensionality proportional to routing)
   - Gate 5: Stability (sustained two-step commit window)
3. Frozen pathway management: reversible via epoch review

**Test gate**:
```
- Scars are append-only (no deletion, only relevance decay)
- Epoch review correctly identifies stale scars
- Gate 0 blocks overlap-dependent candidates regardless of gates 1-5
- All five gates must pass in order (gate 3 not evaluated if gate 2 fails)
- Frozen pathways remain reversible
- Crystallisation only possible in SETTLEMENT governance state
- Integration: full lifecycle from motif creation → scar → potential freeze
```

**Estimated scope**: ~700 lines code, ~500 lines tests

**Note**: This phase is architecturally specified but experimentally unvalidated (V1.md Section 2.9 disclaimer). Phase 7 produces the implementation; Phase 8 validates it.

---

### Phase 8: Integration Validation
**Goal**: All components work together. The controlled difficulty ladder runs end-to-end. The core claim is tested.

This is V1.md Section 7.8: the experiment that validates the core claim.

**Deliverables**:
1. Full end-to-end run: Bob + swiss-ai/MoE backend
   - 3-5 task classes, repeating structure, drifting surface form
   - N thousand steps with drift injection at defined intervals
   - All three baselines + Bob
   - Full telemetry per V1.md Appendix C
2. Full end-to-end run: Bob + Mixtral backend
   - Same protocol, real model
   - Quantized single-GPU operation
3. Pareto analysis report
   - Per-class cost/quality curves
   - Frontier movement over ladder steps
   - Governance timeline
   - Scar accumulation trajectory
   - Quench results and overlap influence scores

**Test gate (the only one that matters)**:
```
The Pareto curve (cost vs. quality) moves inward over experience.
```

If yes: consequence accumulation works. Bob earns cheaper decisions through experience.
If no: either the mechanism is wrong, the gate is miscalibrated, or the architecture doesn't support commitment at this scale.

**Success criteria from V1.md**:
- Quality holds or improves
- Cost decreases
- Pareto frontier moves inward
- Bob beats top-2 vanilla on absolute cost at equal quality (economic relevance)
- Bob beats top-4 vanilla on cost trajectory (cache contribution)
- Bob beats answer cache on robustness to input drift (structural reuse)

---

## 5. Testing Strategy

### 5.1 Test Layers

```
Layer 1: Unit Tests (per-module)
  - Every class, every method, every edge case
  - Run: python -m pytest bob_core/tests/
  - Gate: 100% pass before any commit

Layer 2: Integration Tests (cross-module)
  - Substrate orchestration, clock interaction, gate → cache feedback
  - Run: python -m pytest bob_core/tests/test_substrate.py
  - Gate: 100% pass before phase completion

Layer 3: Replication Tests (reproduce known results)
  - commitment_cache_experiment.py results through Bob
  - 4E top-2: ~37% cost reduction
  - 8E top-2: ~0% cost reduction
  - Gate: results match within 5%

Layer 4: Validation Experiments (new results)
  - Difficulty ladder on swiss-ai/MoE
  - Difficulty ladder on Mixtral
  - Gate: Pareto frontier moves inward

Layer 5: Green Invariant
  - All existing chronomoe_integration tests pass
  - No existing file modified without test coverage
  - Run: python -m chronomoe_integration.tests
  - Gate: 100% pass always
```

### 5.2 Test Fixtures

Shared fixtures for consistent testing:

```python
# conftest.py
@fixture
def swiss_adapter():
    """4-expert top-2 swiss-ai/MoE adapter."""
    ...

@fixture
def tiny_substrate(swiss_adapter):
    """Minimal BobSubstrate for unit testing."""
    ...

@fixture
def four_class_ladder():
    """Synthetic 4-class difficulty ladder (shift-1/3/7, double)."""
    ...

@fixture
def mock_mixtral_adapter():
    """Mock Mixtral adapter (correct shapes, random routing)."""
    ...
```

### 5.3 Falsifiability

Every cheap-path decision produces a complete trace (V1.md Appendix C). If Bob gets cheaper but quality drops, traces show it. If Bob gets cheaper and quality holds, we have the artifact.

No sampling. No aggregation. Every decision logged. The traces ARE the evidence.

---

## 6. Dependency Graph

```
Phase 0: Foundation
    │
    ▼
Phase 1: Core Decision Loop ◄── Reuses: CommitmentCache, GovernanceStateMachine
    │
    ├──────────────────┐
    ▼                  ▼
Phase 2: Clocks    Phase 5: Evaluation (can develop in parallel)
    │                  │
    ▼                  │
Phase 3: Quench        │
    │                  │
    ▼                  │
Phase 4: Context       │
    │                  │
    ├──────────────────┘
    ▼
Phase 6: Mixtral Integration
    │
    ▼
Phase 7: Scars & Crystallisation
    │
    ▼
Phase 8: Integration Validation ◄── THE EXPERIMENT
```

Phases 2 and 5 can develop in parallel (clocks and evaluation framework are independent).
Phases 3 and 4 depend on Phase 2 (clocks must exist before quench protocol).
Phase 6 depends on Phase 1 (core loop must work before Mixtral integration).
Phase 7 depends on Phase 3 (quench must exist before crystallisation — Gate 0 requires overlap testing).
Phase 8 depends on everything.

---

## 7. Milestones

| Milestone | Phase | Deliverable | Verification |
|-----------|-------|-------------|--------------|
| **M0: Bob exists** | 0 | Package + adapter + smoke test | Router logits extracted correctly |
| **M1: Bob decides** | 1 | Cheap/expensive path working | Replication of commitment_cache_experiment |
| **M2: Bob remembers** | 2 | Three clocks running | Clock EMAs at correct timescales |
| **M3: Bob doubts** | 3 | Quench protocol + contamination | Overlap-dependent decisions blocked |
| **M4: Bob sees** | 4 | Context classes from activations | Comparable to known-class baseline |
| **M5: Bob measures** | 5 | Pareto tracking + ladders | Automated experiment runs |
| **M6: Bob scales** | 6 | Mixtral backend working | 8E top-2/top-4 experiment results |
| **M7: Bob scars** | 7 | Permanent consequence accumulation | Scar lifecycle complete |
| **M8: Bob proves** | 8 | Full validation | Pareto frontier moves inward |

---

## 8. Non-Goals (for Bob v1)

| Not Building | Why |
|-------------|-----|
| Model training | Bob is inference-time control, not a trainer |
| Weight modification | Bob observes and routes, doesn't change weights |
| Multi-GPU distribution | Single-GPU first. Scale later. |
| Production serving | Bob is a research instrument |
| Web UI / dashboard | CLI output and JSON traces. Visualization separate. |
| Multi-model ensemble | One backend at a time |
| Learned context classifier (neural) | Online clustering first. Learn later. |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mixtral 8E top-2 never forms commitments | Medium | High | Phase 6 top-4 override experiment. If neither works, switch backend to OLMoE or scale swiss-ai/MoE. |
| Online clustering too noisy | Medium | Medium | Phase 4 validates against known-class baseline. Fallback: hash-based context classes. |
| Measurement apparatus degradation (V1.md 8.5) | Medium | High | Context classifier degrades under same conditions it measures. Phase 4 inter-class distance monitor is the safeguard, but the risk is fundamental: if router representations collapse, the context classifier, stability estimator, and survival tracker all degrade simultaneously. No complete solution; monitor inter-class distance contraction as early warning. |
| Quench overhead too expensive | Low | Medium | Deterministic floor at 1% overhead. Event triggers add variable cost only at critical moments. |
| Survival inflation despite recency weighting | Low | Medium | Phase 1 tests verify inflation doesn't recur. Scheduled perturbation probes as backup. |
| Mixtral quantization changes routing | Medium | Low | Phase 6 tests verify routing consistency between full-precision and quantized. |
| Package grows too large | Low | Low | Each phase has scope estimates. No phase exceeds ~900 lines. Total Bob: ~6500 lines code + ~4200 lines tests. |

---

## 10. Definition of Done

Bob v1 is done when:

1. **M8 passes**: Pareto frontier moves inward on at least one backend.
2. **All tests pass**: ~3000 lines of tests, 100% green.
3. **Traces are complete**: Every decision logged per V1.md Appendix C.
4. **Baselines are honest**: Bob vs top-2 vanilla vs top-4 vanilla vs answer cache, same data, same quality threshold.
5. **Green invariant holds**: All existing chronomoe_integration tests still pass.

The artifact is the Pareto curve. Everything else is infrastructure to produce it.

---

## Appendix A: Estimated Totals

| Category | Lines |
|----------|-------|
| bob_core code (logic + store) | ~5,700 |
| backends code (adapter + swiss + mixtral) | ~1,000 |
| bob_core tests | ~3,500 |
| backends tests (contract + specific) | ~600 |
| Experiment scripts | ~1,000 |
| **Total new code** | **~11,800** |
| Existing chronomoe_integration (preserved) | ~8,000 |
| Existing tests (preserved) | ~3,600 |

## Appendix B: Key Files from Existing Codebase

| File | Lines | Bob Reuse |
|------|-------|-----------|
| `moe.py` | 146 | Wrapped by `backends/swiss_adapter.py` |
| `gpt.py` | 438 | Development model backbone |
| `commitment_cache.py` | 464 | Evolved into `bob_core/cache/commitment.py` |
| `convergence.py` | 976 | GovernanceStateMachine + detector extracted |
| `controller.py` | 1253 | Pattern reference (observe/decide/apply) |
| `routing_geometry.py` | 471 | Imported for diagnostics |
| `stress_bands.py` | 292 | May integrate as debt signal |
| `commitment_cache_experiment.py` | 736 | Replication target for Phase 1 |
| `commitment_cache_scaling.py` | 591 | Replication target for Phase 1 |
