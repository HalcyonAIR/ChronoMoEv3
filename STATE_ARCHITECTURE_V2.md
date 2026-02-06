# State Architecture V2: Three Clean Subsystems

**Halcyon's rule: No dumping grounds. Three subsystems with clear ownership and influence boundaries.**

---

## The Three-Way Split

### 1. CoherenceState: "Am I aligned with the ensemble, and am I being observed?"

**Owner:** Coherence measurement system
**Computed from:** Expert outputs, mixture outputs
**Influences:** Nothing directly (read-only for lifecycle)

```python
@dataclass
class CoherenceState:
    """
    Pure alignment measurement.

    Answers: Is this expert aligned with the mixture?

    RULE: If a field can influence lifecycle decisions,
          it does NOT belong here.
    """

    expert_id: str
    layer_id: int

    # Three-timescale alignment measurement
    phi_fast: float    # ~10 steps: "Am I aligned right now?"
    phi_mid: float     # ~100 steps: "Have I been aligned?"
    phi_slow: float    # ~1000 steps: "Am I structurally aligned?"

    # Observation tracking
    last_update_step: int
    total_tokens_seen: int

    @property
    def phi_delta(self) -> float:
        """Degradation signal: fast - slow."""
        return self.phi_fast - self.phi_slow

    @property
    def is_being_observed(self) -> bool:
        """Has this expert been updated recently?"""
        return (current_step - self.last_update_step) < 1000
```

**Size:** 40 bytes per expert
**No tensors.** No accumulated history beyond EMAs.
**No lifecycle authority.** Read-only.

---

### 2. RoleState: "What do I do when I'm active?"

**Owner:** Role tracking system
**Computed from:** Expert outputs, routing patterns
**Influences:** Merge detection, split detection, spawn initialization

```python
@dataclass
class RoleState:
    """
    Functional behavior description.

    Answers: What is this expert's job? What does it specialize in?

    Used for: Detecting redundancy (merge), detecting bimodality (split),
              initializing new experts (spawn).
    """

    expert_id: str
    layer_id: int

    # === OUTPUT CHARACTERIZATION ===

    # Typical output direction (for merge detection)
    role_vector: Tensor  # [d_model], α=0.9 (fast decay, ~10 steps)

    # Bimodal basin detection (for split detection)
    centroid_1: Tensor   # [d_model], α=0.9
    centroid_2: Tensor   # [d_model], α=0.9
    centroid_count_1: int
    centroid_count_2: int

    # === INPUT CHARACTERIZATION (future) ===

    # What kinds of tokens does this expert handle?
    # token_type_histogram: Optional[Dict[str, int]] = None

    # Which other experts does this expert co-activate with?
    # co_activation_graph: Optional[Dict[str, float]] = None

    def compute_bimodality(self) -> float:
        """
        Separation * balance.

        Derived quantity, not accumulated.
        """
        if self.centroid_count_1 == 0 or self.centroid_count_2 == 0:
            return 0.0

        separation = 1.0 - F.cosine_similarity(
            self.centroid_1.unsqueeze(0),
            self.centroid_2.unsqueeze(0),
            dim=-1
        ).item()

        balance = min(self.centroid_count_1, self.centroid_count_2) / \
                  max(self.centroid_count_1, self.centroid_count_2)

        return separation * balance

    def compute_redundancy(self, other: 'RoleState') -> float:
        """
        Cosine similarity between role vectors.

        Used for merge detection.
        """
        return F.cosine_similarity(
            self.role_vector.unsqueeze(0),
            other.role_vector.unsqueeze(0),
            dim=-1
        ).item()
```

**Size:** 48 KB per expert (3 tensors)
**Accumulated state:** Yes, but **fast decay** (α=0.9, ~10-step memory)
**Reset policy:** Reset centroids on split, reset role_vector on merge
**Lifecycle authority:** Informs decisions, does not make them

**DISCIPLINE:**
- All EMAs use α ≤ 0.9 (fast decay only)
- No α > 0.95 (prevents long-term divergence from coherence)
- Reset on structural changes (prevents locus formation)

---

### 3. RouterState: "What biases exist and why?"

**Owner:** Router system
**Computed from:** Routing decisions, coherence feedback
**Influences:** Routing (via beta), crisis detection (via disagreement)

```python
@dataclass
class RouterState:
    """
    Routing infrastructure state.

    Answers: How is routing being influenced? Is routing healthy?

    This is where the locus lives: beta_coeff driven by phi_slow.
    """

    layer_id: int

    # === SLOW BIAS (THE LOCUS) ===

    # Scale-free bias coefficients
    beta_coeff: Tensor  # [max_experts], clamped to [-k_max, k_max]
    k_max: float = 0.3

    # Logit scale tracking (for scale-free beta)
    logit_std_ema: float  # α=0.99

    # Temperature
    temperature: float = 1.0

    # === ROUTING HEALTH ===

    # Clean vs biased disagreement
    disagreement_rate_js: float  # JS divergence
    disagreement_rate_flip: float  # Top-1 flip rate

    # Calibrated thresholds (regime-specific)
    threshold_warning: float = 0.15
    threshold_serious: float = 0.30
    threshold_crisis: float = 0.45

    # Active experts (fixed-width router)
    max_experts: int
    active_mask: Tensor  # [max_experts], boolean

    def compute_beta_eff(self) -> Tensor:
        """
        Effective bias: beta_coeff * logit_std.

        Scale-free: same k produces same impact across regimes.
        """
        k = self.beta_coeff.clamp(-self.k_max, self.k_max)
        return k * self.logit_std_ema

    def is_in_crisis(self) -> bool:
        """Is clean/biased disagreement catastrophic?"""
        return self.disagreement_rate_js > self.threshold_crisis

    def needs_beta_decay(self) -> bool:
        """Should we decay beta due to disagreement?"""
        return self.disagreement_rate_js > self.threshold_serious
```

**Size:** ~5 KB per layer (mostly tensors for beta_coeff)
**Accumulated state:** Yes (beta_coeff, logit_std_ema)
**Update source:** Coherence feedback (phi_slow → beta_coeff)
**Lifecycle authority:** None (lifecycle reads disagreement, doesn't write beta)

**The locus lives here:**
```python
# Update rule (executed by router, not by lifecycle)
def update_beta(expert_id, phi_slow, eta=0.01, tau=0.5):
    delta = eta * (phi_slow - tau)
    beta_coeff[expert_id] += delta
    beta_coeff[expert_id] = beta_coeff[expert_id].clamp(-k_max, k_max)
```

---

## Lifecycle: The Reader, Not The Writer

```python
class LifecycleCoordinator:
    """
    Reads CoherenceState, RoleState, RouterState.
    Writes only lifecycle decisions (spawn/prune/split/merge).

    DOES NOT WRITE TO ANY STATE CONTAINER.
    """

    def __init__(
        self,
        coherence: Dict[str, CoherenceState],
        roles: Dict[str, RoleState],
        routers: Dict[int, RouterState],
    ):
        self.coherence = coherence
        self.roles = roles
        self.routers = routers

        # Lifecycle decision log
        self.decisions: List[LifecycleDecision] = []

    def evaluate_layer(self, layer_id: int) -> List[LifecycleAction]:
        """
        Evaluate lifecycle for one layer.

        Reads all three state containers.
        Returns proposed actions (spawn/prune/split/merge).
        """
        layer_experts = self._get_layer_experts(layer_id)

        actions = []

        # PRUNE: Low phi_slow (from CoherenceState)
        for expert_id in layer_experts:
            coh = self.coherence[expert_id]
            if coh.phi_slow < 0.3 and coh.is_being_observed:
                actions.append(PruneAction(expert_id, reason="low_coherence"))

        # SPLIT: High bimodality + decent coherence (from RoleState + CoherenceState)
        for expert_id in layer_experts:
            role = self.roles[expert_id]
            coh = self.coherence[expert_id]

            bimodality = role.compute_bimodality()
            if bimodality > 0.6 and coh.phi_slow > 0.5:
                actions.append(SplitAction(expert_id, reason="bimodal"))

        # MERGE: High redundancy (from RoleState)
        for i, eid1 in enumerate(layer_experts):
            for eid2 in layer_experts[i+1:]:
                role1 = self.roles[eid1]
                role2 = self.roles[eid2]

                redundancy = role1.compute_redundancy(role2)
                if redundancy > 0.9:
                    # Check both are healthy (CoherenceState)
                    if (self.coherence[eid1].phi_slow > 0.5 and
                        self.coherence[eid2].phi_slow > 0.5):
                        actions.append(MergeAction(eid1, eid2, reason="redundant"))

        # SPAWN: Layer-wide misfit (from CoherenceState)
        layer_coherence = self._compute_layer_coherence(layer_id)
        num_experts = len(layer_experts)
        if layer_coherence < 0.5 and num_experts < max_experts:
            actions.append(SpawnAction(layer_id, reason="layer_starvation"))

        return actions

    def execute_action(self, action: LifecycleAction):
        """
        Execute lifecycle action.

        Modifies model structure (add/remove experts).
        Initializes state containers for new experts.
        Archives state for removed experts.

        DOES NOT MODIFY EXISTING STATE.
        """
        if isinstance(action, PruneAction):
            self._prune_expert(action.expert_id)
        elif isinstance(action, SplitAction):
            self._split_expert(action.expert_id)
        elif isinstance(action, MergeAction):
            self._merge_experts(action.expert_id_1, action.expert_id_2)
        elif isinstance(action, SpawnAction):
            self._spawn_expert(action.layer_id)

        self.decisions.append(action)
```

**Key:** Lifecycle **reads** state, **proposes** actions, **executes** structure changes. It does not write into CoherenceState, RoleState, or RouterState.

---

## Ownership Table

| Field | Owner | Computed From | Influences | Can Be Dumping Ground? |
|-------|-------|---------------|------------|----------------------|
| **CoherenceState** |
| `phi_{fast,mid,slow}` | Coherence measurement | Expert output, mixture | Lifecycle (read-only), beta (feedback) | ❌ NO |
| `last_update_step` | Coherence measurement | Forward pass | Observability check | ❌ NO |
| **RoleState** |
| `role_vector` | Role tracker | Expert output EMA (α=0.9) | Merge detection | ⚠️ RISK (accumulated) |
| `centroid_1/2` | Role tracker | Expert output EMA (α=0.9) | Split detection | ⚠️ RISK (accumulated) |
| `bimodality` | Role tracker | Derived from centroids | Split detection | ✅ OK (derived) |
| **RouterState** |
| `beta_coeff` | Router | Phi_slow feedback | Routing (the locus) | ✅ OK (authorized) |
| `logit_std_ema` | Router | Logit statistics | Beta scaling | ✅ OK (infrastructure) |
| `disagreement_rate` | Router | Clean vs biased comparison | Crisis detection | ✅ OK (health metric) |

**Dumping ground prevention:**
- RoleState fields are **fast-decay EMAs** (α=0.9, ~10-step memory)
- RoleState fields **reset on structure change** (split/merge)
- Any new field must answer: "Who owns it? What does it influence?"

---

## The Discipline: One Sentence per Field

**Every state field must have:**

1. **Owner:** Who computes this?
2. **Meaning:** What does this field represent?
3. **Influence:** What is this field allowed to affect?

**Example:**

```python
# GOOD
role_vector: Tensor
# Owner: Role tracker
# Meaning: Fast EMA (α=0.9) of expert's typical output direction
# Influence: Merge detection (redundancy scoring)

# BAD (dumping ground symptom)
misc_state: Dict[str, Any]
# Owner: ???
# Meaning: Various things
# Influence: Whatever needs it
```

**If you can't write the one-sentence definition, the field doesn't belong.**

---

## Checkpoint Schema (Revised)

```python
@dataclass
class ChronoCheckpoint:
    """
    Complete system checkpoint with clean separation.

    Three state containers + lifecycle log.
    """

    version: str = "v3.2"
    step: int
    config: ChronoConfig

    # Three subsystems
    coherence: Dict[str, CoherenceState]  # ~40B per expert
    roles: Dict[str, RoleState]           # ~48KB per expert
    routers: Dict[int, RouterState]       # ~5KB per layer

    # Lifecycle log (decisions, not state)
    lifecycle_decisions: List[LifecycleDecision]
```

**Total size:** ~3 MB for 64 experts (unchanged), but cleanly separated.

---

## The Meta-Rule: One Dumping Ground Only

**Halcyon's rule:** The system is allowed one dumping ground, and we're not building it.

**What this means:**
- No "expert_metadata: Dict[str, Any]" field
- No "extra_state: Optional[Tensor]" field
- No "for convenience" fields that don't have clear ownership

**If you're tempted to add a field:**
1. Write the one-sentence definition
2. Identify the owner
3. Document what it influences
4. If you can't do all three → don't add it

**The only exception:** Explicitly labeled "experimental" fields during prototyping, with a removal date.

---

## Implementation Plan

**Phase 2:**
1. Split current `CoherenceState` into three: `CoherenceState`, `RoleState`, `RouterState`
2. Create `LifecycleCoordinator` that reads all three, writes decisions
3. Implement beta update in `RouterState`, not in lifecycle
4. Add ownership documentation to every field

**Phase 3:**
5. Implement fast-decay EMAs (α=0.9) for role_vector and centroids
6. Add reset logic on split/merge
7. Validate that no second locus emerges

**Phase 4+:**
8. Add fields only with full ownership documentation
9. Periodic audit: "Does every field have a clear owner and influence?"

---

**Status:** Clean three-way split specified. No dumping grounds. One locus (in RouterState.beta_coeff). Ready for Phase 2 implementation.
