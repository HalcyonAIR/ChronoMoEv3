# State Separation: Coherence vs Lifecycle

**Jeff's yellow sticky note: role_vector and centroids are lifecycle state, not coherence state.**

---

## The Boundary

**Clean separation of concerns:**

| Responsibility | State | Purpose |
|----------------|-------|---------|
| **Coherence** | phi_{fast, mid, slow} | Measures directional alignment with mixture |
| **Lifecycle** | Everything else | Informs spawn/prune/split/merge decisions |

**The question to ask:** "Does this measure alignment?" → CoherenceState. "Does this inform lifecycle?" → LifecycleState.

---

## Revised Schema

### CoherenceState (Pure Measurement)

```python
@dataclass
class CoherenceState:
    """
    Pure coherence measurement: directional alignment with mixture.

    This is the single state variable of ChronoMoEv3.
    Everything else derives from or uses this, but is separate.
    """

    expert_id: str
    layer_id: int

    # Three-timescale coherence (the core)
    phi_fast: float   # ~10 steps half-life
    phi_mid: float    # ~100 steps half-life
    phi_slow: float   # ~1000 steps half-life

    # Tracking
    last_update_step: int
    total_tokens_seen: int

    @property
    def phi_delta(self) -> float:
        """Fast - Slow coherence delta."""
        return self.phi_fast - self.phi_slow

    @property
    def is_degrading(self) -> bool:
        """Is fast coherence below slow baseline?"""
        return self.phi_delta < 0
```

**Size:** ~40 bytes per expert (3 floats + metadata)

**Responsibility:** Answer "Is this expert aligned with the mixture?"

---

### LifecycleState (Decision Support)

```python
@dataclass
class LifecycleState:
    """
    State used for lifecycle decisions (spawn/prune/split/merge).

    Separate from coherence measurement to avoid scope creep.
    Can be extended without touching CoherenceState.
    """

    expert_id: str
    layer_id: int

    # Role tracking (for merge detection)
    role_vector: Tensor  # [d_model] - typical output direction
    role_vector_alpha: float = 0.99

    # Bimodality detection (for split detection)
    centroid_1: Tensor  # [d_model]
    centroid_2: Tensor  # [d_model]
    centroid_count_1: int
    centroid_count_2: int
    bimodality_score: float  # separation * balance

    # Lifecycle metadata
    born_step: int
    parent_id: Optional[str]
    active: bool
    cooling_until: Optional[int]

    # Future extensions (examples, not implemented yet):
    # gradient_direction: Optional[Tensor] = None  # For optimization health
    # loss_contribution_ema: Optional[float] = None  # For impact_weight
    # utilization_ema: Optional[float] = None  # For routing mass tracking

    def compute_bimodality(self) -> float:
        """Compute separation * balance."""
        if self.centroid_count_1 == 0 or self.centroid_count_2 == 0:
            return 0.0

        separation = 1.0 - F.cosine_similarity(
            self.centroid_1.unsqueeze(0),
            self.centroid_2.unsqueeze(0),
            dim=-1
        ).item()

        balance = min(self.centroid_count_1, self.centroid_count_2) / \
                  max(self.centroid_count_1, self.centroid_count_2)

        self.bimodality_score = separation * balance
        return self.bimodality_score

    def update_role_vector(self, y_bar_e: Tensor):
        """Update role vector (typical output direction)."""
        self.role_vector = (
            self.role_vector_alpha * self.role_vector
            + (1 - self.role_vector_alpha) * y_bar_e
        )

    def update_centroids(self, y_bar_e: Tensor):
        """Update bimodality centroids."""
        # Assign to closer centroid
        sim1 = F.cosine_similarity(
            y_bar_e.unsqueeze(0),
            self.centroid_1.unsqueeze(0),
            dim=-1
        ).item()

        sim2 = F.cosine_similarity(
            y_bar_e.unsqueeze(0),
            self.centroid_2.unsqueeze(0),
            dim=-1
        ).item()

        alpha_c = 0.95  # Centroid EMA rate

        if sim1 >= sim2:
            self.centroid_1 = alpha_c * self.centroid_1 + (1 - alpha_c) * y_bar_e
            self.centroid_count_1 += 1
        else:
            self.centroid_2 = alpha_c * self.centroid_2 + (1 - alpha_c) * y_bar_e
            self.centroid_count_2 += 1
```

**Size:** ~48 KB per expert (2 tensors for centroids, 1 for role_vector, metadata)

**Responsibility:** Provide signals for lifecycle decisions

**Extensibility:** Can add `gradient_direction`, `impact_weight`, etc. without changing CoherenceState

---

## Unified Registry

```python
class ExpertRegistry:
    """
    Central registry managing both coherence and lifecycle state.

    Keeps them separate but coordinated.
    """

    def __init__(self):
        self.coherence: Dict[str, CoherenceState] = {}
        self.lifecycle: Dict[str, LifecycleState] = {}

    def create_expert(
        self,
        expert_id: str,
        layer_id: int,
        d_model: int,
        born_step: int,
        parent_id: Optional[str] = None,
    ):
        """Create both coherence and lifecycle state."""
        # Coherence state (small, pure measurement)
        self.coherence[expert_id] = CoherenceState(
            expert_id=expert_id,
            layer_id=layer_id,
            phi_fast=0.0,
            phi_mid=0.0,
            phi_slow=0.0,
            last_update_step=born_step,
            total_tokens_seen=0,
        )

        # Lifecycle state (larger, decision support)
        self.lifecycle[expert_id] = LifecycleState(
            expert_id=expert_id,
            layer_id=layer_id,
            role_vector=torch.zeros(d_model),
            centroid_1=torch.randn(d_model).normalize(),
            centroid_2=torch.randn(d_model).normalize(),
            centroid_count_1=0,
            centroid_count_2=0,
            bimodality_score=0.0,
            born_step=born_step,
            parent_id=parent_id,
            active=True,
            cooling_until=born_step + cooling_period,
        )

    def update_from_trace(
        self,
        trace: MoETrace,
        alpha_fast: float,
        alpha_mid: float,
        alpha_slow: float,
        step: int,
        layer_id: int,
    ):
        """Update both states from forward pass trace."""
        phi_raw = trace.compute_coherence()  # [num_active_experts]

        for i, expert_idx in enumerate(trace.active_expert_ids):
            expert_id = f"L{layer_id}_E{expert_idx}"

            # Update coherence (pure measurement)
            self._update_coherence(
                expert_id=expert_id,
                phi_raw=phi_raw[i].item(),
                alpha_fast=alpha_fast,
                alpha_mid=alpha_mid,
                alpha_slow=alpha_slow,
                step=step,
                num_tokens=len(trace.token_row_indices[i]),
            )

            # Update lifecycle state (decision support)
            self._update_lifecycle(
                expert_id=expert_id,
                y_bar_e=trace.expert_mean_outputs[i],
            )

    def _update_coherence(
        self,
        expert_id: str,
        phi_raw: float,
        alpha_fast: float,
        alpha_mid: float,
        alpha_slow: float,
        step: int,
        num_tokens: int,
    ):
        """Update coherence measurement."""
        state = self.coherence[expert_id]

        # Three-clock EMA
        state.phi_fast = alpha_fast * state.phi_fast + (1 - alpha_fast) * phi_raw
        state.phi_mid = alpha_mid * state.phi_mid + (1 - alpha_mid) * phi_raw
        state.phi_slow = alpha_slow * state.phi_slow + (1 - alpha_slow) * phi_raw

        # Tracking
        state.last_update_step = step
        state.total_tokens_seen += num_tokens

    def _update_lifecycle(self, expert_id: str, y_bar_e: Tensor):
        """Update lifecycle decision support state."""
        state = self.lifecycle[expert_id]

        # Update role vector
        state.update_role_vector(y_bar_e)

        # Update bimodality centroids
        state.update_centroids(y_bar_e)
        state.compute_bimodality()

    def get_prune_candidates(self, layer_id: int, threshold: float = 0.3) -> List[str]:
        """
        Get experts eligible for pruning.

        Uses: CoherenceState.phi_slow (primary signal)
        """
        candidates = []
        for expert_id, coh_state in self.coherence.items():
            if coh_state.layer_id != layer_id:
                continue

            lifecycle_state = self.lifecycle[expert_id]
            if not lifecycle_state.active:
                continue

            if coh_state.phi_slow < threshold:
                candidates.append(expert_id)

        return candidates

    def get_split_candidates(
        self,
        layer_id: int,
        bimodality_threshold: float = 0.6,
    ) -> List[str]:
        """
        Get experts eligible for splitting.

        Uses: LifecycleState.bimodality_score (primary signal)
        Also checks: CoherenceState.phi_slow (don't split if low coherence)
        """
        candidates = []
        for expert_id, lifecycle_state in self.lifecycle.items():
            if lifecycle_state.layer_id != layer_id:
                continue

            if not lifecycle_state.active:
                continue

            coh_state = self.coherence[expert_id]

            # High bimodality + decent coherence → split candidate
            if (
                lifecycle_state.bimodality_score > bimodality_threshold
                and coh_state.phi_slow > 0.5  # Don't split failing experts
            ):
                candidates.append(expert_id)

        return candidates

    def get_merge_candidates(
        self,
        layer_id: int,
        similarity_threshold: float = 0.9,
    ) -> List[Tuple[str, str]]:
        """
        Get expert pairs eligible for merging.

        Uses: LifecycleState.role_vector (primary signal)
        Also checks: CoherenceState.phi_slow (merge only healthy experts)
        """
        layer_experts = [
            (eid, self.lifecycle[eid])
            for eid in self.lifecycle
            if self.lifecycle[eid].layer_id == layer_id
            and self.lifecycle[eid].active
        ]

        candidates = []

        for i, (eid1, state1) in enumerate(layer_experts):
            for eid2, state2 in layer_experts[i + 1 :]:
                # Compute role similarity
                similarity = F.cosine_similarity(
                    state1.role_vector.unsqueeze(0),
                    state2.role_vector.unsqueeze(0),
                    dim=-1,
                ).item()

                if similarity > similarity_threshold:
                    # Check both are healthy
                    coh1 = self.coherence[eid1]
                    coh2 = self.coherence[eid2]

                    if coh1.phi_slow > 0.5 and coh2.phi_slow > 0.5:
                        candidates.append((eid1, eid2))

        return candidates
```

---

## Checkpoint Schema (Revised)

```python
@dataclass
class ChronoCheckpoint:
    """
    Complete system checkpoint with clear separation.
    """

    version: str = "v3.1"
    step: int
    config: ChronoConfig

    # Separate checkpoints for separate concerns
    coherence_states: Dict[str, CoherenceState]  # ~40B per expert
    lifecycle_states: Dict[str, LifecycleState]  # ~48KB per expert

    router_states: Dict[int, RouterCheckpoint]
    layer_states: Dict[int, LayerCheckpoint]

    def total_size(self) -> int:
        """Estimate checkpoint size in bytes."""
        num_experts = len(self.coherence_states)

        coherence_size = num_experts * 40  # ~40 bytes per expert
        lifecycle_size = num_experts * 48 * 1024  # ~48KB per expert

        return coherence_size + lifecycle_size  # ~3MB for 64 experts
```

---

## Benefits of Separation

1. **Clear responsibility boundaries**
   - CoherenceState: measurement only
   - LifecycleState: decision support only

2. **Independent evolution**
   - Add `impact_weight` to LifecycleState without touching CoherenceState
   - Swap bimodality detector without affecting coherence tracking

3. **Simpler reasoning**
   - "Is phi_slow dropping?" → look at CoherenceState
   - "Should we split expert X?" → look at LifecycleState.bimodality_score

4. **Easier testing**
   - Test coherence computation in isolation
   - Test lifecycle decisions with mocked coherence

5. **Prevents scope creep**
   - CoherenceState stays minimal (3 floats)
   - LifecycleState is the "extra state" dumping ground (by design)

---

## Migration from Current Schema

**Old (everything in CoherenceState):**
```python
@dataclass
class CoherenceState:
    phi_fast: float
    phi_mid: float
    phi_slow: float
    role_vector: Tensor  # ← Lifecycle concern
    centroid_1: Tensor   # ← Lifecycle concern
    centroid_2: Tensor   # ← Lifecycle concern
```

**New (separated):**
```python
# Pure measurement
@dataclass
class CoherenceState:
    phi_fast: float
    phi_mid: float
    phi_slow: float

# Decision support
@dataclass
class LifecycleState:
    role_vector: Tensor
    centroid_1: Tensor
    centroid_2: Tensor
    bimodality_score: float
```

---

## Recommendation

**Accept Jeff's yellow sticky note.**

Move `role_vector`, `centroid_1`, `centroid_2` out of CoherenceState into LifecycleState.

Keep CoherenceState minimal: just phi_{fast,mid,slow}.

This prevents "extra state" from quietly becoming a second subsystem by making the separation explicit and intentional.

---

**Status:** Architectural boundary clarified. Ready to implement with clean separation.
