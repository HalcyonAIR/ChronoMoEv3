"""
BackendAdapter protocol: the contract between Bob and any MoE backend.

Bob talks ONLY to this interface. No model-specific imports in bob_core/.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple
from enum import Enum
import torch


class OverlapKind(Enum):
    """What kind of overlap state this backend has."""
    NONE = "none"                      # No overlap state (feedforward MoE)
    KV_CACHE = "kv_cache"              # KV cache carries state between passes
    CONTROLLER_STATE = "controller_state"  # External controller state
    BOTH = "both"                      # KV cache + controller state


@dataclass
class LayerSnapshot:
    """Raw observation from one layer of one forward pass."""
    layer_id: int
    router_scores: torch.Tensor        # Pre-top-k scores [B*T, num_experts]
    selected_experts: torch.Tensor     # Expert indices chosen [B*T, top_k]
    routing_weights: torch.Tensor      # Weights for selected experts [B*T, top_k]
    expert_usage: torch.Tensor         # Per-expert usage vector [num_experts]


@dataclass
class LayerMotif:
    """Which experts to fire at a single layer."""
    expert_ids: Tuple[int, ...]
    weights: Tuple[float, ...]


@dataclass
class MotifSpec:
    """
    What to execute for cheap path. Can span multiple layers.

    All-or-nothing: the motif fires across ALL specified layers or NONE.
    No layer-progressive execution. The compound gate decides once.
    """
    layers: Dict[int, LayerMotif]

    def total_expert_invocations(self) -> int:
        """Total expert invocations if this motif is executed."""
        return sum(len(lm.expert_ids) for lm in self.layers.values())


@dataclass
class ForwardResult:
    """Everything returned from a forward pass."""
    loss: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    snapshots: List[LayerSnapshot]
    expert_invocations: int
    tokens_processed: int


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
    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Standard forward pass with full routing."""
        ...

    # --- Intervention ---
    def forward_with_motif(
        self,
        inputs: torch.Tensor,
        motif: MotifSpec,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Force specific expert routing across specified layers (cheap path).
        Unspecified layers route normally. All-or-nothing execution."""
        ...

    def forward_counterfactual(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Run same input with overlap state suppressed (quench test).
        Raises NotImplementedError if supports_overlap is False."""
        ...

    # --- Context ---
    def get_context_embedding(self) -> torch.Tensor:
        """One canonical context embedding per decision. [B*T, D].
        Adapter-defined semantics. Consistent within backend. Versioned."""
        ...
