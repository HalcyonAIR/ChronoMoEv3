"""
SwissAdapter: wraps swiss-ai/MoE models for Bob.

Designed for small GPT-2 style MoE models used during development.
This backend has no overlap state (pure feedforward MoE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from backends.adapter import (
    BackendAdapter,
    ForwardResult,
    LayerMotif,
    LayerSnapshot,
    MotifSpec,
    OverlapKind,
)


class SwissAdapter:
    """
    Adapter for swiss-ai/MoE models.

    Wraps a model that has:
    - An embedding layer
    - One or more MoE layers (from moe.py)
    - A language model head

    Usage::

        model = TinyMoEModel(config)
        adapter = SwissAdapter(model, moe_layers={0: model.moe})
        result = adapter.forward(inputs, targets)
        # result.snapshots[0].router_scores  -> [B*T, num_experts]
    """

    def __init__(
        self,
        model: nn.Module,
        moe_layers: Dict[int, nn.Module],
        embed_fn=None,
        head_fn=None,
        norm_fn=None,
    ):
        """
        Args:
            model: The full model (for parameter access, training)
            moe_layers: Dict of {layer_id: MoE module}
            embed_fn: Function(model, x) -> embedded x. If None, uses model.embedding(x).
            head_fn: Function(model, x) -> logits. If None, uses model.output(x).
            norm_fn: Function(model, x) -> normed x. If None, uses model.norm(x).
        """
        self.model = model
        self.moe_layers = moe_layers
        self._embed_fn = embed_fn or (lambda m, x: m.embedding(x))
        self._head_fn = head_fn or (lambda m, x: m.output(x))
        self._norm_fn = norm_fn or (lambda m, x: m.norm(x))

        self._last_snapshots: List[LayerSnapshot] = []
        self._last_context_embedding: Optional[torch.Tensor] = None
        self._last_expert_invocations: int = 0
        self._last_tokens_processed: int = 0

        # Cache MoE properties from first layer
        first_moe = next(iter(moe_layers.values()))
        self._num_experts = len(first_moe.experts)
        self._top_k = first_moe.top_k

    # --- Identity ---

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_layers(self) -> int:
        return len(self.moe_layers)

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def adapter_version(self) -> str:
        return "swiss-v1"

    # --- Overlap ---

    @property
    def supports_overlap(self) -> bool:
        return False

    @property
    def overlap_kind(self) -> OverlapKind:
        return OverlapKind.NONE

    # --- Observation ---

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Standard forward pass through all MoE layers with full routing."""
        self._last_snapshots = []
        total_invocations = 0

        # Embed
        x = self._embed_fn(self.model, inputs)
        self._last_context_embedding = x.view(-1, x.shape[-1]).detach()
        self._last_tokens_processed = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]

        # Forward through MoE layers in order
        for layer_id in sorted(self.moe_layers.keys()):
            moe = self.moe_layers[layer_id]
            moe_out, routing_info = moe(x)

            # Build snapshot
            router_logits = routing_info["router_logits"]
            selected_experts = routing_info["selected_experts"]
            router_probs = F.softmax(router_logits, dim=-1)

            # Compute routing weights for selected experts
            routing_weights = torch.gather(
                router_probs, 1, selected_experts
            )

            # Expert usage: count tokens routed to each expert
            expert_usage = torch.zeros(self._num_experts, device=inputs.device)
            for e in range(self._num_experts):
                expert_usage[e] = (selected_experts == e).any(dim=-1).float().sum()

            snapshot = LayerSnapshot(
                layer_id=layer_id,
                router_scores=router_probs.detach(),
                selected_experts=selected_experts.detach(),
                routing_weights=routing_weights.detach(),
                expert_usage=expert_usage.detach(),
            )
            self._last_snapshots.append(snapshot)

            # Residual connection
            x = x + moe_out
            total_invocations += self._top_k * self._last_tokens_processed

        # Norm + head
        x = self._norm_fn(self.model, x)
        logits = self._head_fn(self.model, x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        self._last_expert_invocations = total_invocations

        return ForwardResult(
            loss=loss,
            logits=logits,
            snapshots=self._last_snapshots,
            expert_invocations=total_invocations,
            tokens_processed=self._last_tokens_processed,
        )

    # --- Intervention ---

    def forward_with_motif(
        self,
        inputs: torch.Tensor,
        motif: MotifSpec,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Execute specific experts at specified layers. Other layers route normally."""
        self._last_snapshots = []
        total_invocations = 0

        x = self._embed_fn(self.model, inputs)
        self._last_context_embedding = x.view(-1, x.shape[-1]).detach()
        self._last_tokens_processed = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        num_tokens = self._last_tokens_processed

        for layer_id in sorted(self.moe_layers.keys()):
            moe = self.moe_layers[layer_id]

            if layer_id in motif.layers:
                # Cheap path: execute only motif experts
                layer_motif = motif.layers[layer_id]
                moe_out, routing_info = self._execute_motif_at_layer(
                    moe, x, layer_motif
                )
                invocations = len(layer_motif.expert_ids) * num_tokens
            else:
                # Normal routing
                moe_out, routing_info = moe(x)
                invocations = self._top_k * num_tokens

            # Build snapshot
            router_logits = routing_info["router_logits"]
            selected_experts = routing_info["selected_experts"]
            router_probs = F.softmax(router_logits, dim=-1)
            routing_weights = torch.gather(router_probs, 1, selected_experts)

            expert_usage = torch.zeros(self._num_experts, device=inputs.device)
            for e in range(self._num_experts):
                expert_usage[e] = (selected_experts == e).any(dim=-1).float().sum()

            snapshot = LayerSnapshot(
                layer_id=layer_id,
                router_scores=router_probs.detach(),
                selected_experts=selected_experts.detach(),
                routing_weights=routing_weights.detach(),
                expert_usage=expert_usage.detach(),
            )
            self._last_snapshots.append(snapshot)

            x = x + moe_out
            total_invocations += invocations

        x = self._norm_fn(self.model, x)
        logits = self._head_fn(self.model, x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        self._last_expert_invocations = total_invocations

        return ForwardResult(
            loss=loss,
            logits=logits,
            snapshots=self._last_snapshots,
            expert_invocations=total_invocations,
            tokens_processed=self._last_tokens_processed,
        )

    def _execute_motif_at_layer(
        self,
        moe: nn.Module,
        inputs: torch.Tensor,
        layer_motif: LayerMotif,
    ) -> tuple:
        """Execute specific experts with specific weights at one MoE layer."""
        inputs_squashed = inputs.view(-1, inputs.shape[-1])

        # Still compute router logits for telemetry
        router_logits = moe.router(inputs_squashed)

        # Execute only motif experts
        results = torch.zeros_like(inputs_squashed)
        for eid, w in zip(layer_motif.expert_ids, layer_motif.weights):
            expert_out, _ = moe.experts[eid](inputs_squashed)
            results += w * expert_out

        # Build routing_info matching normal MoE output format
        # Selected experts: all tokens see the motif experts
        num_tokens = inputs_squashed.shape[0]
        selected = torch.tensor(
            [list(layer_motif.expert_ids)] * num_tokens,
            device=inputs.device,
        )

        return results.view_as(inputs), {
            "router_logits": router_logits,
            "selected_experts": selected,
        }

    def forward_counterfactual(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Not supported for swiss adapter (no overlap state)."""
        raise NotImplementedError(
            "SwissAdapter has no overlap state (OverlapKind.NONE). "
            "Counterfactual testing not available."
        )

    # --- Context ---

    def get_context_embedding(self) -> torch.Tensor:
        """Pre-MoE embeddings as context signal. [B*T, D]."""
        if self._last_context_embedding is None:
            raise RuntimeError("No forward pass has been run yet")
        return self._last_context_embedding
