"""
Swiss-AI/MoE Integration Wrapper.

Wraps swiss-ai/MoE layer to extract ChronoMoEv3 signals without modifying
the original architecture.

Key operations:
1. Inject ExpertRegistry active mask before topk selection
2. Hook expert loop to capture per-expert outputs
3. Build MoETrace from captured signals
4. Maintain identical forward pass (verified by tests)

Based on swiss-ai/MoE architecture:
- Router: nn.Linear(n_embd, num_experts) → softmax → topk
- Experts: nn.ModuleList of MLP modules
- Dispatch: Loop over experts, accumulate weighted outputs
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..coherence import MoETrace
from ..expert_registry import ExpertRegistry


class SwissMoEWrapper(nn.Module):
    """
    Wrapper for swiss-ai/MoE layer with ChronoMoEv3 signal extraction.

    Hooks into the forward pass to:
    - Capture per-expert outputs (for coherence, redundancy, bimodality)
    - Apply ExpertRegistry active mask (for fixed-width routing)
    - Build MoETrace with full signal suite

    Design:
    - Non-invasive: Doesn't modify swiss-ai/MoE source
    - Drop-in replacement: Same forward signature
    - Zero overhead if registry=None (just passthrough)
    """

    def __init__(
        self,
        moe_layer: nn.Module,
        layer_id: int,
        registry: Optional[ExpertRegistry] = None,
        capture_signals: bool = True,
    ):
        """
        Wrap a swiss-ai/MoE layer.

        Args:
            moe_layer: The swiss-ai MoE layer to wrap
            layer_id: Layer index (for registry matching)
            registry: Optional ExpertRegistry (enables masking)
            capture_signals: Whether to capture per-expert outputs
        """
        super().__init__()

        self.moe_layer = moe_layer
        self.layer_id = layer_id
        self.registry = registry
        self.capture_signals = capture_signals

        # Verify moe_layer has expected attributes
        if not hasattr(moe_layer, 'router'):
            raise ValueError("moe_layer must have 'router' attribute")
        if not hasattr(moe_layer, 'experts'):
            raise ValueError("moe_layer must have 'experts' attribute")

        # Cache some config
        self.num_experts = len(moe_layer.experts)
        self.top_k = getattr(moe_layer, 'top_k', 2)

        # Signal capture buffers (cleared each forward pass)
        self.last_trace: Optional[MoETrace] = None
        self._expert_outputs: List[Tensor] = []
        self._expert_token_indices: List[Tensor] = []
        self._expert_gate_weights: List[Tensor] = []

    def forward(
        self,
        x: Tensor,
        return_trace: bool = False,
    ):
        """
        Forward pass with signal extraction.

        Args:
            x: Input tensor [batch, seq_len, d_model] or [batch*seq_len, d_model]
            return_trace: Whether to return MoETrace

        Returns:
            If return_trace=False: Just output tensor (drop-in replacement)
            If return_trace=True: Tuple of (output, trace)
        """
        # Clear capture buffers
        self._expert_outputs = []
        self._expert_token_indices = []
        self._expert_gate_weights = []

        # Flatten if needed
        original_shape = x.shape
        if x.ndim == 3:
            batch_size, seq_len, d_model = x.shape
            x = x.view(batch_size * seq_len, d_model)
        else:
            batch_size, seq_len = None, None
            d_model = x.shape[-1]

        num_tokens = x.shape[0]

        # Router logits
        router_logits = self.moe_layer.router(x)  # [num_tokens, num_experts]

        # Apply active mask if registry provided
        if self.registry is not None:
            active_mask = self.registry.get_active_mask()  # [max_experts]
            # Zero out inactive experts before softmax
            inactive_mask = ~active_mask[:self.num_experts]
            if inactive_mask.any():
                router_logits = router_logits.clone()
                router_logits[:, inactive_mask] = float('-inf')

        # Router probabilities
        router_probs = F.softmax(router_logits, dim=1)  # [num_tokens, num_experts]

        # Top-k selection
        weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=1
        )  # [num_tokens, top_k]

        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Initialize output accumulator
        output = torch.zeros_like(x)  # [num_tokens, d_model]

        # Expert dispatch loop (with signal capture)
        active_expert_ids = []

        for expert_id in range(self.num_experts):
            # Check if this expert was selected for any token
            expert_mask = (selected_experts == expert_id).any(dim=1)
            if not expert_mask.any():
                continue  # No tokens routed to this expert

            active_expert_ids.append(expert_id)

            # Get tokens routed to this expert
            batch_idx = torch.where(expert_mask)[0]  # [n_tokens_for_expert]

            # Get gate weights for these tokens
            # For each token, find which topk position this expert is at
            expert_positions = (selected_experts[batch_idx] == expert_id).long()
            # Get weights at those positions
            gate_weights = weights[batch_idx].gather(1, expert_positions.argmax(1, keepdim=True)).squeeze(1)

            # Expert forward pass
            expert_input = x[batch_idx]
            expert_output = self.moe_layer.experts[expert_id](expert_input)

            # Accumulate weighted output
            weighted_output = expert_output * gate_weights.unsqueeze(1)
            output[batch_idx] += weighted_output

            # Capture signals if enabled
            if self.capture_signals:
                # Mean expert output (for coherence)
                expert_mean = expert_output.mean(dim=0)
                self._expert_outputs.append(expert_mean)
                self._expert_token_indices.append(batch_idx)
                self._expert_gate_weights.append(gate_weights)

        # Reshape if needed
        if batch_size is not None and seq_len is not None:
            output = output.view(batch_size, seq_len, d_model)

        # Build MoETrace if requested
        trace = None
        if self.capture_signals:
            trace = MoETrace(
                mixture=output.view(-1, d_model),  # Flatten for trace
                router_logits_clean=router_logits,
                router_probs=router_probs,
                active_expert_ids=active_expert_ids,
                expert_mean_outputs=self._expert_outputs,
                token_row_indices=self._expert_token_indices,
                gate_weights=self._expert_gate_weights,
            )
            self.last_trace = trace

        # Return based on return_trace flag
        if return_trace:
            return output, trace
        else:
            return output

    def get_last_trace(self) -> Optional[MoETrace]:
        """Get MoETrace from last forward pass."""
        return self.last_trace

    def set_registry(self, registry: ExpertRegistry):
        """Update the ExpertRegistry (for dynamic switching)."""
        if registry.layer_id != self.layer_id:
            raise ValueError(
                f"Registry layer_id {registry.layer_id} doesn't match "
                f"wrapper layer_id {self.layer_id}"
            )
        self.registry = registry

    def unwrap(self) -> nn.Module:
        """Get the underlying swiss-ai/MoE layer."""
        return self.moe_layer

    def status_summary(self) -> str:
        """Human-readable status."""
        registry_status = (
            self.registry.status_summary() if self.registry else "No registry"
        )
        return (
            f"SwissMoEWrapper(layer={self.layer_id}, "
            f"experts={self.num_experts}, top_k={self.top_k}, "
            f"registry={registry_status})"
        )


def wrap_moe_model(
    model: nn.Module,
    moe_layer_names: List[str],
    registries: Optional[Dict[int, ExpertRegistry]] = None,
) -> Dict[str, SwissMoEWrapper]:
    """
    Wrap all MoE layers in a model.

    Args:
        model: The model containing MoE layers
        moe_layer_names: List of attribute paths to MoE layers
            (e.g., ["transformer.h.0.mlp", "transformer.h.1.mlp"])
        registries: Optional dict mapping layer_id → ExpertRegistry

    Returns:
        Dictionary mapping layer_name → SwissMoEWrapper

    Example:
        >>> model = load_swiss_moe_model()
        >>> registries = {
        ...     0: ExpertRegistry(layer_id=0, max_experts=32, initial_active=8),
        ...     1: ExpertRegistry(layer_id=1, max_experts=32, initial_active=8),
        ... }
        >>> wrappers = wrap_moe_model(
        ...     model,
        ...     moe_layer_names=["transformer.h.0.mlp", "transformer.h.1.mlp"],
        ...     registries=registries,
        ... )
    """
    wrappers = {}

    for layer_id, layer_name in enumerate(moe_layer_names):
        # Navigate to the layer (handle nested attributes)
        parts = layer_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Get the MoE layer
        moe_layer = getattr(parent, parts[-1])

        # Get registry if provided
        registry = registries.get(layer_id) if registries else None

        # Wrap it
        wrapper = SwissMoEWrapper(
            moe_layer=moe_layer,
            layer_id=layer_id,
            registry=registry,
        )

        # Replace in model
        setattr(parent, parts[-1], wrapper)

        wrappers[layer_name] = wrapper

    return wrappers


def extract_traces_from_wrappers(
    wrappers: Dict[str, SwissMoEWrapper]
) -> Dict[str, MoETrace]:
    """
    Extract MoETrace from all wrappers after a forward pass.

    Args:
        wrappers: Dictionary from wrap_moe_model()

    Returns:
        Dictionary mapping layer_name → MoETrace
    """
    traces = {}
    for layer_name, wrapper in wrappers.items():
        trace = wrapper.get_last_trace()
        if trace is not None:
            traces[layer_name] = trace
    return traces
