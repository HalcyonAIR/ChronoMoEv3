"""
HuggingFace MoE adapter: full BackendAdapter for Bob substrate.

Works with any HuggingFace MoE model that has a router/gate module.
Supports forward(), forward_with_motif() (expert forcing via gate hooks),
and capture_routing() (observation-only, backward compatible).

Tested with: OLMoE, Mixtral, Qwen2-MoE.

Usage::

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    adapter = HFMoEAdapter(model)

    # Observation only (eligibility check)
    capture = adapter.capture_routing(tokenizer("Hello", return_tensors="pt")["input_ids"])

    # Full BackendAdapter (Bob substrate)
    result = adapter.forward(input_ids, labels)
    result = adapter.forward_with_motif(input_ids, motif_spec, labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from backends.adapter import (
    ForwardResult,
    LayerMotif,
    LayerSnapshot,
    MotifSpec,
    OverlapKind,
)


@dataclass
class RoutingCapture:
    """Captured routing data from one forward pass."""
    snapshots: List[LayerSnapshot]
    num_experts: int
    top_k: int
    num_layers: int


class HFMoEAdapter:
    """
    Full BackendAdapter for HuggingFace MoE models.

    Captures routing telemetry and supports expert forcing via gate hooks.
    For forward_with_motif: overrides gate outputs to force specific expert
    selection. The MoE block uses the forced logits for routing, so expert
    execution follows the motif exactly.

    Gate override approach: set forced logits to log(weight) + 50.0 for
    desired experts, -1e4 for others. After softmax + top-k, the desired
    experts are selected with approximately the desired weights.
    """

    def __init__(
        self,
        model: nn.Module,
        moe_layer_pattern: str = "auto",
    ):
        self.model = model
        self._hooks: List = []
        self._captured: Dict[int, dict] = {}
        self._last_tokens_processed: int = 0

        # Discover MoE layers
        self._moe_layers = self._find_moe_layers(moe_layer_pattern)
        if not self._moe_layers:
            raise ValueError(
                "No MoE layers found. Check model architecture. "
                "Supported: OLMoE, Mixtral, Qwen2-MoE."
            )

        # Extract config (stored as plain attributes, not @property,
        # matching the naming the existing capture_routing callers expect)
        self.num_experts = self._detect_num_experts()
        self.top_k = self._detect_top_k()
        self.num_layers = len(self._moe_layers)

    # --- BackendAdapter protocol properties ---

    @property
    def adapter_version(self) -> str:
        return "hf-v2"

    @property
    def supports_overlap(self) -> bool:
        return False

    @property
    def overlap_kind(self) -> OverlapKind:
        return OverlapKind.NONE

    # --- MoE layer discovery (unchanged) ---

    def _find_moe_layers(self, pattern: str) -> Dict[int, nn.Module]:
        """Find MoE gate modules in the model."""
        moe_layers = {}
        patterns = [
            ("model.layers", "mlp"),
            ("model.layers", "block_sparse_moe"),
            ("layers", "mlp"),
            ("layers", "moe"),
        ]
        for base_path, moe_attr in patterns:
            try:
                obj = self.model
                for part in base_path.split("."):
                    obj = getattr(obj, part)
                for i, layer in enumerate(obj):
                    moe = getattr(layer, moe_attr, None)
                    if moe is not None and self._is_moe_block(moe):
                        moe_layers[i] = moe
                if moe_layers:
                    break
            except (AttributeError, TypeError):
                continue
        return moe_layers

    def _is_moe_block(self, module: nn.Module) -> bool:
        has_gate = any(
            hasattr(module, name)
            for name in ["gate", "router", "gate_proj"]
        )
        has_experts = any(
            hasattr(module, name)
            for name in ["experts", "expert_list"]
        )
        return has_gate or has_experts

    def _detect_num_experts(self) -> int:
        for attr in ["num_experts", "num_local_experts", "n_routed_experts"]:
            val = getattr(self.model.config, attr, None)
            if val is not None:
                return val
        first_moe = next(iter(self._moe_layers.values()))
        for attr in ["num_experts", "num_local_experts"]:
            val = getattr(first_moe, attr, None)
            if val is not None:
                return val
        experts = getattr(first_moe, "experts", None)
        if experts is not None:
            return len(experts)
        raise ValueError("Cannot detect num_experts from model")

    def _detect_top_k(self) -> int:
        for attr in ["num_experts_per_tok", "num_experts_per_token",
                      "top_k", "num_selected_experts"]:
            val = getattr(self.model.config, attr, None)
            if val is not None:
                return val
        return 2

    # --- Hook management ---

    def _find_gate(self, moe: nn.Module) -> Optional[nn.Module]:
        """Find the gate/router module within an MoE block."""
        for name in ["gate", "router"]:
            gate = getattr(moe, name, None)
            if gate is not None:
                return gate
        return None

    def _install_combined_hooks(
        self, motif: Optional[MotifSpec] = None,
    ) -> None:
        """Install hooks that capture routing and optionally override it.

        For layers in the motif: override gate output with forced logits,
        then capture the resulting routing decision.
        For other layers: capture only.
        """
        self._remove_hooks()
        self._captured = {}

        for layer_id, moe in self._moe_layers.items():
            gate = self._find_gate(moe)
            if gate is None:
                continue

            layer_motif = None
            if motif is not None:
                layer_motif = motif.layers.get(layer_id)

            def make_hook(lid, lm, tk, ne):
                def hook(module, input, output):
                    # --- Override gate output for motif layers ---
                    if lm is not None:
                        if isinstance(output, tuple):
                            orig_logits = output[0]
                        else:
                            orig_logits = output

                        forced = torch.full_like(orig_logits, -1e4)
                        for i, eid in enumerate(lm.expert_ids):
                            w = lm.weights[i] if i < len(lm.weights) else 1.0 / len(lm.expert_ids)
                            forced[:, eid] = math.log(max(w, 1e-8)) + 50.0

                        if isinstance(output, tuple):
                            output = (forced,) + output[1:]
                        else:
                            output = forced

                    # --- Capture routing from (possibly modified) output ---
                    if isinstance(output, tuple):
                        if len(output) >= 3:
                            logits = output[0]
                            weights = output[1]
                            indices = output[2]
                        elif len(output) == 2:
                            logits = output[0]
                            probs = F.softmax(logits.float(), dim=-1)
                            weights, indices = torch.topk(probs, tk, dim=-1)
                        else:
                            logits = output[0]
                            probs = F.softmax(logits.float(), dim=-1)
                            weights, indices = torch.topk(probs, tk, dim=-1)
                    else:
                        logits = output
                        probs = F.softmax(logits.float(), dim=-1)
                        weights, indices = torch.topk(probs, tk, dim=-1)

                    self._captured[lid] = {
                        "router_logits": logits.detach(),
                        "selected_experts": indices.detach(),
                        "routing_weights": weights.detach(),
                    }

                    return output
                return hook

            h = gate.register_forward_hook(
                make_hook(layer_id, layer_motif, self.top_k, self.num_experts)
            )
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _build_snapshots(self) -> List[LayerSnapshot]:
        """Build LayerSnapshots from captured routing data."""
        snapshots = []
        for layer_id in sorted(self._captured.keys()):
            data = self._captured[layer_id]
            logits = data["router_logits"]
            selected = data["selected_experts"]
            weights = data["routing_weights"]

            num_tokens = selected.shape[0]
            total_selections = num_tokens * selected.shape[1]
            flat_selected = selected.view(-1)
            expert_usage = torch.zeros(self.num_experts, device=selected.device)
            expert_usage.scatter_add_(
                0, flat_selected.long(),
                torch.ones_like(flat_selected, dtype=expert_usage.dtype),
            )
            expert_usage = expert_usage / max(total_selections, 1)

            snapshots.append(LayerSnapshot(
                layer_id=layer_id,
                router_scores=logits.float() if logits.dim() == 2 else logits,
                selected_experts=selected,
                routing_weights=weights,
                expert_usage=expert_usage,
            ))
        return snapshots

    # --- BackendAdapter: observation ---

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Standard forward pass with full routing.

        Args:
            inputs: input_ids [B, T]
            targets: labels for loss computation [B, T].
                     For standard LM, pass input_ids as targets
                     (HF handles the internal shift).
        """
        self._install_combined_hooks(motif=None)

        with torch.no_grad():
            kwargs = {"input_ids": inputs}
            if targets is not None:
                kwargs["labels"] = targets
            outputs = self.model(**kwargs)

        snapshots = self._build_snapshots()
        self._remove_hooks()

        loss = getattr(outputs, "loss", None)
        logits = getattr(outputs, "logits", None)

        num_tokens = inputs.shape[0] * inputs.shape[1] if inputs.dim() == 2 else inputs.shape[0]
        total_invocations = num_tokens * self.top_k * self.num_layers
        self._last_tokens_processed = num_tokens

        return ForwardResult(
            loss=loss,
            logits=logits,
            snapshots=snapshots,
            expert_invocations=total_invocations,
            tokens_processed=num_tokens,
        )

    # --- BackendAdapter: intervention ---

    def forward_with_motif(
        self,
        inputs: torch.Tensor,
        motif: MotifSpec,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        """Force specific expert routing at motif layers.

        Overrides gate outputs via hooks so the MoE block selects exactly
        the motif experts. Non-motif layers route normally.
        """
        self._install_combined_hooks(motif=motif)

        with torch.no_grad():
            kwargs = {"input_ids": inputs}
            if targets is not None:
                kwargs["labels"] = targets
            outputs = self.model(**kwargs)

        snapshots = self._build_snapshots()
        self._remove_hooks()

        loss = getattr(outputs, "loss", None)
        logits = getattr(outputs, "logits", None)

        num_tokens = inputs.shape[0] * inputs.shape[1] if inputs.dim() == 2 else inputs.shape[0]

        # Compute invocations: motif layers use motif expert count, others use top_k
        total_invocations = 0
        for layer_id in self._moe_layers:
            if layer_id in motif.layers:
                total_invocations += num_tokens * len(motif.layers[layer_id].expert_ids)
            else:
                total_invocations += num_tokens * self.top_k

        self._last_tokens_processed = num_tokens

        return ForwardResult(
            loss=loss,
            logits=logits,
            snapshots=snapshots,
            expert_invocations=total_invocations,
            tokens_processed=num_tokens,
        )

    def forward_counterfactual(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        raise NotImplementedError(
            "HFMoEAdapter has no overlap state (OverlapKind.NONE). "
            "Counterfactual testing not available."
        )

    def get_context_embedding(self) -> torch.Tensor:
        raise NotImplementedError(
            "Context embedding not yet implemented for HF models."
        )

    # --- Backward-compatible observation-only API ---

    def capture_routing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> RoutingCapture:
        """Run one forward pass and capture routing at every MoE layer.

        Backward compatible with existing eligibility check code.
        """
        self._install_combined_hooks(motif=None)

        with torch.no_grad():
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            self.model(**kwargs)

        snapshots = self._build_snapshots()
        self._remove_hooks()

        return RoutingCapture(
            snapshots=snapshots,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_layers=self.num_layers,
        )

    def info(self) -> Dict:
        """Model and MoE architecture summary."""
        return {
            "model_class": type(self.model).__name__,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_moe_layers": self.num_layers,
            "moe_layer_ids": sorted(self._moe_layers.keys()),
        }
