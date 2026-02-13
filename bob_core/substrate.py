"""
BobSubstrate: the main orchestrator.

observe() -> decide() -> update() -> log()

Bob reads adapter snapshots, evaluates the compound gate,
decides cheap or expensive path, and updates the motif store.
"""

from typing import Dict, List, Optional, Tuple

from backends.adapter import (
    BackendAdapter,
    ForwardResult,
    LayerMotif,
    LayerSnapshot,
    MotifSpec,
)
from bob_core.motifs import (
    CompoundGate,
    GateResult,
    GateSignals,
    GateThresholds,
    MotifRecord,
    MotifStore,
)
from bob_core.telemetry import DecisionTrace


class BobSubstrate:
    """
    Consequence-accumulating control plane for MoE models.

    Usage::

        adapter = SwissAdapter(model, moe_layers={0: model.moe})
        bob = BobSubstrate(adapter)

        for step in range(total_steps):
            task_class, inputs, targets = ladder.get_batch(step)
            trace = bob.step(inputs, targets, task_class, step)
            # trace.path is "cheap" or "full"
            # trace.expert_invocations is the cost
            # trace.loss is the quality

    Args:
        adapter: BackendAdapter implementation
        gate_thresholds: Thresholds for the compound gate
        warmup_steps: Steps before cheap path is offered
        governance_state: Fixed governance state (simplified for v1 experiment)
        motif_store_kwargs: Additional kwargs for MotifStore
    """

    def __init__(
        self,
        adapter: BackendAdapter,
        gate_thresholds: Optional[GateThresholds] = None,
        warmup_steps: int = 500,
        governance_state: str = "EQUILIBRIUM",
        **motif_store_kwargs,
    ):
        self.adapter = adapter
        self.gate = CompoundGate(gate_thresholds)
        self.store = MotifStore(**motif_store_kwargs)
        self.warmup_steps = warmup_steps
        self.governance_state = governance_state

        self.traces: List[DecisionTrace] = []

    def step(
        self,
        inputs,
        targets,
        context_class: int,
        step: int,
    ) -> DecisionTrace:
        """
        Run one decision through Bob.

        1. Compute gate signals
        2. If gate passes and past warmup: cheap path
        3. Otherwise: expensive path
        4. Update motif store
        5. Log trace

        Returns the decision trace (the atomic artifact).
        """
        # --- Decide ---
        signals = self.store.get_gate_signals(context_class, step)

        # During warmup, force expensive path
        if step < self.warmup_steps:
            gate_result = GateResult(
                passed=False,
                signals=signals,
                thresholds_used=self.gate.thresholds,
                governance_state=self.governance_state,
                stability_passed=False,
                debt_passed=False,
                survival_passed=False,
            )
        else:
            gate_result = self.gate.evaluate(signals, self.governance_state)

        # --- Execute ---
        top_motif = self.store.get_top_motif(context_class, step)

        if gate_result.passed and top_motif is not None:
            # Cheap path: force the cached expert set
            result = self.adapter.forward_with_motif(
                inputs, top_motif.motif_spec, targets
            )
            path = "cheap"
            motif_id = top_motif.motif_id
            first_layer = next(iter(top_motif.motif_spec.layers.values()))
            expert_ids = first_layer.expert_ids
        else:
            # Expensive path: full routing
            result = self.adapter.forward(inputs, targets)
            path = "full"
            motif_id = None
            expert_ids = ()  # Will be set by _extract_motif below

        loss_val = result.loss.item() if result.loss is not None else float("inf")

        # --- Update ---
        # Build motif from actual routing pattern (token-level aggregation)
        actual_motif_spec, extracted_ids = self._extract_motif(
            result, self.adapter.num_experts
        )
        if not expert_ids:
            expert_ids = extracted_ids

        self.store.update(
            context_class=context_class,
            expert_ids=expert_ids,
            motif_spec=actual_motif_spec,
            loss=loss_val,
            step=step,
            was_cheap=(path == "cheap"),
        )

        # --- Log ---
        trace = DecisionTrace(
            step=step,
            context_class=context_class,
            governance_state=self.governance_state,
            path=path,
            expert_ids=expert_ids,
            expert_invocations=result.expert_invocations,
            tokens_processed=result.tokens_processed,
            loss=loss_val,
            routing_stability=signals.routing_stability,
            debt_level=signals.debt_level,
            motif_survival=signals.motif_survival,
            gate_passed=gate_result.passed,
            stability_passed=gate_result.stability_passed,
            debt_passed=gate_result.debt_passed,
            survival_passed=gate_result.survival_passed,
            motif_id=motif_id,
        )
        self.traces.append(trace)
        return trace

    def _extract_motif(
        self,
        result: ForwardResult,
        num_experts: int,
    ) -> Tuple[MotifSpec, Tuple[int, ...]]:
        """
        Build a MotifSpec from actual token-level routing.

        Aggregates per-token expert selections across the batch to find
        experts that handle a disproportionate share of tokens. The motif
        includes only experts whose selection frequency exceeds the uniform
        baseline by a meaningful margin.

        If no expert is disproportionately dominant (routing is class-agnostic),
        falls back to the most common top-k pair. The cheap path cost then
        equals full routing cost — correctly reflecting that no savings are
        possible without routing specialization.

        Returns (motif_spec, representative_expert_ids).
        """
        layers = {}
        representative_ids = ()

        # Dominance threshold: expert must handle more than uniform + margin
        uniform_fraction = 1.0 / num_experts
        dominance_margin = 0.15  # 15% above uniform to qualify as "dominant"
        dominance_threshold = uniform_fraction + dominance_margin

        for snap in result.snapshots:
            selected = snap.selected_experts  # [B*T, top_k]
            routing_w = snap.routing_weights  # [B*T, top_k]
            num_tokens = selected.shape[0]
            total_selections = num_tokens * selected.shape[1]

            # Count per-expert: selection frequency and aggregate routing weight
            expert_counts: Dict[int, int] = {}
            expert_weight_sums: Dict[int, float] = {}
            for i in range(num_tokens):
                for j in range(selected.shape[1]):
                    eid = int(selected[i, j].item())
                    expert_counts[eid] = expert_counts.get(eid, 0) + 1
                    expert_weight_sums[eid] = (
                        expert_weight_sums.get(eid, 0.0)
                        + routing_w[i, j].item()
                    )

            if not expert_counts:
                continue

            # Find experts above dominance threshold
            dominant = [
                eid for eid, cnt in expert_counts.items()
                if cnt / total_selections > dominance_threshold
            ]

            if dominant:
                # Specialization exists: use only dominant experts
                dominant.sort(key=lambda e: expert_counts[e], reverse=True)
                expert_ids = tuple(dominant)
            else:
                # No specialization: use the most common top-k pair
                # This preserves quality but doesn't reduce cost
                pair_counts: Dict[Tuple[int, ...], int] = {}
                for i in range(num_tokens):
                    pair = tuple(sorted(
                        int(selected[i, j].item())
                        for j in range(selected.shape[1])
                    ))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                best_pair = max(pair_counts, key=pair_counts.get)
                expert_ids = best_pair

            # Compute weights from aggregate routing weights
            total_weight = sum(expert_weight_sums.get(e, 0) for e in expert_ids)
            if total_weight > 0:
                weights = tuple(
                    expert_weight_sums.get(e, 0) / total_weight
                    for e in expert_ids
                )
            else:
                weights = tuple(1.0 / len(expert_ids) for _ in expert_ids)

            layers[snap.layer_id] = LayerMotif(
                expert_ids=expert_ids,
                weights=weights,
            )

            if not representative_ids:
                representative_ids = expert_ids

        return MotifSpec(layers=layers), representative_ids

    def get_traces_window(self, last_n: int) -> List[DecisionTrace]:
        """Get the last N traces."""
        return self.traces[-last_n:]

    def get_pareto_data(self, window: int = 50) -> List[Dict]:
        """
        Get (cost, quality) pairs per window for Pareto analysis.

        Returns list of dicts with avg_cost, avg_loss, cheap_fraction per window.
        """
        data = []
        for i in range(0, len(self.traces), window):
            window_traces = self.traces[i : i + window]
            if not window_traces:
                continue

            avg_cost = sum(t.expert_invocations for t in window_traces) / len(
                window_traces
            )
            avg_loss = sum(t.loss for t in window_traces) / len(window_traces)
            cheap_count = sum(1 for t in window_traces if t.path == "cheap")
            cheap_frac = cheap_count / len(window_traces)

            data.append(
                {
                    "window_start": window_traces[0].step,
                    "window_end": window_traces[-1].step,
                    "avg_cost": avg_cost,
                    "avg_loss": avg_loss,
                    "cheap_fraction": cheap_frac,
                    "n_traces": len(window_traces),
                }
            )
        return data
