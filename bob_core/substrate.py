"""
BobSubstrate: the main orchestrator.

observe() -> decide() -> update() -> log()

Bob reads adapter snapshots, evaluates the compound gate,
decides cheap or expensive path, and updates the motif store.

Phase 1: optional BobCore + Governor + MediumClock wiring.
If bob_core/governor are None -> existing behavior preserved.
"""

import math
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
        bob_core: Optional BobCore for ledger-based lifecycle
        governor: Optional BobGovernor for commit authorization
        medium_clock: Optional MediumClock for instability detection
        promotion_gate: Optional PromotionGate for stability tracking
        motif_store_kwargs: Additional kwargs for MotifStore
    """

    def __init__(
        self,
        adapter: BackendAdapter,
        gate_thresholds: Optional[GateThresholds] = None,
        warmup_steps: int = 500,
        governance_state: str = "EQUILIBRIUM",
        bob_core=None,
        governor=None,
        medium_clock=None,
        promotion_gate=None,
        **motif_store_kwargs,
    ):
        self.adapter = adapter
        self.gate = CompoundGate(gate_thresholds)
        self.store = MotifStore(**motif_store_kwargs)
        self.warmup_steps = warmup_steps
        self.governance_state = governance_state

        # Phase 1 optional components
        self.bob_core = bob_core
        self.governor = governor
        self.medium_clock = medium_clock
        self.promotion_gate = promotion_gate

        # Track previous step state for medium clock
        self._prev_expert_ids: Optional[Tuple[int, ...]] = None
        self._prev_loss: Optional[float] = None
        self._first_commit_step: Optional[int] = None

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

        With governor/bob_core enabled:
        0. medium_clock.tick() - update instability EMAs
        1. Check forced exploration (from prior governor BLOCK)
        2. Query MotifStore - routing_stability, top_motif, survival
        3. Assemble GateSignals (stability from MotifStore, debt from BobCore if available)
        4. CompoundGate.evaluate() (unchanged)
        5. If gate passes AND governor exists:
           -> governor.evaluate_commit() - ALLOW/BLOCK/ESCALATE
           -> If BLOCK: expensive path, set forced exploration next
        6. Execute (cheap or expensive path)
        7. Build RoutingVector from snapshots (if bob_core exists)
        8. Determine success via bob_core.determine_success()
        9. bob_core.process_outcome() - commitment + scar + cost
        10. store.update() - patterns (unchanged, will slim later)
        11. promotion_gate.record() - track stability
        12. identity weight via identity boundary
        13. Log DecisionTrace (with governor verdict fields)

        Without governor/bob_core: existing behavior preserved.
        """
        was_blocked = False
        forced_exploration = False
        governor_decision = None
        governor_reasons = None
        medium_activation = None
        scar_debt = None
        cost_cheap_fraction = None
        commitment_id = None
        identity_weight = None

        # --- 0. Medium clock tick ---
        if self.medium_clock is not None:
            self.medium_clock.tick(
                prev_expert_ids=self._prev_expert_ids,
                curr_expert_ids=(),  # Updated after execution
                prev_loss=self._prev_loss,
                curr_loss=0.0,  # Updated after execution
                was_blocked=False,  # Updated after governor
                path="full",  # Updated after decision
            )
            medium_activation = self.medium_clock.activation

        # --- 1. Check forced exploration ---
        if self.governor is not None:
            forced_exploration = self.governor.consume_forced_exploration()

        # --- 2-3. Gate signals ---
        signals = self.store.get_gate_signals(context_class, step)

        # If bob_core exists, use scar-based debt instead of motif store debt
        if self.bob_core is not None:
            scar_debt = self.bob_core.scars.total_debt(step)

        # --- 4. Gate evaluation ---
        if step < self.warmup_steps or forced_exploration:
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

        # --- 5. Governor evaluation (if gate passed) ---
        top_motif = self.store.get_top_motif(context_class, step)

        if gate_result.passed and self.governor is not None and top_motif is not None:
            first_layer = next(iter(top_motif.motif_spec.layers.values()))
            candidate_ids = first_layer.expert_ids

            from bob_core.governor import GovernorDecision
            verdict = self.governor.evaluate_commit(
                context_class=context_class,
                expert_ids=candidate_ids,
                gate_result=gate_result,
                step=step,
            )
            governor_decision = verdict.decision.value
            governor_reasons = verdict.reasons
            cost_cheap_fraction = verdict.cost_signal.cheap_fraction

            if verdict.decision != GovernorDecision.ALLOW:
                # Governor blocked: force expensive path
                gate_result = GateResult(
                    passed=False,
                    signals=signals,
                    thresholds_used=self.gate.thresholds,
                    governance_state=self.governance_state,
                    stability_passed=gate_result.stability_passed,
                    debt_passed=False,
                    survival_passed=gate_result.survival_passed,
                )
                was_blocked = True

        # --- 6. Execute ---
        if gate_result.passed and top_motif is not None:
            # Compute bias strength: strong when stable, weak when unstable
            base_bias = 5.0
            instability_scale = 1.0 - (medium_activation or 0.0)  # 1.0=stable, 0.0=chaotic
            promotion_scale = 1.0
            if self.promotion_gate is not None:
                promotion_scale = self.promotion_gate.eligibility_score(context_class)
            bias = base_bias * instability_scale * max(0.3, promotion_scale)

            # Apply bias to all layers in the motif
            for lm in top_motif.motif_spec.layers.values():
                lm.bias_strength = bias

            result = self.adapter.forward_with_motif(
                inputs, top_motif.motif_spec, targets
            )
            path = "cheap"
            motif_id = top_motif.motif_id
            first_layer = next(iter(top_motif.motif_spec.layers.values()))
            expert_ids = first_layer.expert_ids
        else:
            result = self.adapter.forward(inputs, targets)
            path = "full"
            motif_id = None
            expert_ids = ()

        loss_val = result.loss.item() if result.loss is not None else float("inf")

        # --- 7. Extract motif + routing vector ---
        actual_motif_spec, extracted_ids, routing_key = self._extract_motif(
            result, self.adapter.num_experts
        )
        if not expert_ids:
            expert_ids = extracted_ids

        # Build routing vectors from snapshots (if bob_core exists)
        routing_vector = None
        if self.bob_core is not None and result.snapshots:
            from bob_core.ledgers import RoutingVector
            routing_vector = RoutingVector.from_snapshot(result.snapshots[0])

        # --- 8-9. BobCore outcome processing ---
        commitment = None
        if self.bob_core is not None:
            governance_coords = self.bob_core.get_governance_coords(
                context_class=context_class,
                step=step,
                medium=medium_activation or 0.0,
            )
            commitment = self.bob_core.process_outcome(
                context_class=context_class,
                step=step,
                loss=loss_val,
                was_cheap=(path == "cheap"),
                expert_ids=expert_ids,
                routing_vector=routing_vector,
                governance_coords=governance_coords,
                expert_invocations=result.expert_invocations,
                was_exploration=forced_exploration,
                was_blocked=was_blocked,
                motif_id=motif_id,
            )
            if commitment is not None:
                commitment_id = commitment.commitment_id
                if self._first_commit_step is None:
                    self._first_commit_step = step

        # --- 10. Update motif store ---
        self.store.update(
            context_class=context_class,
            expert_ids=expert_ids,
            motif_spec=actual_motif_spec,
            loss=loss_val,
            step=step,
            was_cheap=(path == "cheap"),
            routing_key=routing_key,
        )

        # --- 11. Promotion gate ---
        if self.promotion_gate is not None:
            self.promotion_gate.record(context_class, expert_ids, loss_val, step)

        # --- 12. Identity weight ---
        if self.bob_core is not None:
            from bob_core.identity import is_identity_event
            baseline = self.bob_core.commitments.baseline_loss(context_class)
            identity_weight = is_identity_event(
                path=path,
                step=step,
                first_commit_step=self._first_commit_step,
                loss=loss_val,
                baseline=baseline if baseline != float("inf") else loss_val,
            )

        # Re-tick medium clock with actual values
        if self.medium_clock is not None:
            self.medium_clock.tick(
                prev_expert_ids=self._prev_expert_ids,
                curr_expert_ids=expert_ids,
                prev_loss=self._prev_loss,
                curr_loss=loss_val,
                was_blocked=was_blocked,
                path=path,
            )
            medium_activation = self.medium_clock.activation

        # Update prev state for next step
        self._prev_expert_ids = expert_ids
        self._prev_loss = loss_val

        # --- 13. Log ---
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
            governor_decision=governor_decision,
            governor_reasons=governor_reasons,
            forced_exploration=forced_exploration,
            medium_activation=medium_activation,
            scar_debt=scar_debt,
            cost_cheap_fraction=cost_cheap_fraction,
            commitment_id=commitment_id,
            identity_weight=identity_weight,
        )
        self.traces.append(trace)
        return trace

    def _extract_motif(
        self,
        result: ForwardResult,
        num_experts: int,
    ) -> Tuple[MotifSpec, Tuple[int, ...], Tuple[int, ...]]:
        """
        Build a MotifSpec from actual token-level routing.

        Aggregates per-token expert selections across the batch to find
        experts that handle a disproportionate share of tokens. The motif
        includes only experts whose token-level selection frequency exceeds
        the uniform baseline by a meaningful margin.

        Returns (motif_spec, full_representative_ids, routing_key).
        - full_representative_ids: all dominant experts from first layer (for scars)
        - routing_key: top-2 dominant experts from first layer (for stability)
        """
        layers = {}
        representative_ids = ()
        routing_key = ()
        top_k = getattr(self.adapter, 'top_k', 2)

        # Dominance threshold: token frequency (fraction of tokens where expert
        # appears in top-k). Must be 2x the uniform expectation.
        # For 8E top-2: uniform = 0.25, threshold = 0.50
        # For 64E top-8: uniform = 0.125, threshold = 0.25
        uniform_token_freq = min(1.0, top_k / num_experts)
        dominance_multiplier = 2.0
        dominance_threshold = uniform_token_freq * dominance_multiplier

        for snap in result.snapshots:
            selected = snap.selected_experts  # [B*T, top_k]
            routing_w = snap.routing_weights  # [B*T, top_k]
            num_tokens = selected.shape[0]

            # Count per-expert: how many tokens selected this expert
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

            # Find experts above dominance threshold (token frequency)
            dominant = [
                eid for eid, cnt in expert_counts.items()
                if cnt / num_tokens > dominance_threshold
            ]

            if dominant:
                # Specialization exists: use dominant experts
                dominant.sort(key=lambda e: expert_counts[e], reverse=True)
                # Ensure minimum experts for quality (at least top_k - 2)
                min_experts = max(2, top_k - 2)
                if len(dominant) < min_experts:
                    all_sorted = sorted(
                        expert_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    for eid, cnt in all_sorted:
                        if eid not in dominant:
                            dominant.append(eid)
                        if len(dominant) >= min_experts:
                            break
                expert_ids = tuple(dominant)
            else:
                # No specialization: use the most common top-k set
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
                # Routing key: top-2 dominant, sorted for stable comparison
                key_experts = dominant[:2] if dominant else list(expert_ids[:2])
                routing_key = tuple(sorted(key_experts))

        return MotifSpec(layers=layers), representative_ids, routing_key

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
