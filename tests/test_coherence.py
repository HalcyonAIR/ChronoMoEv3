"""
Tests for coherence computation.

Critical: If phi_e doesn't track functional participation, nothing else matters.
"""

import pytest
import torch
from chronomoe_v3.coherence import (
    MoETrace,
    CoherenceState,
    compute_coherence,
    update_coherence_ema,
    batch_update_coherence,
)


class TestMoETrace:
    """Test MoETrace construction and coherence computation."""

    def test_empty_trace(self):
        """Empty trace should return empty coherence."""
        trace = MoETrace(
            mixture=torch.randn(10, 64),
            active_expert_ids=[],
            expert_mean_outputs=[],
            token_row_indices=[],
            gate_weights=[],
        )

        phi = trace.compute_coherence()
        assert phi.shape == (0,)

    def test_single_expert_perfect_alignment(self):
        """Expert perfectly aligned with mixture should have phi=1."""
        d_model = 64
        direction = torch.randn(d_model)
        direction = direction / direction.norm()  # Normalize

        trace = MoETrace(
            mixture=direction.unsqueeze(0).repeat(10, 1),  # All tokens same direction
            active_expert_ids=[0],
            expert_mean_outputs=[direction],
            token_row_indices=[torch.arange(10)],
            gate_weights=[torch.ones(10)],
        )

        phi = trace.compute_coherence()
        assert phi.shape == (1,)
        assert torch.isclose(phi[0], torch.tensor(1.0), atol=1e-5)

    def test_single_expert_opposite_direction(self):
        """Expert opposite to mixture should have phi=-1."""
        d_model = 64
        direction = torch.randn(d_model)
        direction = direction / direction.norm()

        trace = MoETrace(
            mixture=direction.unsqueeze(0).repeat(10, 1),
            active_expert_ids=[0],
            expert_mean_outputs=[-direction],  # Opposite direction
            token_row_indices=[torch.arange(10)],
            gate_weights=[torch.ones(10)],
        )

        phi = trace.compute_coherence()
        assert phi.shape == (1,)
        assert torch.isclose(phi[0], torch.tensor(-1.0), atol=1e-5)

    def test_single_expert_orthogonal(self):
        """Expert orthogonal to mixture should have phi≈0."""
        d_model = 64

        direction1 = torch.zeros(d_model)
        direction1[0] = 1.0  # Unit vector in first dimension

        direction2 = torch.zeros(d_model)
        direction2[1] = 1.0  # Unit vector in second dimension (orthogonal)

        trace = MoETrace(
            mixture=direction1.unsqueeze(0).repeat(10, 1),
            active_expert_ids=[0],
            expert_mean_outputs=[direction2],
            token_row_indices=[torch.arange(10)],
            gate_weights=[torch.ones(10)],
        )

        phi = trace.compute_coherence()
        assert phi.shape == (1,)
        assert torch.isclose(phi[0], torch.tensor(0.0), atol=1e-5)

    def test_multiple_experts_varying_alignment(self):
        """Multiple experts with different alignments."""
        d_model = 64
        mixture_direction = torch.randn(d_model)
        mixture_direction = mixture_direction / mixture_direction.norm()

        # Expert 0: Aligned
        expert0_out = mixture_direction

        # Expert 1: Partially aligned
        expert1_out = 0.5 * mixture_direction + 0.5 * torch.randn(d_model)
        expert1_out = expert1_out / expert1_out.norm()

        # Expert 2: Opposite
        expert2_out = -mixture_direction

        trace = MoETrace(
            mixture=mixture_direction.unsqueeze(0).repeat(30, 1),
            active_expert_ids=[0, 1, 2],
            expert_mean_outputs=[expert0_out, expert1_out, expert2_out],
            token_row_indices=[
                torch.arange(10),
                torch.arange(10, 20),
                torch.arange(20, 30),
            ],
            gate_weights=[torch.ones(10), torch.ones(10), torch.ones(10)],
        )

        phi = trace.compute_coherence()
        assert phi.shape == (3,)
        assert phi[0] > 0.9  # Aligned
        assert phi[2] < -0.9  # Opposite
        assert -0.5 < phi[1] < 0.9  # Partial

    def test_layer_coherence(self):
        """Test layer-wide coherence computation."""
        d_model = 64
        mixture_direction = torch.randn(d_model)
        mixture_direction = mixture_direction / mixture_direction.norm()

        # Two experts, both aligned
        trace = MoETrace(
            mixture=mixture_direction.unsqueeze(0).repeat(20, 1),
            active_expert_ids=[0, 1],
            expert_mean_outputs=[mixture_direction, mixture_direction],
            token_row_indices=[torch.arange(10), torch.arange(10, 20)],
            gate_weights=[torch.ones(10), torch.ones(10)],
        )

        Psi = trace.compute_layer_coherence()
        assert 0.95 < Psi <= 1.0  # Should be near 1.0


class TestCoherenceState:
    """Test CoherenceState tracking."""

    def test_initialization(self):
        """Test default initialization."""
        state = CoherenceState(expert_id="L0_E0", layer_id=0, d_model=64)

        assert state.phi_fast == 0.0
        assert state.phi_mid == 0.0
        assert state.phi_slow == 0.0
        assert state.phi_delta == 0.0
        assert not state.is_degrading

    def test_ema_update(self):
        """Test EMA update logic."""
        state = CoherenceState(expert_id="L0_E0", layer_id=0, d_model=64)

        # Update with phi=1.0
        update_coherence_ema(
            state=state,
            phi_raw=1.0,
            y_bar_e=torch.randn(64),
            alpha_fast=0.9,
            alpha_mid=0.99,
            alpha_slow=0.999,
            step=1,
            num_tokens=10,
        )

        # Fast should respond quickest
        assert state.phi_fast > state.phi_mid > state.phi_slow

        # After many updates, all should converge to 1.0
        for step in range(2, 1000):
            update_coherence_ema(
                state=state,
                phi_raw=1.0,
                y_bar_e=torch.randn(64),
                alpha_fast=0.9,
                alpha_mid=0.99,
                alpha_slow=0.999,
                step=step,
                num_tokens=10,
            )

        assert state.phi_fast > 0.99
        assert state.phi_mid > 0.99
        assert state.phi_slow > 0.98  # Slowest to converge

    def test_degradation_detection(self):
        """Test detection of degrading expert."""
        state = CoherenceState(expert_id="L0_E0", layer_id=0, d_model=64)

        # Build up slow coherence
        for step in range(1000):
            update_coherence_ema(
                state=state,
                phi_raw=0.9,
                y_bar_e=torch.randn(64),
                alpha_fast=0.9,
                alpha_mid=0.99,
                alpha_slow=0.999,
                step=step,
                num_tokens=10,
            )

        assert state.phi_slow > 0.85
        assert not state.is_degrading

        # Sudden drop in fast coherence
        for step in range(1000, 1010):
            update_coherence_ema(
                state=state,
                phi_raw=0.1,  # Dropped
                y_bar_e=torch.randn(64),
                alpha_fast=0.9,
                alpha_mid=0.99,
                alpha_slow=0.999,
                step=step,
                num_tokens=10,
            )

        # Fast should drop, slow should stay high, delta negative
        assert state.phi_fast < 0.5
        assert state.phi_slow > 0.8
        assert state.is_degrading
        assert state.phi_delta < 0


class TestBatchUpdate:
    """Test batch updating from traces."""

    def test_batch_update_creates_states(self):
        """Test that batch_update creates new states."""
        states = {}
        d_model = 64

        trace = MoETrace(
            mixture=torch.randn(10, d_model),
            active_expert_ids=[0, 1],
            expert_mean_outputs=[torch.randn(d_model), torch.randn(d_model)],
            token_row_indices=[torch.arange(5), torch.arange(5, 10)],
            gate_weights=[torch.ones(5), torch.ones(5)],
        )

        states = batch_update_coherence(
            states=states,
            trace=trace,
            alpha_fast=0.9,
            alpha_mid=0.99,
            alpha_slow=0.999,
            step=1,
            layer_id=0,
        )

        assert "L0_E0" in states
        assert "L0_E1" in states
        assert states["L0_E0"].last_update_step == 1
        assert states["L0_E1"].total_tokens_seen == 5

    def test_batch_update_accumulates_tokens(self):
        """Test that token counts accumulate correctly."""
        states = {}
        d_model = 64

        for step in range(10):
            trace = MoETrace(
                mixture=torch.randn(10, d_model),
                active_expert_ids=[0],
                expert_mean_outputs=[torch.randn(d_model)],
                token_row_indices=[torch.arange(10)],
                gate_weights=[torch.ones(10)],
            )

            batch_update_coherence(
                states=states,
                trace=trace,
                alpha_fast=0.9,
                alpha_mid=0.99,
                alpha_slow=0.999,
                step=step,
                layer_id=0,
            )

        assert states["L0_E0"].total_tokens_seen == 100  # 10 tokens × 10 steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
