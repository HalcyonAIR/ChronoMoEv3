"""
Tests for free energy computation.

Critical: Free energy F_l must correctly aggregate all four terms (misfit,
complexity, redundancy, instability) without double-masking or scaling bugs.
"""

import pytest
import torch
from chronomoe_v3.free_energy import (
    FreeEnergyComponents,
    FreeEnergyState,
    compute_layer_coherence,
    compute_misfit_term,
    compute_complexity_term,
    compute_redundancy_term,
    compute_instability_term,
    compute_free_energy,
    create_free_energy_state,
)


class TestLayerCoherence:
    """Test layer coherence computation (Psi_l)."""

    def test_all_experts_perfect(self):
        """All experts perfectly coherent → Psi_l = 1.0."""
        phi_slow = torch.ones(8)
        utilization = torch.ones(8) * 100

        psi = compute_layer_coherence(phi_slow, utilization)
        assert psi == 1.0

    def test_all_experts_zero(self):
        """All experts zero coherence → Psi_l = 0.0."""
        phi_slow = torch.zeros(8)
        utilization = torch.ones(8) * 100

        psi = compute_layer_coherence(phi_slow, utilization)
        assert psi == 0.0

    def test_weighted_average(self):
        """Layer coherence should weight by utilization."""
        phi_slow = torch.tensor([1.0, 0.0])  # One perfect, one zero
        utilization = torch.tensor([300.0, 100.0])  # 3:1 ratio

        psi = compute_layer_coherence(phi_slow, utilization, weight_by_utilization=True)
        # Weighted: (1.0 * 0.75) + (0.0 * 0.25) = 0.75
        assert abs(psi - 0.75) < 1e-5

    def test_uniform_average(self):
        """Uniform average should ignore utilization."""
        phi_slow = torch.tensor([1.0, 0.0])
        utilization = torch.tensor([300.0, 100.0])  # Different weights

        psi = compute_layer_coherence(phi_slow, utilization, weight_by_utilization=False)
        # Uniform: (1.0 + 0.0) / 2 = 0.5
        assert abs(psi - 0.5) < 1e-5

    def test_min_tokens_filter(self):
        """Experts below min_tokens should be excluded."""
        phi_slow = torch.tensor([1.0, 0.0, 0.5])
        utilization = torch.tensor([100.0, 50.0, 100.0])  # Second expert below threshold

        psi = compute_layer_coherence(
            phi_slow, utilization, weight_by_utilization=False, min_tokens=75
        )
        # Only experts 0 and 2 count: (1.0 + 0.5) / 2 = 0.75
        assert abs(psi - 0.75) < 1e-5

    def test_clamping(self):
        """Psi_l should be clamped to [0, 1]."""
        # Artificially create out-of-range value
        phi_slow = torch.tensor([1.5])  # Above 1
        utilization = torch.tensor([100.0])

        psi = compute_layer_coherence(phi_slow, utilization)
        assert psi == 1.0  # Clamped

        phi_slow = torch.tensor([-0.5])  # Below 0
        psi = compute_layer_coherence(phi_slow, utilization)
        assert psi == 0.0  # Clamped

    def test_no_active_experts(self):
        """No experts above min_tokens → Psi_l = 0.0."""
        phi_slow = torch.ones(4)
        utilization = torch.zeros(4)  # All below threshold

        psi = compute_layer_coherence(phi_slow, utilization, min_tokens=10)
        assert psi == 0.0

    def test_mask_parameter(self):
        """Providing a mask should override min_tokens."""
        phi_slow = torch.tensor([1.0, 0.0, 0.5])
        utilization = torch.tensor([100.0, 50.0, 100.0])

        # Manual mask: include only expert 0 and 2
        mask = torch.tensor([True, False, True])

        psi = compute_layer_coherence(
            phi_slow, utilization, weight_by_utilization=False, mask=mask, min_tokens=1
        )
        # Only experts 0 and 2: (1.0 + 0.5) / 2 = 0.75
        assert abs(psi - 0.75) < 1e-5


class TestMisfitTerm:
    """Test misfit term (1 - Psi_l)."""

    def test_perfect_coherence(self):
        """Perfect coherence → zero misfit."""
        misfit = compute_misfit_term(1.0)
        assert misfit == 0.0

    def test_zero_coherence(self):
        """Zero coherence → maximum misfit."""
        misfit = compute_misfit_term(0.0)
        assert misfit == 1.0

    def test_mid_coherence(self):
        """Mid coherence → mid misfit."""
        misfit = compute_misfit_term(0.6)
        assert abs(misfit - 0.4) < 1e-5


class TestComplexityTerm:
    """Test complexity term (lambda * N_active)."""

    def test_zero_experts(self):
        """Zero experts → zero complexity."""
        complexity = compute_complexity_term(0, lambda_weight=0.01)
        assert complexity == 0.0

    def test_scaling(self):
        """Complexity should scale linearly with expert count."""
        complexity_4 = compute_complexity_term(4, lambda_weight=0.01)
        complexity_8 = compute_complexity_term(8, lambda_weight=0.01)

        assert complexity_4 == 0.04
        assert complexity_8 == 0.08

    def test_weight_scaling(self):
        """Complexity should scale with lambda weight."""
        complexity_low = compute_complexity_term(10, lambda_weight=0.01)
        complexity_high = compute_complexity_term(10, lambda_weight=0.1)

        assert complexity_low == 0.1
        assert complexity_high == 1.0


class TestRedundancyTerm:
    """Test redundancy term (rho * R_l)."""

    def test_identical_experts(self):
        """Identical experts → maximum redundancy."""
        # Two experts with identical outputs
        direction = torch.randn(64)
        role_vectors = torch.stack([direction, direction])
        utilization = torch.tensor([100.0, 100.0])

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, similarity_threshold=0.9
        )

        # 100% redundant (cosine = 1.0 > 0.9)
        assert redundancy > 0.09  # Close to 0.1 * 1.0

        # Similarity matrix should show perfect match
        assert similarity[0, 1] > 0.99

    def test_orthogonal_experts(self):
        """Orthogonal experts → zero redundancy."""
        # Two orthogonal directions
        role_vectors = torch.zeros(2, 64)
        role_vectors[0, 0] = 1.0  # First dimension
        role_vectors[1, 1] = 1.0  # Second dimension
        utilization = torch.tensor([100.0, 100.0])

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, similarity_threshold=0.9
        )

        # 0% redundant (cosine = 0.0 < 0.9)
        assert redundancy == 0.0

        # Similarity should show orthogonality
        assert abs(similarity[0, 1]) < 0.01

    def test_partial_redundancy(self):
        """Some redundant pairs → fractional redundancy."""
        # 4 experts: 0-1 similar, 2-3 orthogonal
        role_vectors = torch.zeros(4, 64)

        # Experts 0 and 1: similar
        direction = torch.randn(64)
        role_vectors[0] = direction
        role_vectors[1] = direction + torch.randn(64) * 0.01  # Tiny noise

        # Experts 2 and 3: orthogonal
        role_vectors[2, 0] = 1.0
        role_vectors[3, 1] = 1.0

        utilization = torch.ones(4) * 100.0

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, similarity_threshold=0.9
        )

        # 1 out of 6 pairs redundant (pair 0-1)
        # R_l ≈ 1/6, weighted = 0.1 * 1/6 ≈ 0.0167
        assert 0.01 < redundancy < 0.03

    def test_min_tokens_filter(self):
        """Experts below min_tokens should be excluded from redundancy check."""
        # Two identical experts, but one has low utilization
        direction = torch.randn(64)
        role_vectors = torch.stack([direction, direction])
        utilization = torch.tensor([200.0, 50.0])  # Second below threshold

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, min_tokens=100
        )

        # Only 1 valid expert → no pairs → no redundancy
        assert redundancy == 0.0

        # But similarity should still be computed (for debugging)
        assert similarity[0, 1] > 0.99

    def test_similarity_always_returned(self):
        """Similarity matrix should always be returned, even with no valid pairs."""
        role_vectors = torch.randn(2, 64)
        utilization = torch.zeros(2)  # Both below threshold

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, min_tokens=100
        )

        assert redundancy == 0.0
        assert similarity.shape == (2, 2)  # Computed despite no valid pairs

    def test_mask_parameter(self):
        """Providing a mask should override min_tokens."""
        direction = torch.randn(64)
        role_vectors = torch.stack([direction, direction, direction])
        utilization = torch.tensor([100.0, 50.0, 100.0])

        # Manual mask: exclude expert 1
        mask = torch.tensor([True, False, True])

        redundancy, similarity = compute_redundancy_term(
            role_vectors, utilization, rho_weight=0.1, mask=mask
        )

        # Only pair (0, 2) compared, both identical → 100% redundant
        assert redundancy > 0.09


class TestInstabilityTerm:
    """Test instability term (kappa * I_l)."""

    def test_no_bimodality(self):
        """All experts unimodal → zero instability."""
        bimodality_scores = torch.zeros(8)
        utilization = torch.ones(8) * 100

        instability = compute_instability_term(
            bimodality_scores, kappa_weight=0.1, utilization=utilization
        )

        assert instability == 0.0

    def test_all_bimodal(self):
        """All experts bimodal → high instability."""
        bimodality_scores = torch.ones(8) * 0.5  # All moderately bimodal
        utilization = torch.ones(8) * 100

        instability = compute_instability_term(
            bimodality_scores, kappa_weight=0.1, utilization=utilization
        )

        # Weighted mean = 0.5, kappa * 0.5 = 0.05
        assert abs(instability - 0.05) < 1e-5

    def test_utilization_weighting(self):
        """Instability should weight by utilization, not raw sum."""
        # One heavily-used bimodal expert, one rarely-used
        bimodality_scores = torch.tensor([1.0, 1.0])
        utilization = torch.tensor([900.0, 100.0])  # 9:1 ratio

        instability = compute_instability_term(
            bimodality_scores, kappa_weight=0.1, utilization=utilization
        )

        # Weighted mean = (1.0 * 0.9) + (1.0 * 0.1) = 1.0
        # kappa * 1.0 = 0.1
        assert abs(instability - 0.1) < 1e-5

    def test_no_scaling_with_expert_count(self):
        """Instability should NOT scale with number of experts (use mean, not sum)."""
        bimodality_scores_4 = torch.ones(4) * 0.5
        bimodality_scores_8 = torch.ones(8) * 0.5

        utilization_4 = torch.ones(4) * 100
        utilization_8 = torch.ones(8) * 100

        instability_4 = compute_instability_term(
            bimodality_scores_4, kappa_weight=0.1, utilization=utilization_4
        )
        instability_8 = compute_instability_term(
            bimodality_scores_8, kappa_weight=0.1, utilization=utilization_8
        )

        # Both should give same result (mean, not sum)
        assert abs(instability_4 - instability_8) < 1e-5

    def test_min_tokens_filter(self):
        """Experts below min_tokens should be excluded."""
        bimodality_scores = torch.tensor([1.0, 0.0])
        utilization = torch.tensor([100.0, 50.0])  # Second below threshold

        instability = compute_instability_term(
            bimodality_scores, kappa_weight=0.1, utilization=utilization, min_tokens=75
        )

        # Only first expert counts: mean = 1.0, kappa * 1.0 = 0.1
        assert abs(instability - 0.1) < 1e-5

    def test_mask_parameter(self):
        """Providing a mask should override min_tokens."""
        bimodality_scores = torch.tensor([1.0, 0.5, 0.0])
        utilization = torch.ones(3) * 100

        # Manual mask: exclude middle expert
        mask = torch.tensor([True, False, True])

        instability = compute_instability_term(
            bimodality_scores, kappa_weight=0.1, utilization=utilization, mask=mask
        )

        # Mean of experts 0 and 2: (1.0 + 0.0) / 2 = 0.5
        # kappa * 0.5 = 0.05
        assert abs(instability - 0.05) < 1e-5


class TestFreeEnergyComponents:
    """Test FreeEnergyComponents dataclass."""

    def test_total_property(self):
        """Total should sum all four components."""
        components = FreeEnergyComponents(
            misfit=0.4, complexity=0.08, redundancy=0.02, instability=0.05
        )

        assert abs(components.total - 0.55) < 1e-5

    def test_to_dict(self):
        """to_dict should include all components and total."""
        components = FreeEnergyComponents(
            misfit=0.4, complexity=0.08, redundancy=0.02, instability=0.05
        )

        d = components.to_dict()
        assert d["misfit"] == 0.4
        assert d["complexity"] == 0.08
        assert d["redundancy"] == 0.02
        assert d["instability"] == 0.05
        assert abs(d["total"] - 0.55) < 1e-5


class TestComputeFreeEnergy:
    """Test full free energy computation."""

    def test_healthy_layer(self):
        """Healthy layer (high coherence, no redundancy/instability) → low F_l."""
        phi_slow = torch.ones(4) * 0.9  # High coherence
        utilization = torch.ones(4) * 200

        # All different directions (no redundancy)
        role_vectors = torch.zeros(4, 64)
        role_vectors[0, 0] = 1.0
        role_vectors[1, 1] = 1.0
        role_vectors[2, 2] = 1.0
        role_vectors[3, 3] = 1.0

        bimodality_scores = torch.zeros(4)  # No instability

        components, similarity, f_l = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        # Misfit = 1 - 0.9 = 0.1
        # Complexity = 0.01 * 4 = 0.04
        # Redundancy = 0 (all orthogonal)
        # Instability = 0
        # Total = 0.14

        assert abs(components.misfit - 0.1) < 1e-5
        assert abs(components.complexity - 0.04) < 1e-5
        assert components.redundancy == 0.0
        assert components.instability == 0.0
        assert abs(f_l - 0.14) < 1e-5

    def test_high_misfit_scenario(self):
        """Low coherence → high misfit dominates."""
        phi_slow = torch.ones(4) * 0.2  # Low coherence
        utilization = torch.ones(4) * 200
        role_vectors = torch.randn(4, 64)
        bimodality_scores = torch.zeros(4)

        components, _, f_l = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        # Misfit = 1 - 0.2 = 0.8 (dominates)
        assert components.misfit > 0.7

    def test_high_complexity_scenario(self):
        """Many experts → high complexity."""
        phi_slow = torch.ones(16) * 0.9
        utilization = torch.ones(16) * 200
        role_vectors = torch.randn(16, 64)
        bimodality_scores = torch.zeros(16)

        components, _, f_l = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        # Complexity = 0.01 * 16 = 0.16
        assert abs(components.complexity - 0.16) < 1e-5

    def test_redundancy_mask_stricter(self):
        """Redundancy should use stricter mask (redundancy_min_tokens)."""
        phi_slow = torch.ones(4) * 0.9
        utilization = torch.tensor([200.0, 150.0, 50.0, 200.0])

        # Experts 0 and 1 identical (but 1 has low utilization for redundancy)
        direction = torch.randn(64)
        role_vectors = torch.stack([direction, direction, torch.randn(64), torch.randn(64)])

        bimodality_scores = torch.zeros(4)

        components, _, f_l = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
            min_tokens=10,  # All experts active
            redundancy_min_tokens=175,  # Excludes expert 1
        )

        # Redundancy should be 0 because expert 1 excluded by stricter threshold
        assert components.redundancy == 0.0

        # But num_active_experts should be 3 (expert 2 excluded by min_tokens=10)
        # Actually, with min_tokens=10, all 4 are active
        # Let me recalculate: utilization [200, 150, 50, 200] with min_tokens=10
        # All >= 10, so all active
        # But expert 2 has only 50, so might want to exclude it

    def test_no_double_masking(self):
        """Ensure consistent masking across all terms."""
        phi_slow = torch.ones(4) * 0.8
        utilization = torch.tensor([200.0, 150.0, 50.0, 200.0])
        role_vectors = torch.randn(4, 64)
        bimodality_scores = torch.ones(4) * 0.3

        components, _, f_l = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
            min_tokens=100,  # Excludes expert 2
        )

        # Active experts: 0, 1, 3 (3 total)
        assert abs(components.complexity - 0.03) < 1e-5  # 0.01 * 3

    def test_returns_all_three_values(self):
        """compute_free_energy should return (components, similarity, f_l)."""
        phi_slow = torch.ones(4) * 0.9
        utilization = torch.ones(4) * 200
        role_vectors = torch.randn(4, 64)
        bimodality_scores = torch.zeros(4)

        result = compute_free_energy(
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        assert len(result) == 3
        components, similarity, f_l = result

        assert isinstance(components, FreeEnergyComponents)
        assert similarity.shape == (4, 4)
        assert isinstance(f_l, float)
        assert abs(f_l - components.total) < 1e-5


class TestCreateFreeEnergyState:
    """Test FreeEnergyState creation."""

    def test_state_creation(self):
        """Create FreeEnergyState and verify all fields."""
        phi_slow = torch.ones(4) * 0.8
        utilization = torch.ones(4) * 200
        role_vectors = torch.randn(4, 64)
        bimodality_scores = torch.ones(4) * 0.2

        state = create_free_energy_state(
            layer_id=2,
            step=1000,
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        assert state.layer_id == 2
        assert state.step == 1000
        assert state.num_active_experts == 4
        assert abs(state.layer_coherence - 0.8) < 1e-5
        assert state.expert_coherence.shape == (4,)
        assert state.expert_utilization.shape == (4,)
        assert state.expert_bimodality.shape == (4,)
        assert state.expert_similarity.shape == (4, 4)

    def test_raw_scores_unweighted(self):
        """Raw scores should be unweighted (before lambda/rho/kappa)."""
        phi_slow = torch.ones(4) * 0.8
        utilization = torch.ones(4) * 200
        role_vectors = torch.randn(4, 64)
        bimodality_scores = torch.ones(4) * 0.3

        state = create_free_energy_state(
            layer_id=0,
            step=0,
            phi_slow=phi_slow,
            utilization=utilization,
            role_vectors=role_vectors,
            bimodality_scores=bimodality_scores,
            lambda_complexity=0.01,
            rho_redundancy=0.1,
            kappa_instability=0.1,
        )

        # Instability raw should be ~0.3 (the mean), not 0.03 (weighted)
        assert abs(state.instability_score - 0.3) < 0.05
