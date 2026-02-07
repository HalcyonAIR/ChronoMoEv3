"""
Capacity Whiplash Test: Earned Divergence (Scientific Upgrade)

This version removes the injected beta_init seeding and instead creates
divergence through asymmetric training environments. Two systems experience
different input distributions during Phase 1, which causes beta to drift
naturally through coherence feedback. Then we test whether this earned
divergence still reveals itself under constraint.

Key difference from capacity_whiplash_test.py:
- No beta_init seeding
- Asymmetric input distributions create natural specialization
- Proves trails emerge from interaction, not just initial conditions

Hypothesis: Even with earned (not injected) divergence, constraint reveals
accumulated identity.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronomoe_v3 import (
    ChronoRouter,
    CoherenceBuffer,
    update_beta_from_buffer,
    compute_neff,
)


class AsymmetricEnvironment:
    """
    Generate asymmetric input distributions for two systems.

    System A: Prefers low-frequency features (smooth patterns)
    System B: Prefers high-frequency features (sharp patterns)
    """

    def __init__(self, d_model=64, device="cpu"):
        self.d_model = d_model
        self.device = device

        # Create frequency basis
        # Low freq: first half of spectrum
        # High freq: second half of spectrum
        self.low_freq_mask = torch.zeros(d_model, device=device)
        self.low_freq_mask[: d_model // 2] = 1.0

        self.high_freq_mask = torch.zeros(d_model, device=device)
        self.high_freq_mask[d_model // 2 :] = 1.0

    def sample_for_system_a(self, batch_size):
        """Sample inputs biased toward low-frequency features."""
        # Base random input
        hidden = torch.randn(batch_size, self.d_model, device=self.device)

        # Amplify low-frequency components
        hidden = hidden * (1.0 + 2.0 * self.low_freq_mask)

        return hidden

    def sample_for_system_b(self, batch_size):
        """Sample inputs biased toward high-frequency features."""
        # Base random input
        hidden = torch.randn(batch_size, self.d_model, device=self.device)

        # Amplify high-frequency components
        hidden = hidden * (1.0 + 2.0 * self.high_freq_mask)

        return hidden


class EarnedDivergenceExperiment:
    """Test whether earned divergence (not injected) reveals itself under constraint."""

    def __init__(self, num_experts=8, d_model=64, device="cpu"):
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        self.env = AsymmetricEnvironment(d_model=d_model, device=device)

    def create_system(self, layer_id):
        """Create a router + coherence system with NO beta initialization."""
        router = ChronoRouter(
            d_model=self.d_model,
            num_experts=self.num_experts,
            layer_id=layer_id,
            device=self.device,
        )

        coherence = CoherenceBuffer(
            layer_id=layer_id,
            num_experts=self.num_experts,
            d_model=self.d_model,
            device=self.device,
        )

        # NO beta initialization - start from zero
        return router, coherence

    def train_phase(
        self,
        router,
        coherence,
        num_steps,
        top_k,
        environment_sampler,
        update_beta=False,
        system_name="",
    ):
        """
        Train for num_steps with given environment sampler.

        Args:
            router: ChronoRouter instance
            coherence: CoherenceBuffer instance
            num_steps: Number of training steps
            top_k: Number of experts to select
            environment_sampler: Function that returns batch of inputs
            update_beta: Whether to update beta from coherence
            system_name: Name for logging

        Returns:
            dict with routing statistics
        """
        torch.manual_seed(42 + num_steps)  # Reproducible

        expert_selections = torch.zeros(self.num_experts)
        neff_values = []
        neff_nan_count = 0
        first_nan_step = None

        for step in range(num_steps):
            # Generate input from asymmetric environment
            hidden = environment_sampler(32)

            # Route
            top_k_probs, top_k_indices, z_clean, z_biased = router(
                hidden, top_k=top_k, use_relevance=True
            )

            # Compute biased probabilities for Neff
            p_biased = F.softmax(z_biased / router.router_state.temperature, dim=-1)

            # Track selections
            for expert_id in range(self.num_experts):
                expert_selections[expert_id] += (top_k_indices == expert_id).sum()

            # Track Neff (with NaN handling)
            try:
                neff = compute_neff(p_biased)
                if torch.isnan(torch.tensor(neff)):
                    neff_nan_count += 1
                    if first_nan_step is None:
                        first_nan_step = step
                        print(f"  [{system_name}] First NaN at step {step}")
                        print(
                            f"    p_biased stats: min={p_biased.min():.6f}, max={p_biased.max():.6f}"
                        )
                else:
                    neff_values.append(neff)
            except Exception as e:
                neff_nan_count += 1
                if first_nan_step is None:
                    first_nan_step = step
                    print(f"  [{system_name}] Neff error at step {step}: {e}")

            # Simulate expert outputs (for coherence)
            # Each expert has a preferred direction
            y_experts = []
            for expert_id in range(self.num_experts):
                # Expert direction varies by ID (deterministic but varied)
                torch.manual_seed(1000 + expert_id)
                direction = torch.randn(self.d_model, device=self.device)
                direction = direction / direction.norm()
                y_e = direction.unsqueeze(0).expand(32, -1)
                y_experts.append(y_e)

            # Compute mixture (weighted by routing probs)
            y_mix = torch.zeros(32, self.d_model, device=self.device)
            for expert_id in range(self.num_experts):
                weight = p_biased[:, expert_id].unsqueeze(-1)
                y_mix = y_mix + weight * y_experts[expert_id]

            # Update coherence
            active_experts = torch.unique(top_k_indices.flatten()).tolist()
            for expert_id in active_experts:
                y_e = y_experts[expert_id]
                y_e_mean = y_e.mean(dim=0)
                y_mix_mean = y_mix.mean(dim=0)

                # Compute phi (cosine similarity)
                phi_raw = F.cosine_similarity(
                    y_e_mean.unsqueeze(0), y_mix_mean.unsqueeze(0)
                )

                coherence.update(
                    phi_raw=phi_raw,
                    active_expert_ids=torch.tensor([expert_id], device=self.device),
                    step=step,
                    num_tokens=32,
                )

            # Update beta if requested (EARNING divergence through interaction)
            if update_beta and step > 50 and step % 10 == 0:
                update_beta_from_buffer(
                    router.router_state,
                    coherence.phi_slow,
                    coherence.total_tokens_seen,
                    eta=0.02,
                    tau=0.5,
                    min_tokens=100,
                )

        neff_valid_ratio = len(neff_values) / num_steps if num_steps > 0 else 0.0

        return {
            "expert_selections": expert_selections,
            "neff_mean": sum(neff_values) / len(neff_values) if neff_values else 0.0,
            "neff_valid_ratio": neff_valid_ratio,
            "neff_nan_count": neff_nan_count,
            "beta_final": router.router_state.beta_coeff.cpu().clone(),
        }

    def run_experiment(self):
        """Run full earned divergence experiment."""
        print("=" * 70)
        print("CAPACITY WHIPLASH EXPERIMENT: EARNED DIVERGENCE")
        print("=" * 70)
        print("\nScientific upgrade: NO beta initialization")
        print("Divergence earned through asymmetric environment interaction")
        print()

        # System A: Low-frequency environment
        print("Creating System A (low-frequency environment)...")
        router_a, coherence_a = self.create_system(layer_id=0)

        # System B: High-frequency environment
        print("Creating System B (high-frequency environment)...")
        router_b, coherence_b = self.create_system(layer_id=1)

        # Verify both start with zero beta
        print(
            f"\nInitial beta (both systems): {router_a.router_state.beta_coeff.abs().sum():.6f} (should be 0.0)"
        )

        # Phase 1: Plenty (top-4 routing) - EARNING divergence
        print("\n" + "=" * 70)
        print("PHASE 1: PLENTY (top-4 routing, 200 steps)")
        print("=" * 70)
        print("Systems experiencing asymmetric environments...")
        print("  System A: Low-frequency bias (smooth patterns)")
        print("  System B: High-frequency bias (sharp patterns)")

        stats_a_phase1 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=200,
            top_k=4,
            environment_sampler=self.env.sample_for_system_a,
            update_beta=True,
            system_name="A",
        )

        stats_b_phase1 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=200,
            top_k=4,
            environment_sampler=self.env.sample_for_system_b,
            update_beta=True,
            system_name="B",
        )

        print(f"\nSystem A (low-freq environment):")
        print(
            f"  Neff: {stats_a_phase1['neff_mean']:.2f} (valid: {stats_a_phase1['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_a_phase1['expert_selections'].tolist()}")
        beta_a = stats_a_phase1["beta_final"]
        print(
            f"  Beta (top-2): {beta_a.topk(2).indices.tolist()} = {[f'{v:.3f}' for v in beta_a.topk(2).values.tolist()]}"
        )
        print(f"  Beta range: [{beta_a.min():.3f}, {beta_a.max():.3f}]")

        print(f"\nSystem B (high-freq environment):")
        print(
            f"  Neff: {stats_b_phase1['neff_mean']:.2f} (valid: {stats_b_phase1['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_b_phase1['expert_selections'].tolist()}")
        beta_b = stats_b_phase1["beta_final"]
        print(
            f"  Beta (top-2): {beta_b.topk(2).indices.tolist()} = {[f'{v:.3f}' for v in beta_b.topk(2).values.tolist()]}"
        )
        print(f"  Beta range: [{beta_b.min():.3f}, {beta_b.max():.3f}]")

        # Check if divergence was earned
        beta_divergence = (beta_a - beta_b).abs().sum().item()
        print(f"\n>>> Beta divergence (L1): {beta_divergence:.3f}")
        if beta_divergence < 0.1:
            print(
                "    ⚠ WARNING: Minimal divergence earned. May need stronger asymmetry."
            )
        else:
            print("    ✓ Divergence successfully earned through interaction")

        # Phase 2: Constraint (top-1 routing) - SAME neutral environment for both
        print("\n" + "=" * 70)
        print("PHASE 2: CONSTRAINT (top-1 routing, 100 steps)")
        print("=" * 70)
        print("\nBoth systems now experience SAME neutral environment")
        print("Forcing top-1 selection under identical conditions...")

        # Neutral sampler (no bias)
        def neutral_sampler(batch_size):
            return torch.randn(batch_size, self.d_model, device=self.device)

        stats_a_phase2 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=100,
            top_k=1,
            environment_sampler=neutral_sampler,
            update_beta=False,  # Freeze beta during constraint
            system_name="A",
        )

        stats_b_phase2 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=100,
            top_k=1,
            environment_sampler=neutral_sampler,
            update_beta=False,
            system_name="B",
        )

        print(f"\nSystem A (top-1, neutral environment):")
        print(
            f"  Neff: {stats_a_phase2['neff_mean']:.2f} (valid: {stats_a_phase2['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_a_phase2['expert_selections'].tolist()}")

        print(f"\nSystem B (top-1, neutral environment):")
        print(
            f"  Neff: {stats_b_phase2['neff_mean']:.2f} (valid: {stats_b_phase2['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_b_phase2['expert_selections'].tolist()}")

        # Check divergence under constraint
        top_expert_a = int(stats_a_phase2["expert_selections"].argmax())
        top_expert_b = int(stats_b_phase2["expert_selections"].argmax())

        print(f"\n>>> System A chose expert {top_expert_a}")
        print(f">>> System B chose expert {top_expert_b}")

        if top_expert_a != top_expert_b:
            print(
                "\n✓ DIVERGENCE DETECTED: Earned beta history revealed itself under constraint!"
            )
            divergence_detected = True
        else:
            print(
                "\n✗ NO DIVERGENCE: Systems converged (earned history insufficient or asymmetry too weak)"
            )
            divergence_detected = False

        # Phase 3: Release (return to top-4) - neutral environment
        print("\n" + "=" * 70)
        print("PHASE 3: RELEASE (back to top-4, 100 steps)")
        print("=" * 70)
        print("\nReturning to top-4 with neutral environment...")

        stats_a_phase3 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=100,
            top_k=4,
            environment_sampler=neutral_sampler,
            update_beta=True,
            system_name="A",
        )

        stats_b_phase3 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=100,
            top_k=4,
            environment_sampler=neutral_sampler,
            update_beta=True,
            system_name="B",
        )

        print(f"\nSystem A (post-constraint):")
        print(
            f"  Neff: {stats_a_phase3['neff_mean']:.2f} (valid: {stats_a_phase3['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_a_phase3['expert_selections'].tolist()}")

        print(f"\nSystem B (post-constraint):")
        print(
            f"  Neff: {stats_b_phase3['neff_mean']:.2f} (valid: {stats_b_phase3['neff_valid_ratio']:.1%})"
        )
        print(f"  Expert usage: {stats_b_phase3['expert_selections'].tolist()}")

        # Compute hysteresis
        def normalize(x):
            return x / x.sum()

        dist_a_phase1 = normalize(stats_a_phase1["expert_selections"])
        dist_a_phase3 = normalize(stats_a_phase3["expert_selections"])
        hysteresis_a = (dist_a_phase1 - dist_a_phase3).abs().sum().item()

        dist_b_phase1 = normalize(stats_b_phase1["expert_selections"])
        dist_b_phase3 = normalize(stats_b_phase3["expert_selections"])
        hysteresis_b = (dist_b_phase1 - dist_b_phase3).abs().sum().item()

        print(f"\n  Hysteresis (L1 distance Phase 1 vs Phase 3):")
        print(f"    System A: {hysteresis_a:.3f}")
        print(f"    System B: {hysteresis_b:.3f}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: EARNED DIVERGENCE TEST")
        print("=" * 70)

        print(f"\nKey Difference from Injected Test:")
        print(f"  - NO beta initialization (both started at 0.0)")
        print(f"  - Divergence earned through asymmetric environments")
        print(f"  - Phase 2 & 3 use SAME neutral environment for both")

        beta_a_top = stats_a_phase1["beta_final"].topk(2)
        beta_b_top = stats_b_phase1["beta_final"].topk(2)

        print(f"\nPhase 1 (Earning divergence):")
        print(
            f"  System A developed beta favoring experts {beta_a_top.indices.tolist()} "
            f"(values: {[f'{v:.3f}' for v in beta_a_top.values.tolist()]})"
        )
        print(
            f"  System B developed beta favoring experts {beta_b_top.indices.tolist()} "
            f"(values: {[f'{v:.3f}' for v in beta_b_top.values.tolist()]})"
        )
        print(f"  Beta divergence (L1): {beta_divergence:.3f}")

        print(f"\nPhase 2 (Constraint test - neutral environment):")
        print(f"  System A chose expert {top_expert_a}")
        print(f"  System B chose expert {top_expert_b}")

        if divergence_detected:
            print(
                "\n  ✓ EARNED history revealed itself under constraint (not just seeded history)"
            )
            print(
                "  ✓ Proves: Trails emerge from interaction, not just initial conditions"
            )
        else:
            print("\n  ✗ Earned history did NOT reveal divergence")
            print(
                "  → May need: stronger asymmetry, longer Phase 1, or different mechanism"
            )

        print(f"\nPhase 3 (Release to neutral environment):")
        print(f"  System A Neff: {stats_a_phase1['neff_mean']:.2f} → {stats_a_phase3['neff_mean']:.2f}")
        print(f"  System B Neff: {stats_b_phase1['neff_mean']:.2f} → {stats_b_phase3['neff_mean']:.2f}")
        print(f"  Hysteresis: A={hysteresis_a:.3f}, B={hysteresis_b:.3f}")

        return {
            "divergence_detected": divergence_detected,
            "beta_divergence": beta_divergence,
            "hysteresis_a": hysteresis_a,
            "hysteresis_b": hysteresis_b,
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning on: {device}\n")

    experiment = EarnedDivergenceExperiment(num_experts=8, d_model=64, device=device)

    results = experiment.run_experiment()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if results["divergence_detected"]:
        print("\n✅ SUCCESS: Earned divergence reveals itself under constraint")
        print("   This proves trails emerge from interaction, not just seeding")
    else:
        print("\n⚠️  RESULT: No divergence detected")
        print("   Consider: stronger asymmetry or longer training")

    print()
