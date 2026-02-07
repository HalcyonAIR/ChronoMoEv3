"""
Capacity Whiplash Test: Identity under constraint

Hypothesis: "Identity (constraint accumulation) shows up most clearly under
constraint, not under plenty."

When the world is wide, many systems look similar. When options narrow to
almost nothing, only the deepest accumulated constraints (scars, crystallized
reflexes, beta) still exert force.

Test protocol:
1. Phase 1 (Plenty): Train two systems with top-4 routing
   - System A: Develops beta history promoting experts 0,1
   - System B: Develops beta history promoting experts 2,3
2. Phase 2 (Constraint): Force top-1 routing on both
   - Do systems with different beta histories make different choices?
3. Phase 3 (Release): Return to top-4 routing
   - Do systems exhibit hysteresis (remember constraint episode)?

Expected: If beta/constraint state matters, systems should diverge under
constraint. If it's cosmetic, they'll converge.
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


class CapacityWhiplashExperiment:
    """Test whether constraint reveals accumulated identity."""

    def __init__(self, num_experts=8, d_model=64, device="cpu"):
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device

    def create_system(self, layer_id, beta_bias=None):
        """Create a router + coherence system with optional initial beta bias."""
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

        # Apply initial beta bias if provided
        if beta_bias is not None:
            router.router_state.beta_coeff[:] = beta_bias

        return router, coherence

    def train_phase(
        self, router, coherence, num_steps, top_k, input_bias=None, update_beta=False
    ):
        """
        Train for num_steps with given top_k.

        Args:
            router: ChronoRouter instance
            coherence: CoherenceBuffer instance
            num_steps: Number of training steps
            top_k: Number of experts to select
            input_bias: Optional bias to add to hidden state (shape: [d_model])
            update_beta: Whether to update beta from coherence

        Returns:
            dict with routing statistics
        """
        torch.manual_seed(42 + num_steps)  # Reproducible but varied

        expert_selections = torch.zeros(self.num_experts)
        neff_values = []

        for step in range(num_steps):
            # Generate input (with optional bias)
            hidden = torch.randn(32, self.d_model, device=self.device)
            if input_bias is not None:
                hidden = hidden + input_bias

            # Route
            top_k_probs, top_k_indices, p_biased, z_clean = router(
                hidden, top_k=top_k, use_relevance=True
            )

            # Track selections
            for expert_id in range(self.num_experts):
                expert_selections[expert_id] += (top_k_indices == expert_id).sum()

            # Track Neff (with NaN handling)
            try:
                neff = compute_neff(p_biased)
                if not torch.isnan(torch.tensor(neff)):
                    neff_values.append(neff)
            except:
                pass  # Skip if NaN

            # Simulate expert outputs (for coherence)
            # Experts produce outputs with some structure
            y_experts = []
            for expert_id in range(self.num_experts):
                # Each expert has a "preferred direction" in output space
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

            # Update beta if requested
            if update_beta and step > 50 and step % 10 == 0:
                update_beta_from_buffer(
                    router.router_state,
                    coherence.phi_slow,
                    coherence.total_tokens_seen,
                    eta=0.02,
                    tau=0.5,
                    min_tokens=100,
                )

        return {
            "expert_selections": expert_selections,
            "neff_mean": sum(neff_values) / len(neff_values) if neff_values else 0.0,
            "beta_final": router.router_state.beta_coeff.cpu().clone(),
        }

    def run_experiment(self):
        """Run full capacity whiplash experiment."""
        print("=" * 70)
        print("CAPACITY WHIPLASH EXPERIMENT")
        print("=" * 70)
        print("\nHypothesis: Identity emerges under constraint, not plenty")
        print()

        # System A: Initialize with beta favoring experts 0, 1
        print("Creating System A (prefers experts 0, 1)...")
        beta_init_a = torch.zeros(self.num_experts, device=self.device)
        beta_init_a[0] = 0.2  # Favor expert 0
        beta_init_a[1] = 0.2  # Favor expert 1
        router_a, coherence_a = self.create_system(layer_id=0, beta_bias=beta_init_a)
        input_bias_a = None

        # System B: Initialize with beta favoring experts 2, 3
        print("Creating System B (prefers experts 2, 3)...")
        beta_init_b = torch.zeros(self.num_experts, device=self.device)
        beta_init_b[2] = 0.2  # Favor expert 2
        beta_init_b[3] = 0.2  # Favor expert 3
        router_b, coherence_b = self.create_system(layer_id=1, beta_bias=beta_init_b)
        input_bias_b = None

        # Phase 1: Plenty (top-4 routing)
        print("\n" + "=" * 70)
        print("PHASE 1: PLENTY (top-4 routing, 200 steps)")
        print("=" * 70)

        stats_a_phase1 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=200,
            top_k=4,
            input_bias=input_bias_a,
            update_beta=True,
        )

        stats_b_phase1 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=200,
            top_k=4,
            input_bias=input_bias_b,
            update_beta=True,
        )

        print(f"\nSystem A (top-4):")
        print(f"  Neff: {stats_a_phase1['neff_mean']:.2f}")
        print(f"  Expert usage: {stats_a_phase1['expert_selections'].tolist()}")
        print(f"  Beta: {stats_a_phase1['beta_final'].tolist()}")

        print(f"\nSystem B (top-4):")
        print(f"  Neff: {stats_b_phase1['neff_mean']:.2f}")
        print(f"  Expert usage: {stats_b_phase1['expert_selections'].tolist()}")
        print(f"  Beta: {stats_b_phase1['beta_final'].tolist()}")

        # Phase 2: Constraint (top-1 routing)
        print("\n" + "=" * 70)
        print("PHASE 2: CONSTRAINT (top-1 routing, 100 steps)")
        print("=" * 70)
        print("\nForcing both systems to choose only 1 expert per token...")

        stats_a_phase2 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=100,
            top_k=1,
            input_bias=input_bias_a,
            update_beta=False,  # Freeze beta during constraint
        )

        stats_b_phase2 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=100,
            top_k=1,
            input_bias=input_bias_b,
            update_beta=False,
        )

        print(f"\nSystem A (top-1):")
        print(f"  Neff: {stats_a_phase2['neff_mean']:.2f} (should be ~1)")
        print(
            f"  Expert usage: {stats_a_phase2['expert_selections'].tolist()} (which won?)"
        )

        print(f"\nSystem B (top-1):")
        print(f"  Neff: {stats_b_phase2['neff_mean']:.2f} (should be ~1)")
        print(
            f"  Expert usage: {stats_b_phase2['expert_selections'].tolist()} (which won?)"
        )

        # Check divergence under constraint
        top_expert_a = int(stats_a_phase2["expert_selections"].argmax())
        top_expert_b = int(stats_b_phase2["expert_selections"].argmax())

        print(f"\n>>> System A chose expert {top_expert_a}")
        print(f">>> System B chose expert {top_expert_b}")

        if top_expert_a != top_expert_b:
            print(
                "\n✓ DIVERGENCE DETECTED: Systems with different beta histories made different choices under constraint!"
            )
        else:
            print(
                "\n✗ NO DIVERGENCE: Systems converged despite different beta histories (constraint state may be cosmetic)"
            )

        # Phase 3: Release (return to top-4)
        print("\n" + "=" * 70)
        print("PHASE 3: RELEASE (back to top-4, 100 steps)")
        print("=" * 70)
        print("\nReleasing constraint, returning to top-4...")

        stats_a_phase3 = self.train_phase(
            router_a,
            coherence_a,
            num_steps=100,
            top_k=4,
            input_bias=input_bias_a,
            update_beta=True,
        )

        stats_b_phase3 = self.train_phase(
            router_b,
            coherence_b,
            num_steps=100,
            top_k=4,
            input_bias=input_bias_b,
            update_beta=True,
        )

        print(f"\nSystem A (post-constraint):")
        print(f"  Neff: {stats_a_phase3['neff_mean']:.2f}")
        print(
            f"  Expert usage: {stats_a_phase3['expert_selections'].tolist()} (hysteresis?)"
        )

        print(f"\nSystem B (post-constraint):")
        print(f"  Neff: {stats_b_phase3['neff_mean']:.2f}")
        print(
            f"  Expert usage: {stats_b_phase3['expert_selections'].tolist()} (hysteresis?)"
        )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print("\nThe Hypothesis:")
        print(
            '  "In fact, the more constrained the moment, the more identity shows up.'
        )
        print("   When the world is wide, many systems look similar.")
        print(
            "   When the world narrows to almost nothing, only the deepest trails"
        )
        print('   still exert force. That\'s when reflex crystallisation and scars')
        print('   do the talking."')

        print(f"\nPhase 1 (Plenty): Both systems trained with top-4")
        print(
            f"  System A developed beta favoring expert {stats_a_phase1['beta_final'].argmax()}"
        )
        print(
            f"  System B developed beta favoring expert {stats_b_phase1['beta_final'].argmax()}"
        )

        print(f"\nPhase 2 (Constraint): Both systems forced to top-1")
        print(f"  System A chose expert {top_expert_a}")
        print(f"  System B chose expert {top_expert_b}")

        if top_expert_a != top_expert_b:
            print(
                "\n  ✓ Constraint revealed accumulated identity (beta history mattered)"
            )
        else:
            print(
                "\n  ✗ Constraint did not reveal identity (beta history didn't matter)"
            )

        print(f"\nPhase 3 (Release): Both systems returned to top-4")
        print(
            f"  System A Neff: {stats_a_phase1['neff_mean']:.2f} → {stats_a_phase3['neff_mean']:.2f}"
        )
        print(
            f"  System B Neff: {stats_b_phase1['neff_mean']:.2f} → {stats_b_phase3['neff_mean']:.2f}"
        )
        print(
            "\n  Hysteresis = systems remember constraint episode in their routing patterns"
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning on: {device}")

    experiment = CapacityWhiplashExperiment(
        num_experts=8, d_model=64, device=device
    )

    experiment.run_experiment()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
