"""
NanoMoE Instrumentation: Over-Logging for Lifecycle Learning.

Per Halcyon: "Treat nanoMoE as a logging lab. Over-instrument now."

Logs everything per-step:
- Utilization histograms
- Neff, coherence, bimodality
- Stress band, gate state
- Edit proposals with "why"
- Router distributions
- Expert load evolution

Be LOUD - we'll quiet down for swiss-ai/MoE.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import torch
from torch import Tensor


@dataclass
class StepDiagnostics:
    """
    Per-step diagnostic snapshot.

    Everything we need to understand what the lifecycle system is doing.
    """
    step: int
    loss: float

    # Per-layer metrics
    layer_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    # System state
    stress_band: str = "unknown"
    time_in_comfort: int = 0
    gates_state: Dict[str, bool] = field(default_factory=dict)

    # Edit proposals (if any)
    proposals: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "layers": self.layer_diagnostics,
            "stress_band": self.stress_band,
            "time_in_comfort": self.time_in_comfort,
            "gates_state": self.gates_state,
            "proposals": self.proposals,
        }


@dataclass
class LayerDiagnostics:
    """
    Per-layer per-step diagnostics.

    Captures everything about one MoE layer's state.
    """
    layer_id: int
    step: int

    # Expert utilization
    utilization: List[float] = field(default_factory=list)
    utilization_mean: float = 0.0
    utilization_std: float = 0.0
    utilization_max: float = 0.0
    utilization_min: float = 0.0

    # Coherence
    coherence: List[float] = field(default_factory=list)
    coherence_mean: float = 0.0
    coherence_min: float = 0.0

    # Layer-level metrics
    neff: float = 0.0
    psi: float = 0.0  # Layer coherence

    # Free energy
    f_l: float = 0.0
    misfit: float = 0.0
    complexity: float = 0.0
    redundancy: float = 0.0
    instability: float = 0.0

    # Bimodality (if tracked)
    bimodality_scores: List[float] = field(default_factory=list)
    bimodality_max: float = 0.0

    # Active experts
    num_active_experts: int = 0
    num_cooling_experts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "step": self.step,
            "utilization": {
                "values": self.utilization,
                "mean": self.utilization_mean,
                "std": self.utilization_std,
                "min": self.utilization_min,
                "max": self.utilization_max,
            },
            "coherence": {
                "values": self.coherence,
                "mean": self.coherence_mean,
                "min": self.coherence_min,
            },
            "neff": self.neff,
            "psi": self.psi,
            "free_energy": {
                "f_l": self.f_l,
                "misfit": self.misfit,
                "complexity": self.complexity,
                "redundancy": self.redundancy,
                "instability": self.instability,
            },
            "bimodality": {
                "values": self.bimodality_scores,
                "max": self.bimodality_max,
            },
            "experts": {
                "active": self.num_active_experts,
                "cooling": self.num_cooling_experts,
            },
        }


class NanoMoEInstrumentation:
    """
    Comprehensive instrumentation for nanoMoE integration.

    Logs everything to understand lifecycle behavior.
    """

    def __init__(self, log_dir: Path, log_every_n_steps: int = 1):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_steps = log_every_n_steps

        # Log files
        self.diagnostics_log = self.log_dir / "diagnostics.jsonl"
        self.proposals_log = self.log_dir / "proposals.jsonl"
        self.events_log = self.log_dir / "events.jsonl"

        # Open files
        self.diagnostics_f = open(self.diagnostics_log, "w")
        self.proposals_f = open(self.proposals_log, "w")
        self.events_f = open(self.events_log, "w")

        print(f"📊 Instrumentation initialized:")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Logging every {self.log_every_n_steps} steps")

    def log_step(self, diagnostics: StepDiagnostics):
        """Log full step diagnostics."""
        if diagnostics.step % self.log_every_n_steps == 0:
            self.diagnostics_f.write(json.dumps(diagnostics.to_dict()) + "\n")
            self.diagnostics_f.flush()

    def log_proposal(
        self,
        step: int,
        layer_id: int,
        edit_type: str,
        reason: str,
        evidence: Dict[str, Any],
        approved: bool,
    ):
        """
        Log edit proposal with "why" explanation.

        This is the key question: Why did we propose this edit?
        """
        entry = {
            "step": step,
            "layer_id": layer_id,
            "edit_type": edit_type,
            "reason": reason,
            "evidence": evidence,
            "approved": approved,
        }
        self.proposals_f.write(json.dumps(entry) + "\n")
        self.proposals_f.flush()

        # Also print to console (be LOUD)
        status = "✓ APPROVED" if approved else "✗ REJECTED"
        print(f"\n🔧 [{step:06d}] Layer {layer_id}: {edit_type} {status}")
        print(f"   Why: {reason}")
        if evidence:
            print(f"   ΔF_l: {evidence.get('delta_f_l', 'N/A'):.4f}")

    def log_event(self, step: int, event_type: str, details: Dict[str, Any]):
        """Log significant lifecycle events."""
        entry = {
            "step": step,
            "event": event_type,
            "details": details,
        }
        self.events_f.write(json.dumps(entry) + "\n")
        self.events_f.flush()

        # Print to console
        print(f"📌 [{step:06d}] {event_type}: {details.get('message', '')}")

    def print_layer_summary(self, layer_diag: LayerDiagnostics):
        """Print concise per-layer summary to console."""
        print(f"  Layer {layer_diag.layer_id}:")
        print(f"    Experts: {layer_diag.num_active_experts} active, {layer_diag.num_cooling_experts} cooling")
        print(f"    Utilization: μ={layer_diag.utilization_mean:.1f}, σ={layer_diag.utilization_std:.1f}, range=[{layer_diag.utilization_min:.0f}, {layer_diag.utilization_max:.0f}]")
        print(f"    Coherence: μ={layer_diag.coherence_mean:.3f}, min={layer_diag.coherence_min:.3f}")
        print(f"    Neff: {layer_diag.neff:.2f}, Psi: {layer_diag.psi:.3f}")
        print(f"    F_l: {layer_diag.f_l:.4f} (misfit={layer_diag.misfit:.4f}, complexity={layer_diag.complexity:.4f}, redundancy={layer_diag.redundancy:.4f})")

    def print_step_summary(self, diagnostics: StepDiagnostics):
        """Print concise step summary to console."""
        if diagnostics.step % (self.log_every_n_steps * 10) == 0:  # Every 10th logged step
            print(f"\n{'='*70}")
            print(f"Step {diagnostics.step} | Loss: {diagnostics.loss:.4f} | Band: {diagnostics.stress_band} | Calm: {diagnostics.time_in_comfort}")
            print(f"{'='*70}")

            for layer_diag in diagnostics.layer_diagnostics:
                layer_diag_obj = LayerDiagnostics(**layer_diag)
                self.print_layer_summary(layer_diag_obj)

    def close(self):
        """Close log files."""
        self.diagnostics_f.close()
        self.proposals_f.close()
        self.events_f.close()
        print(f"\n📊 Instrumentation closed. Logs saved to {self.log_dir}")


def compute_layer_diagnostics(
    layer_id: int,
    step: int,
    trace,
    free_energy_state,
    registry,
    bimodality_detector=None,
) -> LayerDiagnostics:
    """
    Compute comprehensive layer diagnostics from trace.

    Args:
        layer_id: Layer index
        step: Training step
        trace: MoETrace from wrapper
        free_energy_state: FreeEnergyState
        registry: ExpertRegistry
        bimodality_detector: Optional BimodalityDetector

    Returns:
        LayerDiagnostics with all metrics
    """
    # Coherence
    coherence = trace.compute_coherence()
    coherence_list = coherence.tolist()

    # Utilization
    utilization = trace.get_expert_utilization()
    utilization_list = utilization.tolist()

    # Neff and Psi
    from chronomoe_v3.lifecycle import compute_neff, compute_saturation
    from chronomoe_v3.free_energy import compute_layer_coherence

    router_probs_per_token = trace.router_probs.mean(dim=0)  # Average over tokens
    neff = compute_neff(router_probs_per_token).item()
    psi = compute_layer_coherence(coherence, utilization).item()

    # Bimodality
    bimodality_scores = []
    bimodality_max = 0.0
    if bimodality_detector:
        for expert_id in trace.active_expert_ids:
            state = bimodality_detector.get_state(str(expert_id))
            if state:
                score = state.compute_bimodality()
                bimodality_scores.append(score)
                bimodality_max = max(bimodality_max, score)

    return LayerDiagnostics(
        layer_id=layer_id,
        step=step,
        utilization=utilization_list,
        utilization_mean=float(utilization.mean()),
        utilization_std=float(utilization.std()),
        utilization_max=float(utilization.max()),
        utilization_min=float(utilization.min()),
        coherence=coherence_list,
        coherence_mean=float(coherence.mean()),
        coherence_min=float(coherence.min()),
        neff=neff,
        psi=psi,
        f_l=free_energy_state.f_l,
        misfit=free_energy_state.components.misfit,
        complexity=free_energy_state.components.complexity,
        redundancy=free_energy_state.components.redundancy,
        instability=free_energy_state.components.instability,
        bimodality_scores=bimodality_scores,
        bimodality_max=bimodality_max,
        num_active_experts=registry.num_active,
        num_cooling_experts=registry.num_cooling,
    )
