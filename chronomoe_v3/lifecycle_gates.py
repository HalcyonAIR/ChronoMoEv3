"""
Non-bypassable lifecycle gates.

Enforces double gate at API level: evidence + calmness.

Usage:
    gates = LifecycleGates(stress_state, stress_cfg)

    # This will raise GateViolation if calmness requirement not met
    gates.require_scar_allowed()
    form_scar()  # Only executes if gate passed

    # Or check without raising
    if gates.check_edit_allowed():
        execute_edit()

Design principle: Make it impossible to accidentally bypass calmness gate.
"""

from dataclasses import dataclass
from typing import Optional

from .stress_bands import (
    StressBandsState,
    StressBandsConfig,
    irreversible_gates,
    IrreversibleGates,
)


class GateViolation(Exception):
    """
    Raised when attempting an irreversible operation without calmness requirement.

    This is not a bug. This is the system working correctly.
    """
    pass


@dataclass
class LifecycleGates:
    """
    Non-bypassable gate checker.

    Wraps stress bands state and provides enforced gate checks.
    All irreversible operations MUST go through this API.
    """

    stress_state: StressBandsState
    stress_cfg: StressBandsConfig
    _gates: Optional[IrreversibleGates] = None

    def __post_init__(self):
        """Compute gates on initialization."""
        self._gates = irreversible_gates(self.stress_state, self.stress_cfg)

    @property
    def gates(self) -> IrreversibleGates:
        """Current gate state."""
        if self._gates is None:
            self._gates = irreversible_gates(self.stress_state, self.stress_cfg)
        return self._gates

    def refresh(self) -> None:
        """Recompute gates (call after stress_state updates)."""
        self._gates = irreversible_gates(self.stress_state, self.stress_cfg)

    # ========================================================================
    # Checking methods (returns bool)
    # ========================================================================

    def check_scar_allowed(self) -> bool:
        """Check if scar formation is allowed (doesn't raise)."""
        return self.gates.allow_scar

    def check_crystallize_allowed(self) -> bool:
        """Check if reflex crystallization is allowed (doesn't raise)."""
        return self.gates.allow_crystallize

    def check_edit_allowed(self) -> bool:
        """Check if structural edits are allowed (doesn't raise)."""
        return self.gates.allow_structural_edits

    # ========================================================================
    # Requiring methods (raises GateViolation if blocked)
    # ========================================================================

    def require_scar_allowed(self) -> None:
        """
        Require scar formation to be allowed.

        Raises:
            GateViolation: If calmness requirement not met
        """
        if not self.gates.allow_scar:
            raise GateViolation(
                f"Scar formation blocked: {self.gates.reason}\n"
                f"Current band: {self.stress_state.current_band}\n"
                f"Time in comfort: {self.stress_state.time_in_comfort} "
                f"(need {self.stress_cfg.scar_calm_steps})\n"
                f"Scars form in calm, not crisis."
            )

    def require_crystallize_allowed(self) -> None:
        """
        Require reflex crystallization to be allowed.

        Raises:
            GateViolation: If calmness requirement not met
        """
        if not self.gates.allow_crystallize:
            raise GateViolation(
                f"Crystallization blocked: {self.gates.reason}\n"
                f"Current band: {self.stress_state.current_band}\n"
                f"Time in comfort: {self.stress_state.time_in_comfort} "
                f"(need {self.stress_cfg.crystallize_calm_steps})\n"
                f"Reflexes crystallize after repeated calm success, not during crisis."
            )

    def require_edit_allowed(self) -> None:
        """
        Require structural edits to be allowed.

        Raises:
            GateViolation: If calmness requirement not met
        """
        if not self.gates.allow_structural_edits:
            raise GateViolation(
                f"Structural edit blocked: {self.gates.reason}\n"
                f"Current band: {self.stress_state.current_band}\n"
                f"Time in comfort: {self.stress_state.time_in_comfort} "
                f"(need {self.stress_cfg.edit_calm_steps})\n"
                f"Identity changes require sustained calm."
            )

    # ========================================================================
    # Context managers (preferred pattern)
    # ========================================================================

    def allow_scar(self):
        """
        Context manager for scar formation.

        Usage:
            with gates.allow_scar():
                form_scar()

        Raises GateViolation if blocked.
        """
        return _GateContext(self, "scar")

    def allow_crystallize(self):
        """
        Context manager for crystallization.

        Usage:
            with gates.allow_crystallize():
                crystallize_reflex()

        Raises GateViolation if blocked.
        """
        return _GateContext(self, "crystallize")

    def allow_edit(self):
        """
        Context manager for structural edits.

        Usage:
            with gates.allow_edit():
                spawn_expert()

        Raises GateViolation if blocked.
        """
        return _GateContext(self, "edit")

    # ========================================================================
    # Status reporting
    # ========================================================================

    def status_summary(self) -> str:
        """Human-readable status summary for logging."""
        return (
            f"LifecycleGates Status:\n"
            f"  Current band: {self.stress_state.current_band}\n"
            f"  Current F_l: {self.stress_state.current_f:.3f}\n"
            f"  Time in comfort: {self.stress_state.time_in_comfort}\n"
            f"  Gates:\n"
            f"    Scar:        {'✓' if self.gates.allow_scar else '✗'}\n"
            f"    Crystallize: {'✓' if self.gates.allow_crystallize else '✗'}\n"
            f"    Edit:        {'✓' if self.gates.allow_structural_edits else '✗'}\n"
            f"  Reason: {self.gates.reason}"
        )


class _GateContext:
    """Internal context manager for gate enforcement."""

    def __init__(self, gate_checker: LifecycleGates, gate_type: str):
        self.gate_checker = gate_checker
        self.gate_type = gate_type

    def __enter__(self):
        # Check gate on entry
        if self.gate_type == "scar":
            self.gate_checker.require_scar_allowed()
        elif self.gate_type == "crystallize":
            self.gate_checker.require_crystallize_allowed()
        elif self.gate_type == "edit":
            self.gate_checker.require_edit_allowed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed
        return False


# ============================================================================
# Frozen survival thresholds (project defaults)
# ============================================================================

# These are conservative: only flag collapse when clearly broken.
# The system should be allowed to be stressed without being declared collapsed.
DEFAULT_COLLAPSE_THRESHOLDS = {
    "psi_critical": 0.1,      # Layer coherence below this = incoherent
    "neff_critical": 1.5,     # Routing to <1.5 experts = collapsed
    "saturation_critical": 0.9,  # One expert >90% load = hogging
    "bimodality_critical": 0.7,  # Mean bimodality >0.7 = widespread instability
}

# Stress band defaults (reasonable starting point)
DEFAULT_STRESS_CONFIG = {
    "comfort_ceiling_init": 1.0,
    "strain_ceiling_init": 2.0,
    "hysteresis_margin": 0.05,
    "scar_calm_steps": 200,
    "crystallize_calm_steps": 2000,
    "edit_calm_steps": 500,
    "target_p_comfort": 0.80,
    "target_p_strain": 0.15,
    "target_p_panic": 0.05,
    "lr_widen": 0.0005,
    "lr_narrow": 0.0005,
    "lr_dist": 0.001,
}


def get_default_collapse_thresholds():
    """Get frozen default collapse thresholds."""
    from .collapse_detection import CollapseThresholds
    return CollapseThresholds(**DEFAULT_COLLAPSE_THRESHOLDS)


def get_default_stress_config():
    """Get frozen default stress band configuration."""
    from .stress_bands import StressBandsConfig
    return StressBandsConfig(**DEFAULT_STRESS_CONFIG)
