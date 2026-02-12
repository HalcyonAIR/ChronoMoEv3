"""
ChronoMoE Integration for swiss-ai/MoE.

Provides lifecycle-aware MoE layers with fixed-width routing.
"""

from chronomoe_integration.chronomoe_layer import ChronoMoE
from chronomoe_integration.expert_registry import (
    ExpertRegistry,
    ExpertState,
    ProbationConfig,
    ExpertInfo,
)
from chronomoe_integration.fixed_width_router import FixedWidthRouter
from chronomoe_integration.stress_bands import (
    StressBandsState,
    StressBandsConfig,
    Band,
    LifecycleGates,
    init_stress_bands,
    step_stress_bands,
    lifecycle_gates,
)
from chronomoe_integration.controller import (
    ChronoController,
    ObservationSnapshot,
    EditProposal,
    EditResult,
    create_controller,
)
from chronomoe_integration.coherence import (
    CoherenceState,
)
from chronomoe_integration.bimodality import (
    BimodalityState,
)
from chronomoe_integration.free_energy import (
    FreeEnergyState,
    FreeEnergyComponents,
)
from chronomoe_integration.delta_bundle import (
    DeltaBundleMerge,
    create_delta_bundle,
    save_delta_bundle_weights,
    save_delta_bundle_metadata,
    load_delta_bundle_weights,
    load_delta_bundle_metadata,
)
from chronomoe_integration.rollback import (
    rollback_merge,
    verify_rollback,
)
from chronomoe_integration.probe_battery import (
    ProbeBatteryResult,
    run_probe_battery,
    check_protected_deltas,
    print_probe_battery_result,
)
from chronomoe_integration.merge_execution import (
    execute_merge,
    find_merge_candidate,
)
from chronomoe_integration.routing_geometry import (
    measure_routing_geometry,
    measure_adaptation_speed,
    compute_effective_rank,
    extract_motifs,
)
from chronomoe_integration.convergence import (
    ConvergenceDetector,
    ConvergenceThresholds,
    ConvergenceState,
    ConvergenceEvent,
    DivergenceEvent,
    DeformationRegime,
    derive_thresholds_from_baseline,
)

__all__ = [
    "ChronoMoE",
    "ExpertRegistry",
    "ExpertState",
    "ProbationConfig",
    "ExpertInfo",
    "FixedWidthRouter",
    "StressBandsState",
    "StressBandsConfig",
    "Band",
    "LifecycleGates",
    "init_stress_bands",
    "step_stress_bands",
    "lifecycle_gates",
    # Milestone A: Controller API
    "ChronoController",
    "ObservationSnapshot",
    "EditProposal",
    "EditResult",
    "create_controller",
    "CoherenceState",
    # Milestone B: Bimodality
    "BimodalityState",
    # Milestone C: Free Energy
    "FreeEnergyState",
    "FreeEnergyComponents",
    # Phase 2: MERGE Execution
    "DeltaBundleMerge",
    "create_delta_bundle",
    "save_delta_bundle_weights",
    "save_delta_bundle_metadata",
    "load_delta_bundle_weights",
    "load_delta_bundle_metadata",
    "rollback_merge",
    "verify_rollback",
    "ProbeBatteryResult",
    "run_probe_battery",
    "check_protected_deltas",
    "print_probe_battery_result",
    "execute_merge",
    "find_merge_candidate",
    # Routing Geometry (Manifold Measurement)
    "measure_routing_geometry",
    "measure_adaptation_speed",
    "compute_effective_rank",
    "extract_motifs",
    # Convergence Detection (Phase 2)
    "ConvergenceDetector",
    "ConvergenceThresholds",
    "ConvergenceState",
    "ConvergenceEvent",
    "DivergenceEvent",
    "DeformationRegime",
    "derive_thresholds_from_baseline",
]
