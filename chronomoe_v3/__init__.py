"""
ChronoMoEv3: Unified multi-timescale MoE lifecycle system.

One mechanism, three projections. Phase coherence tracked at three decay rates.
Fast explores, medium negotiates, slow commits.
"""

from .coherence import (
    MoETrace,
    CoherenceState,
    compute_coherence,
    update_coherence_ema,
    batch_update_coherence,
)

from .clocks import (
    ClockConfig,
    ThreeClockEMA,
)

from .config import ChronoConfig

from .router import (
    RouterState,
    ChronoRouter,
    update_beta,
    update_beta_from_buffer,
    compute_js_divergence,
    compute_flip_rate,
    compute_overlap_only,
    compute_relevance,
)

from .coherence_gpu import (
    CoherenceBuffer,
    MultiLayerCoherenceBuffer,
)

from .lifecycle import (
    PruneDecision,
    LifecycleCoordinator,
    compute_neff,
    compute_saturation,
)

from .bimodality import (
    BimodalityState,
    BimodalityDetector,
)

from .free_energy import (
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

__version__ = "0.1.0"

__all__ = [
    # Coherence
    "MoETrace",
    "CoherenceState",
    "compute_coherence",
    "update_coherence_ema",
    "batch_update_coherence",
    # Coherence GPU
    "CoherenceBuffer",
    "MultiLayerCoherenceBuffer",
    # Clocks
    "ClockConfig",
    "ThreeClockEMA",
    # Config
    "ChronoConfig",
    # Router
    "RouterState",
    "ChronoRouter",
    "update_beta",
    "update_beta_from_buffer",
    "compute_js_divergence",
    "compute_flip_rate",
    "compute_overlap_only",
    "compute_relevance",
    # Lifecycle
    "PruneDecision",
    "LifecycleCoordinator",
    "compute_neff",
    "compute_saturation",
    # Bimodality
    "BimodalityState",
    "BimodalityDetector",
    # Free Energy
    "FreeEnergyComponents",
    "FreeEnergyState",
    "compute_layer_coherence",
    "compute_misfit_term",
    "compute_complexity_term",
    "compute_redundancy_term",
    "compute_instability_term",
    "compute_free_energy",
    "create_free_energy_state",
]
