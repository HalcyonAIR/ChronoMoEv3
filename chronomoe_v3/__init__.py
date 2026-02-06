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
    compute_js_divergence,
    compute_flip_rate,
    compute_relevance,
)

__version__ = "0.1.0"

__all__ = [
    # Coherence
    "MoETrace",
    "CoherenceState",
    "compute_coherence",
    "update_coherence_ema",
    "batch_update_coherence",
    # Clocks
    "ClockConfig",
    "ThreeClockEMA",
    # Config
    "ChronoConfig",
    # Router
    "RouterState",
    "ChronoRouter",
    "update_beta",
    "compute_js_divergence",
    "compute_flip_rate",
    "compute_relevance",
]
