"""
Bob: consequence-accumulating control plane for MoE models.

Bob observes routing decisions, accumulates motifs, and learns when
to take the cheap path. The model does the thinking. Bob decides
how much thinking is necessary.
"""

from bob_core.motifs import MotifStore, MotifRecord, GateSignals, CompoundGate, GateThresholds
from bob_core.substrate import BobSubstrate
from bob_core.telemetry import DecisionTrace

__all__ = [
    "MotifStore",
    "MotifRecord",
    "GateSignals",
    "CompoundGate",
    "GateThresholds",
    "BobSubstrate",
    "DecisionTrace",
]
