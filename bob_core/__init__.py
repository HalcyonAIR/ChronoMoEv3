# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""
Bob: consequence-accumulating control plane for MoE models.

Bob observes routing decisions, accumulates motifs, and learns when
to take the cheap path. The model does the thinking. Bob decides
how much thinking is necessary.

Phase 1: halting + ledgers + medium clock + governor.
"""

from bob_core.motifs import MotifStore, MotifRecord, GateSignals, CompoundGate, GateThresholds, GateResult
from bob_core.substrate import BobSubstrate
from bob_core.telemetry import DecisionTrace
from bob_core.ledgers import (
    BobCore,
    CommitmentLedger,
    ScarLedger,
    CostLedger,
    Commitment,
    Scar,
    RoutingVector,
    CostSignal,
    GovernanceCoords,
)
from bob_core.medium_clock import MediumClock, MediumClockState
from bob_core.governor import BobGovernor, GovernorDecision, GovernorVerdict
from bob_core.identity import is_identity_event
from bob_core.promotion import PromotionGate

__all__ = [
    # Original
    "MotifStore",
    "MotifRecord",
    "GateSignals",
    "GateResult",
    "CompoundGate",
    "GateThresholds",
    "BobSubstrate",
    "DecisionTrace",
    # Phase 1: Ledgers
    "BobCore",
    "CommitmentLedger",
    "ScarLedger",
    "CostLedger",
    "Commitment",
    "Scar",
    "RoutingVector",
    "CostSignal",
    "GovernanceCoords",
    # Phase 1: Medium Clock
    "MediumClock",
    "MediumClockState",
    # Phase 1: Governor
    "BobGovernor",
    "GovernorDecision",
    "GovernorVerdict",
    # Phase 1: Identity + Promotion
    "is_identity_event",
    "PromotionGate",
]
