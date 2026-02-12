#!/usr/bin/env python3
"""
Challenge Receipt Tracking: Behavioral Evidence of Method Diversity

Attack surface exhaustion occurs when the system's method repertoire collapses
into a tight cluster, even if agreement hasn't converged yet.

Challenge receipts are behavioral logs (not self-report) of HOW the system
responds to challenges, capturing method-space diversity over time.

Key distinction from convergence:
- Convergence: Bob and maniac agree on WHAT (claims)
- Exhaustion: System uses same HOW (method family) regardless of challenge

Detection:
1. Log behavioral fingerprints (not opinions)
2. Measure method-space diversity (pairwise distance)
3. Detect clustering (tight method family)
4. Test: Does clustering precede convergence?
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


@dataclass
class ChallengeReceipt:
    """
    Behavioral fingerprint of a single response to a challenge.

    IMPORTANT: This is behavioral evidence, not self-report.
    We don't ask "what did you do?" - we observe what happened.
    """
    step: int                          # When this response occurred
    challenge_id: int                  # Which challenge triggered this

    # Behavioral fingerprint (HOW the system responded)
    routing_pattern: np.ndarray        # Router weights (which experts fired)
    activation_magnitude: float        # How strong was the response
    gradient_direction: np.ndarray     # Which direction did weights move

    # Context
    scar_strength: float              # Total scar strength when this occurred
    regime: str                       # Deformation regime (exploration/transition/saturation)

    # Optional: If this was a successful downgrade
    was_downgrade: bool = False
    downgrade_impact: float = 0.0


@dataclass
class MethodCluster:
    """
    A cluster of similar behavioral responses.

    When multiple receipts cluster tightly in method space,
    it indicates the system is using the same approach repeatedly.
    """
    cluster_id: int
    receipts: List[ChallengeReceipt]
    centroid: np.ndarray               # Average method fingerprint
    radius: float                      # Spread of receipts around centroid

    # Statistics
    first_seen: int                    # Step when cluster first formed
    last_seen: int                     # Most recent receipt in cluster
    regime_distribution: Dict[str, int] = field(default_factory=dict)  # Which regimes


@dataclass
class ExhaustionState:
    """
    Current state of method-space diversity.

    Exhaustion occurs when diversity collapses into tight clusters,
    regardless of whether Bob and maniac agree yet.
    """
    is_exhausted: bool

    # Evidence
    method_diversity: float            # Pairwise distance variance
    num_clusters: int                  # How many distinct method families
    largest_cluster_fraction: float    # Fraction of receipts in dominant cluster

    # Context
    step: int
    regime: str
    num_receipts: int

    # Persistence
    persistence_count: int             # How many intervals has exhaustion held
    exhausted_since: Optional[int] = None


@dataclass
class ExhaustionEvent:
    """
    Event: Transition from diverse → exhausted method space.

    This is the moment when clustering crosses threshold.
    """
    step: int
    regime: str
    scar_strength: float

    # Evidence at transition
    method_diversity: float
    num_clusters: int
    largest_cluster_fraction: float

    # Predictive: Did this precede convergence?
    steps_before_convergence: Optional[int] = None


def compute_method_distance(receipt_a: ChallengeReceipt, receipt_b: ChallengeReceipt) -> float:
    """
    Compute distance between two behavioral responses in method space.

    Method space dimensions:
    1. Routing pattern similarity (cosine distance)
    2. Activation magnitude difference (L1)
    3. Gradient direction similarity (cosine distance)

    Returns:
        Distance in [0, 1] where 0=identical methods, 1=maximally different
    """
    # Routing pattern similarity (cosine distance)
    routing_sim = np.dot(receipt_a.routing_pattern, receipt_b.routing_pattern) / (
        np.linalg.norm(receipt_a.routing_pattern) * np.linalg.norm(receipt_b.routing_pattern) + 1e-8
    )
    routing_dist = (1.0 - routing_sim) / 2.0  # Scale to [0, 1]

    # Activation magnitude difference (normalized L1)
    activation_diff = abs(receipt_a.activation_magnitude - receipt_b.activation_magnitude)
    activation_dist = min(activation_diff, 1.0)  # Clamp to [0, 1]

    # Gradient direction similarity (cosine distance)
    grad_sim = np.dot(receipt_a.gradient_direction, receipt_b.gradient_direction) / (
        np.linalg.norm(receipt_a.gradient_direction) * np.linalg.norm(receipt_b.gradient_direction) + 1e-8
    )
    grad_dist = (1.0 - grad_sim) / 2.0

    # Weighted combination (routing most important, gradient second, activation least)
    distance = 0.5 * routing_dist + 0.3 * grad_dist + 0.2 * activation_dist

    return float(distance)


def compute_method_diversity(receipts: List[ChallengeReceipt], sample_size: int = 50) -> float:
    """
    Compute overall method-space diversity from recent receipts.

    Uses pairwise distance variance as diversity metric:
    - High variance = diverse methods (wide spread)
    - Low variance = exhausted methods (tight cluster)

    Args:
        receipts: Recent challenge receipts
        sample_size: Number of receipts to sample (for efficiency)

    Returns:
        Diversity score in [0, 1] where 0=collapsed, 1=maximally diverse
    """
    if len(receipts) < 2:
        return 1.0  # Single receipt = no clustering yet

    # Sample if too many receipts
    if len(receipts) > sample_size:
        indices = np.random.choice(len(receipts), sample_size, replace=False)
        sampled = [receipts[i] for i in indices]
    else:
        sampled = receipts

    # Compute all pairwise distances
    distances = []
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            dist = compute_method_distance(sampled[i], sampled[j])
            distances.append(dist)

    if not distances:
        return 1.0

    # Diversity = variance of pairwise distances
    # High variance = some pairs very different (diverse)
    # Low variance = all pairs similar (clustered)
    diversity = float(np.std(distances))

    return diversity


def cluster_receipts(
    receipts: List[ChallengeReceipt],
    distance_threshold: float = 0.2,
) -> List[MethodCluster]:
    """
    Cluster receipts by method similarity using simple threshold-based clustering.

    Args:
        receipts: Challenge receipts to cluster
        distance_threshold: Maximum distance for two receipts to be in same cluster

    Returns:
        List of MethodCluster objects
    """
    if not receipts:
        return []

    clusters: List[MethodCluster] = []

    for receipt in receipts:
        # Try to assign to existing cluster
        assigned = False

        for cluster in clusters:
            # Compute distance to cluster centroid
            # For simplicity, use first receipt as representative
            representative = cluster.receipts[0]
            distance = compute_method_distance(receipt, representative)

            if distance <= distance_threshold:
                cluster.receipts.append(receipt)
                cluster.last_seen = receipt.step

                # Update regime distribution
                regime = receipt.regime
                cluster.regime_distribution[regime] = cluster.regime_distribution.get(regime, 0) + 1

                assigned = True
                break

        # Create new cluster if not assigned
        if not assigned:
            new_cluster = MethodCluster(
                cluster_id=len(clusters),
                receipts=[receipt],
                centroid=receipt.routing_pattern.copy(),  # Simplified centroid
                radius=0.0,
                first_seen=receipt.step,
                last_seen=receipt.step,
                regime_distribution={receipt.regime: 1},
            )
            clusters.append(new_cluster)

    # Update cluster statistics
    for cluster in clusters:
        # Compute actual centroid (average routing pattern)
        routing_patterns = np.array([r.routing_pattern for r in cluster.receipts])
        cluster.centroid = np.mean(routing_patterns, axis=0)

        # Compute radius (max distance from centroid)
        distances = []
        for receipt in cluster.receipts:
            # Distance from receipt to centroid (simplified)
            dist = np.linalg.norm(receipt.routing_pattern - cluster.centroid)
            distances.append(dist)

        cluster.radius = float(max(distances)) if distances else 0.0

    return clusters


@dataclass
class ExhaustionThresholds:
    """
    Operational thresholds for exhaustion detection.

    Like convergence thresholds, these should be derived from baseline,
    not intuition.

    IMPORTANT: These should be regime-specific, not global percentiles.
    """
    # Method diversity (minimum variance of pairwise distances) - per regime
    diversity_min_exploration: float = 0.1
    diversity_min_transition: float = 0.08
    diversity_min_saturation: float = 0.05

    # Clustering (maximum fraction in single cluster) - per regime
    max_cluster_fraction_exploration: float = 0.5
    max_cluster_fraction_transition: float = 0.6
    max_cluster_fraction_saturation: float = 0.7

    # Activity gate (minimum mean activation to avoid "system asleep" clustering)
    min_activation_magnitude: float = 0.1

    # Persistence (intervals exhaustion must hold)
    K_persistence: int = 5

    # Minimum receipts for reliable detection
    min_receipts: int = 20


def get_regime_thresholds(regime: str, thresholds: ExhaustionThresholds) -> Tuple[float, float]:
    """
    Get regime-specific diversity and clustering thresholds.

    Args:
        regime: Deformation regime (exploration/transition/saturation)
        thresholds: ExhaustionThresholds with per-regime values

    Returns:
        (diversity_min, max_cluster_fraction) for this regime
    """
    if regime == "exploration":
        return thresholds.diversity_min_exploration, thresholds.max_cluster_fraction_exploration
    elif regime == "transition":
        return thresholds.diversity_min_transition, thresholds.max_cluster_fraction_transition
    else:  # saturation
        return thresholds.diversity_min_saturation, thresholds.max_cluster_fraction_saturation


class ExhaustionDetector:
    """
    Detector for attack surface exhaustion.

    Usage:
        detector = ExhaustionDetector(thresholds)

        for step in training:
            receipt = ChallengeReceipt(...)
            state = detector.update(receipt)

            if state.is_exhausted:
                print(f"Method space exhausted at step {state.exhausted_since}")
    """

    def __init__(
        self,
        thresholds: ExhaustionThresholds,
        window_size: int = 100,
    ):
        """
        Initialize exhaustion detector.

        Args:
            thresholds: Exhaustion detection thresholds (regime-specific)
            window_size: Number of recent receipts to consider
        """
        self.thresholds = thresholds
        self.window_size = window_size

        # Receipt log (sliding window)
        self.receipts: List[ChallengeReceipt] = []

        # Exhaustion state
        self.exhausted = False
        self.exhausted_since = None
        self.persistence_count = 0

        # Event log
        self.exhaustion_events: List[ExhaustionEvent] = []

    def update(
        self,
        receipt: ChallengeReceipt,
    ) -> ExhaustionState:
        """
        Update exhaustion detector with new challenge receipt.

        Args:
            receipt: New behavioral fingerprint

        Returns:
            ExhaustionState (current state after update)
        """
        # Add receipt to sliding window
        self.receipts.append(receipt)
        if len(self.receipts) > self.window_size:
            self.receipts.pop(0)

        # Not enough receipts yet
        if len(self.receipts) < self.thresholds.min_receipts:
            return ExhaustionState(
                is_exhausted=False,
                method_diversity=1.0,
                num_clusters=0,
                largest_cluster_fraction=0.0,
                step=receipt.step,
                regime=receipt.regime,
                num_receipts=len(self.receipts),
                persistence_count=0,
            )

        # Compute method diversity
        diversity = compute_method_diversity(self.receipts)

        # Cluster receipts
        clusters = cluster_receipts(self.receipts)
        num_clusters = len(clusters)

        # Largest cluster fraction
        if clusters:
            largest_cluster = max(clusters, key=lambda c: len(c.receipts))
            largest_cluster_fraction = len(largest_cluster.receipts) / len(self.receipts)
        else:
            largest_cluster_fraction = 0.0

        # Activity gate: check mean activation magnitude (avoid "system asleep" clustering)
        recent_activations = [r.activation_magnitude for r in self.receipts[-20:]]  # Last 20 receipts
        mean_activation = float(np.mean(recent_activations)) if recent_activations else 0.0
        activity_sufficient = mean_activation >= self.thresholds.min_activation_magnitude

        # Get regime-specific thresholds
        diversity_min, max_cluster_fraction = get_regime_thresholds(receipt.regime, self.thresholds)

        # Check exhaustion conditions (all three must hold)
        diversity_collapsed = diversity <= diversity_min
        clustering_high = largest_cluster_fraction >= max_cluster_fraction

        conditions_met = diversity_collapsed and clustering_high and activity_sufficient

        # Update persistence
        if conditions_met:
            self.persistence_count += 1
        else:
            self.persistence_count = 0

        # State transition
        was_exhausted = self.exhausted

        if not was_exhausted and self.persistence_count >= self.thresholds.K_persistence:
            # Transition: diverse → exhausted
            self.exhausted = True
            self.exhausted_since = receipt.step - self.thresholds.K_persistence

            # Log exhaustion event
            event = ExhaustionEvent(
                step=self.exhausted_since,
                regime=receipt.regime,
                scar_strength=receipt.scar_strength,
                method_diversity=diversity,
                num_clusters=num_clusters,
                largest_cluster_fraction=largest_cluster_fraction,
            )
            self.exhaustion_events.append(event)

        elif was_exhausted and not conditions_met:
            # Transition: exhausted → diverse (recovered)
            self.exhausted = False
            self.exhausted_since = None

        # Build current state
        state = ExhaustionState(
            is_exhausted=self.exhausted,
            method_diversity=diversity,
            num_clusters=num_clusters,
            largest_cluster_fraction=largest_cluster_fraction,
            step=receipt.step,
            regime=receipt.regime,
            num_receipts=len(self.receipts),
            persistence_count=self.persistence_count,
            exhausted_since=self.exhausted_since,
        )

        return state

    def get_exhaustion_statistics(self) -> Dict:
        """
        Get statistics about exhaustion events.

        Returns:
            Dict with event counts, timing, regime distribution
        """
        if not self.exhaustion_events:
            return {
                "num_exhaustion_events": 0,
                "exhaustion_by_regime": {
                    "exploration": 0,
                    "transition": 0,
                    "saturation": 0,
                },
                "scar_strength_at_exhaustion": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            }

        # Exhaustion by regime
        regime_counts = defaultdict(int)
        for event in self.exhaustion_events:
            regime_counts[event.regime] += 1

        # Scar strength at exhaustion
        scar_strengths = [e.scar_strength for e in self.exhaustion_events]

        return {
            "num_exhaustion_events": len(self.exhaustion_events),
            "exhaustion_by_regime": {
                "exploration": regime_counts.get("exploration", 0),
                "transition": regime_counts.get("transition", 0),
                "saturation": regime_counts.get("saturation", 0),
            },
            "scar_strength_at_exhaustion": {
                "mean": np.mean(scar_strengths) if scar_strengths else 0.0,
                "std": np.std(scar_strengths) if scar_strengths else 0.0,
                "min": min(scar_strengths) if scar_strengths else 0.0,
                "max": max(scar_strengths) if scar_strengths else 0.0,
            },
        }


def derive_exhaustion_thresholds_from_baseline(
    baseline_data: Dict,
    percentile: float = 10.0,
) -> ExhaustionThresholds:
    """
    Derive exhaustion thresholds from baseline distributions.

    DO NOT use intuition. Use percentiles from natural behavior.

    IMPORTANT: Compute thresholds PER REGIME to avoid mixing distributions.

    Args:
        baseline_data: Dict with baseline distributions:
            - method_diversity_by_regime: Dict[regime, List[float]]
            - cluster_fractions_by_regime: Dict[regime, List[float]]
            - activation_magnitudes: List[float]
        percentile: Percentile to use for threshold (default 10th = low end)

    Returns:
        ExhaustionThresholds derived from data (regime-specific)
    """
    diversity_by_regime = baseline_data.get("method_diversity_by_regime", {})
    cluster_by_regime = baseline_data.get("cluster_fractions_by_regime", {})
    activations = baseline_data.get("activation_magnitudes", [1.0])

    # diversity_min: 10th percentile per regime (low end of natural diversity)
    diversity_min_exploration = float(np.percentile(
        diversity_by_regime.get("exploration", [0.1]),
        percentile
    ))
    diversity_min_transition = float(np.percentile(
        diversity_by_regime.get("transition", [0.08]),
        percentile
    ))
    diversity_min_saturation = float(np.percentile(
        diversity_by_regime.get("saturation", [0.05]),
        percentile
    ))

    # max_cluster_fraction: 90th percentile per regime (high end of natural clustering)
    max_cluster_exploration = float(np.percentile(
        cluster_by_regime.get("exploration", [0.5]),
        100 - percentile
    ))
    max_cluster_transition = float(np.percentile(
        cluster_by_regime.get("transition", [0.6]),
        100 - percentile
    ))
    max_cluster_saturation = float(np.percentile(
        cluster_by_regime.get("saturation", [0.7]),
        100 - percentile
    ))

    # min_activation_magnitude: 10th percentile (low end of natural activity)
    min_activation = float(np.percentile(activations, percentile))

    return ExhaustionThresholds(
        diversity_min_exploration=diversity_min_exploration,
        diversity_min_transition=diversity_min_transition,
        diversity_min_saturation=diversity_min_saturation,
        max_cluster_fraction_exploration=max_cluster_exploration,
        max_cluster_fraction_transition=max_cluster_transition,
        max_cluster_fraction_saturation=max_cluster_saturation,
        min_activation_magnitude=min_activation,
        K_persistence=5,  # Fixed: require 5 intervals
        min_receipts=20,  # Fixed: need 20 receipts minimum
    )
