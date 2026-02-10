"""
Bimodality detection for expert outputs (Phase 3).

Detects when an expert serves two phase-incompatible basins while maintaining
decent average coherence. High coherence ≠ healthy if it's an average over
two distinct modes.

Core mechanism:
- Track two centroids (running means of expert output)
- Assignment: which centroid is closer on each forward pass
- Separation: cosine distance between centroids (directionality)
- Balance: usage ratio (skewed → 0, balanced → 1)
- Bimodality score: separation × balance

Ported from ChronoMoEv3/chronomoe_v3/bimodality.py (simplified, router-agnostic).
"""

from dataclasses import dataclass
from typing import Dict
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class BimodalityState:
    """
    Bimodality tracking state for one expert.

    Tracks two centroids (running means) and their usage statistics.
    """

    expert_id: int
    layer_id: int
    d_model: int

    # Two centroids (running means)
    centroid_a: Tensor  # [d_model]
    centroid_b: Tensor  # [d_model]

    # Assignment counts
    count_a: int = 0  # How many times assigned to centroid A
    count_b: int = 0  # How many times assigned to centroid B

    # EMA decay for centroid updates
    alpha: float = 0.95  # ~20 steps half-life

    def update(self, y_expert_mean: Tensor):
        """
        Update bimodality state with new expert output.

        Args:
            y_expert_mean: [d_model] - mean expert output for this batch
        """
        # Initialize first centroid
        if self.count_a == 0 and self.count_b == 0:
            self.centroid_a = y_expert_mean.clone().detach()
            self.count_a = 1
            return

        # Initialize second centroid if point is far from first
        if self.count_b == 0:
            dist_from_a = (y_expert_mean - self.centroid_a).norm()
            # If far enough, initialize B
            if dist_from_a > 0.5:  # Threshold for "far enough"
                self.centroid_b = y_expert_mean.clone().detach()
                self.count_b = 1
                return
            else:
                # Close to A, update A
                self.centroid_a = (
                    self.alpha * self.centroid_a + (1 - self.alpha) * y_expert_mean
                ).detach()
                self.count_a += 1
                return

        # Both centroids initialized - normal assignment
        dist_a = (y_expert_mean - self.centroid_a).norm()
        dist_b = (y_expert_mean - self.centroid_b).norm()

        # Assign to closer centroid and update it
        if dist_a < dist_b:
            # Closer to A
            self.centroid_a = (
                self.alpha * self.centroid_a + (1 - self.alpha) * y_expert_mean
            ).detach()
            self.count_a += 1
        else:
            # Closer to B
            self.centroid_b = (
                self.alpha * self.centroid_b + (1 - self.alpha) * y_expert_mean
            ).detach()
            self.count_b += 1

    def compute_separation(self) -> float:
        """
        Compute separation between centroids (cosine distance).

        Returns:
            Separation: cosine distance in [0, 2] (0=same, 2=opposite)
        """
        if self.count_a == 0 or self.count_b == 0:
            return 0.0  # No separation if only one mode observed

        # Cosine distance (1 - cosine similarity)
        cos_sim = F.cosine_similarity(
            self.centroid_a.unsqueeze(0), self.centroid_b.unsqueeze(0)
        )
        separation = (1.0 - cos_sim).item()

        return separation

    def compute_balance(self) -> float:
        """
        Compute balance of assignments.

        Returns:
            Balance in [0, 1]: 1.0 = perfectly balanced, 0.0 = all one mode
        """
        total = self.count_a + self.count_b
        if total == 0:
            return 0.0

        # Ratio of minority to majority
        p_a = self.count_a / total
        p_b = self.count_b / total

        balance = min(p_a, p_b) / max(p_a, p_b) if max(p_a, p_b) > 0 else 0.0

        return balance

    def compute_bimodality_score(self) -> float:
        """
        Compute bimodality score: separation × balance.

        High score means:
        - Centroids are far apart (separation)
        - Both centroids are used frequently (balance)

        This indicates expert serving two incompatible basins.

        Returns:
            Bimodality score in [0, 2] (typically [0, 1])
        """
        separation = self.compute_separation()
        balance = self.compute_balance()

        return separation * balance

    def to_dict(self) -> Dict:
        """Serialize state to dict (for logging)."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "separation": round(self.compute_separation(), 4),
            "balance": round(self.compute_balance(), 4),
            "bimodality_score": round(self.compute_bimodality_score(), 4),
            "count_a": self.count_a,
            "count_b": self.count_b,
            "total_observations": self.count_a + self.count_b,
        }


def update_bimodality(
    state: BimodalityState,
    expert_output: Tensor,  # [B*T, d_model] - per-expert output
) -> BimodalityState:
    """
    Update bimodality state with new expert output.

    Args:
        state: Current bimodality state
        expert_output: Per-expert output tensor [B*T, d_model]

    Returns:
        Updated BimodalityState (mutates in place and returns)
    """
    # Compute mean output for this batch
    y_expert_mean = expert_output.mean(dim=0)  # [d_model]

    # Update centroids
    state.update(y_expert_mean)

    return state


def compute_layer_bimodality(
    states: Dict[int, BimodalityState], min_observations: int = 100
) -> float:
    """
    Compute layer-wide bimodality (max score across experts).

    Args:
        states: Dict of expert_id -> BimodalityState
        min_observations: Minimum observations before considering

    Returns:
        Max bimodality score across experts with sufficient observations
    """
    if not states:
        return 0.0

    max_score = 0.0
    for state in states.values():
        total_obs = state.count_a + state.count_b
        if total_obs < min_observations:
            continue

        score = state.compute_bimodality_score()
        max_score = max(max_score, score)

    return max_score
