"""
Bimodality detection for expert outputs.

Detects when an expert is serving two phase-incompatible basins while
maintaining decent average coherence. High coherence ≠ healthy if it's
an average over two distinct modes.

Core mechanism:
- Track two centroids (running means of expert output)
- Assignment: which centroid is closer on each forward pass
- Separation: distance between centroids
- Balance: usage ratio (skewed → 0, balanced → 1)
- Bimodality score: separation × balance

High score → expert serving incompatible basins → split candidate
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class BimodalityState:
    """
    Bimodality tracking state for one expert.

    Tracks two centroids (running means) and their usage statistics.
    """

    expert_id: str
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
            self.centroid_a = y_expert_mean.clone()
            self.count_a = 1
            return

        # Initialize second centroid if point is far from first
        if self.count_b == 0:
            dist_from_a = (y_expert_mean - self.centroid_a).norm()
            # If far enough, initialize B
            if dist_from_a > 0.5:  # Threshold for "far enough"
                self.centroid_b = y_expert_mean.clone()
                self.count_b = 1
                return
            else:
                # Close to A, update A
                self.centroid_a = (
                    self.alpha * self.centroid_a + (1 - self.alpha) * y_expert_mean
                )
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
            )
            self.count_a += 1
        else:
            # Closer to B
            self.centroid_b = (
                self.alpha * self.centroid_b + (1 - self.alpha) * y_expert_mean
            )
            self.count_b += 1

    def compute_separation(self) -> float:
        """
        Compute separation between centroids.

        Returns:
            Separation: normalized distance between centroids
        """
        if self.count_a == 0 or self.count_b == 0:
            return 0.0  # No separation if only one mode observed

        # Cosine distance (1 - cosine similarity)
        cos_sim = torch.nn.functional.cosine_similarity(
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
            Bimodality score in [0, 1]
        """
        separation = self.compute_separation()
        balance = self.compute_balance()

        return separation * balance

    def reset_counts(self):
        """Reset assignment counts (for periodic evaluation)."""
        self.count_a = 0
        self.count_b = 0

    def to_dict(self) -> dict:
        """Serialize state to dict."""
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "separation": self.compute_separation(),
            "balance": self.compute_balance(),
            "bimodality_score": self.compute_bimodality_score(),
            "count_a": self.count_a,
            "count_b": self.count_b,
        }


class BimodalityDetector:
    """
    Detector for bimodal expert behavior.

    Manages BimodalityState for multiple experts, detects split candidates.
    """

    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        d_model: int,
        device: str = "cuda",
        split_threshold: float = 0.3,
        min_observations: int = 100,
    ):
        """
        Initialize bimodality detector.

        Args:
            layer_id: Layer identifier
            num_experts: Number of experts in layer
            d_model: Hidden dimension
            device: Device for tensors
            split_threshold: Bimodality score above this → split candidate
            min_observations: Min total observations before considering split
        """
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = device
        self.split_threshold = split_threshold
        self.min_observations = min_observations

        # Initialize states for all experts
        self.states = {}
        for expert_id in range(num_experts):
            self.states[expert_id] = BimodalityState(
                expert_id=f"L{layer_id}_E{expert_id}",
                layer_id=layer_id,
                d_model=d_model,
                centroid_a=torch.zeros(d_model, device=device),
                centroid_b=torch.zeros(d_model, device=device),
            )

    def update(self, expert_id: int, y_expert_mean: Tensor):
        """
        Update bimodality state for one expert.

        Args:
            expert_id: Expert index
            y_expert_mean: [d_model] - mean expert output
        """
        if expert_id not in self.states:
            # Create state on-the-fly if expert added dynamically
            self.states[expert_id] = BimodalityState(
                expert_id=f"L{self.layer_id}_E{expert_id}",
                layer_id=self.layer_id,
                d_model=self.d_model,
                centroid_a=torch.zeros(self.d_model, device=self.device),
                centroid_b=torch.zeros(self.d_model, device=self.device),
            )

        self.states[expert_id].update(y_expert_mean)

    def detect_split_candidates(self) -> list[int]:
        """
        Detect experts that are split candidates.

        Returns:
            List of expert IDs with high bimodality score
        """
        candidates = []

        for expert_id, state in self.states.items():
            # Check minimum observations
            total_obs = state.count_a + state.count_b
            if total_obs < self.min_observations:
                continue

            # Check bimodality score
            score = state.compute_bimodality_score()
            if score > self.split_threshold:
                candidates.append(expert_id)

        return candidates

    def get_statistics(self, expert_id: int) -> dict:
        """
        Get bimodality statistics for one expert.

        Args:
            expert_id: Expert index

        Returns:
            Dict with separation, balance, score, counts
        """
        if expert_id not in self.states:
            return {}

        return self.states[expert_id].to_dict()

    def snapshot(self) -> dict[int, BimodalityState]:
        """
        Get snapshot of all bimodality states.

        Returns:
            {expert_id: BimodalityState}
        """
        return {eid: state for eid, state in self.states.items()}

    def reset_all_counts(self):
        """Reset assignment counts for all experts."""
        for state in self.states.values():
            state.reset_counts()
