#!/usr/bin/env python3
"""
Synthetic Datasets for Domain Shift Experiments

Creates clean, controlled distribution shifts to measure:
- Scar formation under old distribution
- Scar obsolescence under new distribution
- Reachability contraction/expansion

Datasets:
1. Arithmetic Progressions: Linear patterns (1,2,3,4 or 10,12,14,16)
2. Geometric Progressions: Exponential patterns (1,2,4,8 or 3,9,27,81)

Shift: Arithmetic → Geometric
This is a genuine structural shift, not just "different random seed."
Experts specialized for linear growth will fail on exponential growth.
"""

import torch
import numpy as np
from typing import Iterator, Tuple


class ArithmeticProgressionDataset:
    """
    Dataset of arithmetic progressions (linear patterns).

    Examples:
        [1, 2, 3, 4, 5, 6, ...]
        [10, 12, 14, 16, 18, ...]
        [5, 10, 15, 20, 25, ...]

    Token vocabulary: 0-99
    Sequence length: Configurable
    """

    def __init__(
        self,
        vocab_size: int = 100,
        seq_len: int = 20,
        batch_size: int = 4,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Generate batch of arithmetic progressions."""
        batch = []

        for _ in range(self.batch_size):
            # Random starting point
            start = self.rng.randint(0, self.vocab_size // 2)

            # Random step size (1-5)
            step = self.rng.randint(1, 6)

            # Generate arithmetic progression
            seq = []
            for i in range(self.seq_len):
                token = (start + i * step) % self.vocab_size
                seq.append(token)

            batch.append(seq)

        return torch.tensor(batch, dtype=torch.long)


class GeometricProgressionDataset:
    """
    Dataset of geometric progressions (exponential patterns).

    Examples:
        [1, 2, 4, 8, 16, 32, ...]  (base 2)
        [3, 9, 27, 81, ...]         (base 3)
        [2, 6, 18, 54, ...]         (base 3, start 2)

    Token vocabulary: 0-99 (with wrapping)
    Sequence length: Configurable
    """

    def __init__(
        self,
        vocab_size: int = 100,
        seq_len: int = 20,
        batch_size: int = 4,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Generate batch of geometric progressions."""
        batch = []

        for _ in range(self.batch_size):
            # Random starting point (1-10, avoid 0)
            start = self.rng.randint(1, 11)

            # Random base (2 or 3, small to avoid overflow)
            base = self.rng.choice([2, 3])

            # Generate geometric progression
            seq = []
            current = start
            for i in range(self.seq_len):
                token = current % self.vocab_size
                seq.append(token)
                current = current * base

            batch.append(seq)

        return torch.tensor(batch, dtype=torch.long)


class HybridDataset:
    """
    Mixture of arithmetic and geometric progressions.

    Used to test partial domain shift:
    - Old: 80% arithmetic, 20% geometric
    - New: 20% arithmetic, 80% geometric
    """

    def __init__(
        self,
        vocab_size: int = 100,
        seq_len: int = 20,
        batch_size: int = 4,
        arithmetic_prob: float = 0.8,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.arithmetic_prob = arithmetic_prob

        self.arithmetic_dataset = ArithmeticProgressionDataset(
            vocab_size, seq_len, batch_size=1, seed=seed
        )
        self.geometric_dataset = GeometricProgressionDataset(
            vocab_size, seq_len, batch_size=1, seed=seed + 1
        )

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """Generate batch with mixture of patterns."""
        batch = []

        for _ in range(self.batch_size):
            if np.random.rand() < self.arithmetic_prob:
                # Arithmetic progression
                seq = next(self.arithmetic_dataset)[0]
            else:
                # Geometric progression
                seq = next(self.geometric_dataset)[0]

            batch.append(seq.tolist())

        return torch.tensor(batch, dtype=torch.long)


def create_domain_shift_datasets(
    vocab_size: int = 100,
    seq_len: int = 20,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[Iterator, Iterator]:
    """
    Create paired datasets for domain shift experiment.

    Old dataset: Arithmetic progressions (linear patterns)
    New dataset: Geometric progressions (exponential patterns)

    This is a genuine structural shift:
    - Different inductive bias required
    - Experts specialized for linear growth fail on exponential
    - Scars formed under arithmetic should become obsolete

    Args:
        vocab_size: Vocabulary size (0 to vocab_size-1)
        seq_len: Sequence length
        batch_size: Batch size
        seed: Random seed

    Returns:
        (old_dataset, new_dataset)
    """
    old_dataset = ArithmeticProgressionDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )

    new_dataset = GeometricProgressionDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed + 1000,  # Different seed to avoid correlation
    )

    return old_dataset, new_dataset


if __name__ == "__main__":
    # Test datasets
    print("Testing Arithmetic → Geometric Domain Shift")
    print("=" * 70)

    old_dataset, new_dataset = create_domain_shift_datasets(
        vocab_size=100,
        seq_len=20,
        batch_size=4,
        seed=42,
    )

    print("\nOld Dataset (Arithmetic Progressions):")
    for i in range(3):
        batch = next(old_dataset)
        print(f"  Batch {i}: {batch[0].tolist()[:10]}... (first seq, first 10 tokens)")

    print("\nNew Dataset (Geometric Progressions):")
    for i in range(3):
        batch = next(new_dataset)
        print(f"  Batch {i}: {batch[0].tolist()[:10]}... (first seq, first 10 tokens)")

    print("\n✓ Datasets created successfully")
    print("  Arithmetic: Linear patterns (e.g., 1,2,3,4,5,...)")
    print("  Geometric: Exponential patterns (e.g., 1,2,4,8,16,...)")
    print("  This is a structural shift, not just different random seed.")
