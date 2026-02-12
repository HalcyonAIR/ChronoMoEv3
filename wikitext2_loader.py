#!/usr/bin/env python3
"""
WikiText-2 Dataset Loader

Minimal implementation for merge execution testing.
Downloads WikiText-2 (~2MB) and creates batched iterator.
"""

import torch
from pathlib import Path
import urllib.request
import zipfile


def download_wikitext2(data_dir="./data"):
    """
    Download WikiText-2 dataset if not already present.

    Returns:
        Path to train.txt file
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    # Check if already downloaded
    train_file = data_dir / "wikitext-2" / "wiki.train.tokens"
    if train_file.exists():
        print(f"  WikiText-2 already downloaded: {train_file}")
        return train_file

    # Download
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = data_dir / "wikitext-2-v1.zip"

    print(f"  Downloading WikiText-2 from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    # Extract
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Cleanup
    zip_path.unlink()

    print(f"  Downloaded: {train_file}")
    return train_file


def load_wikitext2_tokens(train_file, vocab_size=10000):
    """
    Load WikiText-2 and tokenize to vocabulary.

    Args:
        train_file: Path to wiki.train.tokens
        vocab_size: Vocabulary size (simple word-level tokenization)

    Returns:
        Tensor of token IDs
    """
    # Read text
    with open(train_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Simple word-level tokenization
    words = text.split()

    # Build vocabulary (most common words)
    from collections import Counter
    word_counts = Counter(words)
    vocab = ['<unk>', '<pad>'] + [w for w, _ in word_counts.most_common(vocab_size - 2)]
    word_to_id = {w: i for i, w in enumerate(vocab)}

    # Tokenize
    tokens = [word_to_id.get(w, 0) for w in words]  # 0 = <unk>

    print(f"  Loaded {len(tokens)} tokens, vocab size {len(vocab)}")

    return torch.tensor(tokens, dtype=torch.long), word_to_id


class WikiText2Dataset:
    """
    WikiText-2 batched iterator.

    Provides fixed-size batches for training.
    """
    def __init__(self, tokens, batch_size=1, seq_len=64, seed=42):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)

        # Calculate number of available sequences
        self.num_batches = (len(tokens) - seq_len) // batch_size

        print(f"  Dataset: {len(tokens)} tokens, {self.num_batches} batches available")

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Sample random starting positions
        batch = []
        for _ in range(self.batch_size):
            start_idx = torch.randint(
                0,
                len(self.tokens) - self.seq_len - 1,
                (1,),
                generator=self.rng
            ).item()

            seq = self.tokens[start_idx:start_idx + self.seq_len + 1]
            batch.append(seq)

        self.current_batch += 1
        return torch.stack(batch)  # [batch_size, seq_len+1]


def setup_wikitext2(data_dir="./data", vocab_size=10000, batch_size=4, seq_len=64, seed=42):
    """
    Setup WikiText-2 dataset.

    Returns:
        (dataset, vocab_size_actual, word_to_id)
    """
    # Download
    train_file = download_wikitext2(data_dir)

    # Tokenize
    tokens, word_to_id = load_wikitext2_tokens(train_file, vocab_size)

    # Create dataset
    dataset = WikiText2Dataset(tokens, batch_size=batch_size, seq_len=seq_len, seed=seed)

    return dataset, len(word_to_id), word_to_id


if __name__ == "__main__":
    # Test
    print("Testing WikiText-2 loader...")
    dataset, vocab_size, word_to_id = setup_wikitext2()

    print(f"\nVocabulary size: {vocab_size}")
    print(f"First 10 words: {list(word_to_id.keys())[:10]}")

    print("\nSample batches:")
    for i, batch in enumerate(dataset):
        if i >= 3:
            break
        print(f"  Batch {i}: shape={batch.shape}, sample tokens={batch[0, :10].tolist()}")

    print("\nWikiText-2 loader working!")
