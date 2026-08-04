"""GloVe embedding loader.

Falls back to random init if the GloVe file isn't present -- this is expected during
the local smoke test (downloading the ~2GB glove.840B.300d.txt is out of scope there);
full Colab training must supply the real file (see scripts/download_glove.sh).
"""
import os
import numpy as np
import torch
import torch.nn as nn

from .vocab import Vocab


def build_embedding_matrix(vocab: Vocab, glove_path: str = None, dim: int = 300, seed: int = 0) -> nn.Embedding:
    rng = np.random.RandomState(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[vocab.word2idx["<pad>"]] = 0.0

    if glove_path and os.path.exists(glove_path):
        found = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab.word2idx:
                    matrix[vocab.word2idx[word]] = np.asarray(parts[1:], dtype=np.float32)
                    found += 1
        print(f"[embeddings] loaded {found}/{len(vocab)} words from GloVe at {glove_path}")
    else:
        print(f"[embeddings] GloVe file not found at {glove_path!r} -- using random init "
              f"(expected during the smoke test; full training must supply glove.840B.300d.txt)")

    return nn.Embedding.from_pretrained(
        torch.from_numpy(matrix), freeze=False, padding_idx=vocab.word2idx["<pad>"]
    )


def load_glove_lookup(glove_path: str, vocab_words: set, dim: int = 300) -> dict:
    """Word -> np.ndarray lookup, used by the EM similarity weighting (src/similarity.py),
    independent of the trainable nn.Embedding above."""
    lookup = {}
    if not (glove_path and os.path.exists(glove_path)):
        return lookup
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab_words:
                lookup[word] = np.asarray(parts[1:], dtype=np.float32)
    return lookup
