"""Similarity measures for the EM-style latent-caption weighting -- paper Sec 3.3, Eq. 7:

  P(c_k | q, t) = s(q, c_k) / sum_{c_j in C} s(q, c_j)

  s(q, c) = alpha * sim_s(q, c) + (1 - alpha) * sim_e(q, c),   alpha = 0.75 (paper-given)

sim_s: Jaccard index over character trigrams of the surface strings (paper: 'Both
strings are broken down to a set of char-based trigrams').

sim_e: cosine similarity between IDF-weighted averaged GloVe embeddings, IDF(x) =
|V| / |{d in D : x in d}| exactly as the paper defines it.
"""
import math
from typing import Dict, Iterable, List

import numpy as np


def char_trigrams(s: str):
    s = s.strip().lower()
    if len(s) < 3:
        return {s} if s else set()
    return set(s[i:i + 3] for i in range(len(s) - 2))


def jaccard_similarity(q: str, c: str) -> float:
    a, b = char_trigrams(q), char_trigrams(c)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_idf(corpus_token_lists: List[List[str]]) -> Dict[str, float]:
    """IDF(x) = |V| / |{d in D : x in d}|, D = corpus of questions/answers/captions (paper Sec 3.3)."""
    n_docs = len(corpus_token_lists)
    doc_freq = {}
    for tokens in corpus_token_lists:
        for w in set(tokens):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    return {w: n_docs / df for w, df in doc_freq.items()}


class EmbeddingSimilarity:
    def __init__(self, embedding_lookup: Dict[str, np.ndarray], idf: Dict[str, float]):
        self.embedding_lookup = embedding_lookup
        self.idf = idf
        self.default_idf = 1.0

    def _weighted_avg(self, tokens: Iterable[str]):
        vecs, weights = [], []
        for w in tokens:
            if w in self.embedding_lookup:
                vecs.append(self.embedding_lookup[w])
                weights.append(self.idf.get(w, self.default_idf))
        if not vecs:
            return None
        weights = np.asarray(weights)
        weights = weights / weights.sum()
        return np.average(np.stack(vecs), axis=0, weights=weights)

    def similarity(self, q_tokens: Iterable[str], c_tokens: Iterable[str]) -> float:
        vq = self._weighted_avg(q_tokens)
        vc = self._weighted_avg(c_tokens)
        if vq is None or vc is None:
            return 0.0
        denom = np.linalg.norm(vq) * np.linalg.norm(vc)
        if denom == 0:
            return 0.0
        return float(np.dot(vq, vc) / denom)


def combined_similarity(q: str, c: str, q_tokens, c_tokens,
                         emb_sim: EmbeddingSimilarity, alpha: float = 0.75) -> float:
    s_s = jaccard_similarity(q, c)
    s_e = emb_sim.similarity(q_tokens, c_tokens)
    return alpha * s_s + (1 - alpha) * s_e
