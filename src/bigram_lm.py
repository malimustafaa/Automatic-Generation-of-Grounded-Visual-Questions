"""Kneser-Ney smoothed bigram language model -- paper Sec 3.2, 'Joint decoding':

  P(w_i | w_{i-1}) = max(count(w_{i-1}, w_i) - d, 0) / count(w_i-1)
                     + lambda(w_{i-1}) * P_KN(w_i)

d = 0.75 is given directly by the paper ('fixed to 0.75 to avoid overfitting').
lambda(w_{i-1}) is the standard KN normalizing constant: (d * |types following w_{i-1}|) / count(w_{i-1}).

At inference (src/generate.py), this is interpolated with the LSTM decoder's own
distribution from t=1 onward:

  P(q_t | q_<t) = (1 - beta) * P_l(q_t | q_<t) + beta * P_b(q_t | q_{t-1})

beta is never given a numeric value by the paper (only defined as beta in [0,1]) --
gap-filled, see configs/default.yaml.
"""
from collections import Counter, defaultdict
from typing import List


class KneserNeyBigram:
    def __init__(self, discount: float = 0.75):
        self.d = discount
        self.bigram_counts = Counter()
        self.unigram_counts = Counter()
        self.continuation_counts = defaultdict(set)  # w -> distinct words preceding w
        self.followers = defaultdict(set)             # w_prev -> distinct words following w_prev
        self.total_bigram_types = 1
        self._prob_vector_cache = {}  # prev_word -> [prob(prev_word, w) for w in vocab], see prob_vector()

    def fit(self, token_sequences: List[List[str]]) -> "KneserNeyBigram":
        for seq in token_sequences:
            for w in seq:
                self.unigram_counts[w] += 1
            for w_prev, w in zip(seq[:-1], seq[1:]):
                self.bigram_counts[(w_prev, w)] += 1
                self.continuation_counts[w].add(w_prev)
                self.followers[w_prev].add(w)
        self.total_bigram_types = max(len(self.bigram_counts), 1)
        return self

    def p_continuation(self, w: str) -> float:
        return max(len(self.continuation_counts.get(w, ())), 1) / self.total_bigram_types

    def prob(self, w_prev: str, w: str) -> float:
        count_prev = self.unigram_counts.get(w_prev, 0)
        if count_prev == 0:
            return self.p_continuation(w)
        count_bigram = self.bigram_counts.get((w_prev, w), 0)
        discounted = max(count_bigram - self.d, 0.0) / count_prev
        n_types_after_prev = len(self.followers.get(w_prev, ()))
        lam = (self.d * n_types_after_prev) / count_prev
        return discounted + lam * self.p_continuation(w)

    def prob_vector(self, prev_word: str, vocab_words: List[str]) -> List[float]:
        """Full P(w | prev_word) over vocab_words, in that order -- cached per
        prev_word, since generate.py's decode loop calls this for the SAME prev_word
        an enormous number of times across a real evaluation run (thousands of images
        x questions x decode steps, mostly sharing a small set of common prev_words
        like "the"/"is"/"a"). Without this, every decode step re-ran prob() once per
        vocabulary word from scratch -- at real eval scale (2000+ images) this was
        billions of redundant Python-level calls, not just slow but potentially
        multi-hour slow, for a step that should be near-instant."""
        if prev_word not in self._prob_vector_cache:
            self._prob_vector_cache[prev_word] = [self.prob(prev_word, w) for w in vocab_words]
        return self._prob_vector_cache[prev_word]
