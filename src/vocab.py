"""Vocabulary for GroundedVQG.

Frequency cutoff (min_count) is a paper gap -- the paper never states its vocabulary
construction rule, so this is a documented, configurable default (see configs/default.yaml).
"""
import json
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD, START, END, UNK]


class Vocab:
    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = []

    def build(self, token_sequences):
        counts = Counter(tok for seq in token_sequences for tok in seq)
        words = [w for w, c in counts.items() if c >= self.min_count]
        self.idx2word = list(SPECIAL_TOKENS) + sorted(words)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    def encode(self, tokens, add_start_end=False):
        ids = [self.word2idx.get(t, self.word2idx[UNK]) for t in tokens]
        if add_start_end:
            ids = [self.word2idx[START]] + ids + [self.word2idx[END]]
        return ids

    def decode(self, ids, stop_at_end=True):
        words = []
        for i in ids:
            w = self.idx2word[i]
            if stop_at_end and w == END:
                break
            if w in (PAD, START):
                continue
            words.append(w)
        return words

    def __len__(self):
        return len(self.idx2word)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.idx2word, f)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path) as f:
            v.idx2word = json.load(f)
        v.word2idx = {w: i for i, w in enumerate(v.idx2word)}
        return v
