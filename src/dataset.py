"""Dataset for real training (Colab). Expects a manifest JSON produced by the
scripts/prepare_*.py pipeline:

[
  {
    "image_id": str,
    "candidates": [{"caption": str, "confidence": float}, ...],  # cached DenseCap output
    "question": str
  },
  ...
]

Image features come from a separate consolidated .npz archive (extract_image_features.py),
loaded ONCE into memory in __init__ rather than one file read per training example --
that used to be one .npy file per image, re-opened from disk (often a Drive mount) on
every single batch across all 128 epochs, which caused a real hang once these files
ended up on Drive (same class of "many small files through Drive's FUSE layer" bug
already fixed for COCO/Visual Genome images).

Question type (Sec 3.3: 'we can directly extract the question type from the question q
by looking at the first few words') is derived here from the question's first token.

Caption sampling follows Sec 3.1's confidence-weighted prior P(c_k|C_i) = softmax(confidence),
which becomes the *proposal* distribution the paper's 'randomly draw a caption each time'
(Sec 3.3) is read against; the EM importance weight (Eq. 7, similarity-based) is computed
separately in src/train.py using src/similarity.py.
"""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .question_type_selector import QUESTION_TYPES
from .vocab import END, Vocab


def infer_type(question: str) -> int:
    first = question.strip().lower().split()[0] if question.strip() else ""
    for i, t in enumerate(QUESTION_TYPES):
        if first == t or first.startswith(t):
            return i
    return 0  # default to "what" -- the paper's dominant type in both datasets


def tokenize(text: str):
    return text.strip().lower().replace("?", " ?").split()


class VQGJsonDataset(Dataset):
    def __init__(self, manifest_path: str, features_path: str, vocab: Vocab, max_len: int = 20):
        with open(manifest_path) as f:
            self.records = json.load(f)
        features_npz = np.load(features_path)
        self.features = {k: features_npz[k] for k in features_npz.files}
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_feat = torch.from_numpy(self.features[str(rec["image_id"])]).float()

        candidates = rec["candidates"]
        conf = np.array([c.get("confidence", 1.0) for c in candidates], dtype=np.float64)
        prior = conf / conf.sum() if conf.sum() > 0 else np.full(len(conf), 1.0 / len(conf))
        chosen_idx = int(np.random.choice(len(candidates), p=prior))
        chosen_caption = candidates[chosen_idx]["caption"]

        caption_tokens = tokenize(chosen_caption)[: self.max_len]
        question_tokens = tokenize(rec["question"])[: self.max_len]

        caption_ids = torch.tensor(self.vocab.encode(caption_tokens), dtype=torch.long)
        decoder_input_ids = torch.tensor(self.vocab.encode(question_tokens), dtype=torch.long)  # w1..wL
        decoder_target_ids = torch.tensor(
            self.vocab.encode(question_tokens) + [self.vocab.word2idx[END]], dtype=torch.long
        )  # w1..wL,<end>

        return {
            "image_feat": image_feat,
            "caption_ids": caption_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_target_ids": decoder_target_ids,
            "decoder_length": len(question_tokens),
            "chosen_caption": chosen_caption,
            "candidate_captions": [c["caption"] for c in candidates],
            "question": rec["question"],
            "type_target": infer_type(rec["question"]),
        }


def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        def pad_stack(seqs):
            lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            maxlen = max(lengths.max().item(), 1)
            out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, : len(s)] = s
            return out, lengths

        image_feat = torch.stack([b["image_feat"] for b in batch])
        caption_ids, caption_lengths = pad_stack([b["caption_ids"] for b in batch])
        decoder_input_ids, decoder_lengths = pad_stack([b["decoder_input_ids"] for b in batch])
        decoder_target_ids, _ = pad_stack([b["decoder_target_ids"] for b in batch])
        type_target = torch.tensor([b["type_target"] for b in batch], dtype=torch.long)

        return {
            "image_feat": image_feat,
            "caption_ids": caption_ids,
            "caption_lengths": caption_lengths,
            "decoder_input_ids": decoder_input_ids,
            "decoder_target_ids": decoder_target_ids,
            "decoder_lengths": decoder_lengths,
            "type_target": type_target,
            "candidate_captions": [b["candidate_captions"] for b in batch],
            "chosen_caption": [b["chosen_caption"] for b in batch],
            "question": [b["question"] for b in batch],
        }

    return collate_fn
