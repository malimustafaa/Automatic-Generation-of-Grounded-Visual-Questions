"""Training loop -- paper Sec 3.3 / 4.4.

Implements the approximate-EM objective (Eq. 5-7):
  1. For each (image, question) instance, a caption c_n is drawn from the DenseCap
     candidate set C_i (confidence-weighted, Sec 3.1 -- see src/dataset.py).
  2. The instance is weighted by P(c_n | q, t) (Eq. 7), computed via the combined
     Jaccard-trigram + IDF-weighted-GloVe similarity in src/similarity.py, normalized
     over all candidates in C_i for that image.
  3. The weighted per-instance loss (Eq. 6) is backpropagated with Adam.

Paper-given: Adam optimizer, batch size 64, alpha=0.75 similarity interpolation,
128 epochs (VQA) / 64 epochs (Visual7W).
Gap-filled (not stated by the paper): learning rate, hidden sizes (see model.py /
correlation.py / decoder.py docstrings), vocab cutoff, dropout, gradient clipping,
beta (bigram/LSTM interpolation weight, used only at inference in generate.py).
All gap-filled values live in configs/default.yaml, not hardcoded here.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VQGJsonDataset, make_collate_fn, tokenize
from src.embeddings import build_embedding_matrix, load_glove_lookup
from src.model import GroundedVQGModel
from src.similarity import EmbeddingSimilarity, build_idf, combined_similarity
from src.vocab import PAD, START, Vocab


def compute_em_weights(questions, candidate_lists, emb_sim: EmbeddingSimilarity, alpha: float, cache: dict):
    """Eq. 7: P(c_n | q, t) = s(q, c_n) / sum_{c_j in C} s(q, c_j), restricted to the
    single drawn candidate c_n vs. the full candidate set C for that image.

    `cache` persists across the whole training run (passed in from train(), not
    recreated per call/epoch). combined_similarity(q, c, ...) is a pure function of
    the (q, c) text pair -- it doesn't depend on the model or training step at all --
    but the dataset is fixed across all epochs, so every one of these similarity
    computations was being redone identically all 128 times before this cache existed.
    This was the actual bottleneck (not the model's forward/backward pass, which is
    genuinely small): up to ~1,280 uncached Python-level similarity computations per
    batch. Epoch 1 still pays full cost (nothing cached yet); every epoch after that
    should be dramatically faster as the cache fills in."""
    weights = []
    for q, candidates in zip(questions, candidate_lists):
        sims = []
        for c in candidates:
            key = (q, c)
            if key not in cache:
                cache[key] = combined_similarity(q, c, tokenize(q), tokenize(c), emb_sim, alpha)
            sims.append(cache[key])
        total = sum(sims)
        weights.append(sims[0] / total if total > 0 else 1.0 / len(candidates))
    return torch.tensor(weights, dtype=torch.float32)


def build_vocab_and_idf(manifest_path: str, min_count: int):
    with open(manifest_path) as f:
        records = json.load(f)
    token_lists = []
    for rec in records:
        token_lists.append(tokenize(rec["question"]))
        for c in rec["candidates"]:
            token_lists.append(tokenize(c["caption"]))
    vocab = Vocab(min_count=min_count).build(token_lists)
    idf = build_idf(token_lists)
    return vocab, idf


def train(config_path: str, manifest_path: str, glove_path: str, out_dir: str, epochs: int = None):
    # Without this, the first checkpoint save (after epoch 1, which at VQA's scale
    # could be a genuinely long wait) crashes with FileNotFoundError if out_dir
    # doesn't already exist -- same failure shape as describe.py's lut_path bug:
    # a long run completing successfully and then dying on the save step.
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, idf = build_vocab_and_idf(manifest_path, cfg["vocab_min_count"])
    embedding = build_embedding_matrix(vocab, glove_path, dim=300).to(device)
    glove_lookup = load_glove_lookup(glove_path, set(vocab.word2idx.keys()))
    emb_sim = EmbeddingSimilarity(glove_lookup, idf)

    model = GroundedVQGModel(
        embedding, vocab_size=len(vocab),
        type_hidden=cfg["type_selector_hidden"], decoder_hidden=cfg["decoder_hidden"],
    ).to(device)

    dataset = VQGJsonDataset(manifest_path, vocab, max_len=cfg["max_question_len"])
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=make_collate_fn(vocab.word2idx[PAD]), num_workers=cfg.get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    n_epochs = epochs or cfg["epochs"]
    pad_id, start_id = vocab.word2idx[PAD], vocab.word2idx[START]
    em_similarity_cache = {}  # persists across all epochs -- see compute_em_weights docstring

    for epoch in range(n_epochs):
        t0 = time.time()
        total_loss = 0.0
        # Previously silent for an entire epoch at VQA's scale (hundreds of thousands
        # of records / batch_size 64 = thousands of batches) before the first print --
        # indistinguishable from actually being stuck without a per-batch progress bar.
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{n_epochs}")
        for step, batch in enumerate(pbar, start=1):
            image_feat = batch["image_feat"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            caption_lengths = batch["caption_lengths"]
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_target_ids = batch["decoder_target_ids"].to(device)
            decoder_lengths = batch["decoder_lengths"]
            type_target = batch["type_target"].to(device)

            type_logits, q_logits = model.forward_step(
                image_feat, caption_ids, caption_lengths, decoder_input_ids, decoder_lengths, start_id
            )
            type_loss, q_loss = GroundedVQGModel.per_instance_loss(
                type_logits, type_target, q_logits, decoder_target_ids, decoder_lengths, pad_id
            )

            em_weight = compute_em_weights(
                batch["question"], batch["candidate_captions"], emb_sim, cfg["similarity_alpha"],
                em_similarity_cache,
            ).to(device)

            loss = (em_weight * (type_loss + q_loss)).mean()

            optimizer.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(avg_loss=f"{total_loss / step:.4f}")

        print(f"[epoch {epoch+1}/{n_epochs}] loss={total_loss / max(len(loader), 1):.4f} "
              f"({time.time() - t0:.1f}s)")

        ckpt_path = f"{out_dir}/checkpoint_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "vocab": vocab.idx2word, "cfg": cfg}, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--glove", default="data/glove.840B.300d.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.manifest, args.glove, args.out_dir, args.epochs)
