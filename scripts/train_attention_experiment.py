"""Standalone training loop for the EXPERIMENTAL attention-augmented model
(src/attention_experiment.py). Deliberately NOT integrated into src/train.py -- keeps
the paper-faithful training pipeline (and its checkpoints/config) completely
untouched. Reuses the same data pipeline and EM-weighting objective as src/train.py
(same manifest.json/features.npz, same compute_em_weights/build_vocab_and_idf -- both
generic/model-agnostic, imported from src.train without modifying it), just swaps in
AttentionGroundedVQGModel and writes to a separate --out_dir so nothing here can
collide with the real reproduction's checkpoints.

This is meant as a proof-of-concept comparison, not a full training commitment: run it
for far fewer epochs than the paper-faithful model's 128 (e.g. --epochs 15-20) to see
whether attention noticeably improves caption-grounding before committing real Colab
GPU time to a longer run.
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attention_experiment import AttentionGroundedVQGModel
from src.dataset import VQGJsonDataset, make_collate_fn
from src.embeddings import build_embedding_matrix, load_glove_lookup
from src.similarity import EmbeddingSimilarity
from src.train import build_vocab_and_idf, compute_em_weights
from src.vocab import PAD, START


def run_batch(batch, model, device, pad_id, start_id, emb_sim, alpha, em_cache):
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
    type_loss, q_loss = AttentionGroundedVQGModel.per_instance_loss(
        type_logits, type_target, q_logits, decoder_target_ids, decoder_lengths, pad_id
    )
    em_weight = compute_em_weights(
        batch["question"], batch["candidate_captions"], emb_sim, alpha, em_cache
    ).to(device)
    return (em_weight * (type_loss + q_loss)).mean()


@torch.no_grad()
def run_validation(val_loader, model, device, pad_id, start_id, emb_sim, alpha, em_cache):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        loss = run_batch(batch, model, device, pad_id, start_id, emb_sim, alpha, em_cache)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(config_path: str, manifest_path: str, features_path: str, glove_path: str,
          out_dir: str, epochs: int):
    os.makedirs(out_dir, exist_ok=True)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, idf = build_vocab_and_idf(manifest_path, cfg["vocab_min_count"])
    embedding = build_embedding_matrix(vocab, glove_path, dim=300).to(device)
    glove_lookup = load_glove_lookup(glove_path, set(vocab.word2idx.keys()))
    emb_sim = EmbeddingSimilarity(glove_lookup, idf)

    model = AttentionGroundedVQGModel(
        embedding, vocab_size=len(vocab),
        type_hidden=cfg["type_selector_hidden"], decoder_hidden=cfg["decoder_hidden"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"AttentionGroundedVQGModel parameter count: {n_params:,}")

    features_npz = np.load(features_path)
    shared_features = {k: features_npz[k] for k in features_npz.files}

    train_dataset = VQGJsonDataset(manifest_path, features_path, vocab, max_len=cfg["max_question_len"],
                                    split="train", features=shared_features)
    val_dataset = VQGJsonDataset(manifest_path, features_path, vocab, max_len=cfg["max_question_len"],
                                  split="val", features=shared_features)
    print(f"train: {len(train_dataset)} records, val: {len(val_dataset)} records")

    collate_fn = make_collate_fn(vocab.word2idx[PAD])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    pad_id, start_id = vocab.word2idx[PAD], vocab.word2idx[START]
    em_similarity_cache = {}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[attention-experiment] epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            loss = run_batch(batch, model, device, pad_id, start_id, emb_sim,
                              cfg["similarity_alpha"], em_similarity_cache)

            optimizer.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(avg_loss=f"{total_loss / step:.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = run_validation(val_loader, model, device, pad_id, start_id, emb_sim,
                                   cfg["similarity_alpha"], em_similarity_cache)
        print(f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"gap={val_loss - train_loss:+.4f} ({time.time() - t0:.1f}s)")

        ckpt_path = f"{out_dir}/checkpoint_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "vocab": vocab.idx2word, "cfg": cfg,
                    "train_loss": train_loss, "val_loss": val_loss}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "vocab": vocab.idx2word, "cfg": cfg,
                        "train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1},
                       f"{out_dir}/checkpoint_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True, help="consolidated .npz from extract_image_features.py")
    parser.add_argument("--glove", default="data/glove.840B.300d.txt")
    parser.add_argument("--out_dir", default="checkpoints_attention_experiment")
    parser.add_argument("--epochs", type=int, required=True,
                         help="proof-of-concept run -- use far fewer than the paper-faithful model's 128, "
                              "e.g. 15-20, to see whether attention helps before committing more GPU time")
    args = parser.parse_args()
    train(args.config, args.manifest, args.features, args.glove, args.out_dir, args.epochs)
