"""End-to-end smoke test with tiny synthetic data -- no real images, no DenseCap, no
GloVe download. Purpose: prove the pipeline (encoder -> correlation -> decoder,
EM-weighted training, bigram-interpolated generation, BLEU eval) is wired correctly
and runs without crashing, before pointing it at real data in Colab.

Run: python -m tests.smoke_test
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.bigram_lm import KneserNeyBigram
from src.dataset import infer_type, tokenize
from src.embeddings import build_embedding_matrix
from src.generate import generate_questions
from src.model import GroundedVQGModel
from src.vocab import PAD, START, Vocab

# --- tiny synthetic dataset: 4 "images", each with a few DenseCap-style candidate
# captions and one gold question. Image features are random vectors (skips VGG-16
# entirely -- we're testing the VQG model, not the frozen feature extractor). ---
SYNTHETIC = [
    {
        "candidates": [
            {"caption": "a red car parked on the street", "confidence": 0.9},
            {"caption": "a man standing next to a car", "confidence": 0.6},
            {"caption": "a tree behind the car", "confidence": 0.3},
        ],
        "question": "what color is the car",
    },
    {
        "candidates": [
            {"caption": "a dog running on the grass", "confidence": 0.8},
            {"caption": "a green field with trees", "confidence": 0.5},
        ],
        "question": "where is the dog",
    },
    {
        "candidates": [
            {"caption": "a woman holding an umbrella", "confidence": 0.85},
            {"caption": "rain falling on the street", "confidence": 0.4},
        ],
        "question": "who is holding the umbrella",
    },
    {
        "candidates": [
            {"caption": "three birds sitting on a wire", "confidence": 0.7},
            {"caption": "a blue sky in the background", "confidence": 0.4},
        ],
        "question": "how many birds are on the wire",
    },
]


def build_vocab():
    token_lists = []
    for rec in SYNTHETIC:
        token_lists.append(tokenize(rec["question"]))
        for c in rec["candidates"]:
            token_lists.append(tokenize(c["caption"]))
    return Vocab(min_count=1).build(token_lists)


def make_batch(vocab, device):
    image_feats, caption_ids_list, decoder_input_list, decoder_target_list = [], [], [], []
    type_targets, candidate_lists, questions = [], [], []

    for rec in SYNTHETIC:
        image_feats.append(torch.randn(300))
        cap = rec["candidates"][0]["caption"]
        cap_tokens = tokenize(cap)
        q_tokens = tokenize(rec["question"])

        caption_ids_list.append(torch.tensor(vocab.encode(cap_tokens), dtype=torch.long))
        decoder_input_list.append(torch.tensor(vocab.encode(q_tokens), dtype=torch.long))
        decoder_target_list.append(
            torch.tensor(vocab.encode(q_tokens) + [vocab.word2idx["<end>"]], dtype=torch.long)
        )
        type_targets.append(infer_type(rec["question"]))
        candidate_lists.append([c["caption"] for c in rec["candidates"]])
        questions.append(rec["question"])

    def pad(seqs, pad_id):
        lengths = torch.tensor([len(s) for s in seqs])
        out = torch.full((len(seqs), lengths.max().item()), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out.to(device), lengths

    caption_ids, caption_lengths = pad(caption_ids_list, vocab.word2idx[PAD])
    decoder_input_ids, decoder_lengths = pad(decoder_input_list, vocab.word2idx[PAD])
    decoder_target_ids, _ = pad(decoder_target_list, vocab.word2idx[PAD])

    return {
        "image_feat": torch.stack(image_feats).to(device),
        "caption_ids": caption_ids,
        "caption_lengths": caption_lengths,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "decoder_lengths": decoder_lengths,
        "type_target": torch.tensor(type_targets, dtype=torch.long).to(device),
        "candidate_captions": candidate_lists,
        "question": questions,
    }


def bleu1(reference: str, hypothesis: str) -> float:
    """Minimal unigram-precision BLEU-1, dependency-free (just to prove the eval
    harness works end-to-end; full BLEU/METEOR/ROUGE-L via pycocoevalcap in eval/)."""
    ref = set(tokenize(reference))
    hyp = tokenize(hypothesis)
    if not hyp:
        return 0.0
    hits = sum(1 for w in hyp if w in ref)
    return hits / len(hyp)


def main():
    device = "cpu"
    vocab = build_vocab()
    print(f"[smoke] vocab size = {len(vocab)}")

    embedding = build_embedding_matrix(vocab, glove_path=None, dim=300)  # no GloVe -> random init
    model = GroundedVQGModel(embedding, vocab_size=len(vocab), type_hidden=64, decoder_hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from src.similarity import EmbeddingSimilarity, build_idf, combined_similarity

    all_tokens = [tokenize(r["question"]) for r in SYNTHETIC] + \
                 [tokenize(c["caption"]) for r in SYNTHETIC for c in r["candidates"]]
    idf = build_idf(all_tokens)
    emb_sim = EmbeddingSimilarity(embedding_lookup={}, idf=idf)  # empty lookup -> sim_e=0, exercises the fallback path

    batch = make_batch(vocab, device)
    start_id, pad_id = vocab.word2idx[START], vocab.word2idx[PAD]

    print("[smoke] running 20 training steps...")
    losses = []
    for step in range(20):
        type_logits, q_logits = model.forward_step(
            batch["image_feat"], batch["caption_ids"], batch["caption_lengths"],
            batch["decoder_input_ids"], batch["decoder_lengths"], start_id,
        )
        type_loss, q_loss = GroundedVQGModel.per_instance_loss(
            type_logits, batch["type_target"], q_logits, batch["decoder_target_ids"],
            batch["decoder_lengths"], pad_id,
        )

        em_weights = []
        for q, cands in zip(batch["question"], batch["candidate_captions"]):
            sims = [combined_similarity(q, c, tokenize(q), tokenize(c), emb_sim, alpha=0.75) for c in cands]
            total = sum(sims)
            em_weights.append(sims[0] / total if total > 0 else 1.0 / len(cands))
        em_weights = torch.tensor(em_weights, dtype=torch.float32)

        loss = (em_weights * (type_loss + q_loss)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[smoke] loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
    assert all(torch.isfinite(torch.tensor(losses))), "non-finite loss encountered"
    assert losses[-1] < losses[0], "loss did not decrease over 20 steps -- training loop is likely broken"

    print("[smoke] running generation...")
    bigram_lm = KneserNeyBigram(discount=0.75).fit([tokenize(r["question"]) for r in SYNTHETIC])

    total_bleu = 0.0
    for i, rec in enumerate(SYNTHETIC):
        image_feat = batch["image_feat"][i].cpu()
        questions = generate_questions(
            model, vocab, bigram_lm, image_feat, rec["candidates"],
            num_questions=3, beta=0.2, max_len=10, device=device,
        )
        assert len(questions) == 3
        assert all(isinstance(q, str) and len(q) > 0 for q in questions)
        print(f"  image {i}: gold={rec['question']!r} generated={questions}")
        total_bleu += max(bleu1(rec["question"], q) for q in questions)

    print(f"[smoke] avg best-of-3 BLEU-1 = {total_bleu / len(SYNTHETIC):.3f} (sanity signal only, not a real eval)")
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
