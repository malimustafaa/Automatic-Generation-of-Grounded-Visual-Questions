"""Inference: generate up to N grounded questions per image (paper Sec 3/4, Fig. 3
sweeps N=1..6).

Per-question generation:
  1. Sample a caption from DenseCap's candidate set C_i, weighted by confidence (Sec 3.1).
  2. Run the question-type selector on that caption; SAMPLE a type from its softmax
     distribution (not argmax) -- generating multiple *diverse* questions per image is
     the paper's whole point, and always taking the argmax type would collapse every
     draw toward the single most likely type. This sampling choice is a gap-filled
     reading of 'samples the most probable question types' (the paper doesn't specify
     argmax vs. sampling).
  3. Build the joint feature map via the encoder + correlation layer.
  4. Autoregressively decode with the LSTM decoder interpolated with the Kneser-Ney
     bigram LM ('Joint decoding', Sec 3.2):
       P(q_t | q_<t) = (1-beta) * P_l(q_t | q_<t) + beta * P_b(q_t | q_{t-1})
     applied from t=1 onward, with the first word fixed to the sampled type (Sec 3.1:
     'we fix the first k words of questions during decoding according to the chosen
     question types'). k=1 here -- the paper never enumerates k per type, so multi-word
     openers (e.g. "how many") are a documented gap; see TYPE_PREFIXES below.
"""
from typing import List

import torch
import torch.nn.functional as F

from .bigram_lm import KneserNeyBigram
from .dataset import tokenize
from .model import GroundedVQGModel
from .question_type_selector import QUESTION_TYPES
from .vocab import END, START, Vocab

TYPE_PREFIXES = {t: [t] for t in QUESTION_TYPES}  # gap-filled: k=1 fixed word per type


@torch.no_grad()
def generate_questions(model: GroundedVQGModel, vocab: Vocab, bigram_lm: KneserNeyBigram,
                        image_feat: torch.Tensor, candidates: List[dict],
                        num_questions: int = 6, beta: float = 0.2, max_len: int = 20,
                        device: str = "cpu") -> List[str]:
    model.eval()
    image_feat = image_feat.to(device).unsqueeze(0)

    conf = torch.tensor([c.get("confidence", 1.0) for c in candidates], dtype=torch.float32)
    prior = conf / conf.sum()

    questions = []
    for _ in range(num_questions):
        cap_idx = int(torch.multinomial(prior, 1).item())
        caption = candidates[cap_idx]["caption"]
        caption_tokens = tokenize(caption)[:max_len]
        caption_ids = torch.tensor([vocab.encode(caption_tokens)], dtype=torch.long, device=device)
        caption_lengths = torch.tensor([max(len(caption_tokens), 1)])

        caption_embeds = model.embedding(caption_ids)
        type_logits = model.type_selector(caption_embeds, caption_lengths)
        type_probs = F.softmax(type_logits, dim=-1).squeeze(0)
        type_idx = int(torch.multinomial(type_probs, 1).item())
        qtype = QUESTION_TYPES[type_idx]

        caption_embedding = model.encoder(image_feat, caption_embeds, caption_lengths)
        joint_feature = model.correlation(caption_embedding, image_feat)

        start_embed = model.embedding(torch.tensor([vocab.word2idx[START]], device=device))
        _, hidden = model.decoder.init_state(joint_feature, start_embed)

        generated = list(TYPE_PREFIXES[qtype])
        prev_word = generated[-1]
        next_input_id = vocab.word2idx.get(generated[-1], vocab.word2idx["<unk>"])

        for _ in range(max_len - len(generated)):
            x_t = model.embedding(torch.tensor([[next_input_id]], device=device))
            logits, hidden = model.decoder.step(x_t, hidden)
            probs_lstm = F.softmax(logits, dim=-1).squeeze(0)

            bigram_probs = torch.tensor(
                [bigram_lm.prob(prev_word, w) for w in vocab.idx2word], dtype=torch.float32, device=device
            )
            combined = (1 - beta) * probs_lstm + beta * bigram_probs
            next_id = int(torch.argmax(combined).item())
            next_word = vocab.idx2word[next_id]

            if next_word == END:
                break
            generated.append(next_word)
            prev_word = next_word
            next_input_id = next_id

        questions.append(" ".join(generated) + " ?")

    return questions
