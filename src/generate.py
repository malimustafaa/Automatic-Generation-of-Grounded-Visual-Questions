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

Decoding search strategy (which word(s) to actually commit to at each step) is not
specified by the paper at all -- another gap. generate_questions()'s `beam_width`
argument defaults to 1, which is exactly the original argmax/greedy behavior (kept as
the default so nothing already validated against Fig. 3 changes unless a caller opts
in); beam_width > 1 uses proper beam search (see _beam_search_decode below), which in
practice produces more coherent completions than greedy on captions the model has weak
associations for.

Two more opt-in, inference-only knobs (both default off, same "nothing changes for
Fig. 3 unless a caller opts in" reasoning as beam_width): `dedup_captions` collapses
DenseCap's often-redundant duplicate-text candidates (see _dedup_captions_by_text)
before the confidence-weighted caption sampling; `distinct_types` masks out question
types already used earlier for the same image before sampling the next one, since the
type-selector's distribution is often skewed enough (~85-90% "what" in practice) that
independent sampling collapses most/all draws onto the same type.

A third opt-in knob, also defaulting off: `lexical_bias` (see _build_lexical_mask) adds
a fixed bonus, in LOG-space, to vocab words that literally appear in the sampled
caption at every decode step, equivalent to multiplying their probability by
exp(lexical_bias) before renormalizing. Unlike attention (src/attention_experiment.py),
which gives the decoder the *ability* to look back at the caption, this doesn't depend
on the model having learned to use that ability -- tested empirically, attention alone
didn't reliably stop the decoder from generating nouns absent from every candidate
caption. Log-space (not a multiplicative boost in probability space, which an earlier
version of this used) matters because a real trained checkpoint's wrong predictions
can be extremely confident (>99%) -- overriding that needs a boost that scales with how
confident the wrong prediction is, which only an exponential (log-additive) bonus does;
a linear-in-probability boost of the same nominal size does essentially nothing against
a near-saturated softmax. Values in the 5-20 range are reasonable starting points
(each +1 is another ~2.7x multiplier); how large a value is actually needed depends on
how dominant the checkpoint's own wrong prediction is for a given image.

`no_repeat_content_words` (default off) blocks a content word from being generated
twice within the same question. Needed because lexical_bias/object_suppression_bias
apply the SAME flat bonus/penalty at every decode step with no memory of what's
already been generated -- confirmed empirically to produce degenerate loops like
"where is the white white white white ... ?" once one word wins a position strongly
enough, since nothing otherwise discourages it from winning every position after that
too. Stopwords (LEXICAL_BIAS_STOPWORDS) are exempt, since legitimately repeating "the"
or "is" within a question is normal.
"""
from typing import List

import torch
import torch.nn.functional as F

from .bigram_lm import KneserNeyBigram
from .dataset import tokenize
from .model import GroundedVQGModel
from .question_type_selector import QUESTION_TYPES
from .vocab import END, PAD, START, UNK, Vocab

TYPE_PREFIXES = {t: [t] for t in QUESTION_TYPES}  # gap-filled: k=1 fixed word per type

# Excluded from the lexical bias (see _build_lexical_mask) -- these are already
# high-probability under the model/bigram LM regardless of caption content, so
# boosting them further wouldn't help; the bias is meant to concentrate on the
# informative nouns/adjectives actually worth grounding the question in.
LEXICAL_BIAS_STOPWORDS = {"a", "an", "the", "is", "of", "on", "in", "at", "to", "and",
                           "with", "this", "that", "it", "are", "was", "were", "be",
                           "been", "being", "?"}


def _build_lexical_mask(caption_tokens: List[str], vocab: Vocab, device: str) -> torch.Tensor:
    """1.0 at vocab positions for each content word (not in LEXICAL_BIAS_STOPWORDS)
    that appears in this specific caption, 0.0 elsewhere. Used by generate_questions'
    `lexical_bias` to boost words the sampled caption actually mentions at every
    decode step -- a hard nudge that doesn't depend on the model having *learned* the
    association, unlike attention (see src/attention_experiment.py), which gives the
    decoder the *ability* to use the caption but, tested empirically, still doesn't
    reliably make it do so on an undertrained checkpoint."""
    mask = torch.zeros(len(vocab.idx2word), device=device)
    for w in caption_tokens:
        if w in LEXICAL_BIAS_STOPWORDS:
            continue
        idx = vocab.word2idx.get(w)
        if idx is not None:
            mask[idx] = 1.0
    return mask


def _dedup_captions_by_text(candidates: List[dict]) -> List[dict]:
    """Collapses candidates whose caption text is identical (after case/whitespace
    normalization) down to one entry -- the highest-confidence copy. DenseCap often
    proposes several overlapping boxes for the same object and describes them with the
    exact same caption ("the hand of a person" showing up 6 times among 20 candidates
    for a single image, in practice); left as-is, confidence-weighted sampling just
    reproduces that redundancy instead of exploring the image's genuinely distinct
    content. Only catches exact (normalized) duplicates, not near-duplicates with
    different wording -- a deliberately narrow, low-risk first pass."""
    best_by_text = {}
    for c in candidates:
        key = c["caption"].strip().lower()
        if key not in best_by_text or c.get("confidence", 1.0) > best_by_text[key].get("confidence", 1.0):
            best_by_text[key] = c
    return list(best_by_text.values())


def _beam_search_decode(model: GroundedVQGModel, vocab: Vocab, bigram_lm: KneserNeyBigram,
                         hidden, first_word: str, beam_width: int, max_len: int,
                         beta: float, device: str, bias_vector: torch.Tensor = None,
                         no_repeat_content_words: bool = False) -> List[str]:
    """Beam search over the same interpolated LSTM+bigram distribution the old greedy
    loop used (paper Sec 3.2's joint-decoding formula), maintaining `beam_width`
    candidate sequences instead of committing to a single argmax word at each step.
    The paper doesn't specify a decoding strategy at all, so this is a gap-filled,
    inference-only choice -- beam_width=1 reduces exactly to the previous argmax/greedy
    behavior (single beam, top-1 == argmax), so it's opt-in rather than a silent change
    to anything already validated against Fig. 3.

    Each beam tracks its own LSTM hidden state (they diverge as soon as two beams pick
    different words) and a cumulative LOG-probability, since summing logs (rather than
    multiplying raw probabilities) avoids underflow over up to max_len steps. A beam
    that emits <end> is frozen into `finished` and competes against still-growing beams
    via a length-normalized score (cum_log_prob / length) at every subsequent pruning
    step, so a short low-quality completion can't win purely by being short, and a
    finished beam isn't force-extended past its natural end.

    Runs one beam at a time through the LSTM (not batched across beams) -- beam_width
    is small (single digits) and this function already operates at image batch size 1,
    so the extra clarity is worth more than the modest speedup from batching.
    """
    active = [([first_word], first_word, hidden, 0.0)]
    finished = []

    # Always masked out, unconditionally -- <pad>/<start>/<unk> are internal
    # bookkeeping tokens, never a legitimate word in a generated question. An
    # undertrained checkpoint can otherwise pick <unk> (or, rarely, <pad>/<start>) as
    # its argmax choice for a word it has weak confidence in, leaking it into the
    # visible output text (seen repeatedly in practice, e.g. "what is <start> <start>
    # mouse <start> ?"). Beyond just looking wrong, this crashed a real Fig.
    # 3-style eval run: pycocoevalcap's METEOR scorer talks to a Java subprocess over
    # a line-based text protocol, and a generated question containing "<...>" tokens
    # desynced it, corrupting every subsequent score in that batch. <end> is
    # deliberately NOT masked here -- it's the legitimate termination signal, handled
    # separately below (a beam that emits it moves to `finished` rather than being
    # treated as a visible word at all).
    special_token_ids = [vocab.word2idx[t] for t in (PAD, START, UNK) if t in vocab.word2idx]

    steps_remaining = max_len - len(active[0][0])
    for _ in range(steps_remaining):
        if not active:
            break
        candidates = []
        for tokens, prev_word, hid, score in active:
            last_id = vocab.word2idx.get(tokens[-1], vocab.word2idx["<unk>"])
            x_t = model.embedding(torch.tensor([[last_id]], device=device))
            logits, new_hidden = model.decoder.step(x_t, hid)
            probs_lstm = F.softmax(logits, dim=-1).squeeze(0)

            bigram_probs = torch.tensor(
                bigram_lm.prob_vector(prev_word, vocab.idx2word), dtype=torch.float32, device=device
            )
            combined = (1 - beta) * probs_lstm + beta * bigram_probs
            combined = combined.clamp(min=1e-12)
            log_probs = torch.log(combined)
            if special_token_ids:
                log_probs = log_probs.clone()
                log_probs[special_token_ids] = float("-inf")
            if bias_vector is not None:
                # Additive in LOG-space (equivalent to a logit bonus), not multiplicative
                # in probability space -- a fixed additive log-bonus is a *multiplicative*
                # factor of exp(bias) on the underlying probability, which scales with how
                # confident the base prediction is. A merely-multiplicative probability
                # boost (combined *= (1 + bias)) turned out to be far too weak in
                # practice: if the model assigns 95% to a wrong word and 0.01% to a
                # caption word, even a 16x boost only reaches ~0.16%, nowhere close to
                # overtaking 95%. exp(bias) grows fast enough to actually win that
                # competition with a much smaller, more sane-looking number. bias_vector
                # can carry both a positive component (lexical_bias, for caption words)
                # and a negative one (object_suppression_bias, for undetected COCO-class
                # words) added together by the caller -- see generate_questions.
                log_probs = log_probs + bias_vector
                log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)

            if no_repeat_content_words:
                # bias_vector applies the SAME flat bonus at every single step,
                # regardless of what's already been generated -- with no memory of the
                # sequence so far, a word that wins one position tends to keep winning
                # every position after it too (nothing in the base decoder or bigram LM
                # strongly discourages self-repetition), producing degenerate loops like
                # "white white white white..." in practice. This blocks a content word
                # (not stopwords -- "the color of the car" legitimately repeats "the")
                # from being generated twice within the same question.
                used = {w for w in tokens if w not in LEXICAL_BIAS_STOPWORDS}
                repeat_ids = [vocab.word2idx[w] for w in used if w in vocab.word2idx]
                if repeat_ids:
                    log_probs = log_probs.clone()
                    log_probs[repeat_ids] = float("-inf")

            k = min(beam_width, log_probs.numel())
            topk_lp, topk_ids = torch.topk(log_probs, k)
            for lp, idx in zip(topk_lp.tolist(), topk_ids.tolist()):
                word = vocab.idx2word[idx]
                candidates.append((tokens + [word], word, new_hidden, score + lp))

        candidates.sort(key=lambda c: c[3] / len(c[0]), reverse=True)
        top = candidates[:beam_width]
        active = [c for c in top if c[1] != END]
        finished.extend(c for c in top if c[1] == END)

    pool = finished if finished else active
    best_tokens, best_word, _, _ = max(pool, key=lambda c: c[3] / len(c[0]))
    if best_word == END:
        best_tokens = best_tokens[:-1]
    return best_tokens


@torch.no_grad()
def generate_questions(model: GroundedVQGModel, vocab: Vocab, bigram_lm: KneserNeyBigram,
                        image_feat: torch.Tensor, candidates: List[dict],
                        num_questions: int = 6, beta: float = 0.2, max_len: int = 20,
                        beam_width: int = 1, dedup_captions: bool = False,
                        distinct_types: bool = False, lexical_bias: float = 0.0,
                        object_suppression_bias: float = 0.0, detected_objects=None,
                        no_repeat_content_words: bool = False,
                        device: str = "cpu") -> List[str]:
    """`object_suppression_bias` + `detected_objects` (a set of COCO class names from
    src/object_detector.py's detect_objects()) subtract a LOG-space penalty from vocab
    words that are COCO class names NOT confirmed present in the image by that
    detector -- directly countering the observed failure mode where the decoder
    confidently generates a common COCO object word (bird, mouse, cake, vase,
    scissors, banana, ...) regardless of caption or image content. Composes with
    lexical_bias in the same bias_vector: a caption word that's also an unconfirmed
    COCO class gets both the positive and negative contribution, so the two signals
    can reinforce or partially cancel depending on whether they agree."""
    model.eval()
    image_feat = image_feat.to(device).unsqueeze(0)

    if dedup_captions:
        candidates = _dedup_captions_by_text(candidates)

    suppression_mask = None
    if object_suppression_bias > 0 and detected_objects is not None:
        from .object_detector import build_object_suppression_mask
        suppression_mask = build_object_suppression_mask(detected_objects, vocab, device)

    conf = torch.tensor([c.get("confidence", 1.0) for c in candidates], dtype=torch.float32)
    prior = conf / conf.sum()

    questions = []
    used_types = set()
    for _ in range(num_questions):
        cap_idx = int(torch.multinomial(prior, 1).item())
        caption = candidates[cap_idx]["caption"]
        caption_tokens = tokenize(caption)[:max_len]
        caption_ids = torch.tensor([vocab.encode(caption_tokens)], dtype=torch.long, device=device)
        caption_lengths = torch.tensor([max(len(caption_tokens), 1)])

        caption_embeds = model.embedding(caption_ids)
        type_logits = model.type_selector(caption_embeds, caption_lengths)
        type_probs = F.softmax(type_logits, dim=-1).squeeze(0)

        if distinct_types:
            # Sampling N independent draws from a type distribution this skewed
            # (~85-90% "what" in practice) tends to just pick "what" every time --
            # masking out types already used among this image's earlier draws (and
            # renormalizing over what's left) forces the N questions to actually cover
            # different types, while still respecting the per-caption type_probs
            # ordering among whatever's still available. Cycles (resets used_types)
            # once all 6 types are exhausted, so num_questions > 6 still works.
            if len(used_types) >= len(QUESTION_TYPES):
                used_types = set()
            mask = torch.tensor([0.0 if t in used_types else 1.0 for t in QUESTION_TYPES])
            masked_probs = type_probs * mask
            if masked_probs.sum() > 0:
                type_idx = int(torch.multinomial(masked_probs / masked_probs.sum(), 1).item())
            else:
                type_idx = int(torch.multinomial(type_probs, 1).item())
        else:
            type_idx = int(torch.multinomial(type_probs, 1).item())
        qtype = QUESTION_TYPES[type_idx]
        used_types.add(qtype)

        caption_embedding = model.encoder(image_feat, caption_embeds, caption_lengths)
        joint_feature = model.correlation(caption_embedding, image_feat)

        start_embed = model.embedding(torch.tensor([vocab.word2idx[START]], device=device))
        _, hidden = model.decoder.init_state(joint_feature, start_embed)

        bias_vector = None
        if lexical_bias > 0:
            bias_vector = lexical_bias * _build_lexical_mask(caption_tokens, vocab, device)
        if suppression_mask is not None:
            penalty = object_suppression_bias * suppression_mask
            bias_vector = (bias_vector - penalty) if bias_vector is not None else -penalty

        first_word = TYPE_PREFIXES[qtype][0]  # k=1 fixed opener, forced regardless of beam_width
        generated = _beam_search_decode(model, vocab, bigram_lm, hidden, first_word,
                                         beam_width, max_len, beta, device, bias_vector,
                                         no_repeat_content_words)

        # tokenize() treats "?" as its own token, so a well-trained decoder learns to
        # generate it naturally as the last word before <end> -- appending another one
        # unconditionally produced "...bedspread ? ?" in practice. Only add it if the
        # model didn't already.
        text = " ".join(generated)
        questions.append(text if generated[-1] == "?" else text + " ?")

    return questions
