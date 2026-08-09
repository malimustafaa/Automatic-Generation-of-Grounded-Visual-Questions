"""Evaluation matching paper Sec 4.3/5 exactly: BLEU-1..4, METEOR, ROUGE-L via
pycocoevalcap (the maintained Python-3 fork of tylin/coco-caption, the exact tool the
paper's own footnote points at), computed two ways:

  - precision-style: each generated question scored against its best-matching
    reference (paper: 'computed against the reference question with the highest
    score among all reference questions in the same image').
  - recall/coverage-style: each *reference* question scored against its best-matching
    *generated* question, swept over N=1..6 generated questions per image (paper Fig. 3).

Requires `pip install pycocoevalcap` (needs a Java runtime for METEOR -- present by
default on Colab).
"""
from collections import defaultdict
from typing import Dict, List

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def _score_all(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
    scores = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for i, s in enumerate(bleu_scores):
        scores[f"BLEU-{i+1}"] = s
    scores["METEOR"], _ = Meteor().compute_score(gts, res)
    scores["ROUGE-L"], _ = Rouge().compute_score(gts, res)
    return scores


def _score_all_per_key(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Per-(gts/res)-key scores for each metric, instead of just the aggregate mean --
    needed by recall_scores' max-reduce-per-reference. All three scorers iterate
    gts.keys() in order and return per-key results in that same order (confirmed
    directly against pycocoevalcap's source for Bleu/Meteor/Rouge, and empirically)."""
    keys = list(gts.keys())
    per_key = {}
    _, bleu_scores = Bleu(4).compute_score(gts, res)
    for n in range(4):
        per_key[f"BLEU-{n+1}"] = dict(zip(keys, bleu_scores[n]))
    _, meteor_scores = Meteor().compute_score(gts, res)
    per_key["METEOR"] = dict(zip(keys, meteor_scores))
    _, rouge_scores = Rouge().compute_score(gts, res)
    per_key["ROUGE-L"] = dict(zip(keys, rouge_scores))
    return per_key


def precision_scores(references: Dict[str, List[str]], generated: Dict[str, List[str]]) -> Dict[str, float]:
    """references[image_id] = list of gold questions; generated[image_id] = list of
    generated questions. Each generated question is scored against ALL references for
    that image (pycocoevalcap picks the best match internally, matching the paper's
    'highest score among all reference questions') -- this is exactly pycocoevalcap's
    native N-references-vs-1-hypothesis support, no restructuring needed."""
    gts, res = {}, {}
    for image_id, gens in generated.items():
        refs = references.get(image_id, [])
        if not refs or not gens:
            continue
        for i, g in enumerate(gens):
            key = f"{image_id}_{i}"
            gts[key] = refs
            res[key] = [g]
    return _score_all(gts, res)


def recall_scores(references: Dict[str, List[str]], generated: Dict[str, List[str]]) -> Dict[str, float]:
    """Paper Sec 4.3: for each reference question, compare against EACH generated
    candidate individually and take the best-matching score -- an estimate of coverage/
    diversity ('analogy of recall'). This is the reverse of precision_scores' direction,
    and pycocoevalcap's scorers only support N references vs 1 hypothesis natively, not
    N hypotheses vs 1 reference -- so this builds one (1 ref, 1 hyp) pair per
    (reference, candidate) combination, scores everything in one batched call via
    _score_all_per_key, then max-reduces per reference before averaging across
    references.

    (Previously this passed all N candidates as a single multi-item "hypothesis" list
    directly to _score_all, which crashed BLEU's assert(len(hypo) == 1) the moment
    N > 1 -- i.e. always, since sweep_num_questions calls this with N up to 6.)"""
    gts, res = {}, {}
    pair_keys_by_ref = defaultdict(list)
    for image_id, refs in references.items():
        gens = generated.get(image_id, [])
        if not refs or not gens:
            continue
        for i, r in enumerate(refs):
            ref_key = f"{image_id}_{i}"
            for j, g in enumerate(gens):
                pair_key = f"{ref_key}__{j}"
                gts[pair_key] = [r]
                res[pair_key] = [g]
                pair_keys_by_ref[ref_key].append(pair_key)

    if not gts:
        return {}

    per_key = _score_all_per_key(gts, res)

    scores = {}
    for metric, key_scores in per_key.items():
        best_per_ref = [max(key_scores[pk] for pk in pair_keys) for pair_keys in pair_keys_by_ref.values()]
        scores[metric] = sum(best_per_ref) / len(best_per_ref)
    return scores


def sweep_num_questions(references: Dict[str, List[str]], generated_pool: Dict[str, List[str]],
                         max_n: int = 6) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Reproduces paper Fig. 3: for N in 1..max_n, truncate each image's generated pool
    to its first N questions and recompute both precision and recall scores."""
    results = {}
    for n in range(1, max_n + 1):
        truncated = {k: v[:n] for k, v in generated_pool.items()}
        results[n] = {
            "precision": precision_scores(references, truncated),
            "recall": recall_scores(references, truncated),
        }
    return results


def group_references_by_image(records: List[dict]) -> Dict[str, List[str]]:
    """records: [{"image_id": ..., "question": ...}, ...] -> {image_id: [questions...]}"""
    out = defaultdict(list)
    for r in records:
        out[str(r["image_id"])].append(r["question"])
    return dict(out)
