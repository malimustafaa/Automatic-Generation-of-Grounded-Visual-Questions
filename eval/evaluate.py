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


def precision_scores(references: Dict[str, List[str]], generated: Dict[str, List[str]]) -> Dict[str, float]:
    """references[image_id] = list of gold questions; generated[image_id] = list of
    generated questions. Each generated question is scored against ALL references for
    that image (pycocoevalcap picks the best match internally, matching the paper's
    'highest score among all reference questions')."""
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
    """Swapped roles vs. precision_scores: each *reference* is scored against all
    generated questions for that image -- an estimate of coverage (paper Sec 4.3)."""
    gts, res = {}, {}
    for image_id, refs in references.items():
        gens = generated.get(image_id, [])
        if not refs or not gens:
            continue
        for i, r in enumerate(refs):
            key = f"{image_id}_{i}"
            gts[key] = [r]
            res[key] = gens
    return _score_all(gts, res)


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
