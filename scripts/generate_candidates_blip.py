"""Generates candidate captions for every image referenced in a questions JSON, using
BLIP + the Faster R-CNN detector + geometric enrichment (src/blip_captioner.py,
src/object_detector.py) instead of DenseCap -- see notebooks/colab_train.ipynb's
DenseCap-vs-BLIP discussion for why: DenseCap trained from scratch measured ~0.09 mAP
and repeatedly produced wrong/irrelevant content on real test images, while BLIP's
captions were verified dramatically more accurate on the same images.

Produces the exact same output format as scripts/run_densecap.py:
  {image_id_str: {"image_id": ..., "candidates": [{"caption":..., "confidence":...}]}}
so it's a drop-in --candidates argument for build_manifest.py -- nothing downstream
needs to know which tool produced the captions.

Per image: src/blip_captioner.py's caption_image_batched() (one whole-image caption +
BLIP on the top --max_regions detected boxes, issued as a single batched BLIP call,
not max_regions+1 separate ones) plus the zero-cost detector/relation candidates
(color/size/position + geometric "on"/"near" relations) already used in the notebook
demo. Fewer, higher-quality candidates per image than DenseCap's raw ~20 -- that was
always the point of today's dedup_captions/lexical_bias work, not something this
script tries to undo by matching DenseCap's volume.

Resumable (already-computed image_ids in --out are skipped) and periodically
checkpointed (--save_every) -- BLIP inference is much slower per-image than
DenseCap's single forward pass, so this is a genuinely long run over a full dataset,
and losing partial progress to a Colab disconnect is exactly the failure this project
can't afford to repeat (see run_densecap.py's own ~3-hour-loss docstring). --max_images
bounds the run to a random sample instead of the full dataset -- given the realistic
runtime (print an estimate after the first --save_every batch and decide from there),
a full-dataset run may not be practical in one sitting; a large-enough sample still
gives the retrain meaningfully better signal than DenseCap's captions did.
"""
import argparse
import json
import os
import random
import time

import torch

from src.blip_captioner import caption_image_batched
from src.object_detector import detect_objects_as_candidates, detect_relations_as_candidates


def caption_one_image(image_path: str, device: str, max_regions: int, region_confidence: float) -> list:
    candidates = caption_image_batched(image_path, max_regions=max_regions,
                                        region_confidence_threshold=region_confidence, device=device)
    candidates += detect_objects_as_candidates(image_path, confidence_threshold=0.5, device=device)
    candidates += detect_relations_as_candidates(image_path, confidence_threshold=0.5, device=device)
    return candidates


def main(questions_path: str, image_root: str, out_path: str, max_regions: int,
         region_confidence: float, save_every: int, max_images: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + ("" if device == "cuda" else "  (WARNING: no GPU -- this will be very slow)"))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(questions_path) as f:
        records = json.load(f)
    unique_images = {r["image_id"]: r["image_filename"] for r in records}
    print(f"{len(unique_images)} unique images referenced by {questions_path}")

    merged = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            merged = json.load(f)
        print(f"Resuming: {len(merged)} images already captioned, loaded from {out_path}.")

    items = [(iid, fn) for iid, fn in unique_images.items() if str(iid) not in merged]
    if max_images is not None and len(items) > max_images:
        random.Random(seed).shuffle(items)
        items = items[:max_images]
        print(f"--max_images={max_images}: sampling this many (of {len(unique_images) - len(merged)} "
              f"still-uncaptioned) images this run, seed={seed}.")
    print(f"{len(items)} images to process this run.")

    since_last_save = 0
    t0 = time.time()
    for i, (image_id, filename) in enumerate(items, start=1):
        img_path = os.path.join(image_root, filename)
        try:
            candidates = caption_one_image(img_path, device, max_regions, region_confidence)
        except (OSError, RuntimeError) as e:
            print(f"[warn] failed on {img_path}: {e}")
            continue

        merged[str(image_id)] = {"image_id": image_id, "candidates": candidates}
        since_last_save += 1

        if i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining_this_run = (len(items) - i) / max(rate, 1e-9)
            print(f"  {i}/{len(items)} done ({rate:.2f} img/s, "
                  f"~{remaining_this_run/60:.1f} min left in this run)")

        if since_last_save >= save_every:
            with open(out_path, "w") as f:
                json.dump(merged, f)
            print(f"  checkpointed {len(merged)} total images to {out_path}")
            since_last_save = 0

    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"Done this run. {len(merged)} total images written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="output of prepare_vqa.py / prepare_visual7w.py")
    parser.add_argument("--image_root", required=True, help="directory containing train2014/val2014")
    parser.add_argument("--out", default="data/blip_candidates.json")
    parser.add_argument("--max_regions", type=int, default=4,
                         help="top-N detected regions to caption with BLIP per image, "
                              "in addition to one whole-image caption")
    parser.add_argument("--region_confidence", type=float, default=0.3)
    parser.add_argument("--save_every", type=int, default=200,
                         help="re-save the candidates file every N newly processed images -- lower than "
                              "extract_image_features.py's default since BLIP inference is much slower per-item")
    parser.add_argument("--max_images", type=int, default=None,
                         help="cap this run to a random sample of this many still-uncaptioned images, "
                              "instead of the full remaining dataset -- re-run (resumable) to keep extending "
                              "the sample across sessions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.questions, args.image_root, args.out, args.max_regions,
         args.region_confidence, args.save_every, args.max_images, args.seed)
