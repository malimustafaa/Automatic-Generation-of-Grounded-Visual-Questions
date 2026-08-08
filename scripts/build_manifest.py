"""Merges questions (prepare_vqa.py / prepare_visual7w.py) + image features
(extract_image_features.py, ONE consolidated .npz keyed by str(image_id)) + DenseCap
candidate captions (run_densecap.py) into the final manifest.json consumed by
src/dataset.py's VQGJsonDataset.

Records without a matched image feature or candidate-caption set are dropped (mirrors
the paper Sec 4.1 note: image-answer pairs with no aligned visual hints/captions are
excluded when the mismatch is due to detector/NLP-tool error).

manifest.json no longer stores a per-image feature file path (there isn't one anymore --
just image_id, looked up against the consolidated .npz by src/dataset.py at load time).

Carries through each record's "split" field (VQA's/Visual7W's own official train/val
partition, set by prepare_vqa.py/prepare_visual7w.py) so train.py can train only on
the official train portion and hold out val for monitoring, matching paper Sec 4.4
("hyperparameters were tuned on the validation sets") -- previously this distinction
was discarded upstream, training on both merged together with nothing held out.
"""
import argparse
import json

import numpy as np


def main(questions_path: str, features_path: str, candidates_path: str, out_path: str):
    with open(questions_path) as f:
        records = json.load(f)
    with open(candidates_path) as f:
        candidates_by_image = json.load(f)  # keys are strings after JSON round-trip
    features = np.load(features_path)
    feature_keys = set(features.files)

    manifest = []
    dropped = 0
    for r in records:
        image_id = str(r["image_id"])
        if image_id not in feature_keys or image_id not in candidates_by_image:
            dropped += 1
            continue
        manifest.append({
            "image_id": r["image_id"],
            "candidates": candidates_by_image[image_id]["candidates"],
            "question": r["question"],
            "split": r.get("split", "train"),
        })

    with open(out_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote {len(manifest)} records to {out_path} ({dropped} dropped: missing feature/captions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--features_path", required=True, help="consolidated .npz from extract_image_features.py")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.questions, args.features_path, args.candidates, args.out)
