"""Merges questions (prepare_vqa.py / prepare_visual7w.py) + image features
(extract_image_features.py) + DenseCap candidate captions (run_densecap.py) into the
final manifest.json consumed by src/dataset.py's VQGJsonDataset.

Records without a matched image feature or candidate-caption set are dropped (mirrors
the paper Sec 4.1 note: image-answer pairs with no aligned visual hints/captions are
excluded when the mismatch is due to detector/NLP-tool error).
"""
import argparse
import json
import os


def main(questions_path: str, features_dir: str, candidates_path: str, out_path: str):
    with open(questions_path) as f:
        records = json.load(f)
    with open(candidates_path) as f:
        candidates_by_image = json.load(f)  # keys are strings after JSON round-trip

    manifest = []
    dropped = 0
    for r in records:
        image_id = str(r["image_id"])
        feat_path = os.path.join(features_dir, f"{r['image_id']}.npy")
        if not os.path.exists(feat_path) or image_id not in candidates_by_image:
            dropped += 1
            continue
        manifest.append({
            "image_id": r["image_id"],
            "image_feat_path": feat_path,
            "candidates": candidates_by_image[image_id]["candidates"],
            "question": r["question"],
        })

    with open(out_path, "w") as f:
        json.dump(manifest, f)
    print(f"Wrote {len(manifest)} records to {out_path} ({dropped} dropped: missing feature/captions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.questions, args.features_dir, args.candidates, args.out)
