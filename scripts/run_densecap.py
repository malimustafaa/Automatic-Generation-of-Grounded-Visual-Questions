"""Driver for soloist97/densecap-pytorch's describe.py -- the DenseCap substitute
(see README.md 'Fidelity notes' for why the original Torch DenseCap isn't used).

This script does NOT reimplement DenseCap; it assumes densecap-pytorch is already
cloned and its pretrained checkpoint downloaded/trained (see README).

Runs describe.py once per image directory given via --img_dirs (VQA v1 references
images from BOTH COCO train2014 and val2014 -- pointing this at only one split means
any question whose image lives in the other gets silently dropped later by
build_manifest.py, which is exactly what happened before this was split-aware), each
into its own subfolder of --result_dir so one split's run never clobbers another's,
then merges all of them into one candidate-caption file:
  {"image_id": ..., "candidates": [{"caption": ..., "confidence": ...}, ...]}

Each split's result.json is checked for existence before (re-)running describe.py --
these runs take hours, so an already-completed split is never redone just because
another split still needs processing or previously failed. This does NOT protect
against a crash *mid-split* (describe.py itself holds all results for a split in
memory and only writes once, at the end -- see run_describe()'s docstring); the
protection here is at the split level, not finer-grained than that.
"""
import argparse
import json
import os
import subprocess


def run_describe(densecap_repo: str, config_json: str, checkpoint: str,
                  img_dir: str, result_dir: str, box_per_img: int = 20):
    """box_per_img=20 (not describe.py's default of 100): the paper draws a handful of
    candidate captions per image (Sec 3.1's confidence-weighted sampling), not the full
    100-box firehose; kept low here to bound the size of the per-image candidate set
    that src/similarity.py has to score against at training time. Gap-filled, tune freely.

    lut_path is passed explicitly (absolute path) rather than relying on describe.py's
    own default ('./data/VG-regions-dicts-lite.pkl', a RELATIVE path) -- that default
    only resolves correctly if describe.py's cwd happens to be densecap_repo. This bit
    us for real once already: a full 82,783-image, ~3-hour run completed successfully
    end-to-end and then crashed on the very last line (saving results, which needs
    lut_path) because of exactly this, losing the entire run since describe.py holds
    everything in memory and only writes to disk once, at the end -- there is no
    incremental/partial save within a single describe.py run to fall back on. Also
    runs with cwd=densecap_repo as further insurance against any other undiscovered
    relative-path assumption in their code."""
    lut_path = os.path.join(densecap_repo, "data", "VG-regions-dicts-lite.pkl")
    cmd = [
        "python", os.path.join(densecap_repo, "describe.py"),
        "--config_json", config_json,
        "--model_checkpoint", checkpoint,
        "--img_path", img_dir,
        "--result_dir", result_dir,
        "--box_per_img", str(box_per_img),
        "--lut_path", lut_path,
        "--verbose",  # without this, describe.py's tqdm progress bar is disabled entirely
        # and nothing prints until the whole run finishes.
    ]
    subprocess.run(cmd, check=True, cwd=densecap_repo)


def merge_result_json(describe_result_path: str, image_id_lookup: dict, merged: dict):
    """Merges one split's result.json entries into the running `merged` dict in place."""
    with open(describe_result_path) as f:
        raw = json.load(f)  # {image_path: [{"box":..., "score":..., "cap":...}, ...]}

    for image_path, dets in raw.items():
        filename = os.path.basename(image_path)
        image_id = image_id_lookup.get(filename)
        if image_id is None:
            continue
        merged[image_id] = {
            "image_id": image_id,
            "candidates": [{"caption": d["cap"], "confidence": d["score"]} for d in dets],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--densecap_repo", required=True, help="path to cloned soloist97/densecap-pytorch")
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--img_dirs", required=True, nargs="+",
        help="one or more image directories to process, e.g. .../train2014 .../val2014 -- "
             "VQA v1 references images from both COCO splits, so both need to be listed "
             "or questions pointing at whichever split is omitted get silently dropped "
             "later by build_manifest.py",
    )
    parser.add_argument("--result_dir", default="data/densecap_raw")
    parser.add_argument("--questions", required=True, help="prepare_vqa.py/prepare_visual7w.py output, for the filename->image_id lookup")
    parser.add_argument("--out", default="data/densecap_candidates.json")
    parser.add_argument("--box_per_img", type=int, default=20)
    args = parser.parse_args()

    with open(args.questions) as f:
        records = json.load(f)
    image_id_lookup = {os.path.basename(r["image_filename"]): r["image_id"] for r in records}

    merged = {}
    for img_dir in args.img_dirs:
        split_name = os.path.basename(os.path.normpath(img_dir))
        split_result_dir = os.path.join(args.result_dir, split_name)
        result_json_path = os.path.join(split_result_dir, "result.json")

        if os.path.exists(result_json_path):
            print(f"[{split_name}] result.json already exists, skipping describe.py, reusing it.")
        else:
            os.makedirs(split_result_dir, exist_ok=True)
            print(f"[{split_name}] running describe.py over {img_dir} ...")
            run_describe(args.densecap_repo, args.config_json, args.checkpoint,
                         img_dir, split_result_dir, args.box_per_img)

        merge_result_json(result_json_path, image_id_lookup, merged)
        print(f"[{split_name}] merged, running total: {len(merged)} images")

    with open(args.out, "w") as f:
        json.dump(merged, f)
    print(f"Wrote {len(merged)} images' candidate captions to {args.out}")
