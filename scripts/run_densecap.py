"""Driver for soloist97/densecap-pytorch's describe.py -- the DenseCap substitute
(see README.md 'Fidelity notes' for why the original Torch DenseCap isn't used).

This script does NOT reimplement DenseCap; it assumes densecap-pytorch is already
cloned and its pretrained checkpoint downloaded (see README for the exact clone/setup
steps -- that repo's own README/OneDrive link, not something we can automate here).

It calls describe.py once per image directory, then reshapes its result.json
(one entry per image, list of {box, score, cap}) into this project's candidate format:
  {"image_id": ..., "candidates": [{"caption": ..., "confidence": ...}, ...]}
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
    only resolves correctly if describe.py's cwd happens to be densecap_repo, which it
    isn't when launched as a subprocess from here. This bit us for real: a full
    82,783-image, ~3-hour run completed successfully end-to-end and then crashed on
    the very last line (saving results, which needs lut_path) because of exactly this,
    losing the entire run since describe.py holds everything in memory and only writes
    to disk once, at the end -- there's no incremental/partial save to fall back on."""
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
        # (`disable=not console_args.verbose`) and nothing prints until the whole run
        # finishes -- describe.py also only writes result.json once, at the very end,
        # after ALL images are processed (no incremental output either way), so
        # --verbose's progress bar is the only way to see this step moving at all.
    ]
    # Belt-and-suspenders on top of the explicit --lut_path above: run with cwd set to
    # densecap_repo, matching how upstream itself expects to be invoked (their own docs
    # assume you've cd'd into the repo first, same as cells 6b/6c already do for
    # preprocess.py/train.py) -- in case describe.py has any OTHER undiscovered
    # relative-path assumption, this is cheap insurance against a repeat of the same
    # multi-hour-run-lost-at-the-last-line failure.
    subprocess.run(cmd, check=True, cwd=densecap_repo)


def reshape_to_candidates(describe_result_path: str, out_path: str, image_id_lookup: dict):
    """image_id_lookup: image_filename -> image_id, so downstream files are keyed by
    the same image_id used in the questions/manifest JSON."""
    with open(describe_result_path) as f:
        raw = json.load(f)  # {image_path: [{"box":..., "score":..., "cap":...}, ...]}

    out = {}
    for image_path, dets in raw.items():
        filename = os.path.basename(image_path)
        image_id = image_id_lookup.get(filename)
        if image_id is None:
            continue
        out[image_id] = {
            "image_id": image_id,
            "candidates": [{"caption": d["cap"], "confidence": d["score"]} for d in dets],
        }

    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} images' candidate captions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--densecap_repo", required=True, help="path to cloned soloist97/densecap-pytorch")
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--result_dir", default="data/densecap_raw")
    parser.add_argument("--questions", required=True, help="prepare_vqa.py/prepare_visual7w.py output, for the filename->image_id lookup")
    parser.add_argument("--out", default="data/densecap_candidates.json")
    parser.add_argument("--box_per_img", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    run_describe(args.densecap_repo, args.config_json, args.checkpoint,
                 args.img_dir, args.result_dir, args.box_per_img)

    with open(args.questions) as f:
        records = json.load(f)
    image_id_lookup = {os.path.basename(r["image_filename"]): r["image_id"] for r in records}

    reshape_to_candidates(os.path.join(args.result_dir, "result.json"), args.out, image_id_lookup)
