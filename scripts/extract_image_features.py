"""Runs the frozen VGG-16 extractor (src/image_features.py) over every image referenced
in a questions JSON (from prepare_vqa.py / prepare_visual7w.py), saving ALL features
into ONE consolidated .npz archive (keyed by str(image_id)) rather than one file per
image.

This used to write one .npy file per image. Combined with src/dataset.py loading a
feature file per training example -- on every batch, across all 128 epochs, not just
once -- that caused a real hang once these files ended up on a Drive mount: the same
"many small files through Drive's FUSE layer" problem already fixed for COCO/Visual
Genome images, just missed here since features get re-read every epoch rather than
read once. A single consolidated file loaded into memory once at dataset construction
time (src/dataset.py) avoids this entirely.

Resumable: an existing output file's already-computed features are loaded and skipped,
and the archive is periodically re-saved (--save_every) so a disconnect partway through
doesn't lose everything -- np.savez rewrites the whole archive each time (not
incremental), so this trades a bit of redundant I/O for crash safety, same tradeoff
made throughout this pipeline for exactly the same reason.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.image_features import VGG16FeatureExtractor

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def open_image_with_retry(img_path: str, attempts: int = 3, delay: float = 1.5):
    """Reading many small files off a Google Drive FUSE mount at scale is prone to
    sporadic OSError: [Errno 5] Input/output error on individual files -- not because
    the file is actually corrupt, just a flaky mount hiccup. A short retry clears most
    of these; if it still fails after `attempts`, the caller skips this one image."""
    last_err = None
    for attempt in range(attempts):
        try:
            return Image.open(img_path).convert("RGB")
        except OSError as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(delay)
    raise last_err


def main(questions_path: str, image_root: str, out_path: str, batch_size: int = 32, save_every: int = 5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(questions_path) as f:
        records = json.load(f)
    unique_images = {r["image_id"]: r["image_filename"] for r in records}
    print(f"{len(unique_images)} unique images to featurize")

    features = {}
    if os.path.exists(out_path):
        existing = np.load(out_path)
        features = {k: existing[k] for k in existing.files}
        print(f"Resuming: {len(features)} features already computed, loaded from {out_path}.")

    model = VGG16FeatureExtractor(out_dim=300).to(device).eval()

    items = [(iid, fn) for iid, fn in unique_images.items() if str(iid) not in features]
    print(f"{len(items)} images remaining to process.")

    since_last_save = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tensors, ids = [], []
        for image_id, filename in batch:
            img_path = os.path.join(image_root, filename)
            try:
                img = open_image_with_retry(img_path)
            except OSError as e:
                print(f"[warn] failed to read {img_path} after retries: {e}")
                continue
            tensors.append(PREPROCESS(img))
            ids.append(image_id)

        if not tensors:
            continue
        with torch.no_grad():
            feats = model(torch.stack(tensors).to(device)).cpu().numpy()
        for image_id, feat in zip(ids, feats):
            features[str(image_id)] = feat.astype(np.float32)
        since_last_save += len(ids)

        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(items)} remaining images done ({len(features)} total in archive)")

        if since_last_save >= save_every:
            np.savez(out_path, **features)
            print(f"  checkpointed {len(features)} features to {out_path}")
            since_last_save = 0

    np.savez(out_path, **features)
    print(f"Done. {len(features)} features written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="output of prepare_vqa.py / prepare_visual7w.py")
    parser.add_argument("--image_root", required=True, help="directory containing train2014/val2014")
    parser.add_argument("--out_path", default="data/image_features.npz")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=5000,
                         help="re-save the consolidated archive every N newly processed images")
    args = parser.parse_args()
    main(args.questions, args.image_root, args.out_path, args.batch_size, args.save_every)
