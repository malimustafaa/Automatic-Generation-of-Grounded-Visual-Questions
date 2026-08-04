"""Runs the frozen VGG-16 extractor (src/image_features.py) over every image referenced
in a questions JSON (from prepare_vqa.py / prepare_visual7w.py), saving one .npy
(300-d) feature file per unique image.
"""
import argparse
import json
import os
import sys

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


def main(questions_path: str, image_root: str, out_dir: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    with open(questions_path) as f:
        records = json.load(f)
    unique_images = {r["image_id"]: r["image_filename"] for r in records}
    print(f"{len(unique_images)} unique images to featurize")

    model = VGG16FeatureExtractor(out_dim=300).to(device).eval()

    items = list(unique_images.items())
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tensors, ids = [], []
        for image_id, filename in batch:
            out_path = os.path.join(out_dir, f"{image_id}.npy")
            if os.path.exists(out_path):
                continue
            img_path = os.path.join(image_root, filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"[warn] missing image: {img_path}")
                continue
            tensors.append(PREPROCESS(img))
            ids.append(image_id)

        if not tensors:
            continue
        with torch.no_grad():
            feats = model(torch.stack(tensors).to(device)).cpu().numpy()
        for image_id, feat in zip(ids, feats):
            np.save(os.path.join(out_dir, f"{image_id}.npy"), feat.astype(np.float32))

        if i % (batch_size * 20) == 0:
            print(f"  {i}/{len(items)} images done")

    print(f"Done. Features written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="output of prepare_vqa.py / prepare_visual7w.py")
    parser.add_argument("--image_root", required=True, help="directory containing train2014/val2014")
    parser.add_argument("--out_dir", default="data/image_features")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.questions, args.image_root, args.out_dir, args.batch_size)
