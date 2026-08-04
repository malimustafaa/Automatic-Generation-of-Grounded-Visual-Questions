"""Downloads and flattens VQA v1 (Antol et al. 2015) into a simple per-question record
list. Paper Sec 4.1: 'VQA-Dataset [Antol et al., 2015]'.

Official VQA v1 download URLs (visualqa.org/vqa_v1_download.html):
  Questions:   https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
               https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
  Annotations: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
               https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip
Images: MS-COCO train2014/val2014 (scripts/download_coco_images.sh) -- VQA v1 does not
bundle images itself, only question/annotation JSON referencing COCO image ids.

Output: data/vqa_questions.json, a list of
  {"question_id": int, "image_id": int, "image_filename": str, "question": str}
"""
import argparse
import json
import os
import zipfile
from urllib.request import urlretrieve

BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
FILES = {
    "train": ("Questions_Train_mscoco.zip", "OpenEnded_mscoco_train2014_questions.json", "train2014"),
    "val": ("Questions_Val_mscoco.zip", "OpenEnded_mscoco_val2014_questions.json", "val2014"),
}


def download_and_extract(zip_name: str, dest_dir: str) -> str:
    zip_path = os.path.join(dest_dir, zip_name)
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_name}...")
        urlretrieve(f"{BASE}/{zip_name}", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    return dest_dir


def coco_filename(image_id: int, split: str) -> str:
    # Standard MS-COCO 2014 filename convention.
    return f"{split}/COCO_{split}_{image_id:012d}.jpg"


def main(dest_dir: str, out_path: str):
    os.makedirs(dest_dir, exist_ok=True)
    records = []
    for split, (zip_name, json_name, coco_split) in FILES.items():
        download_and_extract(zip_name, dest_dir)
        with open(os.path.join(dest_dir, json_name)) as f:
            data = json.load(f)
        for q in data["questions"]:
            records.append({
                "question_id": q["question_id"],
                "image_id": q["image_id"],
                "image_filename": coco_filename(q["image_id"], coco_split),
                "question": q["question"],
            })
        print(f"[{split}] {len(data['questions'])} questions")

    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} total records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest_dir", default="data/vqa_raw")
    parser.add_argument("--out", default="data/vqa_questions.json")
    args = parser.parse_args()
    main(args.dest_dir, args.out)
