"""Downloads and flattens Visual7W (Zhu et al. 2015) into a simple per-question record
list. Paper Sec 4.1: 'Visual7W [Zhu et al., 2015]'.

Source: http://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip
(this is the exact URL used by yukezhu/visual7w-toolkit's own download_dataset.sh).
Images referenced by Visual7W are MS-COCO images (scripts/download_coco_images.sh).

Output: data/visual7w_questions.json, a list of
  {"question_id": ..., "image_id": ..., "image_filename": str, "question": str, "split": "train"|"val"|"test"}

"split" is Visual7W's own official partition, embedded per-image in the dataset's own
schema -- train.py filters on this the same way as prepare_vqa.py's split field.
"""
import argparse
import json
import os
import zipfile
from urllib.request import urlretrieve

URL = "http://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"


def main(dest_dir: str, coco_dir: str, out_path: str):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "dataset_v7w_telling.zip")
    if not os.path.exists(zip_path):
        print("Downloading Visual7W (telling)...")
        urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    with open(os.path.join(dest_dir, "dataset_v7w_telling.json")) as f:
        data = json.load(f)

    records = []
    for image in data["images"]:
        # Visual7W stores COCO filenames directly (or COCO ids for COCO-sourced images);
        # 'filename' is present for the majority of the dataset per the toolkit's schema.
        filename = image.get("filename")
        split = image.get("split", "train")
        for qa in image.get("qa_pairs", []):
            records.append({
                "question_id": qa["qa_id"],
                "image_id": image["image_id"],
                "image_filename": filename,
                "question": qa["question"],
                "split": split,
            })

    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} total records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest_dir", default="data/visual7w_raw")
    parser.add_argument("--coco_dir", default="data/coco")
    parser.add_argument("--out", default="data/visual7w_questions.json")
    args = parser.parse_args()
    main(args.dest_dir, args.coco_dir, args.out)
