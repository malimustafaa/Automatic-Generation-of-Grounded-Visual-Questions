#!/usr/bin/env bash
# Both VQA v1 and Visual7W are built on MS-COCO images (train2014/val2014).
set -euo pipefail
DEST_DIR="${1:-data/coco}"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

for split in train2014 val2014; do
  if [ -d "$split" ]; then
    echo "$split already present, skipping."
    continue
  fi
  echo "Downloading $split.zip (large, ~13-19GB)..."
  curl -L -o "${split}.zip" "http://images.cocodataset.org/zips/${split}.zip"
  unzip -q "${split}.zip"
  rm "${split}.zip"
done
echo "Done: $DEST_DIR/train2014, $DEST_DIR/val2014"
