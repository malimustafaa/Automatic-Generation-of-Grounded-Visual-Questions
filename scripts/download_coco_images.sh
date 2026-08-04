#!/usr/bin/env bash
# Both VQA v1 and Visual7W are built on MS-COCO images (train2014/val2014).
#
# Writes a "<split>.done" marker only after the zip passes an integrity check and
# extracts successfully -- a bare `[ -d "$split" ]` check (the previous version of
# this script) treats *any* partial/interrupted download as "already present" and
# never retries, which silently leaves you with a directory missing thousands of
# images (this is what produces "[warn] missing image" spam in extract_image_features.py).
set -euo pipefail
DEST_DIR="${1:-data/coco}"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

for split in train2014 val2014; do
  marker="${split}.done"
  if [ -f "$marker" ]; then
    echo "$split already fully downloaded (marker found), skipping."
    continue
  fi

  echo "Downloading $split.zip (large, ~13-19GB)..."
  curl -L -o "${split}.zip" "http://images.cocodataset.org/zips/${split}.zip"

  echo "Verifying archive integrity..."
  if ! unzip -tq "${split}.zip" > /dev/null; then
    echo "ERROR: ${split}.zip failed integrity check -- the download was likely" >&2
    echo "interrupted or incomplete. Delete $DEST_DIR/${split}.zip and re-run this script." >&2
    exit 1
  fi

  rm -rf "$split"  # clear any partial extraction left over from an earlier failed attempt
  unzip -q "${split}.zip"
  rm "${split}.zip"
  touch "$marker"
  echo "Done: $DEST_DIR/$split"
done
echo "Done: $DEST_DIR/train2014, $DEST_DIR/val2014"
