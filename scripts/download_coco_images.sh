#!/usr/bin/env bash
# Both VQA v1 and Visual7W are built on MS-COCO images (train2014/val2014).
#
# Extracts to LOCAL disk, not a Google Drive mount. COCO train2014+val2014 is ~123k
# individual small JPEGs; reading/writing that many small files through Drive's FUSE
# mount is both dramatically slower than local disk AND prone to sporadic
# "OSError: [Errno 5] Input/output error" on individual files under sustained load --
# both things we hit in practice with an earlier version of this script that extracted
# straight onto Drive. Only the ~19GB zip files themselves are cached on Drive (a single
# large file is fine there -- it's *many small files* that Drive's FUSE layer struggles
# with), so re-running this on a fresh Colab session doesn't re-download from the network.
#
# If you already have train2014/val2014 fully extracted on Drive from a previous run of
# the old (Drive-extracting) version of this script, don't bother re-downloading -- copy
# what you already have to local disk instead:
#   rsync -a --info=progress2 "$DRIVE_COCO_DIR/train2014" /content/coco/
#   rsync -a --info=progress2 "$DRIVE_COCO_DIR/val2014"   /content/coco/
#   touch /content/coco/train2014.zip.done /content/coco/val2014.zip.done
#
# Usage: download_coco_images.sh <local_extract_dir> [<zip_cache_dir>]
set -euo pipefail
LOCAL_DIR="${1:-/content/coco}"
ZIP_CACHE_DIR="${2:-$LOCAL_DIR}"
mkdir -p "$LOCAL_DIR" "$ZIP_CACHE_DIR"

for split in train2014 val2014; do
  marker="$LOCAL_DIR/${split}.zip.done"
  if [ -f "$marker" ]; then
    echo "$split already extracted locally, skipping."
    continue
  fi

  zip_path="$ZIP_CACHE_DIR/${split}.zip"
  if [ -f "$zip_path" ]; then
    echo "$zip_path already present, reusing cached download."
  else
    echo "Downloading $split.zip (large, ~13-19GB) to $zip_path ..."
    curl -L -o "$zip_path" "http://images.cocodataset.org/zips/${split}.zip"
  fi

  echo "Verifying archive integrity..."
  if ! unzip -tq "$zip_path" > /dev/null; then
    echo "ERROR: $zip_path failed integrity check -- the download was likely" >&2
    echo "interrupted or incomplete. Delete it and re-run this script." >&2
    exit 1
  fi

  echo "Extracting $split to local disk ($LOCAL_DIR)..."
  rm -rf "${LOCAL_DIR:?}/$split"
  unzip -q "$zip_path" -d "$LOCAL_DIR"
  touch "$marker"
  echo "Done: $LOCAL_DIR/$split"
done
echo "Done: $LOCAL_DIR/train2014, $LOCAL_DIR/val2014"
