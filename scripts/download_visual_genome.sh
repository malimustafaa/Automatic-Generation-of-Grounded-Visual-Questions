#!/usr/bin/env bash
# Visual Genome images + region annotations -- needed only if training densecap-pytorch
# ourselves (see README's "Training DenseCap ourselves" section) rather than using the
# author's pretrained checkpoint, which turned out to be unreachable (OneDrive: broken
# share link ["item deleted or don't have permission"]; BaiduYun: blocks anonymous
# browser downloads over ~1GB without a China-verified account). All four URLs below
# were verified live via `curl -I` (200 OK, plausible Content-Length) against the same
# source the official ranjaykrishna/visual_genome Hugging Face dataset loader uses --
# not guessed.
#
# Extracts to LOCAL disk, same reasoning as download_coco_images.sh: ~108k small image
# files unzipped directly onto a Drive mount is dramatically slower than local disk.
#
# Usage: download_visual_genome.sh <local_extract_dir> [<zip_cache_dir>]
#   local_extract_dir: fast ephemeral disk (e.g. /content/visual-genome)
#   zip_cache_dir:     persisted across sessions (e.g. a Drive path) -- avoids
#                       re-downloading ~15GB every time you restart the runtime
set -euo pipefail
LOCAL_DIR="${1:-/content/visual-genome}"
ZIP_CACHE_DIR="${2:-$LOCAL_DIR}"
mkdir -p "$LOCAL_DIR" "$ZIP_CACHE_DIR"

# images.zip -> VG_100K/, images2.zip -> VG_100K_2/ (confirmed extraction behavior,
# not assumed -- matches what densecap-pytorch's preprocess.py/train.py expect under
# IMG_DIR_ROOT = './data/visual-genome').
NAMES=(images.zip images2.zip region_descriptions.json.zip image_data.json.zip)
URLS=(
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
  "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  url="${URLS[$i]}"
  marker="$LOCAL_DIR/${name}.done"
  if [ -f "$marker" ]; then
    echo "$name already extracted, skipping."
    continue
  fi

  zip_path="$ZIP_CACHE_DIR/$name"
  if [ -f "$zip_path" ]; then
    echo "$zip_path already present, reusing cached download."
  else
    echo "Downloading $name (~$(( $(curl -sI "$url" | grep -i content-length | tr -dc '0-9') / 1000000 ))MB)..."
    curl -L -o "$zip_path" "$url"
  fi

  echo "Verifying archive integrity..."
  if ! unzip -tq "$zip_path" > /dev/null; then
    echo "ERROR: $zip_path failed integrity check -- delete it and re-run." >&2
    exit 1
  fi

  echo "Extracting $name to $LOCAL_DIR ..."
  unzip -q "$zip_path" -d "$LOCAL_DIR"
  touch "$marker"
done

echo "Done."
echo "  Images:      $LOCAL_DIR/VG_100K, $LOCAL_DIR/VG_100K_2"
echo "  Annotations: $LOCAL_DIR/region_descriptions.json, $LOCAL_DIR/image_data.json"
