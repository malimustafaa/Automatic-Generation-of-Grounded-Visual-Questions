#!/usr/bin/env bash
# Downloads GloVe 840B.300d (paper Sec 3.2: word embeddings, 300-d, trained on 840B words).
set -euo pipefail
DEST_DIR="${1:-data}"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

if [ -f glove.840B.300d.txt ]; then
  echo "glove.840B.300d.txt already present, skipping download."
  exit 0
fi

echo "Downloading GloVe 840B.300d (~2.2GB zipped, ~5.6GB unzipped)..."
curl -L -o glove.840B.300d.zip https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
echo "Done: $DEST_DIR/glove.840B.300d.txt"
