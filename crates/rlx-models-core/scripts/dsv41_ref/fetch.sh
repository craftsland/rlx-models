#!/usr/bin/env bash
# Fetch the released DeepSeek-V4.1 reference sources next to this harness.
#
# They are DeepSeek's own MIT-licensed code and are deliberately NOT vendored:
# the point of the harness is to compare against whatever the upstream repo
# currently says, and a stale vendored copy would quietly stop doing that.
set -euo pipefail
cd "$(dirname "$0")"
REPO="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main"
BASE="$REPO/inference"
for f in model.py engram.py vision.py image_processor.py; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "$f"
done
curl -fsSL "$REPO/config.json" -o config.json   # the released shapes
# `kernel.py` here is OUR CPU transliteration of the tilelang kernels; keep it.
echo "done — run: RLX_REF_NOQUANT=1 python3 dump.py"
