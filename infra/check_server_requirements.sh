#!/usr/bin/env bash
set -euo pipefail

# infra/check_server_requirements.sh
# Quick checks to run on the server after syncing scripts.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/data"

echo "Project root: $ROOT_DIR"

echo "Checking Gemma4 local dir (example: models/gemma4-e2b)"
if [ -d "$MODELS_DIR/gemma4-e2b" ]; then
  echo " - Found models/gemma4-e2b"
  ls -lah "$MODELS_DIR/gemma4-e2b" | sed -n '1,5p'
else
  echo " - Missing models/gemma4-e2b"
fi

echo "Checking for LLaVA/answerer models (models/*llava* or models/*llava*)"
ls -d $MODELS_DIR/*llava* 2>/dev/null || echo " - No llava-like model directories found"

echo "Checking image corpus (data/image_corpus or data/infoseek)"
if [ -d "$DATA_DIR/image_corpus" ]; then
  echo " - image_corpus exists: $(du -sh "$DATA_DIR/image_corpus" | awk '{print $1}')"
  echo " - sample files:"
  find "$DATA_DIR/image_corpus" -maxdepth 2 -type f | head -n 10
elif [ -d "$DATA_DIR/infoseek" ]; then
  echo " - infoseek dataset exists: $(du -sh "$DATA_DIR/infoseek" | awk '{print $1}')"
  echo " - sample files:"
  find "$DATA_DIR/infoseek" -maxdepth 3 -type f | head -n 10
else
  echo " - No image corpus directory found under data/"
fi

echo "Checking GPU availability (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo " - nvidia-smi not found; no NVIDIA GPU detected or drivers missing"
fi

echo "Checking free disk space (root project filesystem)"
df -h "$ROOT_DIR" | sed -n '1,2p'

echo "Check complete. If anything is missing, run infra/setup_gemma4.sh on this machine (requires HF_TOKEN if gated)."
