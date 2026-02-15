#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
  else
    echo "[WARN] conda not found, continue with current python: $(command -v python || echo '<missing>')"
  fi
fi

python main.py \
  --model-local-dir ./models/llava-onevision-qwen2-7b-ov \
  --hf-home ./models/huggingface-mrag \
  --hf-endpoint "${HF_ENDPOINT:-https://hf-mirror.com}"
