#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/github/MRAG-Bench"

# Avoid allocator internal assert seen on some nodes with expandable_segments=True.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CUDA_VISIBLE_DEVICES=0,1 \
MRAG_NUM_BEAMS=5 \
MRAG_DO_SAMPLE=false \
MRAG_MAX_NEW_TOKENS=32 \
MRAG_MAX_RAG_IMAGES=1 \
MRAG_HF_HOME="${ROOT_DIR}/models/huggingface-mrag" \
MRAG_MODEL_LOCAL_DIR="${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov" \
MRAG_HF_OFFLINE="${MRAG_HF_OFFLINE:-0}" \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
bash eval/models/run_model.sh

if [[ ! -s "llava_one_vision_gt_rag_results.jsonl" ]]; then
  echo "[ERROR] Empty or missing results file: llava_one_vision_gt_rag_results.jsonl"
  exit 2
fi

python eval/score.py -i llava_one_vision_gt_rag_results.jsonl
