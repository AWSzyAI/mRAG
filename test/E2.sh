#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E2] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-single}"
export LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
export JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-1}"
export TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-1}"

export USE_GT="${USE_GT:-0}"
export DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-0}"
export ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/github/MRAG-Bench/results/e2_magiclens_retrieved_rerank_results.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-${ROOT_DIR}/log/e2_magiclens_retrieved_rerank_summary.json}"
export LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/llava_one_vision_retrieved_rag_results.jsonl}"

log "exp=E2 MagicLens+Retrieved-RAG (rerank on)"
log "script=${BASH_SOURCE[0]}"
log "cwd=$(pwd)"
echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ENV] LLAVA_DEVICE_MAP=${LLAVA_DEVICE_MAP}"
echo "[ENV] LLAVA_ATTN_IMPLEMENTATION=${LLAVA_ATTN_IMPLEMENTATION}"
echo "[ENV] JAX_CUDA_REQUIRED=${JAX_CUDA_REQUIRED}"
echo "[ENV] TORCH_CUDA_REQUIRED=${TORCH_CUDA_REQUIRED}"
echo "[ENV] USE_GT=${USE_GT}"
echo "[ENV] DISABLE_MAGICLENS_RERANK=${DISABLE_MAGICLENS_RERANK}"
echo "[ENV] ANSWERS_FILE=${ANSWERS_FILE}"
echo "[ENV] SUMMARY_OUT=${SUMMARY_OUT}"
echo "[ENV] LLAVA_GREEDY_JSONL=${LLAVA_GREEDY_JSONL}"
log "delegate=test/benchmark_magiclens.sh"

exec bash "${SCRIPT_DIR}/benchmark_magiclens.sh" "$@"
