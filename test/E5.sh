#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E5] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_START_TS="$(date +%s)"

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-single}"
export LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
export JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-1}"
export TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-1}"

export EXP_DIR="${EXP_DIR:-${ROOT_DIR}/log/E5}"
mkdir -p "${EXP_DIR}"
export USE_GT="${USE_GT:-1}"
export DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-1}"
export ANSWERS_FILE="${ANSWERS_FILE:-${EXP_DIR}/e5_magiclens_gt_norerank_results.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-${EXP_DIR}/e5_magiclens_gt_norerank_summary.json}"
export LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/llava_one_vision_gt_rag_results.jsonl}"
SCORE_BASENAME="$(basename "${ANSWERS_FILE}")"
SCORE_STEM="${SCORE_BASENAME%.*}"
SCORE_JSON="${ROOT_DIR}/results/${SCORE_STEM}_score.json"
SCORE_GPT_JSON="${ROOT_DIR}/results/${SCORE_STEM}_score_gpt_extracted.json"

log "exp=E5 MagicLens+GT-RAG (rerank off)"
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
echo "[ENV] SCORE_JSON=${SCORE_JSON}"
log "delegate=test/benchmark_magiclens.sh"

bash "${SCRIPT_DIR}/benchmark_magiclens.sh" "$@"

RUN_END_TS="$(date +%s)"
log "duration_sec=$((RUN_END_TS - RUN_START_TS))"

bash "${SCRIPT_DIR}/archive_exp_outputs.sh" "${EXP_DIR}" \
  "${ANSWERS_FILE}" \
  "${SUMMARY_OUT}" \
  "${SCORE_JSON}" \
  "${SCORE_GPT_JSON}" \
  "${ROOT_DIR}/github/MRAG-Bench/results/e5_magiclens_gt_norerank_results.jsonl" \
  "${ROOT_DIR}/log/e5_magiclens_gt_norerank_summary.json" \
  "${ROOT_DIR}/E5.log"
