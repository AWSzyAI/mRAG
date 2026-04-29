#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E6] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_START_TS="$(date +%s)"

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-single}"
export LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
export JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-1}"
export TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-1}"

export EXP_DIR="${EXP_DIR:-${ROOT_DIR}/log/E6}"
mkdir -p "${EXP_DIR}"
export CORPUS_DIR="${CORPUS_DIR:-/public/home/hzh/mRAG/data/image_corpus}"
export DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-0}"
export ANSWERS_FILE="${ANSWERS_FILE:-${EXP_DIR}/e6_clip_corpus_magiclens_rerank_results.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-${EXP_DIR}/e6_clip_corpus_magiclens_rerank_summary.json}"
export LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/results/e3_clip_corpus_rag_results.jsonl}"
export TOP_K="${TOP_K:-5}"
SCORE_BASENAME="$(basename "${ANSWERS_FILE}")"
SCORE_STEM="${SCORE_BASENAME%.*}"
SCORE_JSON="${ROOT_DIR}/results/${SCORE_STEM}_score.json"
SCORE_GPT_JSON="${ROOT_DIR}/results/${SCORE_STEM}_score_gpt_extracted.json"

log "exp=E6 CLIP-RAG corpus + MagicLens rerank"
log "script=${BASH_SOURCE[0]}"
log "cwd=$(pwd)"
echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ENV] LLAVA_DEVICE_MAP=${LLAVA_DEVICE_MAP}"
echo "[ENV] LLAVA_ATTN_IMPLEMENTATION=${LLAVA_ATTN_IMPLEMENTATION}"
echo "[ENV] JAX_CUDA_REQUIRED=${JAX_CUDA_REQUIRED}"
echo "[ENV] TORCH_CUDA_REQUIRED=${TORCH_CUDA_REQUIRED}"
echo "[ENV] CORPUS_DIR=${CORPUS_DIR}"
echo "[ENV] TOP_K=${TOP_K}"
echo "[ENV] DISABLE_MAGICLENS_RERANK=${DISABLE_MAGICLENS_RERANK}"
echo "[ENV] ANSWERS_FILE=${ANSWERS_FILE}"
echo "[ENV] SUMMARY_OUT=${SUMMARY_OUT}"
echo "[ENV] LLAVA_GREEDY_JSONL=${LLAVA_GREEDY_JSONL}"
echo "[ENV] SCORE_JSON=${SCORE_JSON}"
log "delegate=test/benchmark_corpus_rag.sh"

bash "${SCRIPT_DIR}/benchmark_corpus_rag.sh" "$@"

RUN_END_TS="$(date +%s)"
log "duration_sec=$((RUN_END_TS - RUN_START_TS))"

bash "${SCRIPT_DIR}/archive_exp_outputs.sh" "${EXP_DIR}" \
  "${ANSWERS_FILE}" \
  "${SUMMARY_OUT}" \
  "${SCORE_JSON}" \
  "${SCORE_GPT_JSON}" \
  "${ROOT_DIR}/github/MRAG-Bench/results/e6_clip_corpus_magiclens_rerank_results.jsonl" \
  "${ROOT_DIR}/log/e6_clip_corpus_magiclens_rerank_summary.json" \
  "${ROOT_DIR}/E6.log"
