#!/usr/bin/env bash
# E10: Query-rewrite ablation for E8.
# Difference from E8_full: no 5-dimensional Gemma4 rewriting. The original
# question+choices are sent directly to MagicLens as one retrieval instruction.
# Other major settings stay aligned with E8: MagicLens Top-5, RRF-compatible
# final Top-5, Gemma4 evidence descriptions, and LLaVA-OneVision final answerer.
#
# Usage:
#   bash test/E10_no_query_rewrite.sh
#   MAX_SAMPLES=20 bash test/E10_no_query_rewrite.sh
#   CORPUS_DIR=/public/home/hzh/mRAG/data/image_corpus bash test/E10_no_query_rewrite.sh
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E10] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate llava 2>/dev/null || true
  fi
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
GEMMA4_LOCAL_DIR="${GEMMA4_LOCAL_DIR:-models/gemma4-e2b}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-E2B-it}"
GEMMA4_DEVICE="${GEMMA4_DEVICE:-cuda:${GPU_ID}}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-models/llava-onevision-qwen2-7b-ov}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

OUT_DIR="${OUT_DIR:-log/E10}"
mkdir -p "${OUT_DIR}"

ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

log "corpus_dir=${CORPUS_DIR}"
log "gemma4_local_dir=${GEMMA4_LOCAL_DIR}"
log "gemma4_device=${GEMMA4_DEVICE}"
log "llava_model_path=${LLAVA_MODEL_PATH}"
log "max_samples=${MAX_SAMPLES}"
log "out_dir=${OUT_DIR}"

nohup python test/pipeline_multi_dim_rag.py \
  --dataset-name uclanlp/MRAG-Bench \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type raw_question \
  --gemma4-local-dir "${GEMMA4_LOCAL_DIR}" \
  --gemma4-model-id "${GEMMA4_MODEL_ID}" \
  --gemma4-device "${GEMMA4_DEVICE}" \
  --n-dims 1 \
  --dim-top-k 5 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --final-answerer llava \
  --describe-final-images \
  --magiclens-platform cpu \
  --resume-from-existing \
  --llava-model-path "${LLAVA_MODEL_PATH}" \
  --llava-max-images 1 \
  --llava-max-new-tokens 64 \
  --answers-file "${OUT_DIR}/e10_no_rewrite.jsonl" \
  --summary-out "${OUT_DIR}/e10_no_rewrite_summary.json" \
  --save-dimensions-jsonl "${OUT_DIR}/e10_no_rewrite_dims.jsonl" \
  --trace-jsonl "${OUT_DIR}/e10_no_rewrite_trace.jsonl" \
  "${ARGS[@]}" > "${OUT_DIR}/E10_no_query_rewrite.log" 2>&1 &

echo $! > "${OUT_DIR}/E10_no_query_rewrite.pid"
log "started pid=$(cat "${OUT_DIR}/E10_no_query_rewrite.pid")"
log "tail log: tail -f ${OUT_DIR}/E10_no_query_rewrite.log"
log "summary:  ${OUT_DIR}/e10_no_rewrite_summary.json"
