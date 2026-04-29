#!/usr/bin/env bash
# E15: E8/E9 pipeline with Gemma4 E4B-it as the Gemma4 model.
# Keep Gemma4 multi-dim planning + MagicLens + RRF + Gemma4 final answering,
# but use google/gemma-4-E4B-it instead of google/gemma-4-E2B-it.
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E15] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate llava 2>/dev/null || true
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
GEMMA4_LOCAL_DIR="${GEMMA4_LOCAL_DIR:-models/gemma4-e4b-it}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-E4B-it}"
GEMMA4_DEVICE="${GEMMA4_DEVICE:-cuda:${GPU_ID}}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-log/E15}"
mkdir -p "${OUT_DIR}"

ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

log "corpus_dir=${CORPUS_DIR}"
log "gemma4_model_id=${GEMMA4_MODEL_ID}"
log "gemma4_local_dir=${GEMMA4_LOCAL_DIR}"
log "gemma4_device=${GEMMA4_DEVICE}"
log "max_samples=${MAX_SAMPLES}"
log "out_dir=${OUT_DIR}"

nohup python test/pipeline_multi_dim_rag.py \
  --dataset-name uclanlp/MRAG-Bench \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type gemma4_local \
  --gemma4-local-dir "${GEMMA4_LOCAL_DIR}" \
  --gemma4-model-id "${GEMMA4_MODEL_ID}" \
  --gemma4-device "${GEMMA4_DEVICE}" \
  --gemma4-dim-rationale \
  --n-dims 5 \
  --dim-top-k 5 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --final-answerer gemma4 \
  --describe-final-images \
  --gemma4-answer-max-images 6 \
  --gemma4-answer-max-new-tokens 64 \
  --magiclens-platform cpu \
  --resume-from-existing \
  --answers-file "${OUT_DIR}/e15_gemma4_e4b_answer.jsonl" \
  --summary-out "${OUT_DIR}/e15_gemma4_e4b_answer_summary.json" \
  --save-dimensions-jsonl "${OUT_DIR}/e15_gemma4_e4b_answer_dims.jsonl" \
  --trace-jsonl "${OUT_DIR}/e15_gemma4_e4b_answer_trace.jsonl" \
  "${ARGS[@]}" > "${OUT_DIR}/E15_gemma4_e4b_answer.log" 2>&1 &

echo $! > "${OUT_DIR}/E15_gemma4_e4b_answer.pid"
log "started pid=$(cat "${OUT_DIR}/E15_gemma4_e4b_answer.pid")"
log "tail log: tail -f ${OUT_DIR}/E15_gemma4_e4b_answer.log"
log "summary:  ${OUT_DIR}/e15_gemma4_e4b_answer_summary.json"
