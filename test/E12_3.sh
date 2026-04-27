#!/usr/bin/env bash
# E12_3: parameter analysis for per-dimension MagicLens Top-K, K_dim=3.
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E12_3] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate llava 2>/dev/null || true
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
GEMMA4_LOCAL_DIR="${GEMMA4_LOCAL_DIR:-models/gemma4-e2b}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-E2B-it}"
GEMMA4_DEVICE="${GEMMA4_DEVICE:-cuda:0}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-models/llava-onevision-qwen2-7b-ov}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-log/E12/E12_3}"
mkdir -p "${OUT_DIR}"

ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

log "n_dims=5 dim_top_k=3 final_top_k=5 out_dir=${OUT_DIR}"
nohup python test/pipeline_multi_dim_rag.py \
  --dataset-name uclanlp/MRAG-Bench \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type gemma4_local \
  --gemma4-local-dir "${GEMMA4_LOCAL_DIR}" \
  --gemma4-model-id "${GEMMA4_MODEL_ID}" \
  --gemma4-device "${GEMMA4_DEVICE}" \
  --gemma4-dim-rationale \
  --n-dims 5 \
  --dim-top-k 3 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --final-answerer llava \
  --describe-final-images \
  --magiclens-platform cpu \
  --resume-from-existing \
  --llava-model-path "${LLAVA_MODEL_PATH}" \
  --llava-max-images 1 \
  --llava-max-new-tokens 64 \
  --answers-file "${OUT_DIR}/e12_3.jsonl" \
  --summary-out "${OUT_DIR}/e12_3_summary.json" \
  --save-dimensions-jsonl "${OUT_DIR}/e12_3_dims.jsonl" \
  --trace-jsonl "${OUT_DIR}/e12_3_trace.jsonl" \
  "${ARGS[@]}" > "${OUT_DIR}/E12_3.log" 2>&1 &

echo $! > "${OUT_DIR}/E12_3.pid"
log "started pid=$(cat "${OUT_DIR}/E12_3.pid")"
log "tail log: tail -f ${OUT_DIR}/E12_3.log"
