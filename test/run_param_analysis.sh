#!/usr/bin/env bash
# One-shot parameter analysis sweep replacing the old E11/E12 one-by-one runs.
#
# Default mode uses stratified sampling across MRAG-Bench scenarios:
#   bash test/run_param_analysis.sh
#
# Quick smoke:
#   SAMPLES_PER_SCENARIO=2 bash test/run_param_analysis.sh
#
# Full dataset, if ever needed:
#   SAMPLES_PER_SCENARIO=0 bash test/run_param_analysis.sh
#
# Pull results back from remote after running there:
#   make pull param_results y
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [ParamSweep] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate llava 2>/dev/null || true
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# cuda:0

GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
GEMMA4_LOCAL_DIR="${GEMMA4_LOCAL_DIR:-models/gemma4-e2b}"
GEMMA4_MODEL_ID="${GEMMA4_MODEL_ID:-google/gemma-4-E2B-it}"
GEMMA4_DEVICE="${GEMMA4_DEVICE:-cuda:${GPU_ID}}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-models/llava-onevision-qwen2-7b-ov}"
OUT_ROOT="${OUT_ROOT:-log/ParamSweep}"
SAMPLES_PER_SCENARIO="${SAMPLES_PER_SCENARIO:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

mkdir -p "${OUT_ROOT}"

BASE_ARGS=(
  --dataset-name uclanlp/MRAG-Bench
  --corpus-dir "${CORPUS_DIR}"
  --dim-generator-type gemma4_local
  --gemma4-local-dir "${GEMMA4_LOCAL_DIR}"
  --gemma4-model-id "${GEMMA4_MODEL_ID}"
  --gemma4-device "${GEMMA4_DEVICE}"
  --gemma4-dim-rationale
  --final-top-k 5
  --fusion-strategy rrf
  --final-answerer llava
  --describe-final-images
  --magiclens-platform cpu
  --resume-from-existing
  --llava-model-path "${LLAVA_MODEL_PATH}"
  --llava-max-images 1
  --llava-max-new-tokens 64
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  BASE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${SAMPLES_PER_SCENARIO}" != "0" ]]; then
  BASE_ARGS+=(--samples-per-scenario "${SAMPLES_PER_SCENARIO}")
fi

# tag sweep n_dims dim_top_k
EXPERIMENTS=(
  "P01_dims_n1 n_dims 1 5"
  "P02_dims_n2 n_dims 2 5"
  "P03_dims_n3 n_dims 3 5"
  "P04_dims_n4 n_dims 4 5"
  "P05_dimtopk_k1 dim_top_k 5 1"
  "P06_dimtopk_k2 dim_top_k 5 2"
  "P07_dimtopk_k3 dim_top_k 5 3"
  "P08_dimtopk_k4 dim_top_k 5 4"
)

log "out_root=${OUT_ROOT}"
log "samples_per_scenario=${SAMPLES_PER_SCENARIO} max_samples=${MAX_SAMPLES}"
log "total_experiments=${#EXPERIMENTS[@]}"

for line in "${EXPERIMENTS[@]}"; do
  read -r TAG SWEEP N_DIMS DIM_TOP_K <<< "${line}"
  EXP_DIR="${OUT_ROOT}/${TAG}"
  PREFIX="$(echo "${TAG}" | tr '[:upper:]' '[:lower:]')"
  mkdir -p "${EXP_DIR}"

  {
    echo "{"
    echo "  \"tag\": \"${TAG}\","
    echo "  \"prefix\": \"${PREFIX}\","
    echo "  \"sweep\": \"${SWEEP}\","
    echo "  \"n_dims\": ${N_DIMS},"
    echo "  \"dim_top_k\": ${DIM_TOP_K},"
    echo "  \"final_top_k\": 5,"
    echo "  \"fusion_strategy\": \"rrf\","
    echo "  \"samples_per_scenario\": ${SAMPLES_PER_SCENARIO},"
    echo "  \"max_samples\": ${MAX_SAMPLES}"
    echo "}"
  } > "${EXP_DIR}/config.json"

  log "start ${TAG}: sweep=${SWEEP} n_dims=${N_DIMS} dim_top_k=${DIM_TOP_K}"
  set +e
  python test/pipeline_multi_dim_rag.py \
    "${BASE_ARGS[@]}" \
    --n-dims "${N_DIMS}" \
    --dim-top-k "${DIM_TOP_K}" \
    --answers-file "${EXP_DIR}/${PREFIX}.jsonl" \
    --summary-out "${EXP_DIR}/${PREFIX}_summary.json" \
    --save-dimensions-jsonl "${EXP_DIR}/${PREFIX}_dims.jsonl" \
    --trace-jsonl "${EXP_DIR}/${PREFIX}_trace.jsonl" \
    2>&1 | tee "${EXP_DIR}/${PREFIX}.log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" != "0" ]]; then
    log "FAILED ${TAG} rc=${rc}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  else
    log "done ${TAG}"
  fi

  python scripts/summarize_param_sweep.py \
    --root "${OUT_ROOT}" \
    --csv-out "${OUT_ROOT}/param_sweep_results.csv" \
    --json-out "${OUT_ROOT}/param_sweep_results.json" \
    --md-out "${OUT_ROOT}/param_sweep_results.md" \
    --include-e8-baseline
done

log "all parameter experiments finished"
python scripts/summarize_param_sweep.py \
  --root "${OUT_ROOT}" \
  --csv-out "${OUT_ROOT}/param_sweep_results.csv" \
  --json-out "${OUT_ROOT}/param_sweep_results.json" \
  --md-out "${OUT_ROOT}/param_sweep_results.md" \
  --include-e8-baseline
log "summary: ${OUT_ROOT}/param_sweep_results.md"
