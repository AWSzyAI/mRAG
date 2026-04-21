#!/usr/bin/env bash
# E8 Multi-Dimension Query Decomposition RAG — experiment sweep
# Runs all combinations from the testing matrix.
# Usage:
#   DIM_GENERATOR_API_KEY="sk-xxx" bash test/E8_multi_dim_sweep.sh
#   MAX_SAMPLES=50 bash test/E8_multi_dim_sweep.sh   # quick smoke test
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E8-sweep] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava 2>/dev/null || true
  fi
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CORPUS_DIR="${CORPUS_DIR:-/public/home/hzh/mRAG/data/image_corpus}"
export DIM_GENERATOR_API_KEY="${DIM_GENERATOR_API_KEY:-}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

SILICONFLOW_BASE="https://api.siliconflow.cn/v1"
DEEPINFRA_BASE="https://api.deepinfra.com/v1/openai"

# ─── Testing Matrix ───────────────────────────────────────────────────────────
# Each row: EXP_TAG  GENERATOR_MODEL  API_BASE  N_DIMS  DIM_TOP_K  FINAL_TOP_K  FUSION
declare -a EXPERIMENTS=(
  # --- Vary generator model (API, n=3, rrf) ---
  "D01_qwen7b          Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  5  5  rrf"
  "D02_qwen3b          Qwen/Qwen2.5-3B-Instruct          ${SILICONFLOW_BASE}  3  5  5  rrf"
  "D03_deepseek_r1_7b  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  ${SILICONFLOW_BASE}  3  5  5  rrf"
  "D04_llama31_8b      meta-llama/Meta-Llama-3.1-8B-Instruct    ${SILICONFLOW_BASE}  3  5  5  rrf"
  "D05_phi35_mini      microsoft/Phi-3.5-mini-instruct          ${SILICONFLOW_BASE}  3  5  5  rrf"

  # --- Vary n_dims (best model from above, rrf) ---
  "D06_qwen7b_n1       Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  1  5  5  rrf"
  "D07_qwen7b_n5       Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  5  5  5  rrf"

  # --- Vary fusion strategy (n=3) ---
  "D08_qwen7b_score    Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  5  5  score_sum"
  "D09_qwen7b_vote     Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  5  5  voting"

  # --- Vary dim_top_k & final_top_k ---
  "D10_qwen7b_k3       Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  3  5  rrf"
  "D11_qwen7b_k10      Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  10 5  rrf"
  "D12_qwen7b_fk3      Qwen/Qwen2.5-7B-Instruct          ${SILICONFLOW_BASE}  3  5  3  rrf"
)

log "total_experiments=${#EXPERIMENTS[@]}"
log "max_samples=${MAX_SAMPLES}"

RESULTS_CSV="${ROOT_DIR}/log/E8/sweep_results.csv"
mkdir -p "${ROOT_DIR}/log/E8"
echo "exp_tag,model,n_dims,dim_top_k,final_top_k,fusion,accuracy,processed,dim_gen_failures,avg_dim_time,avg_ret_time" \
  > "${RESULTS_CSV}"

for exp_line in "${EXPERIMENTS[@]}"; do
  read -r TAG MODEL API_BASE N_DIMS DIM_TOP_K FINAL_TOP_K FUSION <<< "${exp_line}"
  EXP_DIR="${ROOT_DIR}/log/E8/${TAG}"
  mkdir -p "${EXP_DIR}"
  ANSWERS="${EXP_DIR}/results.jsonl"
  SUMMARY="${EXP_DIR}/summary.json"
  DIMS_JSONL="${EXP_DIR}/dimensions.jsonl"

  log "────────────────────────────────────────"
  log "exp=${TAG} model=${MODEL} n=${N_DIMS} dk=${DIM_TOP_K} fk=${FINAL_TOP_K} fusion=${FUSION}"

  python "${SCRIPT_DIR}/benchmark_multi_dimension_rag.py" \
    --corpus-dir "${CORPUS_DIR}" \
    --dim-generator-type api \
    --dim-generator-model "${MODEL}" \
    --dim-generator-api-base "${API_BASE}" \
    --dim-generator-api-key "${DIM_GENERATOR_API_KEY}" \
    --n-dims "${N_DIMS}" \
    --dim-top-k "${DIM_TOP_K}" \
    --final-top-k "${FINAL_TOP_K}" \
    --fusion-strategy "${FUSION}" \
    --answers-file "${ANSWERS}" \
    --summary-out "${SUMMARY}" \
    --save-dimensions-jsonl "${DIMS_JSONL}" \
    --max-samples "${MAX_SAMPLES}" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run.log"

  if [[ -f "${SUMMARY}" ]]; then
    ACC=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(d['accuracy'])")
    PROC=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(d['processed'])")
    FAIL=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(d['dim_gen_failures'])")
    DT=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(d['avg_dim_gen_time_sec'])")
    RT=$(python3 -c "import json; d=json.load(open('${SUMMARY}')); print(d['avg_retrieval_time_sec'])")
    echo "${TAG},${MODEL},${N_DIMS},${DIM_TOP_K},${FINAL_TOP_K},${FUSION},${ACC},${PROC},${FAIL},${DT},${RT}" \
      >> "${RESULTS_CSV}"
    log "result=${TAG} accuracy=${ACC}%"
  else
    log "WARNING: summary not found for ${TAG}"
    echo "${TAG},${MODEL},${N_DIMS},${DIM_TOP_K},${FINAL_TOP_K},${FUSION},FAIL,0,0,0,0" \
      >> "${RESULTS_CSV}"
  fi

  log "done=${TAG}"
done

log "All experiments complete. Results: ${RESULTS_CSV}"
column -t -s',' "${RESULTS_CSV}"
