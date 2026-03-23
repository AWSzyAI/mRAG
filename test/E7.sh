#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E7] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_START_TS="$(date +%s)"

log "script=${BASH_SOURCE[0]}"
log "cwd=$(pwd)"
log "root_dir=${ROOT_DIR}"
log "conda_env_before=${CONDA_DEFAULT_ENV:-<unset>}"
log "python_before=$(command -v python || echo '<missing>')"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  log "step=conda_activate target_env=llava"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk '{print $1}' | grep -qx "llava"; then
      conda activate llava
      log "conda_activate_result=ok"
    else
      echo "[WARN] conda env 'llava' not found; using current python: $(command -v python || echo '<missing>')"
      log "conda_activate_result=missing_env"
    fi
  else
    echo "[WARN] conda not found; using current python: $(command -v python || echo '<missing>')"
    log "conda_activate_result=conda_not_found"
  fi
fi

log "conda_env_after=${CONDA_DEFAULT_ENV:-<unset>}"
log "python_after=$(command -v python || echo '<missing>')"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CORPUS_DIR="${CORPUS_DIR:-/public/home/hzh/mRAG/data/image_corpus}"
export RETRIEVER_TYPE="${RETRIEVER_TYPE:-magiclens}"
export DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-1}"
export EXP_DIR="${EXP_DIR:-${ROOT_DIR}/log/E7}"
mkdir -p "${EXP_DIR}"
export ANSWERS_FILE="${ANSWERS_FILE:-${EXP_DIR}/e7_magiclens_corpus_rag_results.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-${EXP_DIR}/e7_magiclens_corpus_rag_summary.json}"
export LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-}"
export TOP_K="${TOP_K:-5}"

log "exp=E7 MagicLens-RAG corpus baseline (direct retrieval, no rerank)"
log "delegate=test/benchmark_corpus_rag.sh"
bash "${SCRIPT_DIR}/benchmark_corpus_rag.sh" "$@"

bash "${SCRIPT_DIR}/archive_exp_outputs.sh" "${EXP_DIR}" \
  "${ROOT_DIR}/github/MRAG-Bench/results/e7_magiclens_corpus_rag_results.jsonl" \
  "${ROOT_DIR}/log/e7_magiclens_corpus_rag_summary.json" \
  "${ROOT_DIR}/E7.log"
