#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [CORPUS] $*"; }
print_cmd() {
  printf "[RUN] "
  printf "%q " "$@"
  printf "\n"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_START_TS="$(date +%s)"
cd "${ROOT_DIR}"

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
      echo "[WARN] conda env 'llava' not found, continue with current python: $(command -v python || echo '<missing>')"
      log "conda_activate_result=missing_env"
    fi
  else
    echo "[WARN] conda not found, continue with current python: $(command -v python || echo '<missing>')"
    log "conda_activate_result=conda_not_found"
  fi
fi

log "conda_env_after=${CONDA_DEFAULT_ENV:-<unset>}"
log "python_after=$(command -v python || echo '<missing>')"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-0}"
TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-0}"

CORPUS_DIR="${CORPUS_DIR:-/public/home/hzh/mRAG/data/image_corpus}"
CORPUS_CACHE_DIR="${CORPUS_CACHE_DIR:-${ROOT_DIR}/results/corpus_index}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-64}"
TOP_K="${TOP_K:-5}"
RETRIEVER_TYPE="${RETRIEVER_TYPE:-clip}"
DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-0}"
MAGICLENS_MODEL_SIZE="${MAGICLENS_MODEL_SIZE:-base}"
MAGICLENS_MODEL_PATH="${MAGICLENS_MODEL_PATH:-${ROOT_DIR}/models/magic_lens_clip_base.pkl}"
MAGICLENS_BATCH_SIZE="${MAGICLENS_BATCH_SIZE:-16}"
MAGICLENS_DISABLE_JIT="${MAGICLENS_DISABLE_JIT:-0}"
MAGICLENS_CLEAR_CACHE_EVERY="${MAGICLENS_CLEAR_CACHE_EVERY:-0}"
START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/github/MRAG-Bench/results/corpus_rag_results.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-${ROOT_DIR}/log/corpus_rag_summary.json}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov}"
LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-single}"
LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
LLAVA_LOAD_4BIT="${LLAVA_LOAD_4BIT:-0}"
LLAVA_LOAD_8BIT="${LLAVA_LOAD_8BIT:-0}"
LLAVA_ALLOW_CPU_OFFLOAD="${LLAVA_ALLOW_CPU_OFFLOAD:-0}"
LLAVA_MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-4096}"
LLAVA_NUM_BEAMS="${LLAVA_NUM_BEAMS:-1}"
LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-}"

if [[ ! -d "${CORPUS_DIR}" ]]; then
  echo "[ERROR] CORPUS_DIR not found: ${CORPUS_DIR}"
  exit 2
fi

if [[ "${DISABLE_MAGICLENS_RERANK}" != "1" && ! -f "${MAGICLENS_MODEL_PATH}" ]]; then
  echo "[ERROR] MagicLens model not found: ${MAGICLENS_MODEL_PATH}"
  exit 2
fi

if [[ -d "${LLAVA_MODEL_PATH}" && ! -f "${LLAVA_MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] LLaVA local model dir missing config.json: ${LLAVA_MODEL_PATH}"
  exit 2
fi

if [[ "${DISABLE_MAGICLENS_RERANK}" == "1" && "${RETRIEVER_TYPE}" != "magiclens" ]]; then
  export JAX_PLATFORMS="cpu"
fi

cmd=(
  python test/benchmark_corpus_rag.py
  --corpus-dir "${CORPUS_DIR}"
  --corpus-cache-dir "${CORPUS_CACHE_DIR}"
  --retriever-type "${RETRIEVER_TYPE}"
  --clip-model-name "${CLIP_MODEL_NAME}"
  --clip-batch-size "${CLIP_BATCH_SIZE}"
  --top-k "${TOP_K}"
  --answers-file "${ANSWERS_FILE}"
  --summary-out "${SUMMARY_OUT}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
  --magiclens-model-path "${MAGICLENS_MODEL_PATH}"
  --magiclens-model-size "${MAGICLENS_MODEL_SIZE}"
  --magiclens-batch-size "${MAGICLENS_BATCH_SIZE}"
  --magiclens-clear-cache-every "${MAGICLENS_CLEAR_CACHE_EVERY}"
  --llava-model-path "${LLAVA_MODEL_PATH}"
  --llava-device-map "${LLAVA_DEVICE_MAP}"
  --llava-attn-implementation "${LLAVA_ATTN_IMPLEMENTATION}"
  --llava-max-new-tokens "${LLAVA_MAX_NEW_TOKENS}"
  --llava-num-beams "${LLAVA_NUM_BEAMS}"
)

if [[ "${LLAVA_LOAD_4BIT}" == "1" ]]; then
  cmd+=(--llava-load-4bit)
fi
if [[ "${LLAVA_LOAD_8BIT}" == "1" ]]; then
  cmd+=(--llava-load-8bit)
fi
if [[ "${LLAVA_ALLOW_CPU_OFFLOAD}" == "1" ]]; then
  cmd+=(--llava-allow-cpu-offload)
fi
if [[ "${DISABLE_MAGICLENS_RERANK}" == "1" ]]; then
  cmd+=(--disable-magiclens-rerank)
fi
if [[ "${MAGICLENS_DISABLE_JIT}" == "1" ]]; then
  cmd+=(--magiclens-disable-jit)
fi
if [[ -n "${LLAVA_GREEDY_JSONL}" && -f "${LLAVA_GREEDY_JSONL}" ]]; then
  cmd+=(--llava-greedy-jsonl "${LLAVA_GREEDY_JSONL}")
fi
if [[ -f "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz" ]]; then
  cmd+=(--bpe-path "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz")
fi

echo "[ENV] CORPUS_DIR=${CORPUS_DIR}"
echo "[ENV] CORPUS_CACHE_DIR=${CORPUS_CACHE_DIR}"
echo "[ENV] CLIP_MODEL_NAME=${CLIP_MODEL_NAME}"
echo "[ENV] CLIP_BATCH_SIZE=${CLIP_BATCH_SIZE}"
echo "[ENV] TOP_K=${TOP_K}"
echo "[ENV] RETRIEVER_TYPE=${RETRIEVER_TYPE}"
echo "[ENV] DISABLE_MAGICLENS_RERANK=${DISABLE_MAGICLENS_RERANK}"
echo "[ENV] MAGICLENS_MODEL_PATH=${MAGICLENS_MODEL_PATH}"
echo "[ENV] MAGICLENS_BATCH_SIZE=${MAGICLENS_BATCH_SIZE}"
echo "[ENV] LLAVA_MODEL_PATH=${LLAVA_MODEL_PATH}"
echo "[ENV] LLAVA_DEVICE_MAP=${LLAVA_DEVICE_MAP}"
echo "[ENV] LLAVA_ATTN_IMPLEMENTATION=${LLAVA_ATTN_IMPLEMENTATION}"
echo "[ENV] LLAVA_LOAD_4BIT=${LLAVA_LOAD_4BIT}"
echo "[ENV] LLAVA_LOAD_8BIT=${LLAVA_LOAD_8BIT}"
echo "[ENV] LLAVA_ALLOW_CPU_OFFLOAD=${LLAVA_ALLOW_CPU_OFFLOAD}"
echo "[ENV] LLAVA_MAX_NEW_TOKENS=${LLAVA_MAX_NEW_TOKENS}"
echo "[ENV] LLAVA_NUM_BEAMS=${LLAVA_NUM_BEAMS}"
echo "[ENV] START_INDEX=${START_INDEX}"
echo "[ENV] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[ENV] ANSWERS_FILE=${ANSWERS_FILE}"
echo "[ENV] SUMMARY_OUT=${SUMMARY_OUT}"

log "step=run_corpus_benchmark"
print_cmd "${cmd[@]}"
"${cmd[@]}"

if [[ -s "${ANSWERS_FILE}" ]]; then
  log "step=score_results"
  print_cmd python github/MRAG-Bench/eval/score.py -i "${ANSWERS_FILE}"
  python github/MRAG-Bench/eval/score.py -i "${ANSWERS_FILE}"
fi

echo "[OK] answers_file=${ANSWERS_FILE}"
echo "[OK] summary_file=${SUMMARY_OUT}"
RUN_END_TS="$(date +%s)"
log "duration_sec=$((RUN_END_TS - RUN_START_TS))"
