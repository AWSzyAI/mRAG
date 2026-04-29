#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E4] $*"; }
print_cmd() {
  printf "[RUN] "
  printf "%q " "$@"
  printf "\n"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MRAG_DIR="${ROOT_DIR}/github/MRAG-Bench"
RUN_START_TS="$(date +%s)"

log "exp=E4 LLaVA+No-RAG baseline"
log "script=${BASH_SOURCE[0]}"
log "cwd=$(pwd)"
log "root_dir=${ROOT_DIR}"
log "mrag_dir=${MRAG_DIR}"
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
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export EXP_DIR="${EXP_DIR:-${ROOT_DIR}/log/E4}"
mkdir -p "${EXP_DIR}"

# Keep the same ENV print shape as MagicLens scripts for easy comparison.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-0}"
TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-0}"
MAGICLENS_MODEL_PATH="${MAGICLENS_MODEL_PATH:-${ROOT_DIR}/models/magic_lens_clip_base.pkl}"
MAGICLENS_MODEL_SIZE="${MAGICLENS_MODEL_SIZE:-base}"
MAGICLENS_DISABLE_JIT="${MAGICLENS_DISABLE_JIT:-1}"
MAGICLENS_CLEAR_CACHE_EVERY="${MAGICLENS_CLEAR_CACHE_EVERY:-200}"

LLAVA_MODEL_PATH="${MRAG_MODEL_LOCAL_DIR:-${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov}"
LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-auto}"
LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
LLAVA_LOAD_4BIT="${LLAVA_LOAD_4BIT:-0}"
LLAVA_LOAD_8BIT="${LLAVA_LOAD_8BIT:-0}"
LLAVA_ALLOW_CPU_OFFLOAD="${LLAVA_ALLOW_CPU_OFFLOAD:-0}"
LLAVA_MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-64}"
LLAVA_NUM_BEAMS="${LLAVA_NUM_BEAMS:-1}"

if [[ ! -f "${LLAVA_MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] invalid local model dir: ${LLAVA_MODEL_PATH}"
  echo "[ERROR] missing file: ${LLAVA_MODEL_PATH}/config.json"
  exit 2
fi

OUT="${OUT:-results/e4_llava_no_rag_results.jsonl}"

echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ENV] JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "[ENV] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "[ENV] XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR}"
echo "[ENV] XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-<unset>}"
echo "[ENV] JAX_CUDA_REQUIRED=${JAX_CUDA_REQUIRED}"
echo "[ENV] TORCH_CUDA_REQUIRED=${TORCH_CUDA_REQUIRED}"
echo "[ENV] MAGICLENS_MODEL_PATH=${MAGICLENS_MODEL_PATH}"
echo "[ENV] MAGICLENS_MODEL_SIZE=${MAGICLENS_MODEL_SIZE}"
echo "[ENV] MAGICLENS_DISABLE_JIT=${MAGICLENS_DISABLE_JIT}"
echo "[ENV] MAGICLENS_CLEAR_CACHE_EVERY=${MAGICLENS_CLEAR_CACHE_EVERY}"
echo "[ENV] LLAVA_MODEL_PATH=${LLAVA_MODEL_PATH}"
echo "[ENV] LLAVA_DEVICE_MAP=${LLAVA_DEVICE_MAP}"
echo "[ENV] LLAVA_ATTN_IMPLEMENTATION=${LLAVA_ATTN_IMPLEMENTATION}"
echo "[ENV] LLAVA_LOAD_4BIT=${LLAVA_LOAD_4BIT}"
echo "[ENV] LLAVA_LOAD_8BIT=${LLAVA_LOAD_8BIT}"
echo "[ENV] LLAVA_ALLOW_CPU_OFFLOAD=${LLAVA_ALLOW_CPU_OFFLOAD}"
echo "[ENV] LLAVA_MAX_NEW_TOKENS=${LLAVA_MAX_NEW_TOKENS}"
echo "[ENV] LLAVA_NUM_BEAMS=${LLAVA_NUM_BEAMS}"
echo "[ENV] DISABLE_MAGICLENS_RERANK=<unused in E4>"
echo "[ENV] USE_GT=<unused in E4>"
echo "[ENV] ANSWERS_FILE=${MRAG_DIR}/${OUT}"
echo "[ENV] SUMMARY_OUT=<unused in E4>"
echo "[ENV] LLAVA_GREEDY_JSONL=<unused in E4>"

echo "[PARAM] use_rag=False"
echo "[PARAM] use_retrieved_examples=False"
echo "[PARAM] model=llava_qwen"
echo "[PARAM] model_path=${LLAVA_MODEL_PATH}"

cd "${MRAG_DIR}"
mkdir -p results

cmd=(
  python eval/models/llava_one_vision.py
  --model-path "${LLAVA_MODEL_PATH}"
  --attn-implementation "${LLAVA_ATTN_IMPLEMENTATION}"
  --device-map "${LLAVA_DEVICE_MAP}"
  --num_beams "${LLAVA_NUM_BEAMS}"
  --max-new-tokens "${LLAVA_MAX_NEW_TOKENS}"
  --answers-file "${OUT}"
  --use_rag False
  --use_retrieved_examples False
)

log "step=run_llava_eval"
print_cmd "${cmd[@]}"
"${cmd[@]}"

log "step=score_results"
print_cmd python eval/score.py -i "${OUT}"
python eval/score.py -i "${OUT}"

echo "[OK] answers_file=${MRAG_DIR}/${OUT}"
RUN_END_TS="$(date +%s)"
log "duration_sec=$((RUN_END_TS - RUN_START_TS))"

bash "${SCRIPT_DIR}/archive_exp_outputs.sh" "${EXP_DIR}" \
  "${MRAG_DIR}/${OUT}" \
  "${ROOT_DIR}/E4.log"
