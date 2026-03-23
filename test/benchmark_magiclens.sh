#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [BENCH] $*"; }
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
log "user=${USER:-<unknown>} host=$(hostname)"
log "shell=${SHELL:-<unknown>} pid=$$"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-0}"
TORCH_CUDA_REQUIRED="${TORCH_CUDA_REQUIRED:-0}"

log "step=torch_cuda_probe"
if ! python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[CHECK-FAIL] torch import failed: {e}", file=sys.stderr)
    sys.exit(2)

available = bool(torch.cuda.is_available())
count = torch.cuda.device_count() if available else 0
print(f"[CHECK] torch_cuda_available={available}, torch_cuda_device_count={count}")
sys.exit(0 if available and count > 0 else 1)
PY
then
  if [[ "${TORCH_CUDA_REQUIRED}" == "1" ]]; then
    echo "[ERROR] PyTorch CUDA backend is unavailable and TORCH_CUDA_REQUIRED=1."
    exit 4
  fi
  echo "[WARN] PyTorch CUDA backend unavailable; LLaVA may run on CPU and be very slow."
fi

if [[ "${JAX_PLATFORMS}" == *cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
  log "step=nvidia_smi_probe"
  if ! nvidia-smi -L >/dev/null 2>&1; then
    if [[ "${JAX_CUDA_REQUIRED}" == "1" ]]; then
      echo "[ERROR] nvidia-smi cannot detect a visible GPU and JAX_CUDA_REQUIRED=1."
      exit 3
    fi
    echo "[WARN] nvidia-smi cannot detect a visible GPU; forcing JAX_PLATFORMS=cpu."
    export JAX_PLATFORMS=cpu
  fi
fi

if [[ "${JAX_PLATFORMS}" == *cuda* ]]; then
  log "step=jax_cuda_probe"
  if ! python - <<'PY'
import sys
try:
    import jax
    devs = jax.devices("gpu")
    if not devs:
        raise RuntimeError("no JAX gpu devices found")
except Exception as e:
    print(f"[CHECK-FAIL] {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
print(f"[CHECK] jax_gpu_device_count={len(devs)}")
PY
  then
    if [[ "${JAX_CUDA_REQUIRED}" == "1" ]]; then
      echo "[ERROR] JAX CUDA backend is unavailable and JAX_CUDA_REQUIRED=1."
      exit 3
    fi
    echo "[WARN] JAX CUDA backend unavailable, fallback to CPU (set JAX_CUDA_REQUIRED=1 to fail-fast)."
    export JAX_PLATFORMS=cpu
  fi
fi

MAGICLENS_MODEL_SIZE="${MAGICLENS_MODEL_SIZE:-base}"
MAGICLENS_MODEL_PATH="${MAGICLENS_MODEL_PATH:-${ROOT_DIR}/models/magic_lens_clip_base.pkl}"
# Baseline alignment: 1 query image + 5 retrieved images (except Incomplete).
BASELINE_RAG_IMAGES="${BASELINE_RAG_IMAGES:-5}"
MAX_RAG_IMAGES="${MAX_RAG_IMAGES:-${BASELINE_RAG_IMAGES}}"
MAGICLENS_DISABLE_JIT="${MAGICLENS_DISABLE_JIT:-0}"
MAGICLENS_CLEAR_CACHE_EVERY="${MAGICLENS_CLEAR_CACHE_EVERY:-0}"
START_INDEX="${START_INDEX:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/github/MRAG-Bench/magiclens_rerank_llava_results.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-${ROOT_DIR}/log/magiclens_rerank_llava_summary.json}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov}"
LLAVA_DEVICE_MAP="${LLAVA_DEVICE_MAP:-auto}"
LLAVA_ATTN_IMPLEMENTATION="${LLAVA_ATTN_IMPLEMENTATION:-sdpa}"
LLAVA_LOAD_4BIT="${LLAVA_LOAD_4BIT:-0}"
LLAVA_LOAD_8BIT="${LLAVA_LOAD_8BIT:-0}"
LLAVA_ALLOW_CPU_OFFLOAD="${LLAVA_ALLOW_CPU_OFFLOAD:-0}"
LLAVA_MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-4096}"
LLAVA_NUM_BEAMS="${LLAVA_NUM_BEAMS:-1}"
DISABLE_MAGICLENS_RERANK="${DISABLE_MAGICLENS_RERANK:-0}"
USE_GT="${USE_GT:-1}"
LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/llava_one_vision_gt_rag_results.jsonl}"

log "step=resolve_inputs"
if [[ ! -f "${MAGICLENS_MODEL_PATH}" ]]; then
  echo "[ERROR] MagicLens model not found: ${MAGICLENS_MODEL_PATH}"
  exit 2
fi
if [[ -d "${LLAVA_MODEL_PATH}" ]]; then
  if [[ ! -f "${LLAVA_MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] LLaVA local model dir missing config.json: ${LLAVA_MODEL_PATH}"
    exit 2
  fi
else
  log "LLAVA_MODEL_PATH is not a local directory: ${LLAVA_MODEL_PATH}"
  log "LLaVA loader may access HuggingFace hub if local cache is missing."
fi

if [[ "${JAX_PLATFORMS}" == "cpu" ]]; then
  if [[ "${MAGICLENS_DISABLE_JIT}" == "0" ]]; then
    MAGICLENS_DISABLE_JIT=1
  fi
  if [[ "${MAGICLENS_CLEAR_CACHE_EVERY}" == "0" ]]; then
    MAGICLENS_CLEAR_CACHE_EVERY=200
  fi
fi

if [[ "${LLAVA_LOAD_4BIT}" == "1" && "${LLAVA_LOAD_8BIT}" == "1" ]]; then
  echo "[ERROR] LLAVA_LOAD_4BIT and LLAVA_LOAD_8BIT cannot both be 1."
  exit 2
fi

if [[ "${LLAVA_LOAD_4BIT}" == "1" || "${LLAVA_LOAD_8BIT}" == "1" ]]; then
  if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("bitsandbytes") is not None else 1)
PY
  then
    echo "[ERROR] LLAVA_LOAD_4BIT/LLAVA_LOAD_8BIT requires bitsandbytes in current env."
    echo "[HINT] Install in llava env: pip install 'bitsandbytes>=0.43.1'"
    echo "[HINT] Or disable quantization: LLAVA_LOAD_4BIT=0 LLAVA_LOAD_8BIT=0"
    exit 2
  fi
fi

cmd=(
  python test/benchmark_magiclens.py
  --answers-file "${ANSWERS_FILE}"
  --summary-out "${SUMMARY_OUT}"
  --magiclens-model-path "${MAGICLENS_MODEL_PATH}"
  --magiclens-model-size "${MAGICLENS_MODEL_SIZE}"
  --max-rag-images "${MAX_RAG_IMAGES}"
  --magiclens-clear-cache-every "${MAGICLENS_CLEAR_CACHE_EVERY}"
  --start-index "${START_INDEX}"
  --max-samples "${MAX_SAMPLES}"
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
if [[ "${MAGICLENS_DISABLE_JIT}" == "1" ]]; then
  cmd+=(--magiclens-disable-jit)
fi

if [[ "${DISABLE_MAGICLENS_RERANK}" == "1" ]]; then
  cmd+=(--disable-magiclens-rerank)
fi
if [[ "${USE_GT}" == "1" ]]; then
  cmd+=(--use-gt)
else
  cmd+=(--no-use-gt)
fi

if [[ -f "${LLAVA_GREEDY_JSONL}" ]]; then
  cmd+=(--llava-greedy-jsonl "${LLAVA_GREEDY_JSONL}")
fi

if [[ -f "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz" ]]; then
  cmd+=(--bpe-path "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz")
fi

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
echo "[ENV] DISABLE_MAGICLENS_RERANK=${DISABLE_MAGICLENS_RERANK}"
echo "[ENV] USE_GT=${USE_GT}"
echo "[ENV] BASELINE_RAG_IMAGES=${BASELINE_RAG_IMAGES}"
echo "[ENV] MAX_RAG_IMAGES=${MAX_RAG_IMAGES}"
echo "[ENV] START_INDEX=${START_INDEX} MAX_SAMPLES=${MAX_SAMPLES}"
echo "[ENV] ANSWERS_FILE=${ANSWERS_FILE}"
echo "[ENV] SUMMARY_OUT=${SUMMARY_OUT}"
echo "[ENV] LLAVA_GREEDY_JSONL=${LLAVA_GREEDY_JSONL}"

if [[ "${XLA_PYTHON_CLIENT_PREALLOCATE}" == "true" && -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
  if awk "BEGIN {exit !(${XLA_PYTHON_CLIENT_MEM_FRACTION} > 0.30)}"; then
    echo "[WARN] High XLA_PYTHON_CLIENT_MEM_FRACTION may cause LLaVA CPU offload and severe slowdown."
    echo "[WARN] Try: XLA_PYTHON_CLIENT_PREALLOCATE=false (or MEM_FRACTION <= 0.20)."
  fi
fi

log "step=run_magiclens_benchmark"
print_cmd "${cmd[@]}"

"${cmd[@]}"

if [[ -s "${ANSWERS_FILE}" ]]; then
  log "step=score_results"
  print_cmd python eval/score.py -i "${ANSWERS_FILE}"
  (
    cd "${ROOT_DIR}/github/MRAG-Bench"
    python eval/score.py -i "${ANSWERS_FILE}"
  )
fi

echo "[OK] answers_file=${ANSWERS_FILE}"
echo "[OK] summary_file=${SUMMARY_OUT}"
RUN_END_TS="$(date +%s)"
log "duration_sec=$((RUN_END_TS - RUN_START_TS))"
