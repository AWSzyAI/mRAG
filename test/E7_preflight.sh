#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E7-PREFLIGHT] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

STATUS=0

ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; STATUS=1; }

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_file() {
  local path="$1"
  local label="$2"
  if [[ -f "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

log "root_dir=${ROOT_DIR}"
log "cwd=$(pwd)"
log "conda_env_before=${CONDA_DEFAULT_ENV:-<unset>}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk '{print $1}' | grep -qx "llava"; then
      conda activate llava
      ok "conda env activated: llava"
    else
      fail "conda env 'llava' not found"
    fi
  else
    fail "conda not found"
  fi
else
  ok "already in conda env: llava"
fi

log "conda_env_after=${CONDA_DEFAULT_ENV:-<unset>}"
log "python=$(command -v python || echo '<missing>')"

CORPUS_DIR="${CORPUS_DIR:-/public/home/hzh/mRAG/data/image_corpus}"
CORPUS_CACHE_DIR="${CORPUS_CACHE_DIR:-${ROOT_DIR}/results/corpus_index}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov}"
MAGICLENS_MODEL_PATH="${MAGICLENS_MODEL_PATH:-${ROOT_DIR}/models/magic_lens_clip_base.pkl}"
BPE_PATH="${BPE_PATH:-${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz}"
E3_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/results/e3_clip_corpus_rag_results.jsonl}"

check_dir "${CORPUS_DIR}" "corpus dir"
check_path "${CORPUS_CACHE_DIR}" "corpus cache dir"
check_dir "${LLAVA_MODEL_PATH}" "llava model dir"
check_file "${LLAVA_MODEL_PATH}/config.json" "llava config"
check_file "${MAGICLENS_MODEL_PATH}" "magiclens checkpoint"
check_file "${BPE_PATH}" "magiclens bpe"

if [[ -f "${E3_JSONL}" ]]; then
  ok "reference E3 jsonl: ${E3_JSONL}"
else
  warn "reference E3 jsonl not found: ${E3_JSONL} (not required for E7 run)"
fi

python - <<'PY' || STATUS=1
import importlib
import os
import sys
mods = [
    "torch",
    "clip",
    "jax",
    "jaxlib",
    "datasets",
    "transformers",
    "shortuuid",
    "accelerate",
    "einops",
    "requests",
    "PIL",
    "numpy",
]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
        print(f"[OK] python module: {name}")
    except Exception as e:
        missing.append(name)
        print(f"[FAIL] python module missing: {name} ({type(e).__name__}: {e})")

if missing:
    sys.exit(1)

import torch
print(f"[OK] torch.cuda.is_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] torch.cuda.device_count={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        try:
            print(f"[OK] torch.cuda.device[{i}]={torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"[WARN] torch device name failed for {i}: {e}")
else:
    print("[WARN] torch cannot see CUDA; E7 may run very slowly or fail depending on server config")

try:
    import jax
    print(f"[OK] jax.default_backend={jax.default_backend()}")
    print(f"[OK] jax.devices={jax.devices()}")
except Exception as e:
    print(f"[FAIL] jax backend check failed: {type(e).__name__}: {e}")
    sys.exit(1)
PY

echo
if [[ "${STATUS}" -eq 0 ]]; then
  ok "E7 preflight passed. Recommended run command:"
  echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0} LLAVA_DEVICE_MAP=single JAX_CUDA_REQUIRED=1 TORCH_CUDA_REQUIRED=1 bash test/E7.sh"
else
  fail "E7 preflight failed. Fix the [FAIL] items above before running."
fi

exit "${STATUS}"
