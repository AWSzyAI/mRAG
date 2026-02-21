#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
  else
    echo "[WARN] conda not found, continue with current python: $(command -v python || echo '<missing>')"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
SAMPLE_ID="${SAMPLE_ID:-}"
TOP_K="${TOP_K:-3}"
MAX_RAG_IMAGES="${MAX_RAG_IMAGES:-1}"
LLAVA_BEAM_SIZE="${LLAVA_BEAM_SIZE:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAGICLENS_MODEL_SIZE="${MAGICLENS_MODEL_SIZE:-base}"
MRAG_HF_OFFLINE="${MRAG_HF_OFFLINE:-1}"
LLAVA_OOM_FALLBACK_BEAMS="${LLAVA_OOM_FALLBACK_BEAMS:-3,2,1}"
LLAVA_OOM_MAX_NEW_TOKENS="${LLAVA_OOM_MAX_NEW_TOKENS:-24}"
LLAVA_OOM_FINAL_MAX_NEW_TOKENS="${LLAVA_OOM_FINAL_MAX_NEW_TOKENS:-16}"

HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/models/huggingface-mrag/datasets}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-${ROOT_DIR}/models/llava-onevision-qwen2-7b-ov}"
MAGICLENS_MODEL_PATH="${MAGICLENS_MODEL_PATH:-${ROOT_DIR}/models/magic_lens_clip_base.pkl}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/log/one_sample_compare/sample${SAMPLE_INDEX}}"

LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-}"
LLAVA_BEAM_JSONL="${LLAVA_BEAM_JSONL:-}"
SKIP_LLAVA="${SKIP_LLAVA:-0}"

cmd=(
  python test/compare_one_sample.py
  --sample-index "${SAMPLE_INDEX}"
  --hf-cache-dir "${HF_DATASETS_CACHE}"
  --llava-model-path "${LLAVA_MODEL_PATH}"
  --llava-beam-size "${LLAVA_BEAM_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --llava-oom-fallback-beams "${LLAVA_OOM_FALLBACK_BEAMS}"
  --llava-oom-max-new-tokens "${LLAVA_OOM_MAX_NEW_TOKENS}"
  --llava-oom-final-max-new-tokens "${LLAVA_OOM_FINAL_MAX_NEW_TOKENS}"
  --magiclens-model-path "${MAGICLENS_MODEL_PATH}"
  --magiclens-model-size "${MAGICLENS_MODEL_SIZE}"
  --max-rag-images "${MAX_RAG_IMAGES}"
  --top-k "${TOP_K}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SAMPLE_ID}" ]]; then
  cmd+=(--sample-id "${SAMPLE_ID}")
fi

if [[ "${MRAG_HF_OFFLINE}" == "1" ]]; then
  cmd+=(--hf-offline)
fi

if [[ -n "${LLAVA_GREEDY_JSONL}" ]]; then
  cmd+=(--llava-greedy-jsonl "${LLAVA_GREEDY_JSONL}")
fi

if [[ -n "${LLAVA_BEAM_JSONL}" ]]; then
  cmd+=(--llava-beam-jsonl "${LLAVA_BEAM_JSONL}")
fi

if [[ "${SKIP_LLAVA}" == "1" ]]; then
  cmd+=(--skip-llava)
fi

if [[ -f "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz" ]]; then
  cmd+=(--bpe-path "${ROOT_DIR}/models/bpe_simple_vocab_16e6.txt.gz")
fi

echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ENV] JAX_PLATFORMS=${JAX_PLATFORMS}"
echo "[ENV] XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "[ENV] XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR}"
echo "[ENV] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[ENV] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[ENV] MRAG_HF_OFFLINE=${MRAG_HF_OFFLINE}"
echo "[ENV] LLAVA_MODEL_PATH=${LLAVA_MODEL_PATH}"
echo "[ENV] MAGICLENS_MODEL_PATH=${MAGICLENS_MODEL_PATH}"
echo "[ENV] LLAVA_BEAM_SIZE=${LLAVA_BEAM_SIZE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[ENV] LLAVA_OOM_FALLBACK_BEAMS=${LLAVA_OOM_FALLBACK_BEAMS} LLAVA_OOM_MAX_NEW_TOKENS=${LLAVA_OOM_MAX_NEW_TOKENS} LLAVA_OOM_FINAL_MAX_NEW_TOKENS=${LLAVA_OOM_FINAL_MAX_NEW_TOKENS}"
echo "[ENV] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[RUN] ${cmd[*]}"

"${cmd[@]}"

echo "[OK] report_md=${OUTPUT_DIR}/report.md"
echo "[OK] report_json=${OUTPUT_DIR}/report.json"
