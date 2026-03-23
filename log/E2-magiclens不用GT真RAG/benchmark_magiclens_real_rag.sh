#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export USE_GT="${USE_GT:-0}"
export ANSWERS_FILE="${ANSWERS_FILE:-${ROOT_DIR}/github/MRAG-Bench/magiclens_rerank_llava_retrieved_rag_results.jsonl}"
export SUMMARY_OUT="${SUMMARY_OUT:-${ROOT_DIR}/log/magiclens_rerank_llava_retrieved_rag_summary.json}"
export LLAVA_GREEDY_JSONL="${LLAVA_GREEDY_JSONL:-${ROOT_DIR}/github/MRAG-Bench/llava_one_vision_retrieved_rag_results.jsonl}"

exec bash "${ROOT_DIR}/test/benchmark_magiclens.sh" "$@"
