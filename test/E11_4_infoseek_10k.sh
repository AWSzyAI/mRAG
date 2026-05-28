#!/usr/bin/env bash
# E11_4 在 InfoSeek 10K 样本上的完整工作流
#
# 使用方式:
#   bash test/E11_4_infoseek_10k.sh
#   bash test/E11_4_infoseek_10k.sh --sample-size 10000 --output-dir log/E11_4_infoseek_10k
#   FORCE_RESAMPLE=1 bash test/E11_4_infoseek_10k.sh

set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E11_4_InfoSeek_10K] $*"; }

# 默认参数
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${DATA_ROOT:-data/infoseek}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10000}"
SPLIT="${SPLIT:-entity_test}"
RANDOM_SEED="${RANDOM_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-log/E11_4_infoseek_10k}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORCE_RESAMPLE="${FORCE_RESAMPLE:-0}"

# Corpus 和模型配置
CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
GEMMA4_LOCAL_DIR="${GEMMA4_LOCAL_DIR:-models/gemma4-e2b}"
GEMMA4_DEVICE="${GEMMA4_DEVICE:-cuda:0}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-models/llava-onevision-qwen2-7b-ov}"

log "==============================================="
log "E11_4 InfoSeek 10K 完整工作流"
log "==============================================="

# ========== 阶段 1: 准备采样 ==========
log "[Phase 1/2] 准备采样..."

SAMPLE_DIR="${OUTPUT_DIR}/sampling"
mkdir -p "${SAMPLE_DIR}"

sample_cache_matches() {
  [[ -s "${SAMPLE_DIR}/sample_indices.json" ]] || return 1
  [[ -s "${SAMPLE_DIR}/sample_metadata.json" ]] || return 1
  [[ -s "${SAMPLE_DIR}/samples.jsonl" ]] || return 1

  python3 - "${SAMPLE_DIR}/sample_indices.json" "${SPLIT}" "${SAMPLE_SIZE}" "${RANDOM_SEED}" <<'PY'
import json
import sys

path, split, sample_size, random_seed = sys.argv[1:]
with open(path, "r", encoding="utf-8") as f:
    meta = json.load(f)

ok = (
    meta.get("split") == split
    and int(meta.get("sample_size", -1)) == int(sample_size)
    and int(meta.get("random_seed", -1)) == int(random_seed)
)
raise SystemExit(0 if ok else 1)
PY
}

if [[ "${FORCE_RESAMPLE}" != "1" ]] && sample_cache_matches; then
  log "✓ 复用已有采样: ${SAMPLE_DIR}"
  log "  如需重采样: FORCE_RESAMPLE=1 bash test/E11_4_infoseek_10k.sh"
else
  if [[ "${FORCE_RESAMPLE}" == "1" ]]; then
    log "FORCE_RESAMPLE=1，重新生成采样..."
  else
    log "未找到匹配采样缓存，生成采样..."
  fi

  python3 scripts/prepare_infoseek_10k_samples.py \
    --data-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --sample-size "${SAMPLE_SIZE}" \
    --random-seed "${RANDOM_SEED}" \
    --output-dir "${SAMPLE_DIR}"
fi

log "✓ 采样完成: ${SAMPLE_DIR}"

# ========== 阶段 2: 运行 Benchmark ==========
log "[Phase 2/2] 运行 E11_4 benchmark..."

BENCHMARK_OUTPUT="${OUTPUT_DIR}/benchmark"
mkdir -p "${BENCHMARK_OUTPUT}"
RUN_LOG="${BENCHMARK_OUTPUT}/run.log"

BENCH_ARGS=(
  --sample-dir "${SAMPLE_DIR}"
  --image-dir "${DATA_ROOT}/images/all"
  --output-dir "${BENCHMARK_OUTPUT}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  BENCH_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

log "Benchmark 日志: ${RUN_LOG}"
log "另开终端可查看: tail -f ${RUN_LOG}"
PYTHONUNBUFFERED=1 python3 test/benchmark_e11_4_infoseek.py "${BENCH_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"

log "✓ Benchmark 完成: ${BENCHMARK_OUTPUT}"

# ========== 生成最终报告 ==========
log "[Final] 生成报告..."

REPORT_FILE="${OUTPUT_DIR}/REPORT.txt"
cat > "${REPORT_FILE}" <<EOF
E11_4 InfoSeek 10K 运行报告
================================================

配置:
  数据集: ${SPLIT}
  采样大小: ${SAMPLE_SIZE}
  随机种子: ${RANDOM_SEED}
  输出目录: ${OUTPUT_DIR}

阶段 1 - 采样:
  采样列表: ${SAMPLE_DIR}/sample_indices.json
  样本元数据: ${SAMPLE_DIR}/sample_metadata.json
  样本 JSONL: ${SAMPLE_DIR}/samples.jsonl

阶段 2 - Benchmark:
  结果 JSONL: ${BENCHMARK_OUTPUT}/e11_4_infoseek_results.jsonl
  摘要: ${BENCHMARK_OUTPUT}/e11_4_infoseek_summary.json

关键文件:
  - 按样本查看: tail -5 ${BENCHMARK_OUTPUT}/e11_4_infoseek_results.jsonl
  - 查看统计: cat ${BENCHMARK_OUTPUT}/e11_4_infoseek_summary.json
  - 样本列表: head -20 ${SAMPLE_DIR}/samples.jsonl

================================================
EOF

cat "${REPORT_FILE}"
log "✓ 报告已保存: ${REPORT_FILE}"

log "==============================================="
log "✓ E11_4 InfoSeek 10K 工作流完成!"
log "==============================================="
