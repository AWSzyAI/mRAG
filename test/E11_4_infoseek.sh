#!/usr/bin/env bash
# E11_4 InfoSeek run: 100 open-ended samples with 4-dim query rewrite.
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [E11_4_INFOSEEK] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-log/E11/E11_4_infoseek_100}"
mkdir -p "${OUT_DIR}"

log "running 100 InfoSeek Entity samples with n_dims=4"
python3 test/infoseek_open_ended_multidim.py \
  --data-root /mnt/d/mRAG/data/infoseek \
  --split entity_test \
  --max-samples 100 \
  --n-dims 4 \
  --backend auto \
  --output-jsonl "${OUT_DIR}/e11_4_infoseek_rewrite.jsonl" \
  --summary-json "${OUT_DIR}/e11_4_infoseek_rewrite_summary.json"

log "done"
log "summary: ${OUT_DIR}/e11_4_infoseek_rewrite_summary.json"