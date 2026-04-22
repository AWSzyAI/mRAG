#!/usr/bin/env bash
set -euo pipefail

cd /public/home/hzh/mRAG

export CORPUS_DIR="${CORPUS_DIR:-data/image_corpus}"
# Keep blocking off for throughput; set to 1 only for debugging CUDA stack traces.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

mkdir -p log/E8_final log/E8_full

echo "[step-1] running 5-sample E8_final sanity check..."
python test/pipeline_multi_dim_rag.py \
  --dataset-name uclanlp/MRAG-Bench \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type gemma4_local \
  --gemma4-local-dir models/gemma4-e2b \
  --gemma4-model-id google/gemma-4-E2B-it \
  --gemma4-device cuda:0 \
  --gemma4-dim-rationale \
  --n-dims 5 \
  --dim-top-k 5 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --final-answerer llava \
  --describe-final-images \
  --magiclens-platform cpu \
  --max-samples 5 \
  --llava-max-images 1 \
  --llava-max-new-tokens 64 \
  --answers-file log/E8_final/e8_final.jsonl \
  --summary-out log/E8_final/e8_final_summary.json \
  --save-dimensions-jsonl log/E8_final/e8_final_dims.jsonl \
  --trace-jsonl log/E8_final/e8_final_trace.jsonl > log/E8_final/E8_final.log 2>&1

echo "[step-2] validating E8_final outputs..."
python - <<'PY'
import json
from pathlib import Path
import sys

path = Path("log/E8_final/e8_final.jsonl")
if not path.is_file():
    print("ERROR: missing e8_final.jsonl")
    sys.exit(1)

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if s:
            rows.append(json.loads(s))

if not rows:
    print("ERROR: e8_final.jsonl has no rows")
    sys.exit(1)

bad = []
for i, r in enumerate(rows):
    pred = str(r.get("meta_pred_choice", "")).strip()
    err = r.get("meta_llava_error")
    out = str(r.get("output", "")).strip()
    if pred in ("", "N/A") or err not in (None, "") or out == "":
        bad.append((i, pred, err, out))

if bad:
    print("ERROR: sanity check failed, not starting full run")
    for b in bad[:5]:
        print("bad_row", b)
    sys.exit(2)

print(f"OK: {len(rows)} rows passed sanity checks")
PY

echo "[step-3] launching full-dataset E8_full in background..."
nohup python test/pipeline_multi_dim_rag.py \
  --dataset-name uclanlp/MRAG-Bench \
  --corpus-dir "${CORPUS_DIR}" \
  --dim-generator-type gemma4_local \
  --gemma4-local-dir models/gemma4-e2b \
  --gemma4-model-id google/gemma-4-E2B-it \
  --gemma4-device cuda:0 \
  --gemma4-dim-rationale \
  --n-dims 5 \
  --dim-top-k 5 \
  --final-top-k 5 \
  --fusion-strategy rrf \
  --final-answerer llava \
  --describe-final-images \
  --magiclens-platform cpu \
  --resume-from-existing \
  --llava-max-images 1 \
  --llava-max-new-tokens 64 \
  --answers-file log/E8_full/e8_full.jsonl \
  --summary-out log/E8_full/e8_full_summary.json \
  --save-dimensions-jsonl log/E8_full/e8_full_dims.jsonl \
  --trace-jsonl log/E8_full/e8_full_trace.jsonl > log/E8_full/E8_full.log 2>&1 &

echo $! > log/E8_full/E8_full.pid
echo "started pid=$(cat log/E8_full/E8_full.pid)"
echo "tail log: tail -f log/E8_full/E8_full.log"
