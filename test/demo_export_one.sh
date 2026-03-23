#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "llava" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | awk '{print $1}' | grep -qx "llava"; then
      conda activate llava
    fi
  fi
fi

PIPELINE="${PIPELINE:-${1:-E3}}"
if [[ "${PIPELINE}" == "-h" || "${PIPELINE}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  PIPELINE=E3 SAMPLE_INDEX=0 bash test/demo_export_one.sh
  PIPELINE=E7 SAMPLE_ID=128 bash test/demo_export_one.sh

Convenience wrappers:
  bash test/demo_E4.sh
  bash test/demo_E2.sh
  bash test/demo_E3.sh
  bash test/demo_E7.sh
EOF
  exit 0
fi
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
SAMPLE_ID="${SAMPLE_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/log/demo_review/${PIPELINE}/sample${SAMPLE_INDEX}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

cmd=(
  python test/export_demo_bundle.py
  --pipeline "${PIPELINE}"
  --sample-index "${SAMPLE_INDEX}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SAMPLE_ID}" ]]; then
  cmd+=(--sample-id "${SAMPLE_ID}")
fi
if [[ -n "${HF_CACHE_DIR}" ]]; then
  cmd+=(--hf-cache-dir "${HF_CACHE_DIR}")
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"
echo "[OK] demo_bundle=${OUTPUT_DIR}"
