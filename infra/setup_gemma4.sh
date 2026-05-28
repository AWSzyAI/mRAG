#!/usr/bin/env bash
set -euo pipefail

# infra/setup_gemma4.sh
# Download Gemma4 and optional answerer model to the local `models/` directory
# Usage: HF_TOKEN=<token> ./infra/setup_gemma4.sh [--no-llava]

SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SELF_DIR/.." && pwd)
MODELS_DIR="$ROOT_DIR/models"

NO_LLAVA=0
for arg in "$@"; do
  case "$arg" in
    --no-llava) NO_LLAVA=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "Project root: $ROOT_DIR"
mkdir -p "$MODELS_DIR"

check_python(){
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found; please install Python 3.8+" >&2
    exit 2
  fi
}

check_python

HF_TOKEN=${HF_TOKEN:-}
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not set. If the model is gated, download will fail. Export HF_TOKEN before running."
fi

echo "Checking available disk space on $(pwd)"
df -h .

echo "Downloading Gemma4 (google/gemma-4-E2B-it) into $MODELS_DIR/gemma4-e2b"
python3 - <<PY
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN')
try:
    snapshot_download(repo_id='google/gemma-4-E2B-it', local_dir=os.path.join(os.getcwd(), 'models', 'gemma4-e2b'), token=token)
    print('Gemma4 download completed')
except Exception as e:
    print('Gemma4 download failed:', e)
    raise
PY

if [ "$NO_LLAVA" -eq 0 ]; then
  echo "Downloading example answerer model (llava-like) to models/llava-answerer"
  python3 - <<PY
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN')
try:
    snapshot_download(repo_id='lmms-lab/llava-onevision-qwen2-7b-ov', local_dir=os.path.join(os.getcwd(), 'models', 'llava-onevision-qwen2-7b-ov'), token=token)
    print('LLaVA-like model download completed')
except Exception as e:
    print('LLaVA download failed (continuing):', e)
PY
else
  echo "Skipping LLaVA download (--no-llava)"
fi

echo "Done. Please verify models/ contains gemma4-e2b and any answerer models."
echo "Set GEMMA4_LOCAL_DIR=models/gemma4-e2b and GEMMA4_HF_TOKEN in the environment or .env before running experiments."
