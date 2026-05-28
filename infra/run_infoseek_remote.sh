#!/usr/bin/env bash
set -euo pipefail

# Usage:
# REMOTE_HOST=nnu REMOTE_DIR=/home/user/code/mRAG bash infra/run_infoseek_remote.sh

REMOTE_HOST=${REMOTE_HOST:-nnu}
REMOTE_DIR=${REMOTE_DIR:-/home/user/code/mRAG}
ENV_NAME=${ENV_NAME:-llava}
SAMPLE_SIZE=${SAMPLE_SIZE:-10000}
SPLIT=${SPLIT:-entity_test}
RANDOM_SEED=${RANDOM_SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-log/E11_4_infoseek_10k}

echo "Running benchmark on remote ${REMOTE_HOST}:${REMOTE_DIR} -> outputs ${OUTPUT_DIR}"

echo "Remote: activate conda and run benchmark"
ssh ${REMOTE_HOST} /bin/bash <<'SSH_END'
set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh || true
conda activate ${ENV_NAME} || true
cd ${REMOTE_DIR}
echo "Running benchmark script"
SAMPLE_SIZE=${SAMPLE_SIZE} SPLIT=${SPLIT} RANDOM_SEED=${RANDOM_SEED} OUTPUT_DIR=${OUTPUT_DIR} \
    python test/benchmark_e11_4_infoseek.py --retriever magiclens --dim-generator-type gemma4_local --final-answerer gemma4 --resume-from-existing || true
echo "Remote run finished"
SSH_END

echo "Pulling results back to local log/ directory (rsync)"
rsync -avz --progress ${REMOTE_HOST}:${REMOTE_DIR}/${OUTPUT_DIR}/ ./log/$(basename ${OUTPUT_DIR})/

echo "Done. Pulled to ./log/$(basename ${OUTPUT_DIR})/"
