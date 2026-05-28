#!/usr/bin/env bash
# Pack the working AB conda env and restore it on nnu.
#
# Run this from a machine that can ssh to both AB and nnu:
#   bash infra/migrate_ab_llava_env_to_nnu.sh
#
# Defaults are intentionally conservative: the nnu env is unpacked into a new
# path instead of overwriting the existing llava env.

set -euo pipefail

SRC_HOST="${SRC_HOST:-AB}"
DST_HOST="${DST_HOST:-nnu}"
ENV_NAME="${ENV_NAME:-llava}"
SRC_ENV_PATH="${SRC_ENV_PATH:-/public/home/hzh/.conda/envs/${ENV_NAME}}"
DST_ENV_PATH="${DST_ENV_PATH:-/home/user/env/envs/${ENV_NAME}_from_ab}"
REMOTE_TMP="${REMOTE_TMP:-/tmp/mrag_env_migration}"
LOCAL_TMP="${LOCAL_TMP:-${TMPDIR:-/tmp}/mrag_env_migration}"
ARCHIVE_NAME="${ARCHIVE_NAME:-${ENV_NAME}_from_ab.tar.gz}"
PROJECT_SRC_DIR="${PROJECT_SRC_DIR:-/public/home/hzh/mRAG}"
PROJECT_DST_DIR="${PROJECT_DST_DIR:-/home/user/code/mRAG}"
SYNC_CODE_DEPS="${SYNC_CODE_DEPS:-1}"

log() {
  printf '[migrate-ab-env] %s\n' "$*"
}

quote() {
  printf '%q' "$1"
}

mkdir -p "${LOCAL_TMP}"

log "source: ${SRC_HOST}:${SRC_ENV_PATH}"
log "target: ${DST_HOST}:${DST_ENV_PATH}"

log "checking source env and conda-pack on ${SRC_HOST}"
ssh "${SRC_HOST}" bash -s -- "${SRC_ENV_PATH}" "${REMOTE_TMP}" <<'SRC_CHECK'
set -euo pipefail
src_env="$1"
remote_tmp="$2"
if [ ! -d "${src_env}" ]; then
  echo "source env not found: ${src_env}" >&2
  exit 2
fi
mkdir -p "${remote_tmp}"
"${src_env}/bin/python" - <<'PY' || {
  echo "conda-pack is not installed in the source env." >&2
  echo "Install it on AB first, for example:" >&2
  echo "  ${src_env}/bin/python -m pip install conda-pack" >&2
  exit 4
}
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("conda_pack") else 1)
PY
SRC_CHECK

log "packing env on ${SRC_HOST}"
ssh "${SRC_HOST}" bash -s -- "${SRC_ENV_PATH}" "${REMOTE_TMP}/${ARCHIVE_NAME}" <<'SRC_PACK'
set -euo pipefail
src_env="$1"
archive="$2"
rm -f "${archive}"
"${src_env}/bin/conda-pack" -p "${src_env}" -o "${archive}" --force
ls -lh "${archive}"
SRC_PACK

log "downloading archive to ${LOCAL_TMP}/${ARCHIVE_NAME}"
rsync -azP "${SRC_HOST}:${REMOTE_TMP}/${ARCHIVE_NAME}" "${LOCAL_TMP}/${ARCHIVE_NAME}"

log "uploading archive to ${DST_HOST}:${REMOTE_TMP}/${ARCHIVE_NAME}"
ssh "${DST_HOST}" "mkdir -p $(quote "${REMOTE_TMP}")"
rsync -azP "${LOCAL_TMP}/${ARCHIVE_NAME}" "${DST_HOST}:${REMOTE_TMP}/${ARCHIVE_NAME}"

log "unpacking env on ${DST_HOST}"
ssh "${DST_HOST}" bash -s -- "${DST_ENV_PATH}" "${REMOTE_TMP}/${ARCHIVE_NAME}" <<'DST_UNPACK'
set -euo pipefail
dst_env="$1"
archive="$2"
if [ -e "${dst_env}" ]; then
  echo "target env already exists: ${dst_env}" >&2
  echo "set DST_ENV_PATH to a new path, or remove/rename the old env yourself" >&2
  exit 3
fi
mkdir -p "${dst_env}"
tar -xzf "${archive}" -C "${dst_env}"
"${dst_env}/bin/conda-unpack"
DST_UNPACK

if [ "${SYNC_CODE_DEPS}" = "1" ]; then
  log "syncing editable/source deps likely needed by the env"
  mkdir -p "${LOCAL_TMP}/code_deps"
  for rel in github/LLaVA-NeXT github/magiclens github/scenic; do
    mkdir -p "${LOCAL_TMP}/code_deps/${rel}"
    rsync -az --delete "${SRC_HOST}:${PROJECT_SRC_DIR}/${rel}/" "${LOCAL_TMP}/code_deps/${rel}/"
    ssh "${DST_HOST}" "mkdir -p $(quote "${PROJECT_DST_DIR}/${rel}")"
    rsync -az --delete "${LOCAL_TMP}/code_deps/${rel}/" "${DST_HOST}:${PROJECT_DST_DIR}/${rel}/"
  done
fi

log "running smoke test on ${DST_HOST}"
ssh "${DST_HOST}" bash -s -- "${DST_ENV_PATH}" "${PROJECT_DST_DIR}" <<'DST_TEST'
set -euo pipefail
dst_env="$1"
project_dir="$2"
cd "${project_dir}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${project_dir}/github/LLaVA-NeXT:${project_dir}/github/magiclens:${project_dir}/github/scenic:${PYTHONPATH:-}"
"${dst_env}/bin/python" - <<'PY'
import importlib.util
mods = ["torch", "transformers", "jax", "flax", "PIL", "llava", "inference", "scenic"]
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {spec.origin if spec else 'MISSING'}")
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_count:", torch.cuda.device_count())
PY
DST_TEST

log "done"
log "use this env on nnu with:"
log "  export PYTHONNOUSERSITE=1"
log "  export PYTHONPATH=${PROJECT_DST_DIR}/github/LLaVA-NeXT:${PROJECT_DST_DIR}/github/magiclens:${PROJECT_DST_DIR}/github/scenic:\$PYTHONPATH"
log "  ${DST_ENV_PATH}/bin/python test/benchmark_e11_4_infoseek.py --help"
