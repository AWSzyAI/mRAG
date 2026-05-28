#!/usr/bin/env bash
# Pack a local conda environment and restore it on an offline remote host.
#
# This is intentionally driven by environment variables so sync/Makefile can
# provide host-specific defaults from sync/.sync_ssh.

set -euo pipefail

SYNC_HOST="${SYNC_HOST:-nnu}"
REMOTE_DIR="${REMOTE_DIR:-/home/user/code/mRAG}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-/home/user/env/envs/llava}"
LOCAL_CONDA_ENV="${LOCAL_CONDA_ENV:-llava}"
LOCAL_CONDA_ENV_PATH="${LOCAL_CONDA_ENV_PATH:-}"
ARCHIVE_NAME="${ARCHIVE_NAME:-mrag_llava_conda_pack.tar.gz}"
LOCAL_TMP="${LOCAL_TMP:-${TMPDIR:-/tmp}/mrag_env_sync}"
REMOTE_TMP="${REMOTE_TMP:-/tmp/mrag_env_sync}"
ENV_REPLACE="${ENV_REPLACE:-0}"
ENV_SYNC_DEPS="${ENV_SYNC_DEPS:-1}"
ENV_CODE_DEPS="${ENV_CODE_DEPS:-github/LLaVA-NeXT github/magiclens github/scenic}"
ENV_SMOKE_IMPORTS="${ENV_SMOKE_IMPORTS:-torch transformers jax flax PIL llava inference scenic}"
ENV_SMOKE_ONLY="${ENV_SMOKE_ONLY:-0}"
ENV_PACK_IGNORE_MISSING="${ENV_PACK_IGNORE_MISSING:-0}"

log() {
  printf '[sync-conda-env] %s\n' "$*"
}

quote() {
  printf '%q' "$1"
}

remote_smoke_test() {
  log "running remote smoke test"
  local remote_pythonpath=""
  for rel in ${ENV_CODE_DEPS}; do
    remote_pythonpath="${REMOTE_DIR%/}/${rel}:${remote_pythonpath}"
  done

  ssh "${SYNC_HOST}" bash -s -- \
    "${REMOTE_CONDA_ENV}" \
    "${REMOTE_DIR}" \
    "${remote_pythonpath}" \
    "${ENV_SMOKE_IMPORTS}" <<'REMOTE_TEST'
set -euo pipefail
env_path="$1"
project_dir="$2"
pythonpath="$3"
imports="$4"
cd "${project_dir}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${pythonpath}${PYTHONPATH:-}"
"${env_path}/bin/python" - "${imports}" <<'PY'
import importlib.util
import sys

mods = sys.argv[1].split()
missing = []
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {spec.origin if spec else 'MISSING'}")
    if spec is None:
        missing.append(mod)

try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch check failed:", repr(exc))
    missing.append("torch-runtime")

raise SystemExit(1 if missing else 0)
PY
REMOTE_TEST
}

if [ "${ENV_SMOKE_ONLY}" = "1" ]; then
  remote_smoke_test
  exit 0
fi

find_local_env_path() {
  if [ -n "${LOCAL_CONDA_ENV_PATH}" ]; then
    printf '%s\n' "${LOCAL_CONDA_ENV_PATH}"
    return
  fi

  if [ -d "${LOCAL_CONDA_ENV}" ] && [ -x "${LOCAL_CONDA_ENV}/bin/python" ]; then
    printf '%s\n' "${LOCAL_CONDA_ENV}"
    return
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available; set LOCAL_CONDA_ENV_PATH=/path/to/env" >&2
    return 2
  fi

  conda env list | awk -v name="${LOCAL_CONDA_ENV}" '
    $1 == name { print $NF; found = 1; exit }
    END { if (!found) exit 1 }
  '
}

LOCAL_ENV_PATH="$(find_local_env_path)"
LOCAL_ENV_PATH="${LOCAL_ENV_PATH%/}"
ARCHIVE_PATH="${LOCAL_TMP%/}/${ARCHIVE_NAME}"

if [ ! -x "${LOCAL_ENV_PATH}/bin/python" ]; then
  echo "local env python not found: ${LOCAL_ENV_PATH}/bin/python" >&2
  exit 2
fi

mkdir -p "${LOCAL_TMP}"

log "local env: ${LOCAL_ENV_PATH}"
log "remote: ${SYNC_HOST}:${REMOTE_CONDA_ENV}"
log "archive: ${ARCHIVE_PATH}"

if ! "${LOCAL_ENV_PATH}/bin/python" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("conda_pack") else 1)
PY
then
  cat >&2 <<EOF
conda-pack is not installed in the local env.
Install it locally before syncing, for example:
  ${LOCAL_ENV_PATH}/bin/python -m pip install conda-pack
EOF
  exit 4
fi

log "packing local conda env"
rm -f "${ARCHIVE_PATH}"
pack_args=(-p "${LOCAL_ENV_PATH}" -o "${ARCHIVE_PATH}" --force)
if [ "${ENV_PACK_IGNORE_MISSING}" = "1" ]; then
  log "ENV_PACK_IGNORE_MISSING=1, adding conda-pack --ignore-missing-files"
  pack_args+=(--ignore-missing-files)
fi
"${LOCAL_ENV_PATH}/bin/conda-pack" "${pack_args[@]}"
ls -lh "${ARCHIVE_PATH}"

log "uploading archive"
ssh "${SYNC_HOST}" "mkdir -p $(quote "${REMOTE_TMP}")"
rsync -azP "${ARCHIVE_PATH}" "${SYNC_HOST}:${REMOTE_TMP%/}/${ARCHIVE_NAME}"

if [ "${ENV_SYNC_DEPS}" = "1" ]; then
  log "syncing source deps excluded from normal code sync"
  for rel in ${ENV_CODE_DEPS}; do
    if [ ! -d "${rel}" ]; then
      log "skip missing local dir: ${rel}"
      continue
    fi
    ssh "${SYNC_HOST}" "mkdir -p $(quote "${REMOTE_DIR%/}/${rel}")"
    rsync -az --delete "${rel}/" "${SYNC_HOST}:${REMOTE_DIR%/}/${rel}/"
  done
fi

log "unpacking remote env"
ssh "${SYNC_HOST}" bash -s -- \
  "${REMOTE_CONDA_ENV}" \
  "${REMOTE_TMP%/}/${ARCHIVE_NAME}" \
  "${ENV_REPLACE}" <<'REMOTE_UNPACK'
set -euo pipefail
dst_env="$1"
archive="$2"
replace="$3"

if [ -e "${dst_env}" ]; then
  if [ "${replace}" = "1" ]; then
    backup="${dst_env}.bak_$(date +%Y%m%d_%H%M%S)"
    mv "${dst_env}" "${backup}"
    echo "moved existing env to ${backup}"
  else
    echo "target env already exists: ${dst_env}" >&2
    echo "rerun with ENV_REPLACE=1 to move it aside before unpacking" >&2
    exit 3
  fi
fi

mkdir -p "${dst_env}"
tar -xzf "${archive}" -C "${dst_env}"
if [ -x "${dst_env}/bin/conda-unpack" ]; then
  "${dst_env}/bin/conda-unpack"
else
  echo "[WARN] conda-unpack not found in env; archive unpacked without prefix fixups" >&2
fi
REMOTE_UNPACK

remote_smoke_test

log "done"
log "remote python: ${REMOTE_CONDA_ENV}/bin/python"
log "remote command example:"
log "  make cmd CMD='${REMOTE_CONDA_ENV}/bin/python test/benchmark_e11_4_infoseek.py --help'"
