#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <target_dir> <path1> [path2 ...]" >&2
  exit 2
fi

TARGET_DIR="$1"
shift

mkdir -p "${TARGET_DIR}"

for src in "$@"; do
  [[ -z "${src}" ]] && continue
  if [[ ! -e "${src}" ]]; then
    continue
  fi
  base="$(basename "${src}")"
  dst="${TARGET_DIR}/${base}"
  if [[ "${src}" == "${dst}" ]]; then
    continue
  fi
  mv -f "${src}" "${dst}"
  echo "[ARCHIVE] ${src} -> ${dst}"
done
