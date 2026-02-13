#!/usr/bin/env bash
set -euo pipefail

# Download Country.mmdb for Dreamacro clash with multi-source fallback.
# Usage:
#   bash env/fetch_country_mmdb.sh
#   bash env/fetch_country_mmdb.sh /path/to/Country.mmdb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-${SCRIPT_DIR}/Country.mmdb}"
TMP_FILE="${TARGET}.tmp"

URLS=(
  "https://cdn.jsdelivr.net/gh/Dreamacro/maxmind-geoip@release/Country.mmdb"
  "https://raw.githubusercontent.com/Dreamacro/maxmind-geoip/release/Country.mmdb"
  "https://ghproxy.com/https://raw.githubusercontent.com/Dreamacro/maxmind-geoip/release/Country.mmdb"
)

download_with_curl() {
  local url="$1"
  curl -fL --connect-timeout 10 --max-time 120 -o "${TMP_FILE}" "${url}"
}

download_with_wget() {
  local url="$1"
  wget -O "${TMP_FILE}" --timeout=10 --tries=1 "${url}"
}

validate_mmdb() {
  local size
  size=$(wc -c < "${TMP_FILE}" | tr -d ' ')
  # Country.mmdb should be binary and normally several MB.
  if [[ "${size}" -lt 1000000 ]]; then
    return 1
  fi
  return 0
}

mkdir -p "$(dirname "${TARGET}")"
rm -f "${TMP_FILE}"

success=0
for url in "${URLS[@]}"; do
  echo "[INFO] Trying: ${url}"
  if command -v curl >/dev/null 2>&1; then
    if download_with_curl "${url}" && validate_mmdb; then
      success=1
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if download_with_wget "${url}" && validate_mmdb; then
      success=1
      break
    fi
  else
    echo "[ERROR] Neither curl nor wget is installed."
    exit 2
  fi
  echo "[WARN] Download failed or invalid file from: ${url}"
  rm -f "${TMP_FILE}"
done

if [[ "${success}" -ne 1 ]]; then
  cat <<'EOF'
[ERROR] Failed to fetch Country.mmdb from all sources.
You can manually copy it to the clash working directory, for example:
  scp Country.mmdb <user>@<host>:/public/home/<user>/mRAG/env/Country.mmdb
EOF
  exit 1
fi

mv -f "${TMP_FILE}" "${TARGET}"
echo "[OK] Country.mmdb saved to: ${TARGET}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${TARGET}"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${TARGET}"
fi
