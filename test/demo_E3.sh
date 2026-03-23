#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE=E3 exec bash "${SCRIPT_DIR}/demo_export_one.sh" "$@"
