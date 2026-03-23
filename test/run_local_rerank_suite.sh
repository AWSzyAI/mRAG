#!/usr/bin/env bash
set -euo pipefail

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] [SUITE] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Local-rerank related experiments by default.
# You can override, e.g.:
#   EXPS="E1 E2 E3 E4 E5 E6 E7" bash test/run_local_rerank_suite.sh
EXPS="${EXPS:-E1 E2 E5 E6}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/log}"
mkdir -p "${LOG_DIR}"

log "root_dir=${ROOT_DIR}"
log "experiments=${EXPS}"
log "log_dir=${LOG_DIR}"
log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
log "starting..."

for exp in ${EXPS}; do
  script="${SCRIPT_DIR}/${exp}.sh"
  logfile="${LOG_DIR}/${exp}.log"
  if [[ ! -x "${script}" ]]; then
    echo "[ERROR] script missing or not executable: ${script}"
    exit 2
  fi

  log "run=${exp} script=${script}"
  log "logfile=${logfile}"
  if bash "${script}" > "${logfile}" 2>&1; then
    log "status=${exp}:OK"
  else
    rc=$?
    log "status=${exp}:FAIL rc=${rc}"
    log "last_lines(${exp}):"
    tail -n 60 "${logfile}" || true
    exit "${rc}"
  fi
done

log "all_done"
