#!/usr/bin/awk -f

# Read a key from a selected Host block in .sync.
#
# Usage:
#   awk -v host='featurize' -v key='REMOTE_DIR' -f scripts/read_sync_value.awk .sync
#   awk -v host='featurize' -v key='CONDA_ENV'  -f scripts/read_sync_value.awk .sync

function trim(s) {
  sub(/^[[:space:]]+/, "", s)
  sub(/[[:space:]]+$/, "", s)
  return s
}

BEGIN {
  if (host == "" || key == "") {
    exit 0
  }
  host_lc = tolower(host)
  key_lc = tolower(key)
  in_host = 0
}

{
  line = $0
  sub(/\r$/, "", line)

  if (line ~ /^[[:space:]]*$/ || line ~ /^[[:space:]]*#/) {
    next
  }

  if (line ~ /^[[:space:]]*Host[[:space:]]+/) {
    host_name = line
    sub(/^[[:space:]]*Host[[:space:]]+/, "", host_name)
    host_name = trim(host_name)
    in_host = (tolower(host_name) == host_lc)
    next
  }

  if (!in_host) {
    next
  }

  if (line ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\??=/) {
    eq_pos = index(line, "=")
    lhs = trim(substr(line, 1, eq_pos - 1))
    rhs = trim(substr(line, eq_pos + 1))
    sub(/[[:space:]]*\?$/, "", lhs)
    lhs = trim(lhs)
    cur_key = tolower(lhs)
    if (cur_key == key_lc) {
      print rhs
      exit 0
    }
  }
}
