#!/usr/bin/awk -f

# Convert a gitignore-like exclude list into rsync filter rules.
#
# Input format (from .exclude):
# - normal line: exclude pattern
# - line starts with !: include pattern (negation)
# - supports blank lines and # comments
#
# Output format:
# - rsync filter rules using + / - syntax
# - include rules are emitted before exclude rules so negation works

function ltrim(s) { sub(/^[[:space:]]+/, "", s); return s }
function rtrim(s) { sub(/[[:space:]]+$/, "", s); return s }
function trim(s)  { return rtrim(ltrim(s)) }

function add_include_rule(rule) {
  if (!(rule in include_seen)) {
    include_rules[++include_count] = rule
    include_seen[rule] = 1
  }
}

function add_exclude_rule(rule) {
  if (!(rule in exclude_seen)) {
    exclude_rules[++exclude_count] = rule
    exclude_seen[rule] = 1
  }
}

function add_include_with_parents(pattern,   base, n, i, parts, prefix) {
  base = pattern
  sub(/\/\*\*\*$/, "", base)
  sub(/\/$/, "", base)
  if (base == "") {
    return
  }

  # Wildcard paths cannot be safely expanded to parent dirs.
  if (base ~ /[*?\[]/) {
    add_include_rule("+ " pattern)
    return
  }

  n = split(base, parts, "/")
  prefix = ""
  for (i = 1; i <= n; i++) {
    if (parts[i] == "") {
      continue
    }
    if (prefix == "") {
      prefix = parts[i]
    } else {
      prefix = prefix "/" parts[i]
    }
    add_include_rule("+ " prefix "/")
  }

  add_include_rule("+ " base)
  if (pattern !~ /\/\*\*\*$/) {
    add_include_rule("+ " base "/***")
  }
}

{
  line = $0
  sub(/\r$/, "", line)
  line = trim(line)

  if (line == "" || line ~ /^#/) {
    next
  }

  if (substr(line, 1, 1) == "!") {
    include_pattern = trim(substr(line, 2))
    if (include_pattern != "") {
      add_include_with_parents(include_pattern)
    }
    next
  }

  add_exclude_rule("- " line)
}

END {
  for (i = 1; i <= include_count; i++) {
    print include_rules[i]
  }
  for (i = 1; i <= exclude_count; i++) {
    print exclude_rules[i]
  }
}
