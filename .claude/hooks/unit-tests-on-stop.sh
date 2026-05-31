#!/usr/bin/env bash
# Stop hook: if any *.py under tradingagents/ cli/ tests/ changed vs HEAD (tracked
# or untracked), run the unit suite. If it's red, exit 2 to block "done" and feed
# the failures back — enforces "green before finished" without firing on no-code turns.
input=$(cat)
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$root" 2>/dev/null || exit 0

# Avoid loops: if this stop was already continued by a stop hook, don't re-gate.
active=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("stop_hook_active", False))
except Exception: print(False)' 2>/dev/null)
[ "$active" = "True" ] && exit 0

changed=$( { git diff --name-only HEAD -- tradingagents cli tests 2>/dev/null;
             git ls-files --others --exclude-standard -- tradingagents cli tests 2>/dev/null; } \
           | grep -E '\.py$' )
[ -z "$changed" ] && exit 0

py="$root/.venv/bin/python"
[ -x "$py" ] || py=python3
if ! out=$("$py" -m pytest -q -m unit --tb=line 2>&1); then
  echo "Unit tests FAIL — changed .py detected; fix before finishing:" >&2
  echo "$out" | tail -25 >&2
  exit 2
fi
exit 0
