#!/usr/bin/env bash
# PostToolUse(Edit|Write|MultiEdit): byte-compile the edited .py file for instant
# syntax-error feedback. Exit 2 surfaces the error back to Claude; 0 otherwise.
input=$(cat)
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
py="$root/.venv/bin/python"
[ -x "$py" ] || py=python3
fp=$(printf '%s' "$input" | "$py" -c 'import sys,json
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
case "$fp" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$fp" ] || exit 0
if ! err=$("$py" -m py_compile "$fp" 2>&1); then
  echo "py_compile failed for $fp:" >&2
  echo "$err" >&2
  exit 2
fi
exit 0
