#!/usr/bin/env bash
# PreToolUse guard: refuse to let a secret reach the (public) repo. Fires on
# `git commit`/`git add` (scans the staged diff) and on Edit|Write|MultiEdit
# (scans the content about to be written). High-confidence patterns only;
# REPLACE_WITH_* placeholders are explicitly allowed (that's how real values are
# kept out of the repo — they live only on the mini). Exit 2 blocks the call and
# feeds the reason back. This session had FRED keys, the gog keyring password,
# sk-ant tokens and Google refresh tokens flying around in commands — this is the
# backstop against one of them landing in a commit.
input=$(cat)
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"

# sk-ant-… (Anthropic), GOCSPX-… (Google client secret), 1//… (Google refresh
# token), real FRED key assignment, gog keyring password assigned to a real value.
# The pw char class exempts placeholders (REPLACE_WITH_*, <pw>) and shell
# expansions ($VAR / "$k=$(…)") — those are exactly how the real value is kept
# OUT of the repo, and they were false-positive-blocking doc edits.
PATTERNS='sk-ant-[A-Za-z0-9_-]{20}|GOCSPX-[A-Za-z0-9_-]{10}|1//[0-9A-Za-z_-]{30}|FRED_API_KEY[[:space:]]*=[[:space:]]*[0-9a-f]{20}|GOG_KEYRING_PASSWORD[[:space:]]*=[[:space:]]*[^R<$[:space:]"'"'"']'

tool=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_name",""))
except Exception: print("")' 2>/dev/null)

block() {
  echo "secret-guard: $1 looks like it contains a secret — blocked." >&2
  echo "$2" | head -5 >&2
  echo "Use a REPLACE_WITH_* placeholder; real keys/tokens/passwords live only on the mini, never in this repo." >&2
  exit 2
}

if [ "$tool" = "Bash" ]; then
  cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception: print("")' 2>/dev/null)
  case "$cmd" in
    *"git commit"*|*"git add"*)
      hit=$(cd "$root" 2>/dev/null && git diff --cached 2>/dev/null | grep -nE "$PATTERNS")
      [ -n "$hit" ] && block "the staged diff" "$hit" ;;
  esac
elif [ "$tool" = "Edit" ] || [ "$tool" = "Write" ] || [ "$tool" = "MultiEdit" ]; then
  content=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    t=json.load(sys.stdin).get("tool_input",{})
    parts=[t.get("content",""), t.get("new_string","")]
    parts+=[e.get("new_string","") for e in t.get("edits",[])]
    print("\n".join(p for p in parts if p))
except Exception: print("")' 2>/dev/null)
  hit=$(printf '%s' "$content" | grep -nE "$PATTERNS")
  [ -n "$hit" ] && block "this edit/write" "$hit"
fi
exit 0
