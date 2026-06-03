#!/usr/bin/env bash
# SessionStart: best-effort heads-up if the mini's gog OAuth token is near its
# ~7-day expiry (unverified-app rule that has blocked Sheet updates before).
# Uses the keyring file mtime as a no-password proxy for last auth; if the mini
# is unreachable it stays silent (never blocks session start).
KR='Library/Application Support/gogcli/keyring.json.enc'
mt=$(ssh -o ConnectTimeout=4 -o BatchMode=yes macmini-trueknot "stat -f %m \"\$HOME/$KR\" 2>/dev/null" 2>/dev/null)
[ -z "$mt" ] && exit 0
now=$(date +%s)
age=$(( (now - mt) / 86400 ))
if [ "$age" -ge 6 ]; then
  echo "gog token on macmini-trueknot is ~${age}d old (expires ~7d). If a gog/Sheets call returns invalid_grant, re-auth per the update-summary skill (gog auth add trueknotsg, browser on the mini)."
fi
exit 0
