---
name: deploy-mini
description: >-
  Deploy the current TradingAgents main branch to macmini-trueknot (push → pull →
  editable reinstall) and refresh the OAuth token. Use when asked to "deploy",
  "push to the mini", "ship this to the mini", or before running research that
  must use the latest code.
disable-model-invocation: true
---

# Deploy to macmini-trueknot

The MacBook is the only place code is edited; the mini is pull-only and runs the
production daemon. This ships committed `main` to it.

## Steps
1. **Confirm clean + committed**: `git -C "<repo>" status --short`. If there are
   uncommitted changes the user wants shipped, commit them first (Conventional
   Commits prefix; footer `Co-Authored-By: Claude Opus 4.7 (1M context)
   <noreply@anthropic.com>`; NO "Generated with Claude Code" line; one slice per commit).
2. **Do NOT deploy while a research run is in flight** — `pip install -e .`
   mid-run can change modules the running process imports lazily. Check first:
   `ssh macmini-trueknot 'pgrep -fl tradingresearch || echo idle'`. If a run is
   active, wait for it to finish.
3. **Push + pull + reinstall**:
   ```bash
   git push origin main
   ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet'
   ```
4. **Verify HEAD matches**:
   ```bash
   ssh macmini-trueknot 'cd ~/tradingagents && git --no-pager log --oneline -1'
   ```
   Confirm it equals local `git rev-parse --short HEAD`.
5. **Refresh OAuth** (8h TTL; do before any e2e run):
   `ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'`

Report the deployed SHA back to the user.
