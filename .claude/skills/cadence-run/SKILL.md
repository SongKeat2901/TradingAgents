---
name: cadence-run
description: >-
  Run a batch of research tickers on macmini-trueknot one at a time, audit each
  to A+, and promote/register the passes. Use when asked to "run a cadence",
  "research these tickers and register the A+ ones", or to re-run/refresh a set
  of tickers through the hardened stack. Drives: deploy → run → wait → audit →
  promote → register → summary.
disable-model-invocation: true
---

# Cadence Run

Orchestrates the full research-to-register loop on `macmini-trueknot`. One ticker
at a time (rate-limit + one-run-per-host constraint). ~22-37 min per run.

## Inputs
A list of `(ticker, date)` pairs. Default date = latest close (`--date YYYY-MM-DD`,
historical). If only tickers are given, ask for the trade date or use today's last close.

## Preconditions (do these once at the start)
1. **Deploy the current stack** (so runs use the hardened pipeline): use the
   `deploy-mini` skill, or:
   `git push origin main && ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet'`
2. **Refresh OAuth** (8h TTL): `ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'`. Re-refresh if a batch runs > ~7h.

## Per-ticker loop (repeat for each pair, ONE at a time)
1. **Kick the run** — daemonizes, no customer Telegram delivery:
   ```bash
   ssh macmini-trueknot 'cd ~ && TRADINGRESEARCH_NO_TELEGRAM=1 ~/local/bin/tradingresearch --ticker <T> --date <D>'
   ```
   It prints `started pid=<N>` and returns in ~1s. Output lands in
   `<TK>/preaudit/<D>-<T>/` (the `--output-dir` default), where `<TK>` is the
   trueknotsg Google Drive folder
   `~/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research`
   (override base via `TK_RESEARCH_BASE`).
   Gotchas: do NOT add `nohup`, `&`, or `--output-dir`. macOS has no `timeout` binary.
2. **Wait for completion** — run this as a background Bash command (it re-invokes
   you when the run exits, ~30 min):
   ```bash
   ssh macmini-trueknot 'until ! pgrep -f "tradingresearch --ticker <T>" >/dev/null 2>&1; do sleep 30; done; echo DONE'
   ```
3. **Sanity-check the landing**: confirm `decision.md` + `research-<D>-<T>.pdf`
   exist; check `raw/peer_corrections.json` total, `validation_report.json`
   `blocking_violations`, and that the PDF extracts 0 leak markers.
4. **Audit** with the `report-auditor` subagent (RUN_DIR = the preaudit path).
5. **If A+**: promote **into the ISO-week folder** (`final/wk NN YYYY/`; create it if absent).
   ```bash
   ssh macmini-trueknot 'TK="$HOME/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research"; WK="wk <NN> <YYYY>"; mkdir -p "$TK/final/$WK"; mv "$TK/preaudit/<D>-<T>" "$TK/final/$WK/<D>-<T>"; rm -rf ~/.openclaw/data/research/<D>-<T>'
   ```
   Then add a row to `<TK>/REGISTER.md` (date, ticker, rating,
   A+, ref price, EV, EV-vs-spot, QC, PDF) and rebuild the summary:
   `ssh macmini-trueknot 'cd ~/tradingagents && .venv/bin/python -m cli.update_research_summary'`
   (consolidates the PDF into `final/pdf/` and refreshes prices/moves).
   **If NOT A+**: re-run once (non-determinism often clears a transient nit). If a
   defect persists across re-runs, it's a failed run — leave it out of the
   register and report why (per "remove all failed run").
6. **If you HAND-CORRECT a report** (edit `decision.md`/`decision_executive.md` to
   fix a real defect instead of re-running): after editing you MUST **re-run the
   validator to refresh `validation_report.json`** and **regenerate the PDF** —
   downstream QC and the `cadence-followup` bot read the *pre-computed*
   `validation_report.json`, NOT the live markdown, so a stale report re-flags the
   issue you just fixed (wk24 NOW: $6.42B was corrected in the markdown but the
   stale report kept flagging it). Commands (on the mini, in the run dir):
   ```bash
   ~/tradingagents/.venv/bin/python -c "from cli.research_validation import run_phase_7_validators, write_validation_report as w; import sys; d=sys.argv[1]; w(d, run_phase_7_validators(d))" <RUN_DIR>
   ~/tradingagents/.venv/bin/python -c "from cli.research_pdf import build_research_pdf; from pathlib import Path; d=sys.argv[1]; build_research_pdf(output_dir=d, ticker='<T>', date='<D>', decision=Path(d+'/decision.md').read_text())" <RUN_DIR>
   ```

## Rules
- ONE run at a time; if `pgrep -f tradingresearch` shows an active run, wait.
- Register/promote **A+ only**. Failed runs are excluded and their stray output removed.
- Never deploy (pip install) while a run is in flight — wait for it to finish.
- Report a running tally as you go (N/M A+ so far) and a final summary.
