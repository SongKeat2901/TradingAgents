# TradingAgents — fork at SongKeat2901/TradingAgents

Multi-agent equity-research pipeline. Personal Claude subscription, single-user research,
deployed to `macmini-trueknot` via SSH. Production reports delivered as TrueKnot-branded
PDFs to Telegram chat `-1003753140043`.

## Quick Start

```bash
# Unit tests (216 tests, ~2s)
.venv/bin/python -m pytest -q -m unit --tb=line

# Single-test debug
.venv/bin/python -m pytest tests/test_pm_preflight.py::test_X -v

# Local e2e (don't — too slow; use macmini)
# --output-dir is optional; when omitted the run lands in its working-copy
# (preaudit) location under TK Research in the trueknotsg Google Drive (My Drive):
# ~/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/preaudit/<date>-<ticker>
# (override the base with the TK_RESEARCH_BASE env var on dev hosts).
~/local/bin/tradingresearch --ticker MSFT --date 2026-05-05 \
  --telegram-notify=-1003753140043
```

## E2E run on macmini-trueknot (canonical)

```bash
# 1. Push, deploy
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet'

# 2. Refresh OAuth (8h TTL)
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'

# 3. Archive prior run + kick fresh. TK Research lives in the trueknotsg Google
# Drive (My Drive); set a shell var for the long path:
#   TK="$HOME/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research"
# Default output is the working copy: $TK/preaudit/<DATE>-<TICKER> (override base via TK_RESEARCH_BASE).
# Omit --output-dir to use it; pass an explicit dir only to override.
# A run is "finalized" by manually promoting it BY WEEK (preaudit → final/wk NN YYYY/),
# never by the pipeline:  mv "$TK/preaudit/<DATE>-<TICKER>" "$TK/final/wk NN YYYY/"
ssh macmini-trueknot 'TK="$HOME/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research"; mv "$TK/preaudit/<DATE>-<TICKER>" "...run-<SHA>" 2>/dev/null'
ssh macmini-trueknot '~/local/bin/tradingresearch --ticker <T> --date <D> \
  --telegram-notify=-1003753140043'
# Daemon detaches; takes ~22-35 min.

# 4. Watch for completion
ssh macmini-trueknot 'until ! pgrep -f "tradingresearch --ticker <T>" >/dev/null; do sleep 30; done'
```

## Architecture (HEAD bcf4301, Phase 6.6)

```
PM Pre-flight (Opus 4.8 / CLI) — appends calendar block + SEC filing footer to pm_brief.md
  → Researcher (Python, deterministic data + classifier + peer-ratios block)
  → TA v1 → 4 analysts → TA v2 → Bull/Bear → RM → Trader → Risk team → PM → QC
  → write_research_outputs → research_pdf (TrueKnot-branded)
```

**Both LLM tiers route through `claude -p` subprocess** (`deep_via_cli=True`,
`quick_via_cli=True` defaults). Direct ChatAnthropic 429s on Sonnet/Opus on the personal
subscription. See memory: `project_phase5_rate_limit_finding.md`.

**Deterministic Python blocks** (Phase 6.2/6.3/6.4) are appended to `pm_brief.md` after
the PM Pre-flight LLM call so downstream agents see authoritative ground truth they
can't paraphrase. See memory: `project_phase6_deterministic_blocks_pattern.md`.

## Key Files

| File | Role |
|---|---|
| `cli/research.py` | Main CLI entry; default `--deep claude-opus-4-8`, `--quick claude-sonnet-4-6` |
| `cli/research_pdf.py` | TrueKnot-branded PDF generator (executive summary + appendix + vocab cleanup) |
| `cli/research_writer.py` | Writes per-run markdown + state.json with `_meta` block |
| `tradingagents/graph/trading_graph.py` | Graph wiring (`deep_via_cli`/`quick_via_cli` routing) |
| `tradingagents/agents/managers/pm_preflight.py` | Calendar (6.2) + SEC filing (6.3) block appends |
| `tradingagents/agents/researcher.py` | Peer-ratios (6.4) block append + classification |
| `tradingagents/agents/managers/qc_agent.py` | 16-item audit (item 15 filing-anchor, 16 numerical-trace) |
| `tradingagents/agents/utils/{calendar,sec_edgar,peer_ratios,classifier}.py` | Pure-Python deterministic helpers |
| `tradingagents/llm_clients/claude_cli_chat_model.py` | `claude -p` subprocess wrapper + model alias map |

## Commit conventions

- Commits go directly on `main` (no feature branches for this project).
- Footer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` —
  **no** "Generated with [Claude Code]" line.
- Use `feat()` / `fix()` / `docs()` / `test()` Conventional-Commits prefixes.
- Each commit covers one slice (don't bundle e.g. `peer_ratios.py` + `pm_preflight.py`
  wiring together).

## Gotchas

- **Don't add `bind_tools()` to new analysts.** Phase 6 dropped tool calls; analysts
  read pre-fetched data from `raw/` via `format_for_prompt`. Adding tools forces the
  analyst back onto ChatAnthropic direct-API which 429s.
- **PM Pre-flight runs BEFORE Researcher.** If a deterministic block needs `peers.json`
  (written by Researcher), wire it inside `researcher.py` AFTER the `peers.json` write.
  If it only needs `reference.json` + `classification.json`, wire it in `pm_preflight.py`.
  Calendar lives in PM Pre-flight; SEC filing in PM Pre-flight (fetcher); peer ratios
  in Researcher.
- **yfinance data anomalies.** ORCL Q1 capex/revenue can come back as 108% (capex >
  revenue). The deterministic block reports it faithfully; analyst flags as "anomalous"
  and excludes from peer comparison. Don't auto-clamp.
- **Intraday-bar capture on same-day-close runs.** yfinance returns an *in-progress*
  bar for `trade_date` while the US session is open, whose Close is the last trade, not
  the settlement close. `drop_incomplete_session()` (`dataflows/stockstats_utils.py`,
  wired into both `get_YFin_data_online` + `load_ohlcv`) drops it until 16:00 ET so the
  reference price is the settled close. The mini runs **SGT (UTC+8)** → US close 16:00 ET
  = **04:00 SGT next day**; a same-day cadence started before that hits the bug (it
  silently corrupted 8 of 21 refs in the 2026-06-02 batch, e.g. AAPL $308.85 intraday vs
  $315.20 close). Audit a run: compare `decision.md` "Reference price:" against a fresh
  yfinance close; `validation_report.json` carries `total_violations`/`blocking_violations`.
- **CLI subprocess doesn't support `with_structured_output`.** Judges fall back to
  free-text via `agents/utils/structured.py:invoke_structured_or_freetext`. Expected
  log line: `"... provider does not support with_structured_output (...); falling back
  to free-text generation"`.
- **OAuth token has 8h TTL.** Refresh before each e2e run with `claude -p hi`.
- **`max_tokens` defaults to 8192.** Lower it (`--max-tokens 4096`) only if the run
  hits per-call token-spike pressure; not needed for the current Sonnet/Opus-via-CLI
  setup.

## PDF cover branding

TrueKnot brand pack at `~/Documents/Python/TrueKnot Web Setup/docs/brand/`. Color
tokens live in `cli/research_pdf.py` `_CSS`. Cover SVG is inline; corporate footer
text is hardcoded ("TrueKnot Pte. Ltd. · UEN 202608241M · 1 Bukit Batok Cres,
#05-15, Singapore 658064 · trueknot.sg"). See memory:
`reference_trueknot_brand_pack.md`.

## Pointers

- Memory: `~/.claude/projects/-Users-songkeat-Documents-Python-Trading-Agent/memory/`
- Specs: `docs/superpowers/specs/`
- Plans: `docs/superpowers/plans/`
- Research output on macmini: TK Research lives in the **trueknotsg Google Drive (My Drive)** at
  `~/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/` (override base via
  `TK_RESEARCH_BASE`; native summary Sheet lives inside `final/`). Working copies in
  `…/TK Research/preaudit/<date>-<ticker>/` (default when `--output-dir` omitted); promoted A+ reports
  moved by hand, **by week**, to `…/TK Research/final/wk NN YYYY/<date>-<ticker>/`. Pipeline only writes preaudit.
  Older runs (pre-default) live under `~/.openclaw/data/research/<date>-<ticker>.run-<sha>/`.
- **⚠️ Canonical paths (verified 2026-06-18) — two different trees, set `TK_RESEARCH_BASE` or runs scatter:**
  - `final/` is under **`My Drive/True Knot/TK Research/final/wk NN YYYY/`** (note the `True Knot/` prefix; holds wk 22/23/24…). This is `cli/cadence_followup.py`'s `FINAL_BASE`.
  - The pipeline's **default** preaudit base (`_tk_base()` in `cli/research.py`) is **`My Drive/TK Research`** (NO `True Knot/`) — a *different* tree than `final/`. A run with `TK_RESEARCH_BASE` unset lands in `My Drive/TK Research/preaudit/`, scattered away from `final/` and from the local QC tooling.
  - **Cadence runners + the bot CLI use `TK_RESEARCH_BASE=$HOME/tkresearch`** so working copies live at **`~/tkresearch/preaudit/<date>-<ticker>`** (where `cli/cadence_followup.py` `DEFAULT_PREAUDIT` + QC look). Always export it for any standalone `tradingresearch` run, or the output won't be found by promote/QC.
  - There is **no** `~/gsheet-tool/update_register.py`; the Research Summary gsheet digest is done by the bot (LLM) per the `cadence-followup` SKILL.md, not a deterministic script.
- Telegram delivery: `cli/research_telegram.py` (auto-discovers bot token from
  `~/.openclaw/openclaw.json` on the OpenClaw host)

## Macro regime engine (daily)

- Package: `tradingagents/macro/` — standalone daily engine; CLI `tradingmacro`.
  Spec: `docs/superpowers/specs/2026-06-04-macro-regime-engine-design.md`.
- Trading Plan sheet (native GSheet) ID `1ZLq9HuyU0AAzREECpBGpamDBVpbbjq9V8joHqQHp-cw`,
  in the shared `True Knot/TK Research/pdf/` folder (writes to its first tab, `Sheet1`).
  17 cols incl. **Last Px = live `=GOOGLEFINANCE(ticker,price)`** (no fallback — an
  unresolvable symbol shows #N/A, never a stale price), **Intrinsic FV + Margin-of-Safety %**
  (from `raw/intrinsic_value.json`; blank for foreign-ADR / unprofitable / mis-scaled-eps
  names the plausibility guards suppress), and an **Action** column that is the EV/macro
  bias ONLY (Add/Hold · Hold · Trim/Avoid · Caution · Stand-down) — Rating is its own
  column so they never read as a contradiction. Numeric cells are RAW numbers; the % / $
  display + ± conditional colour come from column formats (pre-formatted strings broke
  Sheets' parsing and the colour rules).
- Entry point on the mini: `~/tradingagents/.venv/bin/tradingmacro` (console script
  from `pip install -e .`; there is **no** `~/local/bin/tradingmacro` wrapper).
- Manual run on the mini (gog needs account + keyring password in env; `-a` is added
  by the engine from `GOG_ACCOUNT`):
  `FRED_API_KEY=… GOG_ACCOUNT=trueknotsg@gmail.com GOG_KEYRING_PASSWORD=… ~/tradingagents/.venv/bin/tradingmacro --manifest-scope --reports-dir <base> [--reports-dir <base2> …] --sheet-id <id> --manifest ~/gsheet-tool/pdf_ids.tsv`
  (add `--no-write` to compute without touching the sheet; `--as-of` defaults to the
  mini's local SGT date).
- **Reports source = run dirs scattered across cadence bases** — `--reports-dir` is
  repeatable; `latest_runs` picks the newest-WRITTEN run per ticker on a same
  trade_date (so a corrected rerun beats a same-date original). Current wk23 bases:
  `~/research-staging-2026-06-02` (originals), `~/tkruns-rerun-2026-06-02` (the 4
  rating-flip reruns), `~/tkruns-fix-2026-06-02`, `~/tkresearch` (NOK),
  `~/.openclaw/data/research` (old). **New cadences land in new dirs — add them to
  the plist's `--reports-dir` list (or consolidate into one canonical base).**
  The My Drive `TK Research/final/` holds only published PDFs, not run dirs.
- **`--manifest-scope`** restricts the plan to the manifest (published-to-Drive)
  tickers, so every row links and stale one-off runs drop out. The manifest
  (`~/gsheet-tool/pdf_ids.tsv`) is therefore the plan's active-watchlist universe.
- Sheet write goes through `gog` v0.11.0 at `/opt/homebrew/bin/gog`:
  `gog sheets update <id> <range> --values-json '<2D array>' --input USER_ENTERED -a <acct>`.
  `write_to_sheet` builds this; `tab=None` → first sheet (range `A1`), fixed-height
  100-row grid so a shorter run can't leave stale rows. gog keyring unlocks from
  `GOG_KEYRING_PASSWORD`; 7-day token — re-auth per the update-summary skill on invalid_grant.
- Scheduled via `ops/com.trueknot.macrodaily.plist` →
  `cp ops/com.trueknot.macrodaily.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.trueknot.macrodaily.plist`
  (fill the three `REPLACE_WITH_…` placeholders — FRED key, sheet ID, gog keyring
  password — first; plist sets `PATH` so launchd finds gog). Runs 05:10 SGT (post US close).
- Needs a free FRED API key (Growth/Inflation/Liquidity hard data); yfinance covers
  the market-priced pillars.
- Intrinsic value: `tradingagents/agents/utils/intrinsic_value.py` computes foreign-ADR
  fair value via a data-derived FX (eps×shares/NI) but has 3 plausibility gates so it
  never emits a wrong number (skip if implied P/E∉[3,60] or fair value ∉ 0.2–3× price).
  Most foreign ADRs stay blank — their free-data eps/prices are trough/thin (a data limit,
  not a math bug; **no paid feed** to fix it). Regenerate per-report JSON without a re-run:
  `/tmp/regen_intrinsic.py` on the mini, then re-run the engine.

## Google Sheets ops (gog + Sheets API)

- `gog` v0.11.0 (`/opt/homebrew/bin/gog`, account `trueknotsg@gmail.com`, keyring pw via
  env `GOG_KEYRING_PASSWORD`, on the **mini only**) does values + STATIC `gog sheets format`
  (CellFormat) only — no batchUpdate.
- **For freeze panes / conditional formatting / batchUpdate**, reuse gog's OAuth without a
  new client: `gog auth tokens export <acct> --out <f>` (refresh token) + client id/secret
  from `~/Library/Application Support/gogcli/credentials.json` → POST oauth2.googleapis.com/token
  → access token → call `https://sheets.googleapis.com/v4/...`. Run with
  `~/tradingagents/.venv/bin/python` (has `requests`; system python lacks it). First tab gid = 0.
- Two sheets live in `pdf/` (now FLAT — all report PDFs moved to pdf/ root, file IDs/links
  unchanged): Trading Plan `1ZLq9HuyU0AAzREECpBGpamDBVpbbjq9V8joHqQHp-cw` (macro engine) and
  Research Summary register `1VJowGGdxjCPd0jMpZVHJlC-C6aEspf1iJWOVPH0T7dk` (update-summary skill).
- Mini ops scripts in `~/gsheet-tool/`: `beautify_trading_plan.py` (static styling + number
  formats), `beautify_conditional.py` (freeze + conditional colours, idempotent),
  `update_register.py` (re-digest the register). Formatting persists across value writes —
  only re-apply after a sheet is recreated.
- **Register-update gotchas:** read with `valueRenderOption=FORMATTED_VALUE` (UNFORMATTED
  returns dates as serial numbers → rows invisible), strip $/%/commas to floats; WRITE via
  `gog sheets update`, NOT the REST values endpoint (values.update is PUT, POST→400 HTML);
  sanitise NaN→""; fetch closes with `yf_retry`+sleep or yfinance rate-limits blank them.

## Claude Code automation (this repo)

- Hooks (`.claude/settings.json`): PreToolUse `secret-guard.sh` (blocks committing/writing
  sk-ant / GOCSPX / Google refresh tokens / real FRED key / gog keyring pw — placeholders OK),
  SessionStart `gog-token-age.sh`, PostToolUse `pycompile.sh`, Stop `unit-tests-on-stop.sh`
  (green-before-done gate).
- Skills (`.claude/skills/`, all user-only): `deploy-mini`, `macro-run` (refresh the Trading
  Plan), `publish-report` (idempotent PDF→Drive→manifest→re-run plan), `update-summary`,
  `cadence-run`. Agents: `report-auditor`, `validator-reviewer`. MCP: `context7` (`.mcp.json`).
