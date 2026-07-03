---
name: macro-run
description: >-
  Run the daily macro regime engine on macmini-trueknot to refresh the TrueKnot
  Trading Plan Google Sheet (regime board + per-stock EV/macro overlay). Use when
  asked to "run the macro engine", "refresh the trading plan", "update the macro
  sheet", or after changing macro code. Also covers the intrinsic-regen + beautify
  follow-ups when the engine code changed.
disable-model-invocation: true
---

# Run the macro regime engine → Trading Plan sheet

Standalone daily engine (`tradingagents/macro/`, CLI `tradingmacro`). Writes the
Trading Plan sheet `1ZLq9HuyU0AAzREECpBGpamDBVpbbjq9V8joHqQHp-cw` (first tab) in
`True Knot/TK Research/pdf/`. Decoupled from the research pipeline — it reads
finished run dirs. The scheduled launchd job already runs it 05:10 SGT; this skill
is for an on-demand refresh. See `project_macro_regime_engine_deployed` memory.

## Run (on the mini)

The reports live scattered across cadence bases; `--reports-dir` is repeatable and
the newest-WRITTEN run wins per ticker on a same date. `--manifest-scope` limits
the plan to published (manifest) tickers so every row links. gog needs the account
+ keyring password in env.

```bash
ssh macmini-trueknot 'export PATH=/opt/homebrew/bin:$PATH
  export GOG_KEYRING_PASSWORD=<keyring pw>   # never commit this; it lives on the mini
  export GOG_ACCOUNT=shianpin@trueknot.sg
  FRED_API_KEY=<fred key> ~/tradingagents/.venv/bin/tradingmacro \
    --manifest-scope \
    --reports-dir ~/tkresearch \
    --reports-dir ~/research-staging-2026-06-02 \
    --reports-dir ~/tkruns-rerun-2026-06-02 \
    --reports-dir ~/tkruns-fix-2026-06-02 \
    --reports-dir ~/.openclaw/data/research \
    --sheet-id 1ZLq9HuyU0AAzREECpBGpamDBVpbbjq9V8joHqQHp-cw \
    --manifest ~/gsheet-tool/pdf_ids.tsv'
```

- Add `--no-write` to compute + print the regime only (no sheet write) — good for a
  sanity check first; expect `Regime: … | gate=… | N names`.
- **`~/tkresearch` is the canonical base** (rglob is recursive: covers `preaudit/`
  AND the local published store `final/wk NN YYYY/`). The dated bases are the wk23
  (2026-06-02/03) cohort, kept for history; never point at a
  `~/Library/CloudStorage` mount (GUI-session-tied, unmounts on user switch).
- The secrets (`FRED_API_KEY`, `GOG_KEYRING_PASSWORD`) are on the mini only — read
  them from the live launchd plist `~/Library/LaunchAgents/com.trueknot.macrodaily.plist`
  if you need the current values; do not hardcode them here.

## After a code change (only then)

1. Deploy: use the `deploy-mini` skill (push → pull → `pip install -e .`).
2. If intrinsic/valuation logic changed, regenerate the per-report JSON the sheet
   reads (deterministic, no LLM): `ssh macmini-trueknot '.venv/bin/python /tmp/regen_intrinsic.py'`
   (script iterates manifest tickers' latest run dirs).
3. Re-run the engine (command above).
4. If the sheet was recreated, re-apply formatting: `~/gsheet-tool/beautify_trading_plan.py`
   (static styling + number formats) then `~/tradingagents/.venv/bin/python ~/gsheet-tool/beautify_conditional.py`
   (freeze + conditional colours; idempotent). Normal value re-writes preserve formatting,
   so this is only needed after a fresh sheet.

## Verify
Read back: `gog sheets get <sheet-id> "A1:A27" -a shianpin@trueknot.sg -p` (regime row
+ tickers). Expect the regime banner in A1 and one row per published ticker.
