# Cadence ops progress — full 22-ticker rerun for trade date 2026-07-01

**Authoritative status doc** for the production cadence (CADENCE_OPS_GOAL.md).
Updated + pushed per ticker so the run can be resumed from any session.

- Code: prod `~/tradingagents` @ `da73d21` (FA-101 + pro-deck + debt-ladder blocks). No deploy needed.
- Base: `TK_RESEARCH_BASE=$HOME/tkresearch` → runs in `~/tkresearch/preaudit/2026-07-01-<T>/`.
- Promote target: `My Drive/True Knot/TK Research/final/wk 27 2026/` (ISO wk of 2026-07-01; folder exists, holds the 2026-06-26 cohort).
- Drive publish: idempotent by file ID via `~/gsheet-tool/pdf_ids.tsv` (all 22 present) → trash old ID, upload new to pdf/ root `1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`, overwrite manifest row.
- Phase 3 tools: `~/gsheet-tool/refresh_trading_plan.sh` (self-sources plist creds, reads canonical final/) + `~/gsheet-tool/update_summary.py`.
- No Telegram delivery (`TRADINGRESEARCH_NO_TELEGRAM=1`).

## Phase 1 — runner

- OAuth refreshed 2026-07-03 00:05 SGT; runner refreshes every 3rd run.
- Detached runner launched **2026-07-03 00:05 SGT**, pid 3921:
  `~/tkresearch/cadence-2026-07-01-runner.sh` — 21 tickers sequential (ORCL excluded; fresh audited run already exists), 5-min cooldowns, 60-min per-run cap, 429 detection + 15-min backoff, stale same-date preaudit dirs auto-archived (`.prerun-*`).
- Live status: `~/tkresearch/cadence-2026-07-01-status.tsv` (`TICKER\tSTARTED|DONE|FAIL|FAIL_429|TIMEOUT\tts`), log `~/tkresearch/cadence-2026-07-01-runner.log`.
- ETA ~13h → ~13:00 SGT 2026-07-03.

## Phase 2 — per-ticker status (run → QC → promoted)

Ticker order = runner order. ORCL run pre-existing (earlier 2026-07-02 audited production run).

| # | Ticker | Run | QC | Promoted (final/ + Drive + manifest) |
|---|--------|-----|----|--------------------------------------|
| — | ORCL | ✓ (pre-existing) | pending re-audit | – |
| 1 | AAOI | running | – | – |
| 2 | AAPL | – | – | – |
| 3 | AMKR | – | – | – |
| 4 | ASX | – | – | – |
| 5 | COIN | – | – | – |
| 6 | GOOGL | – | – | – |
| 7 | IFNNY | – | – | – |
| 8 | INTC | – | – | – |
| 9 | MARA | – | – | – |
| 10 | MRVL | – | – | – |
| 11 | MSFT | – | – | – |
| 12 | NOW | – | – | – |
| 13 | ONDS | – | – | – |
| 14 | RKLB | – | – | – |
| 15 | SATS | – | – | – |
| 16 | SOUN | – | – | – |
| 17 | STM | – | – | – |
| 18 | TSM | – | – | – |
| 19 | TXN | – | – | – |
| 20 | VSH | – | – | – |
| 21 | ON | – | – | – |

Tally: 0/22 promoted.

## Phase 3 — sheets (blocked until all 22 promoted)

- [ ] Trading Plan (`refresh_trading_plan.sh`, sheet `1ZLq9…`) — verify row per ticker, links, no #N/A.
- [ ] Research Summary register (sheet `1VJow…`, update-summary skill).

## Blockers / notes

(none yet)
