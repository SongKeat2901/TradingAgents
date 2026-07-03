# Cadence ops progress — full 22-ticker rerun for trade date 2026-07-01

**Authoritative status doc** for the production cadence (CADENCE_OPS_GOAL.md).
Updated + pushed per ticker so the run can be resumed from any session.

- Code: prod `~/tradingagents` @ `da73d21` (FA-101 + pro-deck + debt-ladder blocks). No deploy needed.
- Base: `TK_RESEARCH_BASE=$HOME/tkresearch` → runs in `~/tkresearch/preaudit/2026-07-01-<T>/`.
- Promote target: `final/wk 27 2026/` — **canonical local path is now
  `~/Library/CloudStorage/GoogleDrive-shianpin@trueknot.sg/My Drive/True Knot/TK Research/final/wk 27 2026/`**
  (the trueknotsg@gmail.com Drive mount no longer exists on the mini — identity migrated to
  shianpin@trueknot.sg ~2026-06-26; same shared tree, ORCL + the 2026-06-26 cohort all present).
- Drive publish: idempotent by file ID via `~/gsheet-tool/pdf_ids.tsv` (all 22 present) → trash old ID, upload new to pdf/ root `1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`, overwrite manifest row. Helper `~/tkresearch/publish-2026-07-01.sh` patched for the new mount + `-a shainpin@trueknot.sg`.
- No Telegram delivery (`TRADINGRESEARCH_NO_TELEGRAM=1`).

## Phase 1 — runner: **COMPLETE**

- All 21 tickers ran sequentially 2026-07-03 00:05–14:18 SGT, **21/21 DONE**, zero 429s/timeouts
  (`~/tkresearch/cadence-2026-07-01-status.tsv`, `ALL COMPLETE 14:23:01`).

## Phase 2 — QC + promote (in progress)

Deterministic sweep: all 21 run dirs complete (decision/exec/PDF/state/validation_report all present);
**all 21 reference prices match fresh yfinance settled closes exactly** (no intraday-bar capture).
Forensic audits: report-auditor per ticker (same rigor as ORCL).

| # | Ticker | Run | QC | Promoted (final/ + Drive + manifest) |
|---|--------|-----|----|--------------------------------------|
| — | ORCL | ✓ (pre-existing) | **A+** | ✓ final/wk 27 + Drive `1aUrPl21h5nIWJj9tn7mj7WvMo31k3nD5` + manifest |
| 1 | AAOI | ✓ | C → hand-corrected (convertible-maturity misattribution, unsourced 5.25% coupon, POC/HVN label, leaked meta, typos); re-verify in flight | – |
| 2 | AAPL | ✓ | A → hand-corrected (CoE 8%→9.9%, P/E + fair-value-anchor provenance, DCF/blend labels); re-verify in flight | – |
| 3 | AMKR | ✓ | validator fix (COHU EBITDA ≈−$1M→−$1.2M), 0/0; full audit in flight | – |
| 4 | ASX | ✓ | B → hand-corrected (exec fwd P/E 21.48→20.28, leaked preamble, false input-gap claims, op-leverage gloss) → **re-verified A+** | ✓ final/wk 27; Drive publish pending gog re-auth |
| 5 | COIN | ✓ | B → hand-corrected (stablecoin sub-line vs total S&S −13.5%); re-verify in flight | – |
| 6 | GOOGL | ✓ | B → hand-corrected (58.7→58.6%, IV provenance); re-verify in flight | – |
| 7 | IFNNY | ✓ | B → hand-corrected (surprise history +11.83→+11.22 ×3); re-verify in flight | – |
| 8 | INTC | ✓ | validator fix (net-debt/FCF sentence reorder; both figures true), 0/0; full audit in flight | – |
| 9 | MARA | ✓ | validator fix (prior-quarter total-debt clause reorder; all 4 figures true), 0/0; full audit in flight | – |
| 10 | MRVL | ✓ | **C** (falsely denied relative_multiples.json; EV $239.11B vs authoritative $215.70B; fwd P/E 44.0x vs 38.88x) → **re-run `--reuse-raw` in flight** | – |
| 11 | MSFT | ✓ | validator fix (fabricated April-2026 closes → real Oct-2023 closes from prices.json), 0/0; full audit in flight | – |
| 12 | NOW | ✓ | A → hand-corrected (−20.60→−20.61%, $2.75B provenance, LT-debt label); re-verify in flight | – |
| 13 | ONDS | ✓ | **A+** (0 issues) | ✓ final/wk 27; Drive publish pending gog re-auth |
| 14 | RKLB | ✓ | **A+** (0 issues) | ✓ final/wk 27; Drive publish pending gog re-auth |
| 15 | SATS | ✓ | B → hand-corrected (+$414.8M swing arithmetic gloss); re-verify in flight | – |
| 16 | SOUN | ✓ | **A+** (0 issues) | ✓ final/wk 27; Drive publish pending gog re-auth |
| 17 | STM | ✓ | validator fix (7/1 close $69.79→$70.72), 0/0; full audit in flight | – |
| 18 | TSM | ✓ | **A+** (0 issues) | ✓ final/wk 27; Drive publish pending gog re-auth |
| 19 | TXN | ✓ | validator fix (ND/EBITDA 1.21x subject-prefix disambiguation), 0/0; full audit in flight | – |
| 20 | VSH | ✓ | B → hand-corrected (TTM op margin 4.9→2.45%); re-verify in flight | – |
| 21 | ON | ✓ | A → hand-corrected (leaked QC-retry preamble, typos); re-verify in flight | – |

All hand-corrections followed cadence-run step 6: edit → re-run phase-7 validators (**all 21 now 0
blocking**; ASX/IFNNY/TSM carry the expected MINOR non-USD skip) → regenerate PDF (16 rebuilt). A
global "10-Q the 10-Q" vocab-residue sweep fixed 9 files across 7 tickers.

Tally: 6/22 in final/ (ORCL fully published; ASX/TSM/ONDS/RKLB/SOUN moved, Drive publish queued on re-auth).

## Phase 3 — sheets (blocked until all 22 promoted AND gog re-auth)

- [ ] Trading Plan (`refresh_trading_plan.sh`, sheet `1ZLq9…`) — script still points at the dead
  trueknotsg mount + account; patch before running (FINAL → shianpin mount, GOG_ACCOUNT → shainpin@trueknot.sg).
- [ ] Research Summary register (sheet `1VJow…`, update-summary skill).

## Blockers / owner action needed

- **gog OAuth `invalid_grant`** (token authed 2026-06-26, 7-day unverified-app expiry) — blocks Drive
  PDF publish + both sheet writes. Nothing half-published: the TSM trash/upload no-op'd cleanly, manifest
  untouched. **Owner: on the mini, in a browser signed in as shainpin@trueknot.sg, run**
  `gog auth add shainpin@trueknot.sg --services gmail,calendar,drive,docs,contacts,sheets,tasks,people`
  (with the keyring password exported in the shell, value in the macrodaily plist) → open the printed
  URL → Advanced → Allow. Then re-invoke the bot to finish publishing + Phase 3.
- **Google identity migrated trueknotsg@gmail.com → shainpin@trueknot.sg (~06-26)**, leaving stale refs:
  `com.trueknot.macrodaily.plist` (GOG_ACCOUNT + `--reports-dir` on the dead mount → **the 05:10 daily
  macro job has been failing since 06-26**) and `~/gsheet-tool/refresh_trading_plan.sh` (same two).
  Will patch both in Phase 3.

## Ops notes

- gog v0.31.1: no `drive trash` — use `gog drive delete <id> -y` (moves to trash). Account spelling in
  gog is `shainpin@trueknot.sg` (sic).
- ORCL audit history: first audit A (2 prose nits) → hand-corrected → re-verified **A+** → promoted.
- REGISTER.md "Week 27" section to be added once all 22 are promoted.
