# Cadence ops progress — full 22-ticker rerun for trade date 2026-07-01

**Authoritative status doc** for the production cadence (CADENCE_OPS_GOAL.md).
Updated + pushed per ticker so the run can be resumed from any session.

- Code: prod `~/tradingagents` @ `efd75c7` (FA-101 + pro-deck + debt-ladder blocks + mount decouple).
- Base: `TK_RESEARCH_BASE=$HOME/tkresearch` → runs in `~/tkresearch/preaudit/2026-07-01-<T>/`.
- **Promote target (2026-07-03, decoupled — Option A shipped): LOCAL `~/tkresearch/final/wk 27 2026/`**
  is the canonical published store. The GUI-session-tied Drive mount is no longer touched by any
  publish op (it unmounts when the shared mini switches GUI user — that's what stalled this cadence).
  See `2026-07-03-decouple-drive-mount.md`. The shianpin Drive `final/` tree is a convenience mirror only.
- Drive publish: idempotent by file ID via `~/gsheet-tool/pdf_ids.tsv` (all 22 present) → upload new to
  pdf/ root `1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs`, overwrite manifest row, then `gog drive delete <old> -y`
  (no `trash` subcommand in gog v0.31). Helper `~/tkresearch/publish-2026-07-01.sh` patched: local final,
  upload-before-delete, `-a shianpin@trueknot.sg` (correct gog spelling per `gog auth list`).
- No Telegram delivery (`TRADINGRESEARCH_NO_TELEGRAM=1`).

## Phase 1 — runner: **COMPLETE**

- All 21 tickers ran sequentially 2026-07-03 00:05–14:18 SGT, **21/21 DONE**, zero 429s/timeouts
  (`~/tkresearch/cadence-2026-07-01-status.tsv`, `ALL COMPLETE 14:23:01`).

## Phase 2 — QC + promote (in progress)

Deterministic sweep: all 21 run dirs complete (decision/exec/PDF/state/validation_report all present);
**all 21 reference prices match fresh yfinance settled closes exactly** (no intraday-bar capture).
Forensic audits: report-auditor per ticker (same rigor as ORCL).

| # | Ticker | Run | QC | Promoted (LOCAL final/ + Drive-by-ID + manifest) |
|---|--------|-----|----|--------------------------------------|
| — | ORCL | ✓ (pre-existing) | **A+** | ✓✓ local final/wk 27 + Drive `1aUrPl21h5nIWJj9tn7mj7WvMo31k3nD5` + manifest |
| 1 | AAOI | ✓ | C → corrected (convertible-maturity misattribution, unsourced 5.25% coupon, POC/HVN label, leaked meta) → residuals fixed (af102 + state mirrors); final verify in flight | – |
| 2 | AAPL | ✓ | A → corrected (CoE 8%→9.9%, provenance, DCF/blend labels) → residuals fixed (exec + debates + state); final verify in flight | – |
| 3 | AMKR | ✓ | F (fabricated ROE 9.15% vs cell 9.62 ×3 + leaked preamble; ALL else A+-clean) → corrected; final verify in flight | – |
| 4 | ASX | ✓ | B → corrected → **re-verified A+** | ✓✓ local final/wk 27 + Drive `1qGGf27DQB67GQahf4vXMKpkewk6qQOt_` + manifest (2026-07-03) |
| 5 | COIN | ✓ | B → corrected (stablecoin vs total S&S −13.5%) → debate residual fixed; final verify in flight | – |
| 6 | GOOGL | ✓ | B → corrected (58.7→58.6% ×7, IV provenance, −42.1/−40.2 attribution); final verify in flight | – |
| 7 | IFNNY | ✓ | B → corrected (surprise +11.83→+11.22, all copies incl. debates + state); final verify in flight | – |
| 8 | INTC | ✓ | B (22.46→22.45% ×6, cols 0/4, 36-mo HVN label) → corrected; final verify in flight | – |
| 9 | MARA | ✓ | A (−104.46% "not reconstructible" claim false — it's the TTM op-margin cell; rel-mult attribution) → corrected; final verify in flight | – |
| 10 | MRVL | ✓ | **C** (falsely denied relative_multiples.json; EV $239.11B vs authoritative $215.70B; fwd P/E 44.0x vs 38.88x) → **re-run `--reuse-raw` in flight** | – |
| 11 | MSFT | ✓ | **C** (false "conclusive filesystem check — no accounting-ratios/rel-mult/IV artifacts" claim; all 3 exist; leaked meta) → **re-run `--reuse-raw` queued behind MRVL** | – |
| 12 | NOW | ✓ | A → corrected → **re-verified A+** | ✓✓ local final/wk 27 + Drive `1yfL8VRBuIpK_5kY3qmLstvlzIyVL6OCr` + manifest (2026-07-03) |
| 13 | ONDS | ✓ | **A+** (0 issues) | ✓✓ local final/wk 27 + Drive `1zr3lSQxnfqDeZJLxbgREtEZQArmsAbiL` + manifest (2026-07-03) |
| 14 | RKLB | ✓ | **A+** (0 issues) | ✓✓ local final/wk 27 + Drive `1r5rYZHDy_AaArerlDy4hXtguWpOA1Zzh` + manifest (2026-07-03) |
| 15 | SATS | ✓ | B → corrected (+$414.8M swing gloss) → A; state.json mirrors patched per auditor → **A+** | ✓✓ local final/wk 27 + Drive `1Z7XaSTbZvirRyQ1T5CccJ7tMUmyvypLg` + manifest (2026-07-03) |
| 16 | SOUN | ✓ | **A+** (0 issues) | ✓✓ local final/wk 27 + Drive `1MVfCdTSvyGfyQ6AvpPBrenPH4Ctr4Zmh` + manifest (2026-07-03) |
| 17 | STM | ✓ | validator fix ($69.79→$70.72) → **full audit A+** | ✓✓ local final/wk 27 + Drive `1SimxSkrQ_noRza7x4j-xI4qgO-C0vLt0` + manifest (2026-07-03) |
| 18 | TSM | ✓ | **A+** (0 issues) | ✓✓ local final/wk 27 + Drive `1pkXRawnR-zzC0AuTy0t4RFn4Tw7stn7d` + manifest (2026-07-03) |
| 19 | TXN | ✓ | validator fix (ND/EBITDA subject-prefix) → **full audit A+** | ✓✓ local final/wk 27 + Drive `1l7uohWYVgqJWK0MeKr_vGwA512zTH8RT` + manifest (2026-07-03) |
| 20 | VSH | ✓ | B → corrected (TTM op margin 2.45%) → **re-verified A+** | ✓✓ local final/wk 27 + Drive `1uYpZ-ErnxCTU5QqI6dvFlPKmo7sQPqiR` + manifest (2026-07-03) |
| 21 | ON | ✓ | A → corrected (leaked preamble, typos) → **re-verified A+** | ✓✓ local final/wk 27 + Drive `13KMSLssTWBXg7rg8jg0eW3O3PxBRXDOw` + manifest (2026-07-03) |

All hand-corrections followed cadence-run step 6: edit → re-run phase-7 validators (**all 0
blocking**; ASX/IFNNY/TSM carry the expected MINOR non-USD skip) → regenerate PDF. A global
"10-Q the 10-Q" vocab-residue sweep fixed 9 files across 7 tickers. Deliverable-mirror fields in
state.json synced for all corrected tickers (`.past_context` + historical QC-round records left
as immutable history). A deterministic residual sweep (md + state.json + extracted PDF text per
stale token) is CLEAN for all 9 hand-corrected tickers.

Tally (2026-07-03 21:40 SGT): **12/22 FULLY published** (local final + Drive-by-ID + manifest,
all 12 IDs verified resolving to research-2026-07-01-*.pdf in pdf/ root). The 12 run dirs were
reconciled off the (temporarily re-mounted) shianpin Drive into `~/tkresearch/final/wk 27 2026/`
(file-count-verified vs source; a 520-file stray `Users/…` tree nested inside VSH — byte-identical
duplicates of 11 already-promoted dirs from the stalled promote — was removed from both copies).
In flight: 9 report-auditor final verifications (AAOI AAPL AMKR COIN GOOGL IFNNY INTC MARA + MRVL
full re-audit of its completed 20:15 re-run); MSFT `--reuse-raw` re-run started 21:33 (pid 46100).

## Phase 3 — sheets (after all 22 promoted; gog is VALID again)

- [ ] Trading Plan (`refresh_trading_plan.sh`, sheet `1ZLq9…`) — script PATCHED 2026-07-03:
  `--reports-dir ~/tkresearch` (local, recursive), `GOG_ACCOUNT=shianpin@trueknot.sg`.
- [ ] Research Summary register (sheet `1VJow…`) — `~/gsheet-tool/update_summary.py` PATCHED:
  FINAL → `~/tkresearch/final`, ACCT → shianpin.

## Blockers — none (2026-07-03 21:40 SGT)

- ~~gog OAuth invalid_grant~~ **RESOLVED**: owner re-authed; `gog drive get` + 11 publishes succeeded.
- ~~stale identity refs~~ **PATCHED**: installed `com.trueknot.macrodaily.plist` now
  `GOG_ACCOUNT=shianpin@trueknot.sg` + `--reports-dir /Users/trueknot/tkresearch` (launchctl reloaded —
  the 05:10 daily macro job is fixed); `refresh_trading_plan.sh` + `update_summary.py` same.
- **Decouple shipped (Option A)**: commits 192d850…efd75c7 — see
  `2026-07-03-decouple-drive-mount.md`. No publish op touches ~/Library/CloudStorage anymore.

## Ops notes

- gog v0.31.1: no `drive trash` — use `gog drive delete <id> -y` (moves to trash). Account spelling in
  gog is `shianpin@trueknot.sg` (per `gog auth list` after the 07-03 re-auth; the earlier "shainpin"
  note is obsolete).
- ORCL audit history: first audit A (2 prose nits) → hand-corrected → re-verified **A+** → promoted.
- REGISTER.md "Week 27" section to be added once all 22 are promoted.
