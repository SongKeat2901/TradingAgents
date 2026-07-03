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
| 1 | AAOI | ✓ | C → corrected → re-audit **B** (sole issue: 4 pre-correction Note-3/6/Note-10 attributions in 3 state.json mirror fields; zero numeric errors) → mirrors synced to 10-Q phrasing, validators 0/0, PDF regen, sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `1i9i4uZKa3AfDwFHEYSAKuPDUV_pWbHK9` + manifest (2026-07-03) |
| 2 | AAPL | ✓ | A → corrected → re-audit **B** (3 residual "8% hurdle" phrasings in debates + state mirrors; all numbers traced flawless) → residuals fixed, validators 0/0, PDF regen, token sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `1MUnrkoPKfYV3IzGJTaxemirbQrV2699J` + manifest (2026-07-03) |
| 3 | AMKR | ✓ | F → corrected → re-audit F (residual ~9.1% + preambles) → fixed → re-audit #2 **B** (sole family: "zero buys" claims vs ~$9.6M in-window Kim/Director purchases in insider.json) → all 6 sites + mirrors normalized to data-accurate wording, validators 0/0, PDF regen, sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `18IcmV_I7wiiGqUwCFVnZTwkDVEDyFdeW` + manifest (2026-07-03) |
| 4 | ASX | ✓ | B → corrected → **re-verified A+** | ✓✓ local final/wk 27 + Drive `1qGGf27DQB67GQahf4vXMKpkewk6qQOt_` + manifest (2026-07-03) |
| 5 | COIN | ✓ | B → corrected → re-audit **B** (RM "Why not Sell" still labeled the $305.4M stablecoin sub-line as S&S; "(10-Q Note 4)" survived in 2 state mirrors) → both fixed incl. mirrors, validators 0/0, PDF regen, sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `1VaPi-O15pElQF2yODtC4TvHH7CwUkKFv` + manifest (2026-07-03) |
| 6 | GOOGL | ✓ | B → corrected → re-audit **A** (leaked QC-retry preamble atop decision.md; META-date aside; caveat word-order garble in PDF vocab-sub) → all 3 fixed incl. state variants, validators 0/0, PDF regen, sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `1IAY9f1di6-S6D_lS-4zcWJMQZs1oexNR` + manifest (2026-07-03) |
| 7 | IFNNY | ✓ | B → corrected → re-audit **A+** (0 issues; surprise history verified everywhere incl. ~20 state.json copies) | ✓✓ local final/wk 27 + Drive `1nnl5LL41TWDFj7m_deAYUjtdFftlPrvu` + manifest (2026-07-03) |
| 8 | INTC | ✓ | B → corrected → re-audit **A** (2 prose labels: $4,070M restructuring misattributed to income_statement col 0 [it's 10-Q-only]; AMD mislabeled cheapest-margin peer [GFS 11.0%]) → both fixed incl. state variants, validators 0/0, PDF regen, sweep clean → **A+** | ✓✓ local final/wk 27 + Drive `1Ewk9O58Mh7eCluAl46A-t8H9UdalA_ce` + manifest (2026-07-03) |
| 9 | MARA | ✓ | A → corrected → re-audit **A+** (0 issues; −104.46% independently reconstructed as TTM op-margin) | ✓✓ local final/wk 27 + Drive `1nmk4s8gqDLptvSVMondGaNYBa969oSxi` + manifest (2026-07-03) |
| 10 | MRVL | ✓ | C → re-run #1 (20:15) audited **C again**: repeated the rel-mult denial verbatim (self-built EV $239.1B vs authoritative $215.70B; fwd P/E fixed), qc_passed=false (died at QC: claude CLI exited 1 ×3 ≈ OAuth 8h expiry, NOT a QC verdict), stale Overweight exec vs fresh Underweight decision → **re-run #2 `--reuse-raw` started 22:13 (07-03, pid 56913, fresh OAuth)**; regenerated pm_brief.md verified to carry all deterministic blocks incl. rel-mult | – |
| 11 | MSFT | ✓ | **C** (false "conclusive filesystem check" claim; leaked meta) → re-run `--reuse-raw` (21:33) completed 22:10, qc_passed=true (0 retries), validators 0/0 → **forensic re-audit A+** (all 16 tiers PASS, 0 issues; §D verdict TRUE — prompt-scoped and accurate per portfolio_manager.py ~L352 4-section injection; DCF base $330.92 + CoE 9.815% independently recomputed; market cap matches to the dollar; ref $384.28 = fresh settled close) | ✓✓ local final/wk 27 + Drive `1sBLBXxW6332MwUbXHhYGwpI-C_AzYCTT` + manifest (2026-07-03; old `1m55cX…` trashed) |
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

Tally (2026-07-03 22:45 SGT): **21/22 FULLY published** (local final + Drive-by-ID + manifest).
The earlier 12 run dirs were reconciled off the (temporarily re-mounted) shianpin Drive into
`~/tkresearch/final/wk 27 2026/` (file-count-verified vs source; a 520-file stray `Users/…` tree
nested inside VSH — byte-identical duplicates of 11 already-promoted dirs from the stalled
promote — was removed from both copies). MSFT promoted+published 22:45 after its A+ re-audit.
Remaining 1: **MRVL** re-run #2 in flight (started 22:13 with fresh OAuth — re-run #1's death
was CLI exit-1 ×3 at the QC node ≈ OAuth expiry, so #2 is expected to reach QC).

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
