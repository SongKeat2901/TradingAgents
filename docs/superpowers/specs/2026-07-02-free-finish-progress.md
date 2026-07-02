# Free-finish progress — debt ladder → prod promote → ORCL test → audit loop (2026-07-02)

Tracking doc for `FREE_FINISH_GOAL.md`. Resume from here if interrupted.

## Phase status

| Phase | State |
|---|---|
| 1. Debt maturity ladder (fetch + block + role directive + refi upgrade), TDD + live smoke | 🔄 in progress |
| 1b. Other free gaps (only if Phase-4 audit flags them) | ⬜ not started |
| 2. Promote main → production `~/tradingagents` (safety-gated) | ⬜ not started |
| 3. ORCL run on production | ⬜ not started |
| 4. Audit vs FA-101 + pro-deck; loop until SATISFIED | ⬜ not started |

## Phase 1 design (decided 2026-07-02)

- `sec_edgar.py`: `DEBT_MATURITY_EXCERPT_KEYWORDS` ("maturities of long-term debt",
  "aggregate maturities", "future principal payments", …) → windowed verbatim excerpts
  computed from the FULL stripped filing text (not the truncated head), attached to the
  filing dict as `debt_maturity_excerpts`. Plus `fetch_debt_maturity_note(ticker, date)`:
  when the latest filing is a 10-Q (ladder lives in the 10-K debt note), fall back to
  fetching the latest 10-K on/before trade_date for the debt excerpts only. Fail-soft n/a.
- `pm_preflight.py`: write `raw/debt_maturity.json` + append `## Debt maturity ladder`
  pm_brief block — verbatim excerpt quote + source (form/filing date/URL), or honest
  n/a ("full ladder not disclosed in the fetched filings") with a do-not-fabricate
  directive. Own try/except (fail-open).
- `distress_screens.format_refinancing_block`: gains ladder-awareness — when the full
  ladder block is present, the proxy block points to it; else keeps the "NOT the full
  ladder" caveat.
- Risk & Red-Flags role directive: cite the year-by-year ladder VERBATIM when the block
  discloses it; else cite the proxy + "full ladder not disclosed". Never invent a schedule.
- Live smoke: ORCL (10-K, ladder expected) + a high-debt name (VZ or T) + a 10-Q-latest
  name to prove the 10-K fallback.

## Evidence log

- (starting) Progress doc created; design decided; TDD next.

## Prod promote status

Not yet promoted.

## ORCL run status

Not yet run.

## Audit verdict

Not yet audited.
