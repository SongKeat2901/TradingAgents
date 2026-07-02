# Free-finish progress — debt ladder → prod promote → ORCL test → audit loop (2026-07-02)

Tracking doc for `FREE_FINISH_GOAL.md`. Resume from here if interrupted.

## Phase status

| Phase | State |
|---|---|
| 1. Debt maturity ladder (fetch + block + role directive + refi upgrade), TDD + live smoke | ✅ shipped (`13e2b5a` + `9c50b79`) |
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
- **Ladder shipped (TDD, 24 new tests, suite 978 passed):**
  - `13e2b5a` — sec_edgar: `DEBT_MATURITY_EXCERPT_KEYWORDS` +
    `extract_debt_maturity_excerpts` (own budget, can't be starved by generic
    keywords), attached to `fetch_latest_filing` from the FULL text (survives
    the 60K truncation), `fetch_debt_maturity_note` (latest-10-K fallback),
    `format_debt_maturity_block` (verbatim quote or honest n/a).
  - `9c50b79` — wiring: pm_preflight writes `raw/debt_maturity.json` + block
    (inline excerpts from the latest filing preferred — MSFT repeats the table
    in its 10-Q — else 10-K note fetch; fail-open n/a always lands);
    refi proxy cross-references the full ladder when available; Risk role
    gains REQUIRED `## Debt maturity ladder` section (verbatim or "full
    ladder not disclosed").
- **Live smoke (real EDGAR, 2026-07-02):**
  - ORCL 10-K 2026-06-22: `future principal payments` excerpt carries the full
    ladder verbatim — "Fiscal 2027 $ 7,210 Fiscal 2028 10,145 Fiscal 2029 5,500
    Fiscal 2030 7,250 Fiscal 2031 9,750 Thereafter 90,250 Total $ 130,105"
    (in millions). No refetch (inline from the latest filing).
  - MSFT: latest is the 10-Q 2026-04-29 which repeats the maturities table —
    smoke caught the possessive phrasing "maturities of OUR long-term debt"
    (added keyword; guarded against the "debt investments" near-miss).
  - VZ (high-debt): 10-Q latest has no ladder → 10-K 2026-02-17 fallback found
    "2026 $ 17,267 2027 9,569 2028 13,032 2029 8,115 2030 11,081 Thereafter
    96,753" verbatim.

## Prod promote status

Not yet promoted.

## ORCL run status

Not yet run.

## Audit verdict

Not yet audited.
