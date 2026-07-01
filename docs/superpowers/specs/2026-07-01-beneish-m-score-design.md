# FA-101 WP4b — Annual Data Layer + Beneish M-Score

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Add an annual (yearly) financial-statement data layer, then use it to compute a deterministic Beneish M-score earnings-manipulation screen — completing FA-101 audit §10 (fraud/distress). The annual layer also unlocks the deferred multi-year CAGRs, which remain a follow-up (not in this spec).

---

## Background

WP4a shipped the Altman Z″ distress score but Beneish M-score was deferred: it is an **annual year-over-year** manipulation model and the pipeline fetched only quarterly statements. Exploration confirmed:
- The annual fetch is **already implemented** — `y_finance.py`'s `get_balance_sheet`/`get_income_statement`/`get_cashflow` already branch on `freq="annual"` (`ticker.balance_sheet`/`.income_stmt`/`.cashflow`) with identical `.to_csv()` formatting. Only `researcher._fetch_financials` needs wiring to request them.
- Free yfinance annual data **carries every Beneish input**, including the two rows most likely to be missing — `Selling General And Administration` and depreciation (`Reconciled Depreciation` in the income statement, `Depreciation And Amortization` in cashflow) — verified live on AAPL/MSFT.
- The existing `_parse_quarterly_csv` (format-generic, just misnamed) parses the annual CSVs unchanged; annual returns ~4 usable fiscal years, sometimes with an all-`None` 5th padding column (oldest).
- `distress_screens.py:5` already anticipates Beneish landing there.

## Design principles

- **Deterministic-block pattern:** compute in Python → write `raw/distress_screens.json` (extend the existing file) → append a block to `pm_brief.md`; the LLM cites it, never recomputes.
- **Free-data honesty:** Beneish needs a full current + prior annual year across 11 line-items. If ANY required input (incl. a missing prior year, or missing depreciation/SG&A) is absent → `m_score = None` → block renders `n/a`. Smaller-cap / foreign-ADR tickers will often be `n/a` — that is correct, not a failure.
- **Sector-appropriate:** skip financials (same gate as Altman Z — Beneish is calibrated for non-financials).
- **Bounded QC surface:** no new QC-checklist item; citation mandated in the fundamentals-analyst prompt only.
- **Reuse:** the annual fetch reuses the existing `freq="annual"` fetchers; the parser reuses `_parse_quarterly_csv`; the block lives in `distress_screens.py` beside Altman Z.

## Part 1 — Annual data layer

### Fetch (`tradingagents/agents/researcher.py` `_fetch_financials`)
Add three keys to the returned bundle, calling the existing fetchers with `freq="annual"`:
`balance_sheet_annual`, `income_statement_annual`, `cashflow_annual`. Same reuse/write path (`_gather_raw` → `financials.json`); `sanity=_id` is unaffected (it only checks ticker/trade_date). This also flows to peers harmlessly (peers just carry extra keys nothing reads).

### Parser (`tradingagents/agents/utils/financials_parser.py`)
Parse the three `*_annual` CSVs with `_parse_quarterly_csv` (mirroring lines 58-60). Expose a `beneish_inputs` sub-dict with **current-year (col 0)** and **prior-year (col 1)** values for the 11 line-items:

```
beneish_inputs = {
  "current": { receivables, sales, cogs, current_assets, ppe, total_assets,
               sga, depreciation, net_income, cfo, total_equity },
  "prior":   { receivables, sales, cogs, current_assets, ppe, total_assets,
               sga, depreciation, total_equity },   # prior needs no net_income/cfo (TATA is current-only)
}
```
- Sales = `Total Revenue`; COGS = `Cost Of Revenue`; SG&A = `Selling General And Administration`; PP&E = `Net PPE` / `Net Property Plant And Equipment`; receivables = `Receivables`/`Accounts Receivable`; CFO = `Operating Cash Flow` (cashflow_annual); net_income = `Net Income`.
- Depreciation: `Reconciled Depreciation` (income_statement_annual) with a `Depreciation And Amortization` / `Depreciation Amortization Depletion` (cashflow_annual) fallback.
- total_liabilities is derived downstream as `total_assets − total_equity`.
- Any field absent, or the prior-year column being the all-`None` padding → that value is `None` (compute degrades to n/a). Use `_row_at(rows, 0, ...)` / `_row_at(rows, 1, ...)`.

## Part 2 — Beneish M-score (`distress_screens.py`)

`compute_beneish_m(beneish_inputs: dict) -> dict` and extend `format_distress_block` (or add `format_beneish_block`) to render `## Manipulation screen (Beneish M-score)`.

**M = −4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI**

The 8 ratios (t = current, p = prior):
- **DSRI** = (receivables_t/sales_t) / (receivables_p/sales_p)
- **GMI** = GM_p / GM_t, where GM = (sales − cogs)/sales
- **AQI** = AQ_t / AQ_p, where AQ = 1 − (current_assets + ppe)/total_assets
- **SGI** = sales_t / sales_p
- **DEPI** = DR_p / DR_t, where DR = depreciation / (depreciation + ppe)
- **SGAI** = (sga_t/sales_t) / (sga_p/sales_p)
- **LVGI** = Lev_t / Lev_p, where Lev = total_liabilities/total_assets (total_liabilities = total_assets − total_equity)
- **TATA** = (net_income_t − cfo_t) / total_assets_t

**Flag:** `M > −1.78` → "elevated manipulation risk"; `M ≤ −1.78` → "normal" (the standard Beneish cutoff).

**Degradation:** if any ratio's inputs are missing or a denominator is 0 (via a shared `_div` returning None), OR the prior-year block is absent → `m_score = None`. Return `{"model": "Beneish M", "applicable": ..., "m_score": <float|None>, "flag": "elevated"|"normal"|None, <8 ratios>, "unavailable_reason": ...}`. Skip financials → `applicable: False`.

**Block** (`## Manipulation screen (Beneish M-score)`): M-score, the flag, and the 8 component ratios; the standard "use verbatim; do not recompute" mandate; a caveat that Beneish flags *risk of* manipulation, not proof, and is unreliable for financials/recent-IPOs/heavy-M&A. Skip/unavailable render the analogous "not applicable" / "n/a (data unavailable)" one-liners as Altman does.

## Wiring & citation
- `researcher.py`: after the Altman Z block, compute Beneish from `fin_parsed["beneish_inputs"]`, extend `raw/distress_screens.json` with the Beneish result, append the block to `pm_brief.md` — own try/except, fail-open (matching the sibling blocks).
- `fundamentals_analyst.py`: extend the distress-screen citation mandate so the analyst also cites the M-score + flag verbatim when applicable. No new QC item.

## Testing

- **Parser (`tests/test_financials_parser.py`):** with a fixture bundle carrying `*_annual` CSVs (≥2 real year columns + one all-`None` padding), assert `beneish_inputs["current"]`/`["prior"]` extract the right values and that a missing prior year / missing depreciation yields `None` (not a crash).
- **Beneish (`tests/test_distress_screens.py` or a new module):**
  - Clean-books fixture (stable ratios ≈ 1.0, small accruals) → `m_score` below −1.78, `flag == "normal"`, and equal to the hand-computed value.
  - Manipulation-pattern fixture (spiking DSRI, SGI, TATA) → `m_score` above −1.78, `flag == "elevated"`.
  - Financials fixture → `applicable == False`.
  - Missing prior year / missing depreciation / zero denominator → `m_score is None`, block renders `n/a`, no exception.
  - Block formatter contains `## Manipulation screen (Beneish M-score)`, the flag, and the "verbatim" mandate.
- **Real success measure (not a unit test):** sane M on a few live watchlist tickers (most legitimate large-caps below −1.78 = normal), financials skipped, and graceful `n/a` where the free annual data is thin — spot-check 1-2 smaller/foreign names.

## Out of scope (later / not this phase)

- Multi-year CAGRs (revenue/EPS/FCF 3-5yr) — now trivial on the annual series; a small follow-up addition to the accounting-ratios block.
- WP5 (new free sources), WP6 (qualitative prompts).
- A new QC-checklist item for the manipulation screen.
- Using annual data to re-base other ratios (the existing quarterly/TTM ratios stay as-is).
