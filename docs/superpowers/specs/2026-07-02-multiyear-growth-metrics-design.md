# FA-101 Coverage — Multi-Year Growth Metrics (CAGRs + Operating Leverage)

**Date:** 2026-07-02
**Status:** Approved (self-approved under the standing FA-101 alignment goal), pending plan
**Scope:** Close FA-101 §4 (Growth & Quality) deterministic gaps: multi-year revenue/EPS/FCF CAGRs and operating leverage, computed from the annual-statement layer WP4b already added. Deterministic block extension — no new LLM node, no new data source, no new QC item. First phase of the FA-101-alignment program (coverage → roles → per-role retry).

---

## Background

The 2026-07-02 FA-101 re-audit found §4 still missing multi-year CAGRs, operating leverage, and incremental ROIC — all deferred in Phase 1 for lack of annual data. WP4b's annual data layer (`balance_sheet_annual`/`income_statement_annual`/`cashflow_annual` in the bundle; `financials_parser` parses them via `_parse_quarterly_csv`) now supplies the multi-year series. This phase computes the growth metrics from it and surfaces them in the existing `## Accounting ratios` block.

Incremental ROIC is deferred to a later phase — it needs careful NOPAT/invested-capital year-deltas that are noisy on free data; CAGRs + operating leverage are the clean, high-value §4 items.

## Design principles

- **Deterministic-block pattern:** compute in Python from the annual series, render into the existing `## Accounting ratios` block; the LLM cites, never recomputes.
- **Free-data honesty:** any metric lacking ≥2 valid annual years, or with a non-positive endpoint where a ratio/root is undefined, renders `n/a (data unavailable)`. Never fabricate.
- **Reuse:** the annual CSVs are already parsed; add series extraction beside `beneish_inputs`; compute in `accounting_ratios.py` beside the existing YoY growth.
- **No new node/QC item** — pure extension of an existing block.

## Part 1 — annual series in `financials_parser`

Add to `parse_financials(...)` (the annual CSVs `is_a`/`cf_a` are already parsed for `beneish_inputs`): expose most-recent-first annual series as `annual_series`:

```
annual_series = {
  "revenue": [y0, y1, y2, y3, ...],       # income_statement_annual "Total Revenue"/"Operating Revenue"
  "diluted_eps": [...],                     # "Diluted EPS"
  "ebit": [...],                            # "Operating Income"/"EBIT"
  "fcf": [...],                             # cashflow_annual "Free Cash Flow"; else per-year (Operating Cash Flow + Capital Expenditure)  [capex is negative in yfinance, so CFO + capex = CFO - |capex|]
}
```
Each is the full column list from the parsed annual rows (a helper `_row_series(rows, *aliases)` returning `rows.get(alias)` for the first matching alias, else `[]`). Trailing all-`None` padding columns are left as-is (the CAGR/OL computation filters them).

## Part 2 — CAGRs + operating leverage in `accounting_ratios.py`

`compute_accounting_ratios(fin, ...)` gains (reads `fin.get("annual_series")`):

- **`revenue_cagr_pct`, `eps_cagr_pct`, `fcf_cagr_pct`** — for each series: drop trailing `None`s, require ≥2 values with `oldest > 0` and `latest > 0`; `cagr = ((latest/oldest) ** (1/(n-1)) - 1) * 100`, rounded; `years = n-1`. Else `None`. (Report the span used, e.g. "3y".) A helper `_cagr(series) -> (pct_or_None, years)`.
- **`operating_leverage`** — using annual revenue + ebit series' two most-recent valid years: `(pct_change(ebit)) / (pct_change(revenue))`, where `pct_change(x)=(x_t - x_{t-1})/abs(x_{t-1})`; require prior values present and `revenue` pct-change ≠ 0 and prior revenue/ebit > 0; else `None`. Rounded.

Rendered as new rows in `format_accounting_ratios_block(...)`: "Revenue CAGR (Ny)", "EPS CAGR (Ny)", "FCF CAGR (Ny)", "Operating leverage (ΔEBIT%/ΔRev%, latest yr)". Missing → `n/a (data unavailable)`.

## Testing

- **Parser (`tests/test_financials_parser.py`):** with an annual bundle carrying a 4-year revenue/eps/ebit series (+ a padding `None`), `annual_series["revenue"]` is the full most-recent-first list; a series absent → `[]` (no crash).
- **Ratios (`tests/test_accounting_ratios.py`):**
  - A hand-computed 4-year revenue series (e.g. `[133.1, 121, 110, 100]` → 3-yr CAGR = 10.0%) → `revenue_cagr_pct == 10.0`.
  - Operating leverage: ebit up 20%, revenue up 10% → `operating_leverage == 2.0`.
  - Non-positive base (oldest ≤ 0) or <2 years → `None`.
  - Block renders the new rows; missing → `n/a`.

## Out of scope (later phases of the FA-101-alignment program)

- Incremental ROIC (needs invested-capital deltas; noisy on free data) — later.
- WP5 data sources; red-flag screens (goodwill/commodity); the clear-role multi-agent restructure; per-role retry; macro-in-report — later phases.
