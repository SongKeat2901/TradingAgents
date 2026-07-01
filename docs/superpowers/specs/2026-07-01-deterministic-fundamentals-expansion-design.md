# Deterministic Fundamentals Expansion — Phase 1 Design

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Phase 1 of closing the gaps identified in the 2026-07-01 Fundamental-Analysis-101 audit. Covers WP1 (wire up already-fetched data), WP2 (accounting-ratios block), WP3 (relative-multiples block). All deterministic; no new external data sources.

---

## Background

An audit of the Phase 6 research pipeline against `Fundamental Analysis 101.md` (13 sections, ~101 sub-items) found ~25-30% coverage. The pipeline is deep and bulletproof on a narrow set of load-bearing numbers (net debt, peer ratios, intrinsic value, earnings calendar, SEC-filing anchoring) but broad classic-fundamentals coverage is thin.

Biggest deterministic-fixable gaps:
- Classic accounting ratios almost entirely absent (ROE/ROA/ROIC/ROCE, DuPont, current/quick, turnover DSO/DPO/CCC, payout/dividend/buyback/total-shareholder yield) — despite several being free yfinance fields already in the raw dump that nothing reads (`tradingagents/dataflows/y_finance.py:281-305`).
- Relative multiples beyond P/E missing (EV/EBITDA, EV/EBIT, EV/Sales, P/B, P/S, P/FCF, PEG). EV/EBITDA explicitly flagged as an unfilled data gap at `intrinsic_value.py:252`.
- Two "dead pipeline" outputs: `insider.json` fetched (`researcher.py:150-153`) but never read by any prompt; earnings-surprise magnitude discarded at `calendar.py:114`.

**Sequencing decision:** phased, cheapest-highest-value first. Phase 1 = WP1+WP2+WP3 (deterministic, no new data sources). WP4 (fraud/distress screens), WP5 (new free data sources), WP6 (qualitative prompt upgrades) are later phases with their own specs.

**Enforcement decision:** full "block + cite + QC" — matches the existing net-debt / peer-ratios treatment.

**Foundation decision:** build one shared deterministic financials parser now (pulls WP4's data layer — working-capital line-items — forward), so ratios/multiples/screens all read from a single source of truth.

## Design principles (non-negotiable)

- **Deterministic-block pattern.** Compute in Python, append the result to `pm_brief.md` as ground truth; the LLM cites it, never recomputes it. (Ref: `project_phase6_deterministic_blocks_pattern`.)
- **Free-data honesty.** Every metric renders `n/a (data unavailable)` when a source line-item is missing. Never fabricate, never estimate to fill a blank. Blocks must degrade cleanly for data-sparse ADRs and financials.
- **Don't destabilize the strongest area.** Do NOT refactor `intrinsic_value.py`'s existing `parse_fundamentals` in Phase 1. The new parser is additive. Consolidation is a noted future follow-up, not part of this phase.
- **Bounded QC surface.** Grouped QC checklist items, not one per ratio, to keep rerun risk sane.

---

## Architecture & module layout

Three new modules under `tradingagents/agents/utils/`, appended around `pm_preflight` at the same orchestration point as `net_debt.py` / `peer_ratios.py` / `calendar.py`:

1. **`financials_parser.py`** — single source of truth.
   - `parse_financials(financials_json) -> Financials` extracts every line-item once: income statement, balance sheet, cash flow, both quarterly and annual, and computes TTM (trailing-four-quarters) where applicable.
   - Line-items extracted: revenue, COGS, gross profit, SG&A, R&D, operating income (EBIT), EBITDA, interest expense, net income, EPS (basic & diluted), diluted share count; total assets, current assets, cash & equivalents, short-term investments, receivables, inventory, PP&E, goodwill/intangibles, current liabilities, payables, total debt, total equity; CFO, capex, free cash flow, dividends paid, buybacks (repurchase of stock), stock-based comp.
   - Returns a typed dict/dataclass with `None` for any line-item not present in the raw dump. Includes a small helper for average-balance (mean of current and prior period) used by turnover ratios.
   - **No fabrication:** if a line-item row is absent, the field is `None`; downstream renders `n/a`.

2. **`accounting_ratios.py`**
   - `compute_accounting_ratios(fin: Financials, wacc: float | None) -> dict`
   - `format_accounting_ratios_block(ratios) -> str` → `## Accounting ratios` section for `pm_brief.md`.

3. **`relative_multiples.py`**
   - `compute_relative_multiples(fin, subject_market_cap, peer_data) -> dict`
   - `format_relative_multiples_block(...) -> str` → `## Relative valuation multiples` section for `pm_brief.md`.

---

## WP1 — Wire up already-fetched data

### Insider transactions (fix dead `insider.json`)
- Add `insider.json` to `fundamentals_analyst.py`'s `files=[...]` list.
- Add a mandated prompt subsection: net buy/sell over the last 6-12 months, notable individuals (CEO/CFO/directors), cluster buying/selling. If the file is empty, state "no reported insider transactions in window."

### Earnings-surprise history (stop discarding at `calendar.py:114`)
- Extend `calendar.py` to extract `EPS Estimate` and `Surprise(%)` from the yfinance `earnings_dates` rows it already reads.
- Surface the last 4-8 reported surprises (date, actual, estimate, surprise %) in the reporting-status block written via `pm_preflight`.
- Beat/miss streak summarized (e.g., "beat 6 of last 8").

Note: the "surface unused yfinance summary fields" idea from the audit is intentionally NOT done separately — WP2 computes ROE/P/B/D/E/etc. properly from statements, superseding the point-in-time summary fields.

---

## WP2 — Accounting Ratios block

All computed from `financials_parser` output. Basis conventions: **flow metrics use TTM**; **stock/balance metrics use latest reported period**; **turnover ratios use average balance** (mean of current and prior period) when two periods are available, else latest with a note.

- **Profitability:** gross / operating / net / FCF margins; ROE, ROA, ROIC, ROCE; **ROIC-vs-WACC spread** (WACC pulled from the intrinsic-value block output → value-creation test). ROIC = NOPAT / (total debt + equity − cash); document the exact formula in code.
- **DuPont:** ROE decomposed = net margin × asset turnover × equity multiplier (three factors must reconcile to ROE within rounding).
- **Liquidity:** current, quick, cash ratios.
- **Leverage:** debt/equity; **net-debt/EBITDA for the SUBJECT** (today only peers have it — reuse the net-debt block's figure so the two never disagree); interest coverage (EBIT / interest expense); FCF/total-debt.
- **Efficiency:** asset turnover; inventory turnover & DIO; DSO; DPO; cash-conversion-cycle (DIO + DSO − DPO).
- **Return of capital:** payout ratio (dividends / net income); dividend yield (dividends per share / price); buyback yield (net buybacks / market cap); total shareholder yield (dividend + buyback yield).
- **Growth / quality:** revenue / diluted-EPS / FCF CAGRs over 3yr and 5yr (as annual history allows); CFO-vs-NI accruals ratio (CFO / net income — a quality flag; <1 sustained = low earnings quality).

Rendering: a markdown table grouped by the buckets above; any metric lacking inputs shows `n/a (data unavailable)` with no numeric guess.

---

## WP3 — Relative Multiples block

Subject + **peer-median**, for: P/E (trailing & forward), PEG, EV/EBITDA, EV/EBIT, EV/Sales, P/B, P/S, P/FCF.

- **Peer market cap** must be fetched (currently absent — the root cause of the `intrinsic_value.py:252` EV/EBITDA gap). Extend `peer_ratios.py`'s per-peer fetch to also pull market cap; reuse that in `relative_multiples.py` so peer EV multiples become computable.
- **EV consistency:** EV = market cap + net debt, using the SAME net-debt figure the net-debt block computed — the two blocks must tie out. QC item enforces this.
- PEG = trailing P/E / forward EPS-growth %.
- Self-vs-own-history multiple bands: **DEFERRED** (YAGNI for Phase 1). Flag as a future enhancement only; do not build.

Rendering: subject value | peer median | subject's percentile-vs-peers where meaningful. `n/a` for any multiple missing its inputs.

---

## Enforcement (block + cite + QC)

### Citation mandates
- `fundamentals_analyst.py`: mandate citing the accounting-ratios and relative-multiples blocks (ROE/ROIC/leverage; the multiples table) rather than recomputing.
- `portfolio_manager.py`: reference the ROIC-vs-WACC spread and relative-multiples positioning in the thesis.

### QC checklist (grouped, added to `qc_agent.py`)
1. "Accounting ratios block present; ROE / ROIC / leverage cited; no contradiction with raw statement figures."
2. "Relative multiples block present & internally consistent — EV math ties to the net-debt block; no fabricated peer multiples."

Grouped intentionally (not one item per ratio) to bound rerun risk. Mirror into the PM's own inline checklist where that pattern already exists.

---

## Graceful degradation & testing

### Degradation
- `financials_parser` yields `None` for missing line-items; formatters render `n/a (data unavailable)`.
- No block aborts the pipeline on missing data — it renders with `n/a` cells. (Contrast: peer-extraction empty-peers is a hard abort; ratios are not load-bearing enough to abort on.)

### Tests (follow existing `tests/` patterns)
Unit tests per ratio family using fixture `financials.json` for three archetypes:
1. **Clean compounder** — all line-items present; every ratio computes; DuPont reconciles to ROE; EV ties to net debt.
2. **Cyclical** — negative/volatile earnings; verify no divide-by-zero, sensible `n/a` on negative-denominator ratios (e.g., PEG on negative growth).
3. **Data-sparse ADR** — many missing line-items; verify graceful `n/a` throughout, no fabrication, no crash.

Additional targeted tests: net-debt/EBITDA subject figure matches the net-debt block; EV in relative-multiples matches net-debt block EV.

---

## Out of scope (later phases)

- WP4 — fraud/distress *screens* (Altman Z-score, Beneish M-score) — data layer is pulled forward here; the scores themselves are Phase 2.
- WP5 — new free data sources (SEC EDGAR 8-K, DEF 14A proxy, 13F/institutional holders; yfinance short interest, analyst recommendations).
- WP6 — qualitative prompt upgrades (Porter's Five Forces, moat durability, management/capital-allocation, customer concentration, governance/share-class, litigation, ESG-if-material).
- Consolidating `intrinsic_value.py`'s `parse_fundamentals` onto the new shared parser.
- Self-vs-own-history relative-multiple bands.
