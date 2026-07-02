# FA-101 Phase 5 — Three Deterministic Coverage Blocks (incremental ROIC, goodwill-impairment flag, commodity-exposure flag)

**Date:** 2026-07-02
**Status:** Approved (self-approved under the standing FA-101 alignment goal), pending plan; **build after Phase 4 merges to main** (independent, but keeps a clean base)
**Scope:** Close the last *tractable* FA-101 coverage gaps (Tier A of the coverage audit) as deterministic `pm_brief.md` blocks on data already fetched — no new source, no new LLM node. Covers §4 incremental ROIC + reinvestment runway, and §7 goodwill-impairment risk + commodity input exposure.

---

## Background

The 2026-07-02 coverage audit (`docs/…/2026-07-02-fa101-coverage-audit.md`) found §1-9 largely covered; the only deterministic gaps left on free data are three §4/§7 items deferred earlier. All inputs are already parsed by `financials_parser` (verified): `annual_series.ebit`, annual balance-sheet (`bs_a`), latest `goodwill`/`total_assets`/`total_equity`/`total_debt`/`tax`/`net_income`, and `classification.json`/yfinance `sector`+`industry`.

## Design principles

- **Deterministic-block pattern:** compute in Python → write `raw/<name>.json` → append a markdown block to `pm_brief.md` → the owning role node cites verbatim. Follows [[project_phase6_deterministic_blocks_pattern]].
- **Free-data honesty:** any input missing / non-positive where a ratio is undefined → `n/a (data unavailable)`; never fabricate. Financial-sector names skip the ratios that don't apply (as distress screens already do).
- **Reuse:** extend `financials_parser` to expose the small extra annual series needed; compute in existing helper modules; render into blocks the role nodes already read.
- **Cite in the right clear-role node** (Phase 3): incremental ROIC → Financial-Statement / Competitive-Quality (capital-allocation); goodwill + commodity flags → Risk & Red-Flags. Add applicability-gated citation lines to those prompts and matching required-header/citation checks only if low-noise.

## Block 1 — Incremental ROIC + reinvestment runway (§4)

`compute_incremental_roic(fin)` (new, e.g. in `accounting_ratios.py` or a `quality_metrics.py`):
- Needs per-year **NOPAT** = `ebit_year × (1 − tax_rate)` and **invested capital (IC)** = `total_debt_year + total_equity_year`, for the two most-recent valid annual years.
- `tax_rate` = a single effective rate from the latest year: `tax / pretax_income` clamped to `[0, 0.35]`; if unavailable default `0.21` (US statutory) — **label which was used**.
- `incremental_roic_pct = (NOPAT_t − NOPAT_{t-1}) / (IC_t − IC_{t-1}) × 100`. Guards: both years present; `ΔIC` non-zero and (for interpretability) positive; prior IC/NOPAT present; result within a plausibility band `[-100, +200]%` else `n/a` (noisy free-data deltas). Report the year span.
- Requires new annual series in `financials_parser.annual_series`: `total_debt` and `total_equity` (via `_series(bs_a, "Total Debt")` / `_series(bs_a, "Stockholders Equity", "Common Stock Equity")`) and `pretax`/`tax` for the rate.
- Render row(s) in the accounting-ratios (or a small new) block: `Incremental ROIC (ΔNOPAT/ΔIC, N→M)`, with a one-line note on the tax rate used. Missing → `n/a`.

## Block 2 — Goodwill-impairment risk flag (§7)

`compute_goodwill_flag(fin)` (in `distress_screens.py`, beside Altman/Beneish):
- `goodwill_pct_assets = goodwill / total_assets × 100`; `goodwill_pct_equity = goodwill / total_equity × 100` (guard `total_equity > 0`).
- Optional trend: if an annual goodwill series is available (`_series(bs_a, "Goodwill", "Goodwill And Other Intangible Assets")`), report YoY direction (rising/flat/falling).
- **Flag** `elevated` when `goodwill_pct_equity ≥ 50` OR `goodwill_pct_assets ≥ 30` (heuristic thresholds; a large goodwill/equity cushion is where an impairment most threatens book value); else `normal`. Missing goodwill → `n/a (no goodwill reported)` (many names carry none — that is not a red flag).
- Render `## Goodwill / impairment screen` block: goodwill, % of assets, % of equity, trend, flag. Financial-sector names: still applicable (banks carry goodwill from acquisitions) — no skip.

## Block 3 — Commodity input-exposure flag (§7)

`compute_commodity_exposure(classification)` (pure mapping, new small module `commodity_exposure.py`):
- A static, documented `sector/industry → exposure` map keyed off `classification.json` (or yfinance `sector`/`industry`). Examples: `Energy`, `Basic Materials`, `Airlines`, `Packaged Foods`, `Restaurants`, `Auto Manufacturers`, `Building Products`, `Steel`, `Agriculture` → **exposed** with the primary input named (crude/jet fuel, metals, ags, resins…); most `Technology`/`Software`/`Financials` → **low**.
- Deterministic, no financials. Output: `exposure` (high/moderate/low), `primary_inputs` (list), `rationale` (one line). Unknown sector → `low (not classified)` honestly.
- Render a one-block `## Commodity input exposure`: exposure level + named inputs + a note that input-cost sensitivity feeds margin risk. The Risk & Red-Flags node cites it alongside cyclicality.

## Wiring

- `researcher.py` (or `pm_preflight.py` for the classification-only commodity block): after the sibling deterministic blocks, compute each, write `raw/{incremental_roic,goodwill_flag,commodity_exposure}.json`, append the formatted block to `pm_brief.md`, each in its own fail-open try/except.
- Prompts: add applicability-gated cite lines — Financial-Statement/Competitive-Quality cite incremental ROIC in capital-allocation; Risk & Red-Flags cite the goodwill flag + commodity exposure. Keep the Phase-4 required-header lists unchanged unless a citation check proves low-noise on the live smoke (avoid gratuitous retries).

## Testing

- **Parser:** `annual_series` gains `total_debt`/`total_equity` series (+ pretax/tax) — a fixture with a 3-year balance sheet yields the right most-recent-first lists; absent → `[]`.
- **Incremental ROIC:** hand-computed (e.g. NOPAT 100→120 on IC 500→600 → ΔNOPAT 20 / ΔIC 100 = 20%); non-positive/zero ΔIC → `n/a`; out-of-band → `n/a`; tax-rate fallback labeled.
- **Goodwill flag:** goodwill 60 / equity 100 → 60% → `elevated`; goodwill 5 / equity 100 → `normal`; no goodwill → `n/a`.
- **Commodity map:** `Energy` → high + crude; `Software` → low; unknown → `low (not classified)`.
- **Blocks render** headings + values; missing → `n/a`. **Real success (mini):** sane values on a diverse ticker set (an industrial flags commodity exposure; a software name does not; a serial acquirer shows elevated goodwill).

## Out of scope (Phase 2b / later)

- 13F / 13D-G / DEF 14A / 8-K / debt-maturity-wall — need SEC EDGAR fetch (separate Phase 2b).
- DDM / SOTP / reverse-DCF; cohort/NRR/ARPU/same-store; ESG — Tier C, honest-blank on free data.
