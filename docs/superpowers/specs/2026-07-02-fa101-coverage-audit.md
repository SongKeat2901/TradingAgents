# FA-101 Coverage Audit — post-Phase-4 (2026-07-02)

Definitive map of `Fundamental Analysis 101.md` §1-10 against shipped pipeline coverage, to pin down exactly what "the researcher does ALL FA-101 items" still requires. Verified against the code, not from memory.

## Covered (shipped + deployed)

| FA-101 section | Where covered |
|---|---|
| §1 Top-down (macro/sector/competitive/peers) | Macro regime engine (daily `tradingmacro`); Competitive-Quality role node (Porter/moat); peer set (`peers.json`, classifier) |
| §2 Income statement (revenue/margins/EPS/YoY/share-count) | `financials_parser` + accounting-ratios; YoY in Financial-Statement node |
| §2 Balance sheet — net debt | `net_debt.py` deterministic block |
| §2 Cash flow — FCF, CFO | `financials.json`, intrinsic_value FCF |
| §3 Profitability (margins, ROE/ROA/ROIC, ROIC-WACC, DuPont) | `accounting_ratios.py` |
| §3 Liquidity (current/quick/cash ratios) | `accounting_ratios.py` ✓ (verified 172-174) |
| §3 Leverage (D/E, net debt/EBITDA, interest coverage) | `accounting_ratios.py` ✓ (177) |
| §3 Efficiency (DSO/DIO/DPO/CCC, asset turnover) | `accounting_ratios.py` ✓ (179-182) |
| §3 Return-of-capital (payout, div yield, buyback yield, TSY) | `accounting_ratios.py` ✓ (183-186) |
| §4 Historical CAGRs (rev/EPS/FCF) + operating leverage | multi-year growth block (2026-07-02) |
| §4 Forward growth (consensus) | WP5a sentiment/consensus + calendar |
| §5 Relative multiples | `relative_multiples.py` |
| §5 Intrinsic (DCF/WACC/EPV, scenario, margin of safety) | `intrinsic_value.py` |
| §6 Qualitative (management, business model, moat, governance, capital allocation) | Competitive-Quality + Catalysts & Ownership role nodes |
| §7 Distress/manipulation (Altman Z″, Beneish M) | `distress_screens.py` |
| §8 Consensus/revisions, earnings surprise, catalysts, short interest | WP5a sentiment/consensus + calendar/earnings-surprise |
| §9 10-K/10-Q, Form 4 insider | `sec_edgar.py` filing fetch + `insider.json` |
| §10 Segment economics, backlog/RPO/deferred revenue | Financial-Statement node prompt (cites SEC filing RPO/segments) |

**Multi-agent clear roles (goal b):** Phase 3 — 4 role nodes + aggregator. **Per-role retry (goal c):** Phase 4 — self-loop conditional edges (built, pending merge/deploy).

## Remaining gaps

### Tier A — deterministic blocks on data already fetched (no new source) — "Phase 5"
1. **Incremental ROIC** (§4) — ΔNOPAT / ΔInvested-Capital from `annual_series` (revenue/ebit already there; need invested-capital = debt + equity per year, and NOPAT = EBIT×(1−tax)). Explicitly deferred in Phase 1; the cleanest remaining §4 item. Guard: ≥2 valid years, non-zero ΔIC, plausibility band.
2. **Goodwill-impairment risk flag** (§7) — goodwill as % of total assets and % of equity + YoY trend, from the balance sheet (`goodwill`/`goodwillAndOtherIntangibleAssets` in yfinance). A high/ rising goodwill/equity ratio flags impairment risk. Deterministic, honest-blank when absent.
3. **Commodity input-exposure flag** (§7) — a deterministic sector/industry → commodity-sensitivity mapping keyed off `classification.json`/yfinance `sector`/`industry` (e.g. Materials/Energy/Airlines/Packaged-Foods → flagged input-cost exposure). No new fetch.

### Tier B — needs SEC EDGAR fetch (new data pipeline) — "Phase 2b"
4. **13F institutional ownership** + change (§8) — SEC 13F aggregation.
5. **13D/13G activist stakes** (§8).
6. **DEF 14A proxy** — comp & governance detail (§6/§9).
7. **8-K material events** (§9).
8. **Debt maturity schedule / maturity-wall** (§2/§7) — from 10-K long-term-debt notes.

### Tier C — low-value / niche on free data (accept as honest-blank)
- DDM, SOTP, reverse-DCF (§5) — niche; current EPV/DCF suffices.
- Cohort/NRR/LTV-CAC, same-store sales, ARPU (§10) — model-specific, rarely in free data.
- ESG (§10) — no free structured source; out of scope per no-paid-services.

## Recommended order
Phase 5 (Tier A, 3 deterministic blocks — reuses `annual_series`, fits the established block pattern, high signal) → Phase 2b (Tier B, SEC-fetch — one shared EDGAR fetcher, then 13F/13D/DEF14A/8-K blocks). Tier C stays honest-blank.
