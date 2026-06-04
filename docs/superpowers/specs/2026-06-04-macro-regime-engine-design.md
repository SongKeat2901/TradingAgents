# Macro Regime Engine — Design

**Date:** 2026-06-04
**Status:** Approved (design); pending implementation plan
**Author:** SongKeat + Claude

## Purpose

A standalone, daily "central deciding machine" that judges the current macro
regime independently of any single stock, then biases each researched name's
expected value (EV) by that name's statistical sensitivity to the macro factors.
When the macro board turns broadly red, it gates trading entirely — regardless of
any stock's standalone EV.

Two layers, cleanly separated:

1. **Regime (stock-independent):** read the macro tape across six pillars, score
   each, aggregate into a regime + a Growth×Inflation quadrant + a trade gate.
2. **Per-stock overlay:** each name's *macro betas* (statistical sensitivities to
   the factors) combine with the regime to tilt its EV, scale its conviction/size,
   and inherit the global gate.

Output is written daily to the **Trading Plan** Google Sheet in
`My Drive/True Knot/TK Research/pdf/` (the shared folder).

## Scope

- **In scope:** the daily macro engine, the six-pillar framework, statistical
  per-stock betas, the three-layer bias math, and idempotent writes to the Trading
  Plan sheet.
- **Out of scope:** changes to the research pipeline itself (the engine is fully
  decoupled and reads finished reports); the seeded base of the Trading Plan sheet
  (built separately); intraday/real-time updates (daily cadence only).

## Architecture & data flow

A standalone daily job on `macmini-trueknot`, separate from the research pipeline.
New package `tradingagents/macro/` with small, single-purpose, independently
testable units. Data flows one direction:

```
macro_data.py   → fetch + cache raw series (yfinance + FRED)
pillars.py      → score each of the 6 pillars from series → [-1..+1] + R/A/G status
regime.py       → aggregate pillars → regime label, Growth×Inflation quadrant, gate level
betas.py        → rolling factor regression per ticker → beta vector + R²
bias.py         → regime × betas → EV tilt, conviction/size, gate (the 3 layers)
plan_writer.py  → idempotent write to the Trading Plan gsheet (ID manifest)
macro_daily.py  → orchestrator/CLI; scheduled daily after US close
```

Unit responsibilities:

- **`macro_data.py`** — fetch raw series from yfinance + FRED, cache to disk daily
  (one fetch per series per day). Pure I/O + caching; returns tidy pandas Series.
  Depends on: `yfinance`, FRED HTTP API (key from env/keychain).
- **`pillars.py`** — given the cached series, compute each pillar's score in
  [-1,+1] and a R/A/G status. Pure function of inputs; no I/O.
- **`regime.py`** — given pillar scores, produce the regime label, the
  Growth×Inflation quadrant, and the gate level (GO / CAUTION / STAND-DOWN). Pure.
- **`betas.py`** — given a ticker's price history + factor returns, run the rolling
  OLS and return a beta vector + R² + a confidence flag (with shrinkage for short
  histories). Pure compute on provided frames.
- **`bias.py`** — given regime + a stock's betas + the stock's research EV, return
  adjusted EV, conviction/size score, and the effective action after the gate. Pure.
- **`plan_writer.py`** — given the regime board + per-stock rows, write them to the
  Trading Plan sheet idempotently via the ID manifest (never name-search). I/O via
  `gog`.
- **`macro_daily.py`** — orchestrates: load tickers (from the report set), fetch
  data, score pillars/regime, compute betas, bias each name, write the sheet.

## The six pillars — indicators & sources

| Pillar | Indicators | Source | Cadence |
|---|---|---|---|
| **Growth** | ISM Mfg/Svc PMI, initial jobless claims, curve 10y–2y & 10y–3m, copper/gold ratio | FRED + yfinance (`HG=F`, `GC=F`) | weekly/monthly + daily |
| **Inflation** | CPI & PCE YoY, 10y breakeven (`T10YIE`), oil (`CL=F`), broad commodities (`DBC`) | FRED + yfinance | monthly + daily |
| **Liquidity/Policy** | Fed funds, real 10y (`DFII10`), net liquidity = WALCL − RRPONTSYD − WTREGEN, M2 | FRED | weekly |
| **Financial conditions** | DXY (`DX-Y.NYB`), IG spread (`BAMLC0A0CM`), HY spread (`BAMLH0A0HYM2`), MOVE (`^MOVE`), NFCI | yfinance + FRED | daily/weekly |
| **Risk appetite** | VIX (`^VIX`), hi-beta/low-vol (`SPHB`/`SPLV`), cyclicals/defensives (`XLY`/`XLP`), BTC-USD, gold, breadth (% > 200dma proxy) | yfinance | daily |
| **Positioning** | AAII bull-bear, Fear & Greed, fund flows | scrape/proxy (weakest free data) | weekly |

Mixed cadence is expected: monthly/weekly hard data (CPI, ISM, Fed balance sheet)
sits alongside daily market-priced series. Each indicator carries its own "as-of"
date; the regime uses the latest available value of each.

## Scoring methodology

- Each indicator → **z-score** vs a trailing window (≈1–3 yr) **plus a trend sign**
  (sign of recent change) → mapped to [−1, +1] (positive = supportive of risk
  assets).
- Weighted mean of an indicator set → **pillar score** + R/A/G status via
  thresholds.
- **Regime** = weighted aggregate of pillar scores.
- **Quadrant** = sign(Growth) × sign(Inflation): Goldilocks / Reflation /
  Stagflation / Deflation.
- **Gate** = derived from the aggregate score **and** the breadth of red pillars.
  "All red → don't trade" is operationalized as: ≥4 of 6 pillars red (or aggregate
  below a hard floor) ⇒ STAND-DOWN; a middling band ⇒ CAUTION; otherwise GO.

## Per-stock betas (statistical)

- **Factors:** Δ10Y (rates), ΔDXY (dollar), ΔHY-spread (credit/risk), oil return,
  market return (`SPY`), growth−value (`IWF` − `IWD`). Factor series are
  standardized.
- **Method:** rolling OLS (252-day window) of each stock's daily excess returns on
  the factor returns → beta vector + R².
- **Short-history names** (recent IPOs, e.g. RKLB, SOUN): apply **shrinkage toward
  the sector mean / zero** with a low-confidence flag. This stays pure-statistical
  (no hand-assigned mappings) while preventing noisy betas from driving the tilt.
- Multicollinearity is managed by keeping the factor set small and curated; betas
  with poor R² are down-weighted in the conviction layer.

## The bias math (three layers)

1. **Expected-return tilt.** The regime implies an expected move per factor (e.g.
   easing liquidity → rates ↓; risk-on → spreads ↓, market ↑). Macro contribution
   for a stock = Σ(βᵢ × expected_moveᵢ), annualized to the EV horizon (12 mo).
   **Adjusted EV = Research EV + contribution**, capped at ±15% (configurable
   `EV_TILT_CAP`).
2. **Conviction / size.** A position score scaled by regime quality × beta
   alignment (is the stock's macro exposure tailwind or headwind now?) × beta
   confidence (R²).
3. **Global gate.** Breadth-of-red ⇒ STAND-DOWN forces size → 0 and flags every
   row ("no new risk"), overriding the per-stock numbers. CAUTION applies a
   conviction haircut. GO leaves the per-stock result intact.

## Base EV source

The engine reads each report's 12-month target / EV from `decision.md`. If a report
has no explicit EV, it derives one from (price target − last price). Last price is
refreshed daily from yfinance (settled close, reusing `drop_incomplete_session`).

## Trading Plan sheet layout

**Top — Regime board:** the 6 pillars with current readings, R/A/G status, the
overall regime label, the Growth×Inflation quadrant, and the gate level.

**Per-ticker rows:**

```
Ticker · Rating · Macro Driver (top betas) · Macro Bias (R/A/G) · Research EV% ·
Macro Δ · Adjusted EV% · Conviction/Size · Action Today · Entry/Trim/Stop ·
Last Px · Note · PDF link
```

Written **idempotently** via the ID manifest (`~/gsheet-tool/`-style), never by
name-search — consistent with the no-duplicates rule. Sheet lives in
`My Drive/True Knot/TK Research/pdf/` (the shared folder).

## Ops

- **Schedule:** launchd/cron on the mini, daily ~05:00 SGT (after 16:00 ET close,
  so the settled close and same-day macro prints are available).
- **Auth:** `gog` for the Sheet write (7-day token caveat already handled by the
  `update-summary` skill + token-age hook); FRED key on the mini.
- **Caching:** one fetch per series per day; engine is idempotent and re-runnable.

## Testing

Unit tests per module, mirroring the existing `-m unit` suite:

- `pillars.py` — scoring on fixture series (known z-scores → known statuses).
- `regime.py` — quadrant + gate logic (breadth-of-red thresholds, edge cases).
- `betas.py` — regression on synthetic data with known betas; shrinkage on short
  samples.
- `bias.py` — tilt math, conviction scaling, gate override precedence.
- `plan_writer.py` — idempotent write (replace-by-ID, no dupes) against a fake gog.

## Prerequisites & decisions (approved)

1. **FRED API key** — free; obtained and stored on the mini. Comprehensive
   Growth/Inflation/Liquidity hard data requires it (not in yfinance).
2. **Positioning pillar** — shipped low-weight with proxies for v1 (only pillar
   without good free daily data); upgrade later.
3. **Base EV source** — read from `decision.md`; derive from (target − price) when
   absent.

## Open items (post-v1)

- Upgrade the Positioning pillar with better data (AAII feed, fund-flow source).
- Tune pillar weights and the gate thresholds against historical regimes
  (backtest the gate vs. drawdowns).
- Consider factor orthogonalization if multicollinearity degrades beta stability.
