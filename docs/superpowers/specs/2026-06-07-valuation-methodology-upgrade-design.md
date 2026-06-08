# Valuation Methodology Upgrade — Design

**Date:** 2026-06-07
**Author:** Claude (Opus 4.8) for SK, from Shian Pin's proposal
**Status:** Approved (SK goal: "wire the spec proceed with the built and rerun the researches")
**Touches:** `tradingagents/agents/utils/intrinsic_value.py`, `tradingagents/dataflows/y_finance.py`, `tests/test_intrinsic_value.py`

## Problem

The current engine triangulates `base = median(EPV, peer-P/E, DCF*)` for STANDARD names,
with bear/bull = min/max. Two mechanisms systematically undervalue high-quality
compounders:

1. **EPV (no-growth earnings power) is averaged into the base.** EPV answers "what is this
   worth if growth stops" — a downside metric, not a price target. For growth names it is
   the lowest method and drags the blend down.
2. **The DCF is excluded when FCF/NI < 0.5** (capex guard). Capex-peak compounders (e.g.
   MSFT, FCF/NI = 0.30) lose their DCF leg entirely, leaving `base = median(EPV, peer-P/E)`
   = the midpoint. MSFT: EPV $169 + peer $497 → **base $333**, ~25% below the $441 price and
   $454 scenario EV.

## Goal

Fair values calibrated to how the market prices growth, while keeping downside discipline.
EPV stays **visible as a stress test**, not a blend drag. Bounded by the existing sanity
guards so it can't blow up.

## Design

All changes are deterministic Python (no LLM, no paid data — yfinance/FRED only),
consistent with the existing deterministic-block pattern.

### 1. Data inputs (small fetch addition)

The fundamentals block is built in `dataflows/y_finance.py` (`info`-based field list).

- **Add one field:** `("Revenue Growth", info.get("revenueGrowth"))`.
- **Gross margin** needs no fetch change — `Gross Profit` and `Revenue (TTM)` are already
  emitted; compute `gross_margin = gross_profit / revenue` in `parse_fundamentals`.

`parse_fundamentals` gains: `gross_profit` (from "Gross Profit"), `revenue_growth`
(from "Revenue Growth"). It already parses `revenue` and `sector`.

### 2. Growth-tier classification (within STANDARD only)

New `classify_growth_tier(fund) -> "COMPOUNDER" | "MATURE" | "CYCLICAL"`. Only consulted
for the STANDARD profile; FINANCIAL / UNPROFITABLE / NAV_PROXY keep their current paths.

- **CYCLICAL** if industry contains "Semiconductor" OR sector in {"Energy", "Basic Materials"}
  OR industry matches oil/gas/mining/steel. (Sector/industry-only in v1; earnings-volatility
  detection is a later refinement.)
- **COMPOUNDER** if `revenue_growth >= 0.12` AND `gross_margin >= 0.50`.
- **MATURE** otherwise — **including when `revenue_growth` is unknown** (conservative
  fallback: unverified growth keeps EPV's weight rather than over-crediting a compounder).

Order: cyclical check first, then compounder, else mature.

### 3. Normalized forward-FCF DCF

- **Normalized base FCF:** `FCF_NORM_CONVERSION = 0.80`. If `net_income > 0` and
  `fcf/net_income < 0.80` (incl. negative FCF), use `fcf_base = net_income * 0.80` (a
  through-cycle owner-earnings proxy) instead of excluding the DCF. If `net_income <= 0`,
  keep actual FCF; if that is also <= 0 the DCF leg drops and weights renormalize.
- **3-phase growth** over `HORIZON_YEARS = 5`: years 1–3 at `near_g` (forward-EPS-implied,
  capped 25%), years 4–5 linear decay `near_g → TERMINAL_GROWTH`, then terminal perpetuity
  at `TERMINAL_GROWTH = 0.025`. (Replaces the current single linear ramp.)
- This makes the capex-peak DCF usable and forward-looking on free data. It approximates a
  growth trajectory via forward-EPS growth; it cannot model segment-level (e.g. Azure)
  detail without a paid feed — stated as a limitation in the report.

### 4. Class-weighted blend (replaces median triangulation)

Base fair value = weighted mean of available method values, weights by tier:

| Tier | DCF | Peer-P/E | EPV |
|---|---:|---:|---:|
| COMPOUNDER | 0.65 | 0.35 | 0 (stress only) |
| MATURE | 0.40 | 0.30 | 0.30 |
| CYCLICAL | 0.30 | 0.70 | 0 (stress only) |

If a method is unavailable (None), its weight is dropped and the remaining weights
**renormalize** to sum to 1. If no method is available, fair value stays None (existing
"not computable" path).

### 5. Valuation band + labelled swing variable

- **base** = weighted blend above.
- **bear** = re-blend with DCF leg at `(near_g − GROWTH_DELTA, wacc + DISCOUNT_DELTA)` and
  peer leg at the **25th-percentile** peer P/E; **bull** = `(near_g + GROWTH_DELTA,
  wacc − DISCOUNT_DELTA)` and **75th-percentile** peer P/E. Same tier weights. EPV (when
  weighted, i.e. MATURE) held constant across the band (it is the no-growth floor).
- JSON gains `scenario_drivers`: a label per scenario, e.g.
  `bear: "FCF growth −2pp, WACC +1pp, peer P/E 25th pct"`.

### 6. EPV → downside stress test

EPV is always computed and surfaced as the **"no-growth floor / downside stress"** value.
It is weighted into the base **only for MATURE (0.30)**; never for COMPOUNDER or CYCLICAL.

### 7. Unchanged guards (kept verbatim)

- eps plausibility (suppress if price/eps ∉ [3, 60]).
- output sanity (suppress base ∉ [0.2×, 3×] price).
- foreign-ADR FX caveat + derivation.
- MC-EV reconciliation (AGREE/DIVERGE).

### Output / formatter

`format_intrinsic_value_block` updated to show: tier + the weights used, the bear/base/bull
band with per-scenario swing labels, EPV labelled as downside stress (not a blend member
for compounders), and the existing assumptions/reconciliation lines. `constants_note` gains
the tier weights and `FCF_NORM_CONVERSION`.

## Out of scope (fast-follow)

- **#5 category peer selection** — lives in `pm_preflight.py`'s peer-extraction prompt, not
  this engine. Logged as a separate follow-up.

## Testing

Extend `tests/test_intrinsic_value.py` (keep all existing tests green):

- `classify_growth_tier`: compounder (high growth + margin), mature (low growth), mature
  (growth unknown → fallback), cyclical (semis/energy).
- Normalized-FCF DCF: capex-peak name (FCF/NI < 0.8) uses `NI × 0.8`; healthy name uses
  actual FCF; NI ≤ 0 drops DCF.
- Weighted blend per tier incl. method-missing renormalization.
- Band ordering bear ≤ base ≤ bull; scenario_drivers present.
- EPV excluded from compounder base, included for mature.
- **MSFT regression** (synthetic inputs mirroring the 2026-06-02 run): base rises from ~$333
  into the high-$300s/low-$400s, materially closer to the $454 scenario EV / $441 price.
- Guards still fire (P/E∉[3,60] suppress; 0.2–3× suppress; foreign-ADR caveat).

## Expected effect

Compounders stop being systematically undervalued; EPV remains visible as the downside
floor; output still bounded by the sanity guards. After deploy, **rerun all 21 wk24 tickers
@ 2026-06-05** on the new engine.
