# Intrinsic Value — design spec

Set 2026-06-01. Add a deterministic intrinsic-value (IV) valuation block to the
researcher: a triangulated fair-value range computed in Python from `raw/` data,
reconciled against the existing 12-month Monte-Carlo scenario EV, surfaced as a
report section, and audited as a new Tier 15. The rating engine is unchanged —
IV is decision-support, not a rating driver.

Brainstorm decisions (locked): Q1=C (fair value + reconcile), Q2=B (multi-method
triangulated range), Q3=A (deterministic applicability gate + per-profile method
selection), Q4=A (deterministic/transparent assumptions), Q5=A (decision-support;
rating stays MC-EV-driven).

## Goal / success criteria
- Every fresh (≤1 week) report gains a truthful IV treatment and stays **A+**
  under the Tier 1-15 audit (zero fabricated/hallucinated numbers, no missing
  data presented as data, leak-free customer PDF).
- IV is computed deterministically in Python; the LLM only *interprets* it.
- No method is forced where its inputs don't support it; gaps are stated, not filled.

## Architecture (mirrors peer_ratios / net_debt / forward_probabilities)
New pure-Python module `tradingagents/agents/utils/intrinsic_value.py`:
- `classify_valuation_profile(financials, net_debt) -> str`
- `compute_intrinsic_value(financials, net_debt, reference, peer_ratios, risk_free, fx_rate=None) -> dict`
- `format_intrinsic_value_block(iv) -> str`

Wired in `tradingagents/agents/researcher.py` AFTER the net-debt block (needs
financials + net_debt + reference + peer_ratios):
1. fetch risk-free (`^TNX` 10-Y via yfinance) with a fixed fallback,
2. write `raw/intrinsic_value.json`,
3. append a `## Intrinsic value` block to `pm_brief.md` (verbatim, like the others).

Report side: PM/executive renders an **"Intrinsic Value & Reconciliation"**
section from the block; PDF leak-scrubs it (`raw/intrinsic_value.json` →
"the valuation dataset"). Rating still derives from the MC engine.

Data flow: `researcher` → `raw/intrinsic_value.json` + `pm_brief.md` block → PM
reads → report section + reconciliation → PDF.

## B1 — Applicability classifier (deterministic; picks methods, never forces one)
| Profile | Trigger (from data) | Methods run | Skipped (stated) |
|---|---|---|---|
| STANDARD | NI(TTM) > 0 AND FCF(TTM) > 0, non-financial | DCF + EPV + Multiples + Reverse-DCF | — |
| UNPROFITABLE/EARLY | NI < 0 OR FCF < 0, non-financial | EV/Sales-vs-peers + Reverse-DCF + path-to-profitability note | DCF, EPV |
| FINANCIAL | `Sector` ∈ {Financial Services, Banks, …} | P/B, P/E, dividend-discount | FCF-DCF, EPV |
| NAV-PROXY | configurable ticker set (e.g. MSTR) + "investments ≫ revenue" heuristic | none — "IV ≈ NAV-driven; DCF/EPS IV not meaningful" | all |

Zero applicable methods → block states "Intrinsic value not computable — <reasons>;
rely on scenario EV."

## B2 — Methods (STANDARD; subsets for other profiles)
- **DCF (FCF-based):** base = FCF(TTM); grow at near-term g (Forward-EPS/TTM-EPS
  implied, capped) fading linearly to terminal g=2.5% over a 5-year explicit
  window; discount each year + Gordon terminal at WACC; ΣPV → enterprise value →
  − net debt → ÷ diluted shares → fair value/share.
- **EPV:** normalized EBIT × (1−tax) ÷ WACC (no growth) → EV → equity → /share. Floor.
- **Multiples-implied:** peer-median P/E × EPS; peer-median EV/EBITDA × EBITDA −
  net debt → /share (from peer_ratios.json + fundamentals).
- **Reverse-DCF:** solve the FCF growth that makes DCF fair value = current price;
  report "market implies g ≈ X%" as a reasonableness check.

## B3 — Assumptions (data-derived or named constants; every one printed)
- Cost of equity = risk_free (`^TNX`, live) + ERP 5.0% × beta (data).
- WACC blends after-tax cost of debt (tax rate + net debt, data).
- Near-term growth from Forward EPS vs TTM EPS, capped; terminal g = 2.5%.
- Bear/base/bull = growth ±2 pp, discount ±1 pp.
- Guards: cap implied near-term growth; floor discount rate for very-high-beta names.
- Constants live as named module-level values, echoed into the JSON `constants_note`.

## B4 — Currency
If `financial_currency` ≠ USD, fetch FX and convert per-share fair value to the
price currency; if FX unavailable, skip the comparison with a stated currency
caveat (never compare mismatched units). Affects ADRs (TSM/ASX/FUTU).

## B5 — Triangulation
Applicable methods → fair-value range (bear/base/bull). base = DCF base
(STANDARD) else method median. margin_of_safety = (base − price)/price.

## C1 — IV ↔ EV reconciliation (deterministic)
Compare IV base, 12-mo MC scenario EV (from forward_probabilities/decision), spot:
- margin_of_safety = (IV_base − price)/price
- iv_vs_ev = (IV_base − MC_EV)/MC_EV
- flag = AGREE if IV and EV point the same way vs price AND |iv_vs_ev| ≤ 15%, else DIVERGE.
PM surfaces this and must address a DIVERGE; rating unchanged.

## C2 — Output `raw/intrinsic_value.json`
```
{ trade_date, ticker, profile, applicable_methods, skipped_methods:[{method,reason}],
  inputs:{ risk_free, erp, beta, cost_of_equity, wacc, tax_rate, fcf_ttm, eps_ttm,
           forward_eps, near_term_growth, terminal_growth, horizon_years,
           diluted_shares, net_debt, currency, fx_rate },
  methods:{ dcf:{bear,base,bull}, epv:{value}, multiples:{pe_implied,ev_ebitda_implied},
            reverse_dcf:{implied_growth} },
  fair_value:{bear,base,bull}, margin_of_safety_pct,
  reconciliation:{ mc_ev, iv_base, price, iv_vs_ev_pct, iv_vs_price_pct, flag },
  constants_note }
```

## C3 — Error handling / honesty
Missing input → method → `skipped_methods` with reason. Risk-free fetch fail →
fixed fallback (noted). Currency mismatch w/o FX → skip + caveat. Zero methods →
explicit "not computable". Never fabricate to fill a gap.

## C4 — Testing (TDD, `pytest -m unit`)
Per-method math on fixed fixtures (known inputs → known fair value); classifier
across all four profiles; skip-with-reason on missing inputs; reverse-DCF solver;
reconciliation AGREE/DIVERGE; format block; risk-free stubbed. One golden sanity
check vs a real run (AAPL magnitude).

## C5 — Auditing & customer-facing
- Add **Tier 15 — Intrinsic value** to the audit framework (`/tmp/tk-audit-spec.md`
  + the `report-auditor` agent + the cadence/audit memory): recompute IV from
  `raw/` inputs + stated assumptions; verify applicability honesty; verify the
  reconciliation arithmetic; flag any report IV figure not matching the artifact.
- PDF leak-scrub: `raw/intrinsic_value.json` → "the valuation dataset"; no internal
  labels in the customer section.
- (Deferred to a later pass: a peer-corrector-style output validator that snaps the
  report's cited IV to the artifact. Not in v1.)

## Out of scope (v1)
- IV influencing the rating (Q5=B/C) — decision-support only for now.
- Own-history multiples (only current cross-sectional peer multiples available in raw).
- Sector-specific assumption sets beyond the four profiles.

## Verification / handover bar
Implement TDD → deploy to macmini → re-run each fresh ticker → audit Tier 1-15 →
all fresh reports A+ WITH a truthful IV section → register/summary refreshed.
Hand over only when every fresh report is verified A+ with intrinsic value.
