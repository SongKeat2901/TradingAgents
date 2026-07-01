# FA-101 WP4a — Altman Z″ Distress Screen

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Add a deterministic Altman Z″ (4-variable) financial-distress score as a `pm_brief.md` block — closing part of FA-101 audit §10 (Risk & Red Flags: distress screening). This is **WP4a**. The Beneish M-score (WP4b) is deferred: it needs an annual current-vs-prior-year data layer the pipeline does not fetch today, and gets its own spec.

---

## Background

The 2026-07-01 FA-101 audit found the pipeline had **no quantitative distress/fraud screens** (§10). WP4 addresses that. Data-layer check: `financials_parser` reads only **quarterly** statements (~4-5 quarters), which is enough for Altman Z″ (a point-in-time balance-sheet score) but NOT for Beneish M (an annual year-over-year model). So WP4 splits — this spec ships Altman Z″ now; Beneish waits on annual-statement fetching.

**Why Z″ (not the original 5-variable Z):** the watchlist is tech-heavy, and the original Z's X5 = Sales/Total-Assets term penalizes asset-light firms. Altman's Z″ is the cross-industry / non-manufacturer revision that drops X5, making it sector-robust for a mixed universe. It also uses **book** equity, so it needs no market-cap input.

## Design principles

- **Deterministic-block pattern:** compute in Python → write `raw/distress_screens.json` → append a block to `pm_brief.md` as ground truth; the LLM cites it, never recomputes. (Ref: `project_phase6_deterministic_blocks_pattern`.)
- **Free-data honesty:** any missing input → `Z = n/a (data unavailable)`, never fabricated.
- **Sector-appropriate:** skip financials (banks/insurers) where Z is not meaningful, with an explicit note — mirrors how `intrinsic_value` already skips the FINANCIAL profile.
- **Bounded QC surface:** no new QC-checklist item; distress is an informational risk flag, not a load-bearing number. Citation is mandated in the fundamentals-analyst prompt only.
- **Named for extension:** the module is `distress_screens.py` so Beneish M (WP4b) can be added beside Altman without a rename.

## Formula (Altman Z″, 4-variable)

`Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4`

| Var | Definition | Inputs (all from `financials_parser`) |
|---|---|---|
| X1 | Working capital / total assets | `(current_assets − current_liabilities) / total_assets` |
| X2 | Retained earnings / total assets | `retained_earnings / total_assets` |
| X3 | EBIT (TTM) / total assets | `ebit_ttm / total_assets` |
| X4 | **Book** equity / total liabilities | `total_equity / (total_assets − total_equity)` |

**Zones:** `Z'' > 2.6` → **Safe**; `1.1 ≤ Z'' ≤ 2.6` → **Grey**; `Z'' < 1.1` → **Distress**.

Notes: X3 uses `ebit_ttm` (the trailing-twelve-month EBIT, NOT the single-quarter `ebit`). X4 uses book equity, so no market cap is needed. `total_liabilities = total_assets − total_equity`; if that is ≤ 0 (or any input missing), X4 (and Z″) render `n/a`.

## Components

### 1. New parser field
`tradingagents/agents/utils/financials_parser.py`: add `"retained_earnings": _row_col0(bs, "Retained Earnings")` to the balance-sheet section. Everything else Z″ needs is already parsed.

### 2. New module `tradingagents/agents/utils/distress_screens.py`
- `compute_altman_z(fin: dict) -> dict` — returns:
  - populated: `{"model": "Altman Z''", "z_score": <float>, "zone": "Safe"|"Grey"|"Distress", "x1"..."x4": <float>, "applicable": True}`
  - skipped (financials): `{"applicable": False, "skip_reason": "Altman Z not meaningful for financials (sector: <sector>)"}`
  - unavailable (missing inputs): `{"applicable": True, "z_score": None, "zone": None, "unavailable_reason": "<which input>"}`
  - **Sector gate:** skip when `fin.get("sector")` contains "Financial" (case-insensitive) — e.g. yfinance "Financial Services". (If the run's `raw/classification.json` FINANCIAL profile is readily available at the call site, the plan may use that instead; sector-string gating is the simple deterministic default.)
- `format_distress_block(result: dict) -> str` → a `## Distress screen (Altman Z″)` block:
  - populated: a small table (Z″ score, zone, and X1–X4 with their formulas) + the standard "use verbatim; do not recompute" mandate + a one-line caveat that Z″ is a relative distress indicator, not a default prediction.
  - skipped: `## Distress screen (Altman Z″) — not applicable (financials)`.
  - unavailable: `## Distress screen (Altman Z″) — n/a (data unavailable: <reason>)`.

### 3. Wiring — `researcher.py`
After the accounting-ratios block (where `fin_parsed` already exists), in its own try/except (fail-open like the sibling blocks): `z = compute_altman_z(fin_parsed)` → write `raw/distress_screens.json` (`json.dumps(..., indent=2, default=str)`) → `format_distress_block(z)` → append to `pm_brief_path`.

### 4. Citation — `fundamentals_analyst.py`
Extend the existing accounting-ratios citation mandate so the analyst also cites the Z″ score + zone from the `## Distress screen` block (verbatim, when applicable). No new QC-checklist item.

## Data flow

```
researcher.fetch_research_pack:
  fin_parsed = parse_financials(financials)   # now includes retained_earnings
  ...accounting-ratios block...
  z = compute_altman_z(fin_parsed)            # sector gate + degrade-to-n/a
  write raw/distress_screens.json
  append format_distress_block(z) to pm_brief.md
```

## Testing

- **Unit (`tests/test_distress_screens.py`):**
  - Healthy fixture (positive WC/RE, solid EBIT) → `zone == "Safe"`, Z″ computed and equal to the hand-computed value.
  - Distressed fixture (negative working capital, negative retained earnings, negative EBIT) → `zone == "Distress"`.
  - A grey-zone fixture near the 1.1–2.6 band → `zone == "Grey"` (locks the thresholds).
  - Financial sector fixture → `applicable == False`, block says "not applicable".
  - Missing input (e.g. no `retained_earnings` / no `total_assets`) → `z_score is None`, block renders `n/a`, no exception.
  - `total_liabilities <= 0` → X4/Z″ `n/a` (no divide error).
  - Block formatter: contains `## Distress screen`, the zone, and the "verbatim" mandate.
- **Parser test (`tests/test_financials_parser.py`, extend):** `retained_earnings` parses from the balance-sheet CSV col 0.
- **Real success measure (not a unit test):** sane Z″ on live tickers (e.g. a strong balance sheet → Safe; a leveraged/loss-making name → lower), and financials correctly skipped.

## Out of scope (later / not this phase)

- Beneish M-score (WP4b) — needs annual current+prior-year statements the pipeline doesn't fetch; separate spec (that annual-data layer also unlocks the deferred FA-101 multi-year CAGRs).
- The original 5-variable Z or profile-selected variant — Z″ chosen as the single sector-robust model.
- A new QC-checklist item for the distress score.
- REIT/utility-specific distress models.
