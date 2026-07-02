# Multi-Year Growth Metrics — Implementation Plan (FA-101 §4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add multi-year revenue/EPS/FCF CAGRs and operating leverage (from WP4b's annual layer) to the accounting-ratios block, closing FA-101 §4.

**Architecture:** Extend `financials_parser` to expose annual series (`annual_series`), then compute CAGRs + operating leverage in `accounting_ratios.py` and render them in the existing `## Accounting ratios` block. Deterministic; no new node/data/QC item.

**Tech Stack:** Python 3, pytest (`unit`), no new deps.

## Global Constraints

- **CAGR:** for a most-recent-first series, drop `None`s; require ≥2 values with `oldest>0` and `latest>0`; `cagr% = ((latest/oldest)**(1/(n-1)) - 1)*100`; span years = n-1. Else `None`.
- **Operating leverage:** `(ebit_t−ebit_p)/|ebit_p|  ÷  (rev_t−rev_p)/|rev_p|` using annual col0 (t) vs col1 (p); require both years present, `rev_p>0`, `ebit_p>0`, and rev pct-change ≠ 0; else `None`.
- **Free-data honesty:** any metric that can't be computed → `None` → rendered `n/a (data unavailable)`. Never fabricate; never crash.
- **Reuse:** the annual CSVs are already parsed in `parse_financials` (`is_a`, `cf_a`, and a bs annual parse if needed); add series extraction beside `beneish_inputs`.
- **Test marker:** module has `pytestmark = pytest.mark.unit`.
- **Run:** `.venv/bin/python -m pytest -q -m unit --tb=line` (baseline **791** — do not regress).

---

### Task 1: `annual_series` in `financials_parser`

**Files:** Modify `tradingagents/agents/utils/financials_parser.py`; Test `tests/test_financials_parser.py`.

**Interfaces:** Produces `parse_financials(...)["annual_series"] = {"revenue":[...], "diluted_eps":[...], "ebit":[...], "fcf":[...]}` (most-recent-first). Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_financials_parser.py
_IS_A_SERIES = ("# IS annual\n\n,2025,2024,2023,2022\n"
                "Total Revenue,133.1,121,110,100\nOperating Income,120,100,90,80\n"
                "Diluted EPS,3.3,3.0,2.7,2.5\n")
_CF_A_SERIES = ("# CF annual\n\n,2025,2024,2023,2022\n"
                "Free Cash Flow,50,45,40,35\n")


def test_annual_series_extracted():
    from tradingagents.agents.utils.financials_parser import parse_financials
    b = {"ticker": "ACME", "trade_date": "2026-01-01", "financial_currency": "USD",
         "fundamentals": "# f\n", "balance_sheet": "", "cashflow": "", "income_statement": "",
         "balance_sheet_annual": "", "income_statement_annual": _IS_A_SERIES, "cashflow_annual": _CF_A_SERIES}
    a = parse_financials(b)["annual_series"]
    assert a["revenue"] == [133.1, 121, 110, 100]
    assert a["ebit"] == [120, 100, 90, 80]
    assert a["diluted_eps"] == [3.3, 3.0, 2.7, 2.5]
    assert a["fcf"] == [50, 45, 40, 35]


def test_annual_series_absent_is_empty():
    from tradingagents.agents.utils.financials_parser import parse_financials
    b = {"ticker": "ACME", "trade_date": "2026-01-01", "financial_currency": "USD",
         "fundamentals": "# f\n", "balance_sheet": "", "cashflow": "", "income_statement": "",
         "balance_sheet_annual": "", "income_statement_annual": "", "cashflow_annual": ""}
    a = parse_financials(b)["annual_series"]
    assert a["revenue"] == [] and a["fcf"] == []
```

- [ ] **Step 2: Run — FAIL** (`KeyError: 'annual_series'`).
Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -k annual_series -q`

- [ ] **Step 3: Implement**

In `parse_financials`, the annual CSVs are already parsed (`_, is_a = _parse_quarterly_csv(d.get("income_statement_annual",""))` and `cf_a` — if not present in the current code, add them mirroring the existing annual parses used for `beneish_inputs`). Add a helper + the series:

```python
def _series(rows, *aliases):
    """Full most-recent-first value list for the first matching row label, else []."""
    for a in aliases:
        vals = rows.get(a)
        if vals:
            return vals
    return []


def _fcf_series(cf_rows):
    fcf = _series(cf_rows, "Free Cash Flow")
    if fcf:
        return fcf
    cfo = _series(cf_rows, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _series(cf_rows, "Capital Expenditure")
    if cfo and capex and len(cfo) == len(capex):
        return [ (c + x) if (c is not None and x is not None) else None
                 for c, x in zip(cfo, capex) ]   # capex is negative in yfinance -> CFO + capex
    return cfo or []
```

Then add to the returned dict:

```python
        "annual_series": {
            "revenue": _series(is_a, "Total Revenue", "Operating Revenue"),
            "diluted_eps": _series(is_a, "Diluted EPS"),
            "ebit": _series(is_a, "Operating Income", "EBIT"),
            "fcf": _fcf_series(cf_a),
        },
```

(If `is_a`/`cf_a` aren't already local vars from the beneish parse, add `_, is_a = _parse_quarterly_csv(d.get("income_statement_annual", ""))` and `_, cf_a = _parse_quarterly_csv(d.get("cashflow_annual", ""))`.)

- [ ] **Step 4: Run — PASS.** `.venv/bin/python -m pytest tests/test_financials_parser.py -q`
- [ ] **Step 5: Commit** `feat: expose annual_series (revenue/eps/ebit/fcf) from annual statements`

---

### Task 2: CAGRs + operating leverage in `accounting_ratios.py`

**Files:** Modify `tradingagents/agents/utils/accounting_ratios.py`; Test `tests/test_accounting_ratios.py`.

**Interfaces:** Consumes `fin["annual_series"]`. Produces `revenue_cagr_pct`, `revenue_cagr_years`, `eps_cagr_pct`, `fcf_cagr_pct`, `operating_leverage` in the ratios dict + block rows.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_accounting_ratios.py
def test_cagr_and_operating_leverage():
    from tradingagents.agents.utils.accounting_ratios import compute_accounting_ratios
    fin = {"annual_series": {"revenue": [133.1, 121, 110, 100],
                             "ebit": [120, 100, 90, 80],
                             "diluted_eps": [3.3, 3.0, 2.7, 2.5],
                             "fcf": [50, 45, 40, 35]}}
    r = compute_accounting_ratios(fin)
    assert r["revenue_cagr_pct"] == 10.0 and r["revenue_cagr_years"] == 3
    # OL: ebit +20% / rev +10% = 2.0
    assert r["operating_leverage"] == 2.0


def test_cagr_na_on_bad_data():
    from tradingagents.agents.utils.accounting_ratios import compute_accounting_ratios
    r = compute_accounting_ratios({"annual_series": {"revenue": [100], "ebit": [], "diluted_eps": [], "fcf": []}})
    assert r["revenue_cagr_pct"] is None and r["operating_leverage"] is None
```

- [ ] **Step 2: Run — FAIL.** `.venv/bin/python -m pytest tests/test_accounting_ratios.py -k "cagr or operating_leverage" -q`

- [ ] **Step 3: Implement**

Add helpers + fields in `compute_accounting_ratios` (reuse the module's `_r`; `annual = fin.get("annual_series") or {}`):

```python
def _cagr(series):
    vals = [v for v in (series or []) if v is not None]
    if len(vals) < 2:
        return None, None
    latest, oldest = vals[0], vals[-1]
    if oldest is None or latest is None or oldest <= 0 or latest <= 0:
        return None, None
    years = len(vals) - 1
    return round(((latest / oldest) ** (1 / years) - 1) * 100, 2), years


def _op_leverage(rev, ebit):
    if len(rev) < 2 or len(ebit) < 2:
        return None
    rt, rp, et, ep = rev[0], rev[1], ebit[0], ebit[1]
    if None in (rt, rp, et, ep) or rp <= 0 or ep <= 0:
        return None
    rev_chg = (rt - rp) / abs(rp)
    if rev_chg == 0:
        return None
    return round(((et - ep) / abs(ep)) / rev_chg, 2)
```

In `compute_accounting_ratios`, after the existing YoY block:

```python
    annual = fin.get("annual_series") or {}
    r["revenue_cagr_pct"], r["revenue_cagr_years"] = _cagr(annual.get("revenue"))
    r["eps_cagr_pct"], _ = _cagr(annual.get("diluted_eps"))
    r["fcf_cagr_pct"], _ = _cagr(annual.get("fcf"))
    r["operating_leverage"] = _op_leverage(annual.get("revenue") or [], annual.get("ebit") or [])
```

In `format_accounting_ratios_block`, add rows (using the block's `_cell` helper):

```python
        (f"Revenue CAGR ({r.get('revenue_cagr_years') or '?'}y)", _cell(r.get("revenue_cagr_pct"), "%")),
        ("EPS CAGR", _cell(r.get("eps_cagr_pct"), "%")),
        ("FCF CAGR", _cell(r.get("fcf_cagr_pct"), "%")),
        ("Operating leverage (ΔEBIT%/ΔRev%, latest yr)", _cell(r.get("operating_leverage"), "x")),
```

- [ ] **Step 4: Run — PASS** (focused + full `.venv/bin/python -m pytest -q -m unit --tb=line`).
- [ ] **Step 5: Commit** `feat: add multi-year CAGRs + operating leverage to accounting-ratios block`

---

### Task 3: Full-suite gate

- [ ] **Step 1:** `.venv/bin/python -m pytest -q -m unit --tb=line` — green (baseline 791 + new tests). Investigate any regression (esp. existing accounting-ratios tests + the `_cell` formatter).

---

## Out of scope
Incremental ROIC; WP5 data; red-flag screens; role restructure; per-role retry; macro-in-report.

## Self-Review
- Spec coverage: annual series (Task 1), CAGRs + operating leverage + block rows (Task 2), n/a degradation (both tasks' bad-data tests), gate (Task 3). Mapped.
- Placeholders: none; code complete; the "if is_a/cf_a not already local" note is a concrete conditional instruction.
- Type consistency: `annual_series` dict keys produced in Task 1 consumed by name in Task 2; `_cagr` returns `(pct, years)` unpacked consistently; reuses `_r`/`_cell`.
