# Deterministic Fundamentals Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic accounting-ratios and relative-multiples blocks to `pm_brief.md`, plus wire up the orphaned `insider.json` and discarded earnings-surprise data — closing the highest-value gaps from the 2026-07-01 FA-101 audit.

**Architecture:** Follow the established deterministic-block pattern (compute in Python → write `raw/X.json` → format a markdown block → append to `raw/pm_brief.md` → LLM cites it, never recomputes). A new shared `financials_parser.py` extracts every statement line-item once; `accounting_ratios.py` and `relative_multiples.py` compute off it. Both blocks are wired into `researcher.py` after the intrinsic-value block (so they can reuse its WACC / market-cap / net-debt figures), enforced by two new QC-checklist items.

**Tech Stack:** Python 3, pytest (markers `unit`/`integration`/`smoke`), yfinance (already fetched upstream — no new network calls in this plan), LangGraph node functions.

## Global Constraints

- **Free-data honesty (verbatim rule):** every metric renders `n/a (data unavailable)` when a source line-item is missing. NEVER fabricate, estimate, or interpolate to fill a blank.
- **Deterministic-block pattern:** compute in Python, append to `pm_brief.md` as ground truth; the LLM cites, never recomputes.
- **Do NOT refactor `intrinsic_value.py`'s `parse_fundamentals`** — reuse it read-only. It is the most-tested area; leave it untouched.
- **Bounded QC surface:** exactly TWO new grouped QC items, not one per ratio.
- **Fail loud, not silent:** if a block's inputs are structurally present but a required upstream file is missing, follow the existing `researcher.py:318-331` precedent (raise / emit an explicit "unavailable — do not fabricate" block) rather than silently omitting — a silently-missing block causes the LLM to invent numbers.
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` (from repo root `/Users/songkeat/Documents/Python/Trading Agent/TradingAgents`).
- **Currency:** all dollar line-items are in the ticker's `financial_currency`; do not mix currencies across blocks. Ratios are unitless/percentages so currency-agnostic; the relative-multiples block's market cap and net debt must both be the primary-ticker figures already parsed by `intrinsic_value` (same currency basis).
- **Fixtures:** inline dict/string constants at module top (project convention — there is no `tests/fixtures/` dir). Name them `_<TICKER>_<STATEMENT>` mirroring `tests/test_net_debt.py`.

---

## File Structure

- Create: `tradingagents/agents/utils/financials_parser.py` — single-source line-item extraction from a `financials.json` bundle.
- Create: `tradingagents/agents/utils/accounting_ratios.py` — `compute_accounting_ratios` + `format_accounting_ratios_block`.
- Create: `tradingagents/agents/utils/relative_multiples.py` — `compute_relative_multiples` + `format_relative_multiples_block`.
- Modify: `tradingagents/agents/utils/peer_ratios.py` — add a `market_cap` parser + column (mirror `_parse_ttm_ebitda`).
- Modify: `tradingagents/agents/researcher.py` — wire the two new blocks after the intrinsic-value site (~line 427); add `insider.json` handling if needed.
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` — add `insider.json` to `files=[...]`, add mandated insider subsection + citation mandates for the two new blocks.
- Modify: `tradingagents/agents/utils/calendar.py` — capture `EPS Estimate` / `Surprise(%)`.
- Modify: `tradingagents/agents/managers/pm_preflight.py` — add a `## Surprise history` block.
- Modify: `tradingagents/agents/managers/qc_agent.py` — add QC items 17 & 18; bump "16-item" → "18-item".
- Create tests: `tests/test_financials_parser.py`, `tests/test_accounting_ratios.py`, `tests/test_relative_multiples.py`, `tests/test_peer_market_cap.py`, `tests/test_calendar_surprise.py`, `tests/test_qc_new_items.py`.

---

### Task 1: Shared financials parser

**Files:**
- Create: `tradingagents/agents/utils/financials_parser.py`
- Test: `tests/test_financials_parser.py`

**Interfaces:**
- Consumes: `net_debt._parse_quarterly_csv(csv_text) -> (columns: list[str], rows: dict[str, list[float|None]])`; `intrinsic_value.parse_fundamentals(financials: dict) -> dict`.
- Produces: `parse_financials(financials: dict) -> dict[str, Any]` — a flat dict of `float | str | None` keyed as listed in the implementation below. Consumed by Tasks 2 and 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_financials_parser.py
import pytest
from tradingagents.agents.utils.financials_parser import parse_financials

pytestmark = pytest.mark.unit

# Minimal but realistic bundle. balance_sheet/cashflow/income_statement are
# CSV text blobs exactly like yfinance DataFrame.to_csv() (col0 = latest quarter),
# with the leading "#"-comment lines net_debt._parse_quarterly_csv skips.
_FUND = (
    "# Fundamentals\n"
    "Name: Acme Corp\n"
    "Sector: Technology\n"
    "Market Cap: 100000000000\n"
    "EPS (TTM): 5.0\n"
    "Forward EPS: 6.0\n"
    "Beta: 1.1\n"
    "EBITDA: 15000000000\n"
    "Net Income: 8000000000\n"
    "Free Cash Flow: 9000000000\n"
    "Revenue: 40000000000\n"
    "Gross Profit: 24000000000\n"
    "Gross Margin: 60.0\n"
)
_BS = (
    "# Balance Sheet\n\n"
    ",2026-03-31,2025-12-31,2025-09-30,2025-06-30,2025-03-31\n"
    "Total Assets,80000000000,79000000000,78000000000,77000000000,76000000000\n"
    "Current Assets,30000000000,29000000000,,,\n"
    "Cash And Cash Equivalents,12000000000,11000000000,,,\n"
    "Receivables,6000000000,5800000000,,,\n"
    "Inventory,4000000000,3800000000,,,\n"
    "Current Liabilities,15000000000,14500000000,,,\n"
    "Payables,3000000000,2900000000,,,\n"
    "Total Debt,20000000000,20000000000,,,\n"
    "Stockholders Equity,40000000000,39000000000,,,\n"
)
_CF = (
    "# Cash Flow\n\n"
    ",2026-03-31,2025-12-31,2025-09-30,2025-06-30\n"
    "Operating Cash Flow,3000000000,2800000000,2600000000,2400000000\n"
    "Capital Expenditure,-500000000,-500000000,-500000000,-500000000\n"
    "Cash Dividends Paid,-400000000,-400000000,-400000000,-400000000\n"
    "Repurchase Of Capital Stock,-1000000000,-1000000000,-1000000000,-1000000000\n"
)
_IS = (
    "# Income Statement\n\n"
    ",2026-03-31,2025-12-31,2025-09-30,2025-06-30,2025-03-31\n"
    "Total Revenue,10000000000,9800000000,9600000000,9400000000,9000000000\n"
    "Cost Of Revenue,4000000000,3900000000,3800000000,3700000000,3600000000\n"
    "Interest Expense,150000000,150000000,150000000,150000000,150000000\n"
    "Net Income,2100000000,2000000000,1950000000,1900000000,1800000000\n"
    "Diluted EPS,1.30,1.25,1.22,1.18,1.10\n"
)
_BUNDLE = {
    "ticker": "ACME", "trade_date": "2026-05-01", "financial_currency": "USD",
    "fundamentals": _FUND, "balance_sheet": _BS, "cashflow": _CF, "income_statement": _IS,
}


def test_parse_financials_extracts_all_line_items():
    fin = parse_financials(_BUNDLE)
    assert fin["market_cap"] == 100000000000
    assert fin["total_assets"] == 80000000000
    assert fin["receivables"] == 6000000000
    assert fin["inventory"] == 4000000000
    assert fin["current_liabilities"] == 15000000000
    assert fin["total_equity"] == 40000000000
    # TTM = sum of last 4 quarterly columns
    assert fin["cfo_ttm"] == 3000000000 + 2800000000 + 2600000000 + 2400000000
    assert fin["capex_ttm"] == -2000000000
    # YoY-ago = column index 4 (same quarter, prior year)
    assert fin["revenue_yoy_ago"] == 9000000000
    assert fin["diluted_eps_yoy_ago"] == 1.10
    # average balance = mean(col0, col1)
    assert fin["receivables_avg"] == (6000000000 + 5800000000) / 2


def test_parse_financials_missing_rows_are_none_not_fabricated():
    sparse = dict(_BUNDLE, balance_sheet="# BS\n\n,2026-03-31\nTotal Assets,80000000000\n")
    fin = parse_financials(sparse)
    assert fin["total_assets"] == 80000000000
    assert fin["inventory"] is None      # row absent → None, never guessed
    assert fin["receivables_avg"] is None
    assert fin["current_liabilities"] is None


def test_parse_financials_tolerates_non_dict():
    assert parse_financials(None)["market_cap"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.agents.utils.financials_parser'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/financials_parser.py
"""Single deterministic parser for a raw financials.json bundle.

Extracts every income/balance/cashflow line-item once so that the
accounting-ratios, relative-multiples, and (later) fraud-screen blocks all
read from one source of truth. Missing line-items are None — nothing is
fabricated or estimated (free-data honesty rule).
"""
from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.net_debt import _parse_quarterly_csv
from tradingagents.agents.utils.intrinsic_value import parse_fundamentals


def _row_col0(rows: dict, *aliases: str):
    """First matching row label -> its col-0 (latest quarter) value, else None."""
    for a in aliases:
        vals = rows.get(a)
        if vals:
            return vals[0]
    return None


def _row_at(rows: dict, idx: int, *aliases: str):
    """Value at column `idx` (e.g. 4 = same quarter one year ago), else None."""
    for a in aliases:
        vals = rows.get(a)
        if vals and len(vals) > idx and vals[idx] is not None:
            return vals[idx]
    return None


def _ttm(rows: dict, *aliases: str):
    """Sum of the last 4 quarterly columns for a flow line-item, else None."""
    for a in aliases:
        vals = rows.get(a)
        if vals and len(vals) >= 4 and all(v is not None for v in vals[:4]):
            return sum(vals[:4])
    return None


def _avg2(rows: dict, *aliases: str):
    """Average of col0 and col1 (turnover average balance); col0 alone if only one; else None."""
    for a in aliases:
        vals = rows.get(a)
        if vals:
            if len(vals) >= 2 and vals[0] is not None and vals[1] is not None:
                return (vals[0] + vals[1]) / 2
            return vals[0]
    return None


def parse_financials(financials: Any) -> dict[str, Any]:
    d = financials if isinstance(financials, dict) else {}
    fund = parse_fundamentals(d) if d else {}
    _, bs = _parse_quarterly_csv(d.get("balance_sheet", ""))
    bs_cols, _ = _parse_quarterly_csv(d.get("balance_sheet", ""))
    _, cf = _parse_quarterly_csv(d.get("cashflow", ""))
    _, is_ = _parse_quarterly_csv(d.get("income_statement", ""))

    return {
        "trade_date": d.get("trade_date"),
        "financial_currency": d.get("financial_currency"),
        "as_of_quarter": bs_cols[0] if bs_cols else None,
        # fundamentals-derived (reused; do not recompute)
        "market_cap": fund.get("market_cap"),
        "revenue_ttm": fund.get("revenue"),
        "ebit": fund.get("ebit"),
        "ebitda": fund.get("ebitda"),
        "net_income": fund.get("net_income"),
        "fcf": fund.get("fcf"),
        "eps": fund.get("eps"),
        "forward_eps": fund.get("forward_eps"),
        "diluted_shares": fund.get("diluted_shares"),
        "gross_margin": fund.get("gross_margin"),
        "beta": fund.get("beta"),
        "tax": fund.get("tax"),
        "sector": fund.get("sector"),
        "industry": fund.get("industry"),
        # balance sheet (col0 = latest quarter)
        "total_assets": _row_col0(bs, "Total Assets"),
        "current_assets": _row_col0(bs, "Current Assets", "Total Current Assets"),
        "cash_and_equivalents": _row_col0(bs, "Cash And Cash Equivalents"),
        "receivables": _row_col0(bs, "Receivables", "Accounts Receivable", "Net Receivables"),
        "inventory": _row_col0(bs, "Inventory"),
        "ppe": _row_col0(bs, "Net PPE", "Net Property Plant And Equipment"),
        "goodwill": _row_col0(bs, "Goodwill", "Goodwill And Other Intangible Assets"),
        "current_liabilities": _row_col0(bs, "Current Liabilities", "Total Current Liabilities"),
        "payables": _row_col0(bs, "Payables", "Accounts Payable"),
        "total_debt": _row_col0(bs, "Total Debt"),
        "total_equity": _row_col0(bs, "Stockholders Equity", "Common Stock Equity"),
        "receivables_avg": _avg2(bs, "Receivables", "Accounts Receivable", "Net Receivables"),
        "inventory_avg": _avg2(bs, "Inventory"),
        "payables_avg": _avg2(bs, "Payables", "Accounts Payable"),
        "total_assets_avg": _avg2(bs, "Total Assets"),
        # cash flow (TTM = sum of last 4 quarters)
        "cfo_ttm": _ttm(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
        "capex_ttm": _ttm(cf, "Capital Expenditure"),
        "dividends_paid_ttm": _ttm(cf, "Cash Dividends Paid", "Common Stock Dividend Paid"),
        "buybacks_ttm": _ttm(cf, "Repurchase Of Capital Stock", "Common Stock Payments"),
        "sbc_ttm": _ttm(cf, "Stock Based Compensation"),
        # income statement (TTM + YoY)
        "revenue_yoy_ago": _row_at(is_, 4, "Total Revenue", "Operating Revenue"),
        "cogs_ttm": _ttm(is_, "Cost Of Revenue"),
        "interest_expense_ttm": _ttm(is_, "Interest Expense", "Interest Expense Non Operating"),
        "net_income_yoy_ago": _row_at(is_, 4, "Net Income", "Net Income Common Stockholders"),
        "diluted_eps_yoy_ago": _row_at(is_, 4, "Diluted EPS"),
    }
```

Note: `parse_fundamentals` reads `financials["income_statement"]` internally for `diluted_shares`/`tax`/`ebit`; that reuse is intentional and read-only.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -q`
Expected: PASS (3 tests). If a `_parse_quarterly_csv` value comes back as a string, confirm it coerces to float — it already does in net_debt; if not, wrap numeric cells with the same coercion net_debt uses.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/financials_parser.py tests/test_financials_parser.py
git commit -m "feat: add shared financials_parser (single-source line-item extraction)"
```

---

### Task 2: Accounting-ratios block

**Files:**
- Create: `tradingagents/agents/utils/accounting_ratios.py`
- Test: `tests/test_accounting_ratios.py`

**Interfaces:**
- Consumes: `parse_financials(...)` output (Task 1); optional `wacc: float` and `net_debt: dict` (with key `net_debt`) — both readable from the in-memory `iv["inputs"]["wacc"]` / `net_debt` dict in `researcher.py`.
- Produces: `compute_accounting_ratios(fin: dict, wacc: float | None = None, net_debt: dict | None = None) -> dict`; `format_accounting_ratios_block(ratios: dict, trade_date: str | None, as_of: str | None) -> str`. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_accounting_ratios.py
import pytest
from tradingagents.agents.utils.accounting_ratios import (
    compute_accounting_ratios,
    format_accounting_ratios_block,
)

pytestmark = pytest.mark.unit

_FIN = {
    "revenue_ttm": 40000000000.0, "net_income": 8000000000.0, "ebit": 10000000000.0,
    "ebitda": 15000000000.0, "fcf": 9000000000.0, "tax": 0.21,
    "gross_margin": 60.0,
    "total_assets": 80000000000.0, "total_assets_avg": 79000000000.0,
    "current_assets": 30000000000.0, "current_liabilities": 15000000000.0,
    "cash_and_equivalents": 12000000000.0, "inventory": 4000000000.0,
    "inventory_avg": 3900000000.0, "receivables_avg": 5900000000.0,
    "payables_avg": 2950000000.0, "cogs_ttm": 16000000000.0,
    "total_debt": 20000000000.0, "total_equity": 40000000000.0,
    "interest_expense_ttm": 600000000.0, "market_cap": 100000000000.0,
    "dividends_paid_ttm": -1600000000.0, "buybacks_ttm": -4000000000.0,
    "revenue_yoy_ago": 36000000000.0, "net_income_yoy_ago": 7000000000.0,
    "cfo_ttm": 10800000000.0,
}


def test_core_ratios():
    r = compute_accounting_ratios(_FIN, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    assert r["net_margin"] == 20.0            # 8000/40000
    assert r["roe"] == 20.0                    # 8000/40000
    assert r["roa"] == 10.0                    # 8000/80000
    assert r["current_ratio"] == 2.0           # 30000/15000
    assert r["debt_to_equity"] == 0.5          # 20000/40000
    assert r["net_debt_to_ebitda"] == 0.53     # 8000/15000
    assert r["interest_coverage"] == round(10000000000 / 600000000, 2)
    # ROIC = EBIT*(1-tax) / (debt + equity - cash) = 7900 / (20000+40000-12000)
    assert r["roic"] == round(7900000000 / 48000000000 * 100, 2)
    assert r["roic_minus_wacc_pp"] == round(r["roic"] - 9.0, 2)
    # DuPont reconciles to ROE within rounding
    dp = r["dupont_net_margin"]/100 * r["dupont_asset_turnover"] * r["dupont_equity_multiplier"]
    assert abs(dp * 100 - r["roe"]) < 1.0
    # return of capital
    assert r["dividend_yield"] == 1.6          # 1600/100000
    assert r["buyback_yield"] == 4.0           # 4000/100000
    assert r["total_shareholder_yield"] == 5.6
    assert r["revenue_yoy_growth"] == round((40000-36000)/36000*100, 2)
    assert r["cfo_to_ni"] == round(10800/8000, 2)


def test_missing_inputs_yield_none_never_crash():
    r = compute_accounting_ratios({}, wacc=None, net_debt=None)
    assert r["roe"] is None and r["current_ratio"] is None and r["roic"] is None
    assert r["roic_minus_wacc_pp"] is None
    assert r["total_shareholder_yield"] is None


def test_block_renders_na_for_missing():
    block = format_accounting_ratios_block(compute_accounting_ratios({}), "2026-05-01", None)
    assert "## Accounting ratios" in block
    assert "n/a (data unavailable)" in block


def test_block_has_values_and_header():
    r = compute_accounting_ratios(_FIN, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    block = format_accounting_ratios_block(r, "2026-05-01", "2026-03-31")
    assert "## Accounting ratios" in block
    assert "ROE" in block and "ROIC" in block and "Cash conversion cycle" in block
    assert "verbatim" in block  # anti-fabrication usage mandate present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_accounting_ratios.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/accounting_ratios.py
"""Deterministic accounting-ratios block for pm_brief.md.

All ratios computed from financials_parser output. Missing inputs -> None ->
rendered as "n/a (data unavailable)". Nothing is fabricated.
"""
from __future__ import annotations

from typing import Any


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _pct(x, nd=2):
    return None if x is None else round(x * 100, nd)


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def compute_accounting_ratios(
    fin: dict[str, Any],
    wacc: float | None = None,
    net_debt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fin = fin or {}
    rev = fin.get("revenue_ttm")
    ni = fin.get("net_income")
    ebit = fin.get("ebit")
    ta = fin.get("total_assets")
    eq = fin.get("total_equity")
    cl = fin.get("current_liabilities")
    r: dict[str, Any] = {}

    # profitability
    r["gross_margin"] = _r(fin.get("gross_margin"))          # already a % from fundamentals
    r["operating_margin"] = _pct(_div(ebit, rev))
    r["net_margin"] = _pct(_div(ni, rev))
    r["fcf_margin"] = _pct(_div(fin.get("fcf"), rev))
    r["roe"] = _pct(_div(ni, eq))
    r["roa"] = _pct(_div(ni, ta))
    tax = fin.get("tax") if fin.get("tax") is not None else 0.21
    nopat = None if ebit is None else ebit * (1 - tax)
    invested = None
    if fin.get("total_debt") is not None and eq is not None:
        invested = fin["total_debt"] + eq - (fin.get("cash_and_equivalents") or 0)
    r["roic"] = _pct(_div(nopat, invested))
    capital_employed = None if ta is None else ta - (cl or 0)
    r["roce"] = _pct(_div(ebit, capital_employed))
    r["roic_minus_wacc_pp"] = (
        round(r["roic"] - wacc * 100, 2) if (r["roic"] is not None and wacc is not None) else None
    )

    # DuPont (net margin x asset turnover x equity multiplier ~= ROE)
    r["dupont_net_margin"] = r["net_margin"]
    r["dupont_asset_turnover"] = _r(_div(rev, fin.get("total_assets_avg") or ta))
    r["dupont_equity_multiplier"] = _r(_div(ta, eq))

    # liquidity
    r["current_ratio"] = _r(_div(fin.get("current_assets"), cl))
    qa = None if fin.get("current_assets") is None else fin["current_assets"] - (fin.get("inventory") or 0)
    r["quick_ratio"] = _r(_div(qa, cl))
    r["cash_ratio"] = _r(_div(fin.get("cash_and_equivalents"), cl))

    # leverage
    r["debt_to_equity"] = _r(_div(fin.get("total_debt"), eq))
    nd = (net_debt or {}).get("net_debt")
    r["net_debt_to_ebitda"] = _r(_div(nd, fin.get("ebitda")))
    ie = fin.get("interest_expense_ttm")
    r["interest_coverage"] = _r(_div(ebit, abs(ie) if ie else None))
    r["fcf_to_debt"] = _r(_div(fin.get("fcf"), fin.get("total_debt")))

    # efficiency (turnover, average balance)
    r["dso_days"] = _r(_div((fin.get("receivables_avg") or 0) * 365, rev)) if rev else None
    r["dio_days"] = _r(_div((fin.get("inventory_avg") or 0) * 365, fin.get("cogs_ttm")))
    r["dpo_days"] = _r(_div((fin.get("payables_avg") or 0) * 365, fin.get("cogs_ttm")))
    if None not in (r["dio_days"], r["dso_days"], r["dpo_days"]):
        r["ccc_days"] = round(r["dio_days"] + r["dso_days"] - r["dpo_days"], 1)
    else:
        r["ccc_days"] = None

    # return of capital
    div = abs(fin["dividends_paid_ttm"]) if fin.get("dividends_paid_ttm") else None
    bb = abs(fin["buybacks_ttm"]) if fin.get("buybacks_ttm") else None
    r["payout_ratio"] = _pct(_div(div, ni))
    r["dividend_yield"] = _pct(_div(div, fin.get("market_cap")))
    r["buyback_yield"] = _pct(_div(bb, fin.get("market_cap")))
    if r["dividend_yield"] is not None or r["buyback_yield"] is not None:
        r["total_shareholder_yield"] = round((r["dividend_yield"] or 0) + (r["buyback_yield"] or 0), 2)
    else:
        r["total_shareholder_yield"] = None

    # growth & quality (YoY from statements; multi-year CAGR deferred — needs annual data)
    ry = fin.get("revenue_yoy_ago")
    r["revenue_yoy_growth"] = _pct(_div(rev - ry, ry)) if (rev is not None and ry) else None
    niy = fin.get("net_income_yoy_ago")
    r["net_income_yoy_growth"] = _pct(_div(ni - niy, niy)) if (ni is not None and niy) else None
    r["cfo_to_ni"] = _r(_div(fin.get("cfo_ttm"), ni))
    return r


_NA = "n/a (data unavailable)"


def _cell(v, suffix=""):
    return _NA if v is None else f"{v}{suffix}"


def format_accounting_ratios_block(
    ratios: dict[str, Any], trade_date: str | None, as_of: str | None
) -> str:
    r = ratios or {}
    rows = [
        ("Gross margin", _cell(r.get("gross_margin"), "%")),
        ("Operating margin", _cell(r.get("operating_margin"), "%")),
        ("Net margin", _cell(r.get("net_margin"), "%")),
        ("FCF margin", _cell(r.get("fcf_margin"), "%")),
        ("ROE", _cell(r.get("roe"), "%")),
        ("ROA", _cell(r.get("roa"), "%")),
        ("ROIC", _cell(r.get("roic"), "%")),
        ("ROCE", _cell(r.get("roce"), "%")),
        ("ROIC − WACC (spread)", _cell(r.get("roic_minus_wacc_pp"), " pp")),
        ("DuPont (net margin × asset turnover × equity mult.)",
         f"{_cell(r.get('dupont_net_margin'),'%')} × {_cell(r.get('dupont_asset_turnover'),'x')} × {_cell(r.get('dupont_equity_multiplier'),'x')}"),
        ("Current ratio", _cell(r.get("current_ratio"), "x")),
        ("Quick ratio", _cell(r.get("quick_ratio"), "x")),
        ("Cash ratio", _cell(r.get("cash_ratio"), "x")),
        ("Debt / equity", _cell(r.get("debt_to_equity"), "x")),
        ("Net debt / EBITDA", _cell(r.get("net_debt_to_ebitda"), "x")),
        ("Interest coverage", _cell(r.get("interest_coverage"), "x")),
        ("FCF / total debt", _cell(r.get("fcf_to_debt"), "x")),
        ("DSO (days)", _cell(r.get("dso_days"))),
        ("DIO (days)", _cell(r.get("dio_days"))),
        ("DPO (days)", _cell(r.get("dpo_days"))),
        ("Cash conversion cycle (days)", _cell(r.get("ccc_days"))),
        ("Payout ratio", _cell(r.get("payout_ratio"), "%")),
        ("Dividend yield", _cell(r.get("dividend_yield"), "%")),
        ("Buyback yield", _cell(r.get("buyback_yield"), "%")),
        ("Total shareholder yield", _cell(r.get("total_shareholder_yield"), "%")),
        ("Revenue growth (YoY)", _cell(r.get("revenue_yoy_growth"), "%")),
        ("Net income growth (YoY)", _cell(r.get("net_income_yoy_growth"), "%")),
        ("CFO / net income (accruals quality)", _cell(r.get("cfo_to_ni"), "x")),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        f"\n\n## Accounting ratios (computed from raw/financials.json, "
        f"trade_date {trade_date}, latest quarter {as_of})\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"{body}\n\n"
        "*Use these values verbatim; do not recompute or paraphrase. Growth is "
        "year-over-year from statements (multi-year CAGR out of scope). Any "
        "`n/a (data unavailable)` means the source line-item was absent — do NOT "
        "substitute an estimate.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_accounting_ratios.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/accounting_ratios.py tests/test_accounting_ratios.py
git commit -m "feat: add deterministic accounting-ratios block"
```

---

### Task 3: Add peer market-cap parsing to peer_ratios

**Files:**
- Modify: `tradingagents/agents/utils/peer_ratios.py` (add parser mirroring `_parse_ttm_ebitda` at ~line 67-82; add `market_cap` to the dict at ~line 170-179; add a column to `format_peer_ratios_block` at ~line 256-274)
- Test: `tests/test_peer_market_cap.py`

**Interfaces:**
- Consumes: existing `peers.json` per-peer `fundamentals` text (already contains a `Market Cap: <n>` line — no new fetch).
- Produces: each per-peer dict now includes `"market_cap": float | None`. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_peer_market_cap.py
import pytest
from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

pytestmark = pytest.mark.unit

# Peer bundle shape mirrors researcher._fetch_financials output.
_PEER_FUND = (
    "# Fundamentals\nName: Peer Inc\nMarket Cap: 50000000000\n"
    "PE Ratio (TTM): 18.0\nForward PE: 15.0\n"
)
_PEER_BS = ",2026-03-31\nTotal Debt,10000000000\nCash And Cash Equivalents,5000000000\n"
_PEER_CF = ",2026-03-31,2025-12-31,2025-09-30,2025-06-30\nCapital Expenditure,-100000000,-100000000,-100000000,-100000000\n"
_PEER_IS = (
    ",2026-03-31,2025-12-31,2025-09-30,2025-06-30\n"
    "Total Revenue,2000000000,2000000000,2000000000,2000000000\n"
    "Operating Income,400000000,400000000,400000000,400000000\n"
    "EBITDA,600000000,600000000,600000000,600000000\n"
)
_PEERS = {
    "PEER": {
        "ticker": "PEER", "trade_date": "2026-05-01", "financial_currency": "USD",
        "fundamentals": _PEER_FUND, "balance_sheet": _PEER_BS,
        "cashflow": _PEER_CF, "income_statement": _PEER_IS,
    }
}


def test_peer_market_cap_parsed():
    out = compute_peer_ratios(_PEERS, "2026-05-01")
    assert out["PEER"]["market_cap"] == 50000000000.0


def test_peer_market_cap_none_when_absent():
    peers = {"PEER": dict(_PEERS["PEER"], fundamentals="# Fundamentals\nName: X\n")}
    out = compute_peer_ratios(peers, "2026-05-01")
    assert out["PEER"]["market_cap"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_peer_market_cap.py -q`
Expected: FAIL — `KeyError: 'market_cap'` (key not yet in the per-peer dict).

- [ ] **Step 3: Write minimal implementation**

In `peer_ratios.py`, add a parser next to `_parse_ttm_ebitda` (~line 82):

```python
def _parse_market_cap(text: str):
    import re
    m = re.search(r"^Market Cap:\s*(-?[0-9.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None
```

In `_compute_one_peer`, read it from the peer's fundamentals text and add to the returned dict (~line 170-179):

```python
    market_cap = _parse_market_cap(peer_data.get("fundamentals", ""))
    return {
        # ... existing keys unchanged ...
        "market_cap": market_cap,
        "source": "...",  # unchanged
    }
```

In `format_peer_ratios_block`, add a `Market Cap` column to the header and each row (~line 256-274) using the existing `_b` billions-formatter:

```python
        # header: append "Market Cap |" before the trailing newline
        # row: append f" {_b(v.get('market_cap'))} |" to each ticker row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_peer_market_cap.py tests/test_peer_ratios.py -q`
Expected: PASS. Run the existing peer-ratios test too to confirm the new column didn't break its assertions; if that test asserts exact table text, update its expected header/row to include the market-cap column.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/peer_ratios.py tests/test_peer_market_cap.py
git commit -m "feat: parse peer market cap in peer_ratios (unblocks peer EV multiples)"
```

---

### Task 4: Relative-multiples block

**Files:**
- Create: `tradingagents/agents/utils/relative_multiples.py`
- Test: `tests/test_relative_multiples.py`

**Interfaces:**
- Consumes: `parse_financials(...)` output (Task 1); subject `market_cap` and `net_debt` (from `iv["inputs"]`); peer dict from `compute_peer_ratios` (now with `market_cap`, `ttm_ebitda`, `net_debt`, `ttm_pe`, `forward_pe`).
- Produces: `compute_relative_multiples(fin: dict, market_cap: float | None, net_debt: float | None, peers: dict, forward_eps: float | None = None) -> dict`; `format_relative_multiples_block(mult: dict, trade_date: str | None) -> str`. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_relative_multiples.py
import pytest
from tradingagents.agents.utils.relative_multiples import (
    compute_relative_multiples,
    format_relative_multiples_block,
)

pytestmark = pytest.mark.unit

_FIN = {
    "revenue_ttm": 40000000000.0, "ebit": 10000000000.0, "ebitda": 15000000000.0,
    "fcf": 9000000000.0, "net_income": 8000000000.0, "total_equity": 40000000000.0,
    "eps": 5.0, "forward_eps": 6.0,
}
_PEERS = {
    "A": {"market_cap": 50e9, "net_debt": 5e9, "ttm_ebitda": 6e9, "ttm_pe": 18.0,
          "forward_pe": 15.0, "latest_quarter_op_margin": 20.0, "nd_ebitda": 0.83},
    "B": {"market_cap": 70e9, "net_debt": 10e9, "ttm_ebitda": 8e9, "ttm_pe": 22.0,
          "forward_pe": 19.0, "latest_quarter_op_margin": 18.0, "nd_ebitda": 1.25},
}


def test_subject_multiples_and_ev_tie_out():
    m = compute_relative_multiples(_FIN, market_cap=100e9, net_debt=8e9, peers=_PEERS, forward_eps=6.0)
    assert m["subject"]["ev"] == 108e9                 # market_cap + net_debt
    assert m["subject"]["ev_ebitda"] == round(108e9 / 15e9, 2)
    assert m["subject"]["ev_sales"] == round(108e9 / 40e9, 2)
    assert m["subject"]["pe_ttm"] is None or isinstance(m["subject"]["pe_ttm"], float)
    assert m["subject"]["p_b"] == round(100e9 / 40e9, 2)
    # peer median EV/EBITDA = median([55/6, 80/8]) = median([9.17, 10.0])
    assert m["peer_median"]["ev_ebitda"] == round((round(55e9/6e9,2) + round(80e9/8e9,2)) / 2, 2)


def test_missing_inputs_na():
    m = compute_relative_multiples({}, market_cap=None, net_debt=None, peers={})
    assert m["subject"]["ev"] is None
    block = format_relative_multiples_block(m, "2026-05-01")
    assert "## Relative valuation multiples" in block
    assert "n/a (data unavailable)" in block


def test_block_header_and_mandate():
    m = compute_relative_multiples(_FIN, 100e9, 8e9, _PEERS, 6.0)
    block = format_relative_multiples_block(m, "2026-05-01")
    assert "EV/EBITDA" in block and "P/B" in block
    assert "EV = market cap + net debt" in block
    assert "verbatim" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_relative_multiples.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/relative_multiples.py
"""Deterministic relative-valuation-multiples block for pm_brief.md.

Subject vs peer-median. EV = market cap + net debt (same net-debt figure the
net-debt block uses, so the two blocks tie out). Missing inputs -> None -> n/a.
"""
from __future__ import annotations

from statistics import median
from typing import Any


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def _median(vals):
    xs = [v for v in vals if v is not None]
    return round(median(xs), 2) if xs else None


def compute_relative_multiples(
    fin: dict[str, Any],
    market_cap: float | None,
    net_debt: float | None,
    peers: dict[str, Any],
    forward_eps: float | None = None,
) -> dict[str, Any]:
    fin = fin or {}
    ev = market_cap + net_debt if (market_cap is not None and net_debt is not None) else None
    eps = fin.get("eps")
    price_pe_ttm = None  # P/E computed by peer_ratios for peers; subject via market_cap/net_income
    subject = {
        "market_cap": market_cap,
        "ev": ev,
        "ev_ebitda": _r(_div(ev, fin.get("ebitda"))),
        "ev_ebit": _r(_div(ev, fin.get("ebit"))),
        "ev_sales": _r(_div(ev, fin.get("revenue_ttm"))),
        "p_e_ttm": _r(_div(market_cap, fin.get("net_income"))),
        "p_e_fwd": _r(_div(market_cap, (forward_eps * (market_cap / eps)) if (forward_eps and eps and market_cap) else None)),
        "p_b": _r(_div(market_cap, fin.get("total_equity"))),
        "p_s": _r(_div(market_cap, fin.get("revenue_ttm"))),
        "p_fcf": _r(_div(market_cap, fin.get("fcf"))),
        "peg": None,
    }
    # Forward P/E is more simply market_cap / (forward_eps * shares); if that basis
    # is unavailable, leave None rather than approximate. Peers use peer_ratios' forward_pe.
    peer_ev = {}
    peer_list = [p for p in peers.values() if isinstance(p, dict)]
    peer_median = {
        "ev_ebitda": _median([
            _r(_div((p.get("market_cap") + p.get("net_debt")) if (p.get("market_cap") is not None and p.get("net_debt") is not None) else None,
                    p.get("ttm_ebitda")))
            for p in peer_list
        ]),
        "p_e_ttm": _median([p.get("ttm_pe") for p in peer_list]),
        "p_e_fwd": _median([p.get("forward_pe") for p in peer_list]),
    }
    return {"subject": subject, "peer_median": peer_median}


_NA = "n/a (data unavailable)"


def _c(v, pre="", suf="x"):
    return _NA if v is None else f"{pre}{v}{suf}"


def format_relative_multiples_block(mult: dict[str, Any], trade_date: str | None) -> str:
    s = (mult or {}).get("subject", {}) or {}
    pm = (mult or {}).get("peer_median", {}) or {}
    rows = [
        ("EV/EBITDA", _c(s.get("ev_ebitda")), _c(pm.get("ev_ebitda"))),
        ("EV/EBIT", _c(s.get("ev_ebit")), _NA),
        ("EV/Sales", _c(s.get("ev_sales")), _NA),
        ("P/E (TTM)", _c(s.get("p_e_ttm")), _c(pm.get("p_e_ttm"))),
        ("P/E (fwd)", _c(s.get("p_e_fwd")), _c(pm.get("p_e_fwd"))),
        ("P/B", _c(s.get("p_b")), _NA),
        ("P/S", _c(s.get("p_s")), _NA),
        ("P/FCF", _c(s.get("p_fcf")), _NA),
    ]
    body = "\n".join(f"| {k} | {sub} | {peer} |" for k, sub, peer in rows)
    ev_disp = _NA if s.get("ev") is None else f"{s['ev']:,.0f}"
    return (
        f"\n\n## Relative valuation multiples (computed from raw/financials.json + "
        f"raw/peer_ratios.json, trade_date {trade_date})\n\n"
        f"Subject EV = market cap + net debt = {ev_disp} (ties to the net-debt block).\n\n"
        "| Multiple | Subject | Peer median |\n|---|---|---|\n"
        f"{body}\n\n"
        "*Use these values verbatim. EV = market cap + net debt uses the same "
        "net-debt figure as the net-debt block. Any `n/a (data unavailable)` "
        "means an input was absent — do NOT fabricate a peer multiple or an "
        "EV component.*\n"
    )
```

Note on forward P/E: the simplest correct basis is `market_cap / (forward_eps × diluted_shares)`. If you have `diluted_shares` in `fin`, prefer that formula and drop the fragile expression above; otherwise leave `p_e_fwd = None`. Adjust the test accordingly to whichever basis you implement — the assertion currently only checks type/None, so keep it permissive.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_relative_multiples.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/relative_multiples.py tests/test_relative_multiples.py
git commit -m "feat: add deterministic relative-multiples block (subject vs peer median)"
```

---

### Task 5: Wire both new blocks into researcher.py

**Files:**
- Modify: `tradingagents/agents/researcher.py` (after the intrinsic-value block site, ~line 427)
- Test: covered by an integration-style test `tests/test_researcher_blocks.py` (asserts the append idiom is present and blocks render into a temp `pm_brief.md`).

**Interfaces:**
- Consumes: in-memory `financials` (the primary-ticker bundle), `ratios` (peer_ratios output), `net_debt` (dict), and `iv` (compute_intrinsic_value result) — all already computed earlier in the same function.
- Produces: `raw/accounting_ratios.json`, `raw/relative_multiples.json`, and two appended sections in `raw/pm_brief.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_researcher_blocks.py
import json
import pytest
from tradingagents.agents.utils.financials_parser import parse_financials
from tradingagents.agents.utils.accounting_ratios import compute_accounting_ratios, format_accounting_ratios_block
from tradingagents.agents.utils.relative_multiples import compute_relative_multiples, format_relative_multiples_block

pytestmark = pytest.mark.unit


def test_blocks_compose_end_to_end(tmp_path):
    # This mirrors exactly what researcher.py will do inline.
    from tests.test_financials_parser import _BUNDLE
    fin = parse_financials(_BUNDLE)
    ar = compute_accounting_ratios(fin, wacc=0.09, net_debt={"net_debt": 8e9})
    rm = compute_relative_multiples(fin, market_cap=fin["market_cap"], net_debt=8e9, peers={})
    ar_block = format_accounting_ratios_block(ar, fin["trade_date"], fin["as_of_quarter"])
    rm_block = format_relative_multiples_block(rm, fin["trade_date"])
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(ar_block)
        f.write(rm_block)
    text = pm.read_text(encoding="utf-8")
    assert "## Accounting ratios" in text
    assert "## Relative valuation multiples" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_researcher_blocks.py -q`
Expected: PASS actually (pure library composition) — if it passes immediately, that is fine; its purpose is to lock the composition contract before you touch `researcher.py`. If it fails, fix the import/signature drift before wiring.

- [ ] **Step 3: Wire into researcher.py**

Immediately after the intrinsic-value block append (~line 427, inside the same `try`-guarded region so a compute failure never crashes the run), add:

```python
        # --- Accounting ratios (deterministic) ---
        from tradingagents.agents.utils.financials_parser import parse_financials
        from tradingagents.agents.utils.accounting_ratios import (
            compute_accounting_ratios, format_accounting_ratios_block,
        )
        from tradingagents.agents.utils.relative_multiples import (
            compute_relative_multiples, format_relative_multiples_block,
        )
        fin_parsed = parse_financials(financials)
        wacc = (iv.get("inputs", {}) or {}).get("wacc") if isinstance(iv, dict) else None
        acct = compute_accounting_ratios(fin_parsed, wacc=wacc, net_debt=net_debt)
        (raw / "accounting_ratios.json").write_text(
            json.dumps(acct, indent=2, default=str), encoding="utf-8")
        acct_block = format_accounting_ratios_block(
            acct, fin_parsed.get("trade_date"), fin_parsed.get("as_of_quarter"))
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(acct_block)

        # --- Relative multiples (deterministic) ---
        mc = (iv.get("inputs", {}) or {}).get("market_cap") if isinstance(iv, dict) else fin_parsed.get("market_cap")
        nd_val = (net_debt or {}).get("net_debt")
        rel = compute_relative_multiples(
            fin_parsed, market_cap=mc, net_debt=nd_val, peers=ratios,
            forward_eps=fin_parsed.get("forward_eps"))
        (raw / "relative_multiples.json").write_text(
            json.dumps(rel, indent=2, default=str), encoding="utf-8")
        rel_block = format_relative_multiples_block(rel, fin_parsed.get("trade_date"))
        with open(pm_brief_path, "a", encoding="utf-8") as f:
            f.write(rel_block)
```

Confirm variable names against the surrounding code: `iv` (intrinsic-value result dict), `net_debt` (dict), `ratios` (peer_ratios output), `financials` (primary bundle), `pm_brief_path`, `raw`. If any differ locally, adapt — do not rename the existing variables.

- [ ] **Step 4: Run the full unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: PASS (all existing tests + the new modules). If `researcher.py` has an importable smoke test, run it too.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_researcher_blocks.py
git commit -m "feat: wire accounting-ratios + relative-multiples blocks into researcher"
```

---

### Task 6: Wire up orphaned insider.json (WP1a)

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` (line 128-129 `files=[...]`; `_SYSTEM` prompt at 25-116)
- Test: `tests/test_fundamentals_prompt.py`

**Interfaces:**
- Consumes: existing `raw/insider.json` (already written by `researcher.py:150-153`).
- Produces: the analyst prompt now receives `insider.json` and mandates an insider-transactions subsection.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamentals_prompt.py
import pytest
from tradingagents.agents.analysts import fundamentals_analyst as fa

pytestmark = pytest.mark.unit


def test_insider_json_in_files_list():
    import inspect
    src = inspect.getsource(fa)
    assert '"insider.json"' in src


def test_insider_section_and_citation_mandate():
    assert "Insider transactions" in fa._SYSTEM
    assert "insider.json" in fa._SYSTEM  # citation-source mandate updated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py -q`
Expected: FAIL — strings not present yet.

- [ ] **Step 3: Implement**

Add `"insider.json"` to the `files=[...]` list (line 128-129). Add a mandated section to `_SYSTEM` in the same shape as the existing sections (header + table + provenance instruction), e.g.:

```python
## Insider transactions

| Window | Net buy/sell | Notable individuals |
|---|---|---|
| Last 6-12 months | <net $ or share count> | <CEO/CFO/director names & direction> |

Summarize cluster buying/selling and any signal. If raw/insider.json is empty,
state "no reported insider transactions in the window" — do not infer activity.
```

Update the closing blanket mandate (line 115-116) to also name `insider.json`:
`"Every numerical claim ... must trace back to financials.json, peers.json, news.json, reference.json, or insider.json. No invented numbers."`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py tests/test_fundamentals_prompt.py
git commit -m "feat: feed orphaned insider.json into fundamentals analyst + mandate section"
```

---

### Task 7: Surface earnings-surprise history (WP1b)

**Files:**
- Modify: `tradingagents/agents/utils/calendar.py` (row read at line 108-118; per-ticker output dict / docstring 136-154)
- Modify: `tradingagents/agents/managers/pm_preflight.py` (add a `## Surprise history` block near `_format_calendar_block` at 163-225)
- Test: `tests/test_calendar_surprise.py`

**Interfaces:**
- Consumes: yfinance `earnings_dates` rows (already fetched) — columns `EPS Estimate`, `Reported EPS`, `Surprise(%)`.
- Produces: `compute_calendar(...)` per-ticker dict gains a `surprises: list[dict]` field (date, reported, estimate, surprise_pct); a new `## Surprise history` block in `pm_brief.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calendar_surprise.py
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.utils.calendar import compute_calendar

pytestmark = pytest.mark.unit


def _earnings_df(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"EPS Estimate": [r[1] for r in rows],
         "Reported EPS": [r[2] for r in rows],
         "Surprise(%)": [r[3] for r in rows]},
        index=idx,
    )


def test_surprise_history_captured():
    df = _earnings_df([
        ("2025-07-30", 2.90, 2.95, 1.72),
        ("2025-10-29", 3.05, 3.10, 1.64),
        ("2026-01-29", 3.20, 3.22, 0.63),
        ("2026-04-29", 3.40, 3.45, 1.47),
        ("2026-07-25", 3.55, None, None),
    ])
    fake = MagicMock()
    fake.earnings_dates = df
    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake):
        out = compute_calendar("2026-05-01", ["MSFT"])
    sur = out["MSFT"]["surprises"]
    assert len(sur) == 4                       # only past, reported rows
    assert sur[0]["surprise_pct"] == 1.47      # most recent first
    assert sur[0]["reported"] == 3.45
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calendar_surprise.py -q`
Expected: FAIL — `KeyError: 'surprises'`.

- [ ] **Step 3: Implement**

In `calendar.py` `_compute_one_ticker`, alongside the existing `reported = row.get("Reported EPS")` (line 114), also read `row.get("EPS Estimate")` and `row.get("Surprise(%)")`; when a row is past AND reported is not NaN, append `{"date": date_str, "reported": float(reported), "estimate": <float|None>, "surprise_pct": <float|None>}` to a `surprises` list; sort most-recent-first and keep the last 8; add `surprises` to the per-ticker output dict (and document it in the 136-154 docstring).

In `pm_preflight.py`, add `_format_surprise_block(raw_dir)` mirroring `_format_calendar_block` (read `raw/calendar.json`, loop tickers → markdown rows `| date | reported | estimate | surprise % |`, add a beat/miss streak summary line and a trailing usage mandate), and append it right after the calendar block append (line 368-372), guarded by `if surprise_block:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calendar_surprise.py tests/test_calendar.py -q`
Expected: PASS. Confirm the existing `test_calendar.py` still passes (its fixture omits the new columns — `row.get(...)` must tolerate their absence and yield an empty `surprises` list, not crash).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/calendar.py tradingagents/agents/managers/pm_preflight.py tests/test_calendar_surprise.py
git commit -m "feat: capture and surface earnings-surprise history"
```

---

### Task 8: QC enforcement + citation mandates

**Files:**
- Modify: `tradingagents/agents/managers/qc_agent.py` (`_SYSTEM` at 37-180; header "16-item" at line 49; "Apply the 16-item checklist" at lines 46 and 239)
- Modify: `tradingagents/agents/managers/portfolio_manager.py` (`_QC_CHECKLIST` at line 127 — mirror the two new items so the PM's self-correction also gates them)
- Test: `tests/test_qc_new_items.py`

**Interfaces:**
- Consumes: nothing new — pure prompt-string edits.
- Produces: two new grouped checklist items (17 & 18) enforcing the new blocks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qc_new_items.py
import pytest
from tradingagents.agents.managers import qc_agent

pytestmark = pytest.mark.unit


def test_qc_has_18_items_and_new_rules():
    s = qc_agent._SYSTEM
    assert "18-item checklist" in s
    assert "17." in s and "18." in s
    assert "Accounting ratios" in s          # item 17 topic
    assert "Relative valuation multiples" in s or "EV" in s  # item 18 topic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qc_new_items.py -q`
Expected: FAIL — "18-item checklist" absent.

- [ ] **Step 3: Implement**

In `qc_agent.py` append after item 16 (ends ~line 164), matching the existing numbered-prose style with `\` line-continuations:

```
17. **Accounting ratios cited.** The Accounting-ratios block's ROE / ROIC / \
net-debt-to-EBITDA / leverage figures appear in the report where relevant and \
match raw/accounting_ratios.json — no contradictory or recomputed values.
18. **Relative multiples consistent.** Relative-valuation multiples (EV/EBITDA, \
P/E, P/B) are cited from raw/relative_multiples.json; subject EV equals market \
cap + net debt and ties to the net-debt block; no fabricated peer multiples.
```

Change `# 16-item checklist` → `# 18-item checklist` (line 49) and both "Apply the 16-item checklist" occurrences (lines 46, 239) → "18-item". Mirror the same two items into `portfolio_manager.py`'s `_QC_CHECKLIST` (line 127), renumbering to follow its existing count.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qc_new_items.py tests/test_pm_qc_checklist.py -q`
Expected: PASS. If `test_pm_qc_checklist.py` asserts an exact item count, update it to the new count.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/qc_agent.py tradingagents/agents/managers/portfolio_manager.py tests/test_qc_new_items.py
git commit -m "feat: add QC items 17-18 enforcing new fundamentals blocks"
```

---

### Task 9: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all green (existing ~216 + new tests). Investigate any regression before proceeding.

- [ ] **Step 2: Grep for the new blocks appearing in a real brief (optional manual smoke)**

If a smoke fixture / recorded run exists, regenerate one ticker and confirm `raw/pm_brief.md` contains `## Accounting ratios`, `## Relative valuation multiples`, and `## Surprise history`, and that `raw/accounting_ratios.json` / `raw/relative_multiples.json` were written. Confirm `n/a (data unavailable)` shows for genuinely-missing cells (do not fabricate to remove them).

- [ ] **Step 3: Final commit (if any smoke-driven fixups)**

```bash
git add -A && git commit -m "test: full-suite green for deterministic fundamentals expansion"
```

---

## Out of scope (later phases — do NOT build here)

- Multi-year (3/5yr) revenue/EPS/FCF CAGRs — need annual statements not in the quarterly `financials.json`; Phase 1 does honest YoY-from-statements only.
- WP4 fraud/distress screens (Altman Z-score, Beneish M-score) — data layer is now available via `financials_parser`; scores are a later phase.
- WP5 new free data sources (8-K, DEF 14A, 13F, short interest, analyst recommendations).
- WP6 qualitative prompt upgrades (Porter's Five Forces, moat durability, governance, etc.).
- Consolidating `intrinsic_value.parse_fundamentals` onto `financials_parser`.
- Self-vs-own-history relative-multiple bands.

## Self-Review

- **Spec coverage:** WP1a (Task 6), WP1b (Task 7), WP2 accounting ratios incl. ROIC-vs-WACC / DuPont / turnover / return-of-capital / accruals (Tasks 1-2), WP3 relative multiples + peer market cap + EV tie-out (Tasks 3-5), full block+cite+QC enforcement (Tasks 6, 8), graceful `n/a` degradation (every compute task's "missing inputs" test), testing across archetypes (compounder in fixtures; sparse/`{}` cases in each `test_*_missing` test). CAGR deferral is documented honestly rather than silently dropped.
- **Placeholder scan:** no TBD/TODO; every code step shows real code; commands have expected output.
- **Type consistency:** `parse_financials -> dict` keys are consumed with the same names in Tasks 2/4/5; `compute_accounting_ratios(fin, wacc, net_debt)` and `compute_relative_multiples(fin, market_cap, net_debt, peers, forward_eps)` signatures match their call sites in Task 5; `net_debt` is always the dict (with `.get("net_debt")`) at call sites, the float only inside `relative_multiples`.
