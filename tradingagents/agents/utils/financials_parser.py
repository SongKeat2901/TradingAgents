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


def _series(rows: dict, *aliases: str):
    """Full most-recent-first value list for the first matching row label, else []."""
    for a in aliases:
        vals = rows.get(a)
        if vals:
            return vals
    return []


def _fcf_series(cf_rows: dict):
    fcf = _series(cf_rows, "Free Cash Flow")
    if fcf:
        return fcf
    cfo = _series(cf_rows, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _series(cf_rows, "Capital Expenditure")
    if cfo and capex and len(cfo) == len(capex):
        return [(c + x) if (c is not None and x is not None) else None
                for c, x in zip(cfo, capex)]  # capex is negative in yfinance -> CFO + capex
    return cfo or []


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
    bs_cols, bs = _parse_quarterly_csv(d.get("balance_sheet", ""))
    _, cf = _parse_quarterly_csv(d.get("cashflow", ""))
    _, is_ = _parse_quarterly_csv(d.get("income_statement", ""))
    _, bs_a = _parse_quarterly_csv(d.get("balance_sheet_annual", ""))
    _, is_a = _parse_quarterly_csv(d.get("income_statement_annual", ""))
    _, cf_a = _parse_quarterly_csv(d.get("cashflow_annual", ""))

    def _dep(rows_is, rows_cf, idx):
        # depreciation: income-statement 'Reconciled Depreciation', else cashflow D&A
        return (_row_at(rows_is, idx, "Reconciled Depreciation")
                or _row_at(rows_cf, idx, "Depreciation And Amortization",
                           "Depreciation Amortization Depletion"))

    def _annual_side(idx):
        return {
            "receivables": _row_at(bs_a, idx, "Receivables", "Accounts Receivable", "Net Receivables"),
            "current_assets": _row_at(bs_a, idx, "Current Assets", "Total Current Assets"),
            "ppe": _row_at(bs_a, idx, "Net PPE", "Net Property Plant And Equipment"),
            "total_assets": _row_at(bs_a, idx, "Total Assets"),
            "total_equity": _row_at(bs_a, idx, "Stockholders Equity", "Common Stock Equity"),
            "sales": _row_at(is_a, idx, "Total Revenue", "Operating Revenue"),
            "cogs": _row_at(is_a, idx, "Cost Of Revenue"),
            "sga": _row_at(is_a, idx, "Selling General And Administration"),
            "depreciation": _dep(is_a, cf_a, idx),
        }

    current_side = _annual_side(0)
    current_side["net_income"] = _row_at(is_a, 0, "Net Income", "Net Income Common Stockholders")
    current_side["cfo"] = _row_at(cf_a, 0, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    beneish_inputs = {"current": current_side, "prior": _annual_side(1)}

    return {
        "trade_date": d.get("trade_date"),
        "financial_currency": d.get("financial_currency"),
        "as_of_quarter": bs_cols[0] if bs_cols else None,
        # fundamentals-derived (reused; do not recompute)
        "market_cap": fund.get("market_cap"),
        "revenue_ttm": fund.get("revenue"),
        # NOTE: "ebit" is a SINGLE-QUARTER value (parse_fundamentals._col0 on the
        # income statement) — do not use it as a denominator/numerator against TTM
        # figures. "ebit_ttm" below is the trailing-twelve-month sum the
        # accounting-ratios and relative-multiples blocks must use instead.
        "ebit": fund.get("ebit"),
        "ebit_ttm": _ttm(is_, "Operating Income", "EBIT"),
        "ebitda": fund.get("ebitda"),  # already TTM, from yfinance info
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
        "long_term_debt": _row_col0(bs, "Long Term Debt"),
        "total_equity": _row_col0(bs, "Stockholders Equity", "Common Stock Equity"),
        "retained_earnings": _row_col0(bs, "Retained Earnings"),
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
        # NOTE: growth must compare like-for-like (single quarter vs the same
        # quarter a year ago) -- the *_latest_q fields below pair with
        # *_yoy_ago; never diff a TTM figure against a single-quarter one
        # (that produced a ~4x-inflated growth rate on real data).
        "revenue_latest_q": _row_at(is_, 0, "Total Revenue", "Operating Revenue"),
        "revenue_yoy_ago": _row_at(is_, 4, "Total Revenue", "Operating Revenue"),
        "cogs_ttm": _ttm(is_, "Cost Of Revenue"),
        "interest_expense_ttm": _ttm(is_, "Interest Expense", "Interest Expense Non Operating"),
        "net_income_latest_q": _row_at(is_, 0, "Net Income", "Net Income Common Stockholders"),
        "net_income_yoy_ago": _row_at(is_, 4, "Net Income", "Net Income Common Stockholders"),
        "diluted_eps_yoy_ago": _row_at(is_, 4, "Diluted EPS"),
        "beneish_inputs": beneish_inputs,
        # latest annual pretax + tax provision -> effective tax rate for NOPAT
        # (incremental-ROIC block, FA-101 Phase 5)
        "pretax_income_annual": _row_at(is_a, 0, "Pretax Income"),
        "tax_provision_annual": _row_at(is_a, 0, "Tax Provision"),
        "annual_series": {
            "revenue": _series(is_a, "Total Revenue", "Operating Revenue"),
            "diluted_eps": _series(is_a, "Diluted EPS"),
            "ebit": _series(is_a, "Operating Income", "EBIT"),
            "fcf": _fcf_series(cf_a),
            "total_debt": _series(bs_a, "Total Debt"),
            "total_equity": _series(bs_a, "Stockholders Equity", "Common Stock Equity"),
        },
    }
