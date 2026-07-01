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
    r["gross_margin"] = _pct(fin.get("gross_margin"))         # fraction (0-1) from fundamentals -> %
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

    # efficiency (turnover, average balance) — a missing average balance must
    # yield None, never fabricate 0 days.
    ra = fin.get("receivables_avg")
    r["dso_days"] = _r(_div(ra * 365, rev)) if (ra is not None and rev) else None
    ia = fin.get("inventory_avg")
    r["dio_days"] = _r(_div(ia * 365, fin.get("cogs_ttm"))) if ia is not None else None
    pa = fin.get("payables_avg")
    r["dpo_days"] = _r(_div(pa * 365, fin.get("cogs_ttm"))) if pa is not None else None
    if None not in (r["dio_days"], r["dso_days"], r["dpo_days"]):
        r["ccc_days"] = round(r["dio_days"] + r["dso_days"] - r["dpo_days"], 1)
    else:
        r["ccc_days"] = None

    # return of capital — a genuine 0.0 (non-payer) must render as 0, not n/a.
    div = abs(fin["dividends_paid_ttm"]) if fin.get("dividends_paid_ttm") is not None else None
    bb = abs(fin["buybacks_ttm"]) if fin.get("buybacks_ttm") is not None else None
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
        "substitute an estimate. ROIC uses a 21% statutory tax rate when the "
        "issuer's effective tax rate is unavailable.*\n"
    )
