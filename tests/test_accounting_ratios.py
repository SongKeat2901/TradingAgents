import pytest
from tradingagents.agents.utils.accounting_ratios import (
    compute_accounting_ratios,
    format_accounting_ratios_block,
)

pytestmark = pytest.mark.unit

_FIN = {
    "revenue_ttm": 40000000000.0, "net_income": 8000000000.0, "ebit_ttm": 10000000000.0,
    "ebitda": 15000000000.0, "fcf": 9000000000.0, "tax": 0.21,
    "gross_margin": 0.60,
    "total_assets": 80000000000.0, "total_assets_avg": 79000000000.0,
    "current_assets": 30000000000.0, "current_liabilities": 15000000000.0,
    "cash_and_equivalents": 12000000000.0, "inventory": 4000000000.0,
    "inventory_avg": 3900000000.0, "receivables_avg": 5900000000.0,
    "payables_avg": 2950000000.0, "cogs_ttm": 16000000000.0,
    "total_debt": 20000000000.0, "total_equity": 40000000000.0,
    "interest_expense_ttm": 600000000.0, "market_cap": 100000000000.0,
    "dividends_paid_ttm": -1600000000.0, "buybacks_ttm": -4000000000.0,
    # revenue_latest_q/revenue_yoy_ago are the single-quarter figures growth
    # must use; kept deliberately far from revenue_ttm (~4x it, like real
    # TTM-vs-Q data) so a regression that diffs TTM against a year-ago
    # quarter is caught -- see test_revenue_and_net_income_growth_use_quarter_not_ttm.
    "revenue_latest_q": 82900000000.0, "revenue_yoy_ago": 70100000000.0,
    "net_income_latest_q": 22000000000.0, "net_income_yoy_ago": 18000000000.0,
    "cfo_ttm": 10800000000.0,
}


def test_core_ratios():
    r = compute_accounting_ratios(_FIN, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    assert r["gross_margin"] == 60.0           # 0.60 fraction -> 60%
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
    # growth compares the latest single quarter to the same quarter a year
    # ago -- NOT revenue_ttm/net_income (TTM), which are ~4x a single quarter.
    assert r["revenue_yoy_growth"] == round((82900-70100)/70100*100, 2)
    assert r["net_income_yoy_growth"] == round((22000-18000)/18000*100, 2)
    assert r["cfo_to_ni"] == round(10800/8000, 2)


def test_revenue_and_net_income_growth_use_quarter_not_ttm():
    """Regression for the TTM-vs-quarter units mismatch: growth must diff
    revenue_latest_q/net_income_latest_q against *_yoy_ago (both single
    quarters), never revenue_ttm/net_income (TTM) against a single quarter.
    On the old code this produced ~354%/~385% on live MSFT data instead of
    the true ~18%/~23%."""
    fin = dict(_FIN)
    fin["revenue_ttm"] = 320000000000.0  # ~4x revenue_latest_q, as on real data
    fin["net_income"] = 88000000000.0    # ~4x net_income_latest_q
    r = compute_accounting_ratios(fin, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    assert r["revenue_yoy_growth"] == round((82900000000.0 - 70100000000.0) / 70100000000.0 * 100, 2)
    assert r["revenue_yoy_growth"] == 18.26
    assert r["net_income_yoy_growth"] == round((22000000000.0 - 18000000000.0) / 18000000000.0 * 100, 2)
    # sanity: nowhere near the TTM-vs-quarter-inflated ~354%/~385%
    assert r["revenue_yoy_growth"] < 50
    assert r["net_income_yoy_growth"] < 50


def test_missing_inputs_yield_none_never_crash():
    r = compute_accounting_ratios({}, wacc=None, net_debt=None)
    assert r["roe"] is None and r["current_ratio"] is None and r["roic"] is None
    assert r["roic_minus_wacc_pp"] is None
    assert r["total_shareholder_yield"] is None


def test_missing_receivables_avg_yields_none_dso_not_fabricated_zero():
    fin = dict(_FIN)
    fin.pop("receivables_avg")
    r = compute_accounting_ratios(fin, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    assert r["dso_days"] is None
    block = format_accounting_ratios_block(r, "2026-05-01", "2026-03-31")
    assert "| DSO (days) | n/a (data unavailable) |" in block


def test_zero_dividends_and_buybacks_render_as_real_zero_not_na():
    fin = dict(_FIN)
    fin["dividends_paid_ttm"] = 0.0
    fin["buybacks_ttm"] = 0.0
    r = compute_accounting_ratios(fin, wacc=0.09, net_debt={"net_debt": 8000000000.0})
    assert r["dividend_yield"] == 0.0
    assert r["buyback_yield"] == 0.0


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
    assert "21% statutory tax rate" in block  # ROIC tax assumption disclosed


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
