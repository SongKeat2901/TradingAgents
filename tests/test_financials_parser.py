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
    "Operating Income,2600000000,2500000000,2400000000,2300000000,2200000000\n"
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
    # latest-quarter counterparts = column index 0 (pair with *_yoy_ago so
    # growth diffs quarter-vs-year-ago-quarter, never TTM-vs-quarter)
    assert fin["revenue_latest_q"] == 10000000000
    assert fin["net_income_latest_q"] == 2100000000
    # average balance = mean(col0, col1)
    assert fin["receivables_avg"] == (6000000000 + 5800000000) / 2
    # ebit_ttm = sum of last 4 quarterly "Operating Income" columns (TTM,
    # not the single-quarter "ebit" from parse_fundamentals)
    assert fin["ebit_ttm"] == 2600000000 + 2500000000 + 2400000000 + 2300000000


def test_parse_financials_missing_rows_are_none_not_fabricated():
    sparse = dict(_BUNDLE, balance_sheet="# BS\n\n,2026-03-31\nTotal Assets,80000000000\n")
    fin = parse_financials(sparse)
    assert fin["total_assets"] == 80000000000
    assert fin["inventory"] is None      # row absent → None, never guessed
    assert fin["receivables_avg"] is None
    assert fin["current_liabilities"] is None


def test_parse_financials_tolerates_non_dict():
    assert parse_financials(None)["market_cap"] is None


def test_parse_financials_retained_earnings():
    from tradingagents.agents.utils.financials_parser import parse_financials
    bundle = {"ticker": "ACME", "trade_date": "2026-05-01", "financial_currency": "USD",
              "fundamentals": "# f\nMarket Cap: 1\n",
              "balance_sheet": "# BS\n\n,2026-03-31,2025-12-31\nTotal Assets,80,79\nRetained Earnings,40000000000,38000000000\n",
              "cashflow": "", "income_statement": ""}
    assert parse_financials(bundle)["retained_earnings"] == 40000000000
