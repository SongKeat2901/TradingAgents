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
