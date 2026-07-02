"""Cash-flow momentum (QoQ) block + segment-trend / call-takeaway directives
(pro-deck technique D, deck pp66+70).

The deck's p70 'earnings call takeaways' quantitative half (OCF +100% QoQ,
capex -11% QoQ, FCF -1.8B vs -11.5B) is exactly reproducible from the
quarterly cashflow columns we already fetch — so it becomes a deterministic
block, not prose the LLM re-derives.
"""
import pytest

from tradingagents.agents.utils.cashflow_momentum import (
    compute_cashflow_momentum,
    format_cashflow_momentum_block,
)

pytestmark = pytest.mark.unit


def _fin(cashflow_csv: str) -> dict:
    return {"cashflow": cashflow_csv}


# Mirrors the live ORCL quarterly columns (most-recent-first, like yfinance)
_ORCL_CF = (
    ",2026-05-31,2026-02-28,2025-11-30,2025-08-31,2025-05-31\n"
    "Operating Cash Flow,14620000000,7151000000,2066000000,8140000000,6157000000\n"
    "Capital Expenditure,-16493000000,-18635000000,-12033000000,-8502000000,-9080000000\n"
    "Free Cash Flow,-1873000000,-11484000000,-9967000000,-362000000,-2923000000\n"
)


def test_happy_path_matches_deck_p70():
    m = compute_cashflow_momentum(_fin(_ORCL_CF))
    assert m["available"] is True
    assert len(m["quarters"]) == 5
    # chronological ascending
    assert m["quarters"][0]["end"] == "2025-05-31"
    assert m["quarters"][-1]["end"] == "2026-05-31"
    latest = m["latest_qoq"]
    # OCF +104% QoQ (deck: ">100%")
    assert latest["ocf_pct"] == pytest.approx(1.044, abs=0.01)
    # capex -11.5% QoQ on spend magnitude (deck: "-11%")
    assert latest["capex_pct"] == pytest.approx(-0.115, abs=0.005)
    # FCF -1.87B vs -11.48B (deck: "-1.8B, a large improvement from -11.5B")
    assert latest["fcf_delta"] == pytest.approx(9.611e9, rel=1e-3)
    assert latest["fcf_latest"] == pytest.approx(-1.873e9)
    assert latest["fcf_prev"] == pytest.approx(-11.484e9)


def test_fcf_derived_from_ocf_plus_capex_when_row_missing():
    csv = (
        ",2026-03-31,2025-12-31\n"
        "Operating Cash Flow,10000000000,8000000000\n"
        "Capital Expenditure,-4000000000,-5000000000\n"
    )
    m = compute_cashflow_momentum(_fin(csv))
    assert m["available"] is True
    assert m["quarters"][-1]["fcf"] == pytest.approx(6e9)
    assert m["latest_qoq"]["fcf_delta"] == pytest.approx(3e9)


def test_unavailable_with_fewer_than_two_quarters():
    csv = ",2026-03-31\nOperating Cash Flow,10000000000\nCapital Expenditure,-4000000000\n"
    m = compute_cashflow_momentum(_fin(csv))
    assert m["available"] is False
    assert m["reason"]


def test_unavailable_on_missing_cashflow():
    m = compute_cashflow_momentum({"cashflow": ""})
    assert m["available"] is False


def test_pct_none_when_prior_nonpositive():
    csv = (
        ",2026-03-31,2025-12-31\n"
        "Operating Cash Flow,10000000000,-2000000000\n"
        "Capital Expenditure,-4000000000,-5000000000\n"
    )
    m = compute_cashflow_momentum(_fin(csv))
    assert m["latest_qoq"]["ocf_pct"] is None  # % change vs a negative base is meaningless
    assert m["latest_qoq"]["fcf_delta"] is not None  # $ delta still valid


def test_format_block_happy_path():
    block = format_cashflow_momentum_block(compute_cashflow_momentum(_fin(_ORCL_CF)), "2026-07-01")
    assert "## Cash-flow momentum (QoQ)" in block
    assert "2026-07-01" in block
    assert "$14.6B" in block  # latest OCF in table
    assert "+104" in block    # OCF QoQ %
    assert "−11.5%" in block or "-11.5%" in block  # capex QoQ %
    assert "verbatim" in block


def test_format_block_unavailable():
    block = format_cashflow_momentum_block(compute_cashflow_momentum({"cashflow": ""}), "2026-07-01")
    assert "## Cash-flow momentum (QoQ)" in block
    assert "unavailable" in block.lower()
    assert "Do not cite" in block


def test_researcher_wires_cashflow_momentum():
    import inspect
    from tradingagents.agents import researcher
    src = inspect.getsource(researcher)
    assert "cashflow_momentum" in src
    assert "format_cashflow_momentum_block" in src


def test_financial_role_has_segment_trend_directive():
    from tradingagents.agents.analysts.fundamentals_roles import _SYSTEM_FINANCIAL
    # multi-quarter YoY/QoQ segment view from disclosed figures only
    assert "YoY" in _SYSTEM_FINANCIAL and "QoQ" in _SYSTEM_FINANCIAL
    low = _SYSTEM_FINANCIAL.lower()
    assert "segment" in low and "trend" in low


def test_financial_role_has_earnings_takeaways_directive():
    from tradingagents.agents.analysts.fundamentals_roles import _SYSTEM_FINANCIAL
    assert "Cash-flow momentum (QoQ)" in _SYSTEM_FINANCIAL
    assert "takeaway" in _SYSTEM_FINANCIAL.lower()
