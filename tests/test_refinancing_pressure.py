import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.distress_screens import (
    compute_refinancing_pressure, format_refinancing_block,
)


def test_elevated_high_current_low_cash():
    # current=50 of total=100 (50%) >=40, cash 20 < 50 -> elevated
    r = compute_refinancing_pressure({"total_debt": 100, "long_term_debt": 50, "cash_and_equivalents": 20})
    assert r["pct_current_of_total"] == 50.0 and r["cash_cover_current"] == 0.4
    assert r["flag"] == "elevated"


def test_low_when_mostly_long_term_and_cash_rich():
    # current=10 of 100 (10%), cash 50 covers -> low
    r = compute_refinancing_pressure({"total_debt": 100, "long_term_debt": 90, "cash_and_equivalents": 50})
    assert r["flag"] == "low"


def test_moderate_high_current_but_cash_covers():
    r = compute_refinancing_pressure({"total_debt": 100, "long_term_debt": 50, "cash_and_equivalents": 80})
    assert r["flag"] == "moderate"  # 50% near-term but cash covers


def test_na_when_split_missing():
    r = compute_refinancing_pressure({"total_debt": 100, "long_term_debt": None})
    assert r["applicable"] is False


def test_na_when_ltd_exceeds_total():
    r = compute_refinancing_pressure({"total_debt": 100, "long_term_debt": 120, "cash_and_equivalents": 10})
    assert r["applicable"] is False


def test_block_render_and_na():
    block = format_refinancing_block(compute_refinancing_pressure(
        {"total_debt": 100, "long_term_debt": 50, "cash_and_equivalents": 20}))
    assert "## Refinancing / maturity-wall proxy" in block and "elevated" in block
    assert "verbatim" in block and "10-K" in block
    na = format_refinancing_block(compute_refinancing_pressure({"total_debt": None}))
    assert "n/a" in na
