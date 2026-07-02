import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.dividend_discount import compute_ddm, format_ddm_block


def _fin(ni=1000, eq=5000, divs=-300, sh=100):
    return {"net_income": ni, "total_equity": eq, "dividends_paid_ttm": divs, "diluted_shares": sh}


def test_mature_payer_value():
    # mature payer: ROE 10%, payout 60% -> g=(1-.6)*.10=4% < r=9% (valid Gordon Growth)
    # D0/share = 600/100 = 6; value = 6*1.04/(0.09-0.04) = 124.8
    r = compute_ddm(_fin(ni=1000, eq=10000, divs=-600, sh=100), cost_of_equity=0.09)
    assert r["applicable"] is True
    assert r["payout_pct"] == 60.0 and r["roe_pct"] == 10.0
    assert r["g"] == pytest.approx(0.04, abs=1e-6)
    assert r["value"] == pytest.approx(124.8, abs=0.5)
    assert r["value_g_minus"] is not None and r["value_g_plus"] is not None


def test_growth_compounder_declined():
    # high ROE (20%), low payout (30%) -> g=14% >= r=10% -> DDM invalid, honest n/a
    r = compute_ddm(_fin(ni=1000, eq=5000, divs=-300, sh=100), cost_of_equity=0.10)
    assert r["applicable"] is False
    assert "cost of equity" in r["reason"] and "DCF" in r["reason"]


def test_non_payer_na():
    r = compute_ddm(_fin(divs=0), cost_of_equity=0.09)
    assert r["applicable"] is False
    assert "not a stable dividend payer" in r["reason"] or "not a payer" in r["reason"]


def test_negative_earnings_na():
    r = compute_ddm(_fin(ni=-500), cost_of_equity=0.09)
    assert r["applicable"] is False


def test_payout_over_100_na():
    r = compute_ddm(_fin(ni=100, divs=-150), cost_of_equity=0.09)
    assert r["applicable"] is False and "100%" in r["reason"]


def test_missing_coe_na():
    r = compute_ddm(_fin(), cost_of_equity=None)
    assert r["applicable"] is False


def test_block_render_and_na():
    block = format_ddm_block(compute_ddm(_fin(ni=1000, eq=10000, divs=-600, sh=100), cost_of_equity=0.09))
    assert "## Dividend Discount Model" in block and "DDM fair value" in block
    assert "verbatim" in block and "Sensitivity" in block
    na = format_ddm_block(compute_ddm(_fin(divs=0), cost_of_equity=0.10))
    assert "n/a" in na and "stable dividend payer" in na.lower()
