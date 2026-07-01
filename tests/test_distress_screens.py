import pytest
from tradingagents.agents.utils.distress_screens import compute_altman_z, format_distress_block

pytestmark = pytest.mark.unit

# healthy: ta=100 eq=60 ca=50 cl=20 re=40 ebit=15 -> tl=40 wc=30
# x1=.3 x2=.4 x3=.15 x4=1.5 -> Z=6.56*.3+3.26*.4+6.72*.15+1.05*1.5=5.855 -> Safe
_HEALTHY = {"sector": "Technology", "total_assets": 100.0, "total_equity": 60.0,
            "current_assets": 50.0, "current_liabilities": 20.0,
            "retained_earnings": 40.0, "ebit_ttm": 15.0}
# distressed: ta=100 eq=5 ca=20 cl=40 re=-30 ebit=-10 -> Z=-2.907 -> Distress
_DISTRESS = {"sector": "Industrials", "total_assets": 100.0, "total_equity": 5.0,
             "current_assets": 20.0, "current_liabilities": 40.0,
             "retained_earnings": -30.0, "ebit_ttm": -10.0}
# grey: ta=100 eq=30 ca=40 cl=30 re=10 ebit=5 -> Z=1.768 -> Grey
_GREY = {"sector": "Consumer", "total_assets": 100.0, "total_equity": 30.0,
         "current_assets": 40.0, "current_liabilities": 30.0,
         "retained_earnings": 10.0, "ebit_ttm": 5.0}


def test_healthy_safe_zone():
    r = compute_altman_z(_HEALTHY)
    assert r["applicable"] is True
    assert r["z_score"] == 5.85  # round(5.855, 2); float repr(5.855)=5.8549999999999995 -> rounds to 5.85
    assert r["zone"] == "Safe"
    assert r["x4"] == 1.5


def test_distress_zone():
    r = compute_altman_z(_DISTRESS)
    assert r["zone"] == "Distress" and r["z_score"] < 1.1


def test_grey_zone():
    r = compute_altman_z(_GREY)
    assert r["zone"] == "Grey" and 1.1 <= r["z_score"] <= 2.6


def test_financials_skipped():
    r = compute_altman_z(dict(_HEALTHY, sector="Financial Services"))
    assert r["applicable"] is False
    assert "financial" in r["skip_reason"].lower()


def test_missing_input_na():
    r = compute_altman_z(dict(_HEALTHY, retained_earnings=None))
    assert r["applicable"] is True and r["z_score"] is None and r["zone"] is None


def test_zero_total_liabilities_na():
    # total_equity == total_assets -> total_liabilities == 0 -> x4 undefined
    r = compute_altman_z(dict(_HEALTHY, total_equity=100.0))
    assert r["z_score"] is None


def test_block_populated():
    block = format_distress_block(compute_altman_z(_HEALTHY))
    assert "## Distress screen (Altman Z″)" in block
    assert "Safe" in block and "5.85" in block and "verbatim" in block


def test_block_skipped_and_na():
    fin_block = format_distress_block(compute_altman_z(dict(_HEALTHY, sector="Financial Services")))
    assert "not applicable" in fin_block
    na_block = format_distress_block(compute_altman_z(dict(_HEALTHY, total_assets=None)))
    assert "n/a" in na_block
