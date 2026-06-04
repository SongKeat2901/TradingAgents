import pytest

from tradingagents.macro import bias, regime as regime_mod
from tradingagents.macro.betas import Betas
from tradingagents.macro.pillars import PillarScore

pytestmark = pytest.mark.unit

_ALL = ["growth", "inflation", "liquidity", "financial_conditions",
        "risk_appetite", "positioning"]


def _regime(score_by_name):
    ps = []
    for n in _ALL:
        sc = score_by_name.get(n, 0.0)
        status = "G" if sc >= 0.2 else "R" if sc <= -0.2 else "A"
        ps.append(PillarScore(name=n, score=sc, status=status))
    return regime_mod.build(ps)


def test_expected_factor_moves_rates_rise_on_growth_inflation():
    r = _regime({"growth": 1.0, "inflation": -1.0, "liquidity": 0.0})
    # inflation pillar -1.0 == rising inflation; growth +1.0 → rates rise
    moves = bias.expected_factor_moves(r)
    assert moves["d_10y"] > 0


def test_positive_tilt_when_betas_align_with_regime():
    r = _regime({n: 0.6 for n in _ALL})           # risk-on, GO
    b = Betas("X", {"d_10y": 0, "d_dxy": 0, "d_hy_spread": 0,
                    "oil_ret": 0, "mkt": 1.5, "growth_value": 0}, 0.8, "high", 300)
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.10)
    assert sb.macro_delta_pct > 0
    assert sb.adjusted_ev_pct > 0.10
    assert sb.conviction > 0


def test_tilt_capped_at_ev_tilt_cap():
    r = _regime({n: 1.0 for n in _ALL})
    b = Betas("X", {f: 50.0 for f in ["d_10y", "d_dxy", "d_hy_spread",
              "oil_ret", "mkt", "growth_value"]}, 0.9, "high", 300)
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.10)
    assert abs(sb.macro_delta_pct) <= bias.EV_TILT_CAP + 1e-9


def test_gate_stand_down_zeroes_conviction_and_flags_action():
    r = _regime({n: -0.6 for n in _ALL})          # all red → STAND_DOWN
    b = Betas("X", {f: 0.0 for f in
              ["d_10y", "d_dxy", "d_hy_spread", "oil_ret", "mkt", "growth_value"]},
              0.5, "high", 300)
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.20)
    assert sb.conviction == 0.0
    assert "STAND DOWN" in sb.action.upper() or "NO NEW RISK" in sb.action.upper()


def test_driver_names_top_two_betas_by_abs():
    b = Betas("X", {"d_10y": 0.1, "d_dxy": -2.0, "d_hy_spread": 1.5,
                    "oil_ret": 0.0, "mkt": 0.2, "growth_value": 0.0}, 0.7, "high", 300)
    drv = bias.describe_driver(b)
    assert "d_dxy" in drv and "d_hy_spread" in drv
