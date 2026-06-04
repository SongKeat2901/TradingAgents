import pytest

from tradingagents.macro import regime
from tradingagents.macro.pillars import PillarScore

pytestmark = pytest.mark.unit


def _pillars(score_by_name):
    out = []
    for name, sc in score_by_name.items():
        status = "G" if sc >= 0.2 else "R" if sc <= -0.2 else "A"
        out.append(PillarScore(name=name, score=sc, status=status))
    return out


_ALL = ["growth", "inflation", "liquidity", "financial_conditions",
        "risk_appetite", "positioning"]


def test_gate_stand_down_when_breadth_of_red():
    ps = _pillars({n: -0.5 for n in _ALL})       # all red
    r = regime.build(ps)
    assert r.gate == "STAND_DOWN"
    assert r.red_count == 6


def test_gate_go_when_broadly_green():
    ps = _pillars({n: 0.5 for n in _ALL})
    r = regime.build(ps)
    assert r.gate == "GO"
    assert r.score > 0


def test_gate_caution_in_the_middle():
    # Two red pillars (breadth < 4) pulling the aggregate to ~-0.18 — below the
    # CAUTION threshold (-0.1) but above the STAND_DOWN floor (-0.4).
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -0.5
    scores["financial_conditions"] = -0.5
    r = regime.build(_pillars(scores))
    assert r.gate == "CAUTION"


def test_quadrant_goldilocks_growth_up_inflation_down():
    # inflation pillar score is HIGH when inflation is FALLING (invert=True),
    # so a positive inflation pillar score == disinflation == quadrant "down".
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = 0.5
    scores["inflation"] = 0.5
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Goldilocks"


def test_quadrant_stagflation_growth_down_inflation_up():
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -0.5
    scores["inflation"] = -0.5     # low pillar score == rising inflation
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Stagflation"
