import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import pillars
from tradingagents.macro.config import IndicatorSpec

pytestmark = pytest.mark.unit


def _rising_series(n=300, start=100.0, step=0.5):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(start + step * np.arange(n), index=idx)


def test_zscore_latest_is_positive_for_rising_series():
    z = pillars.zscore_latest(_rising_series(), window=200)
    assert z > 0


def test_indicator_score_inverts_when_flagged():
    spec_plain = IndicatorSpec("a", "fred", "A", "growth", invert=False)
    spec_inv = IndicatorSpec("b", "fred", "B", "growth", invert=True)
    s = _rising_series()
    assert pillars.indicator_score(spec_plain, s) > 0
    assert pillars.indicator_score(spec_inv, s) < 0


def test_score_pillar_aggregates_and_statuses_green():
    specs = [IndicatorSpec("a", "fred", "A", "growth"),
             IndicatorSpec("b", "fred", "B", "growth")]
    series = {"a": _rising_series(), "b": _rising_series()}
    ps = pillars.score_pillar("growth", specs, series)
    assert ps.name == "growth"
    assert ps.score > 0.2 and ps.status == "G"


def test_score_pillar_handles_missing_series_gracefully():
    specs = [IndicatorSpec("a", "fred", "A", "growth")]
    ps = pillars.score_pillar("growth", specs, series={})  # nothing loaded
    assert ps.status == "A" and ps.score == 0.0


def test_score_all_returns_one_per_pillar():
    from tradingagents.macro.config import INDICATORS, PILLARS
    series = {sp.name: _rising_series() for sp in INDICATORS}
    out = pillars.score_all(series)
    assert {p.name for p in out} == set(PILLARS)
