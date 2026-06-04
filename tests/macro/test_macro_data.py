import json
import pandas as pd
import pytest

from tradingagents.macro import macro_data
from tradingagents.macro.config import IndicatorSpec

pytestmark = pytest.mark.unit


def test_parse_fred_observations_to_series():
    payload = {"observations": [
        {"date": "2026-05-01", "value": "4.5"},
        {"date": "2026-05-02", "value": "."},      # FRED missing marker
        {"date": "2026-05-03", "value": "4.7"},
    ]}
    s = macro_data._parse_fred(payload)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-05-01", "2026-05-03"]
    assert s.iloc[-1] == 4.7


def test_load_series_caches_after_first_fetch(tmp_path, monkeypatch):
    spec = IndicatorSpec("vix", "yfinance", "^VIX", "risk_appetite")
    calls = {"n": 0}

    def fake_fetch(spec_):
        calls["n"] += 1
        return pd.Series([10.0, 11.0], index=pd.to_datetime(["2026-05-01", "2026-05-02"]))

    monkeypatch.setattr(macro_data, "_fetch_yfinance", fake_fetch)
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)

    s1 = macro_data.load_series(spec, as_of="2026-05-02")
    s2 = macro_data.load_series(spec, as_of="2026-05-02")
    assert calls["n"] == 1            # second call served from cache
    assert s1.equals(s2)


def test_load_series_routes_fred(tmp_path, monkeypatch):
    spec = IndicatorSpec("cpi", "fred", "CPIAUCSL", "inflation")
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(macro_data, "_fetch_fred",
                        lambda s: pd.Series([1.0], index=pd.to_datetime(["2026-05-01"])))
    out = macro_data.load_series(spec, as_of="2026-05-01")
    assert out.iloc[-1] == 1.0


def test_load_all_skips_failed_series(monkeypatch):
    from tradingagents.macro.config import IndicatorSpec
    good = IndicatorSpec("good", "yfinance", "G", "growth")
    bad = IndicatorSpec("bad", "fred", "B", "growth")

    def fake_load(spec, as_of):
        if spec.name == "bad":
            raise RuntimeError("boom")
        return pd.Series([1.0], index=pd.to_datetime(["2026-05-01"]))

    monkeypatch.setattr(macro_data, "load_series", fake_load)
    out = macro_data.load_all([good, bad], as_of="2026-06-02")
    assert "good" in out and "bad" not in out


def test_load_prices_drops_incomplete_and_caches(tmp_path, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)
    idx = pd.to_datetime(["2026-05-29", "2026-06-01"]).tz_localize("America/New_York")
    fake_hist = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    fake_hist.index.name = "Date"

    class FakeTicker:
        def __init__(self, code):
            pass

        def history(self, period="2y", auto_adjust=True):
            return fake_hist.copy()

    # Import stockstats_utils BEFORE patching yfinance, since it imports yf at
    # module level.  Then stub drop_incomplete_session so the test stays hermetic.
    import tradingagents.dataflows.stockstats_utils as _stu
    monkeypatch.setattr(_stu, "drop_incomplete_session", lambda df, now=None: df)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    s = macro_data.load_prices("AAA", as_of="2026-06-02")
    assert [round(v, 2) for v in s.values] == [100.0, 101.0]
    assert s.index[-1].strftime("%Y-%m-%d") == "2026-06-01"
    # second call served from cache (no yfinance import needed)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert macro_data.load_prices("AAA", as_of="2026-06-02").equals(s)
