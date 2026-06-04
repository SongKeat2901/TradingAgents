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
