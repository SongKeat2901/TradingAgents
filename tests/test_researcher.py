"""Tests for the Researcher data fetcher."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _stub_state(tmp_path, peers=("GOOG", "META")):
    return {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "peers": list(peers),
        "raw_dir": str(tmp_path / "raw"),
    }


def test_researcher_writes_all_expected_files(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    # Stub the data tools so the test doesn't network out
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {"ticker": t, "revenue": 100})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {"sentiment": 0.5})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {
        "ohlcv": "Date,Open,High,Low,Close,Volume\n2026-05-01,408.0,462.0,379.0,410.0,1000000",
    })
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": 405.0, "close_200_sma": 380.0,
        "rsi": 58.0, "macd": 1.2,
        "boll_ub": 430.0, "boll_lb": 390.0, "atr": 4.2,
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    raw = Path(state["raw_dir"])
    for f in ("financials.json", "peers.json", "news.json", "insider.json",
              "social.json", "prices.json", "reference.json"):
        assert (raw / f).exists(), f"missing: {f}"


def test_researcher_writes_reference_with_required_keys(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {
        "ohlcv": "Date,Open,High,Low,Close,Volume\n2026-05-01,408.0,462.0,379.0,410.0,1000000",
    })
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": 405.0, "close_200_sma": 380.0,
        "rsi": 58.0, "macd": 1.2,
        "boll_ub": 430.0, "boll_lb": 390.0, "atr": 4.2,
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    ref = json.loads((Path(state["raw_dir"]) / "reference.json").read_text())
    assert ref["ticker"] == "MSFT"
    assert ref["trade_date"] == "2026-05-01"
    assert ref["reference_price_source"].startswith("yfinance close")
    assert ref["spot_50dma"] == 405.0
    assert ref["spot_200dma"] == 380.0
    assert ref["atr_14"] == 4.2


def test_researcher_writes_peer_per_ticker(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    fetched = {}

    def fake_financials(t, d):
        fetched.setdefault(t, 0)
        fetched[t] += 1
        return {"ticker": t, "revenue": 100 if t == "MSFT" else 200}

    monkeypatch.setattr(researcher, "_fetch_financials", fake_financials)
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": ""})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {})

    state = _stub_state(tmp_path, peers=("GOOG", "META", "AAPL"))
    researcher.fetch_research_pack(state)

    peers_data = json.loads((Path(state["raw_dir"]) / "peers.json").read_text())
    assert set(peers_data.keys()) == {"GOOG", "META", "AAPL"}
    assert peers_data["GOOG"]["revenue"] == 200
    # Main ticker also fetched
    assert "MSFT" in fetched
