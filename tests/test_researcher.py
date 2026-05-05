"""Tests for the Researcher data fetcher."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# Stubs match the real tool return shapes: get_stock_data returns CSV string
# with comment header; get_indicators returns "## <ind> values from ...\n<DATE>: <val>".
_OHLCV_STUB = (
    "# Stock data for MSFT\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-01-15,395.0,415.0,385.0,400.0,1000000,0.0,0.0\n"
    "2026-03-10,400.0,460.0,395.0,455.0,1500000,0.0,0.0\n"
    "2026-04-30,415.0,420.0,395.0,408.0,1000000,0.0,0.0\n"
    "2026-05-01,408.0,425.0,379.0,410.0,1000000,0.0,0.0\n"
)

_INDICATOR_STUB = lambda val: (
    f"## sample values from 2026-04-01 to 2026-05-01:\n\n"
    f"2026-05-01: {val}\n2026-04-30: {val - 0.1}\n"
)


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
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0),
        "close_200_sma": _INDICATOR_STUB(380.0),
        "rsi": _INDICATOR_STUB(58.0),
        "macd": _INDICATOR_STUB(1.2),
        "boll_ub": _INDICATOR_STUB(430.0),
        "boll_lb": _INDICATOR_STUB(390.0),
        "atr": _INDICATOR_STUB(4.2),
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    raw = Path(state["raw_dir"])
    for f in ("financials.json", "peers.json", "news.json", "insider.json",
              "social.json", "prices.json", "reference.json"):
        assert (raw / f).exists(), f"missing: {f}"


def test_researcher_writes_reference_with_numeric_values(tmp_path, monkeypatch):
    """reference.json must contain real numbers parsed from the OHLCV CSV
    and indicator strings — the PM's QC #7 self-audit depends on this."""
    from tradingagents.agents import researcher
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0),
        "close_200_sma": _INDICATOR_STUB(380.0),
        "rsi": _INDICATOR_STUB(58.0),
        "macd": _INDICATOR_STUB(1.2),
        "boll_ub": _INDICATOR_STUB(430.0),
        "boll_lb": _INDICATOR_STUB(390.0),
        "atr": _INDICATOR_STUB(4.2),
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    ref = json.loads((Path(state["raw_dir"]) / "reference.json").read_text())
    assert ref["ticker"] == "MSFT"
    assert ref["trade_date"] == "2026-05-01"
    assert ref["reference_price"] == 410.0  # parsed from CSV row for 2026-05-01
    assert ref["reference_price_source"].startswith("yfinance close")
    assert ref["spot_50dma"] == 405.0
    assert ref["spot_200dma"] == 380.0
    # YTD high/low computed across 2026 rows up to and including 2026-05-01
    assert ref["ytd_high"] == 460.0  # max High in 2026 rows (2026-03-10 row)
    assert ref["ytd_low"] == 379.0  # min Low in 2026 rows (2026-05-01 row)
    assert ref["atr_14"] == 4.2


def test_close_on_or_before_falls_back_when_date_is_holiday(tmp_path):
    """If trade_date isn't in the OHLCV (weekend/holiday), use the most recent
    prior trading day."""
    from tradingagents.agents.researcher import _close_on_or_before, _parse_ohlcv_rows
    rows = _parse_ohlcv_rows(_OHLCV_STUB)
    # Saturday 2026-05-02 is not a trading day; should pick 2026-05-01
    assert _close_on_or_before(rows, "2026-05-02") == 410.0
    assert _close_on_or_before(rows, "2026-05-01") == 410.0
    # Date before any rows → None
    assert _close_on_or_before(rows, "2025-01-01") is None


def test_latest_indicator_value_handles_na_and_empty():
    from tradingagents.agents.researcher import _latest_indicator_value
    # Real-format string with N/A entries above the latest numeric
    txt = (
        "## atr values from 2026-04-26 to 2026-05-01:\n\n"
        "2026-05-01: 11.21\n"
        "2026-04-30: 10.19\n"
        "2026-04-26: N/A: Not a trading day (weekend or holiday)\n"
    )
    assert _latest_indicator_value(txt) == 11.21
    assert _latest_indicator_value("") is None
    assert _latest_indicator_value(None) is None  # type: ignore[arg-type]


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


def test_researcher_writes_classification_json(tmp_path, monkeypatch):
    """The Researcher must write raw/classification.json with the expected schema."""
    from tradingagents.agents import researcher
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0),
        "close_200_sma": _INDICATOR_STUB(460.0),
        "rsi": _INDICATOR_STUB(58.0),
        "macd": _INDICATOR_STUB(1.2),
        "boll_ub": _INDICATOR_STUB(430.0),
        "boll_lb": _INDICATOR_STUB(390.0),
        "atr": _INDICATOR_STUB(8.0),
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    cls_path = Path(state["raw_dir"]) / "classification.json"
    assert cls_path.exists()
    cls = json.loads(cls_path.read_text())
    # Schema check — every documented key present
    for key in ("setup_class", "gap_to_50dma_pct", "gap_to_200dma_pct",
                "ma_alignment", "recent_volume_signal", "upside_target",
                "upside_pct", "downside_target", "downside_pct",
                "reward_risk_ratio", "rationale"):
        assert key in cls, f"classification.json missing key: {key}"
    # The stubbed fixture (spot=410, 50-DMA=405, 200-DMA=460) is bear-aligned
    # downtrend; should be one of the bear classes.
    assert cls["setup_class"] in {"CAPITULATION", "BREAKDOWN", "DOWNTREND"}


# NOTE: calendar.json is now written by PM Pre-flight (which runs before the
# Researcher and has the peer list), not by the Researcher. See
# tests/test_pm_preflight.py::test_pm_preflight_writes_calendar_json_using_extracted_peers.
