import json
import pytest
import tradingagents.agents.researcher as R

pytestmark = pytest.mark.unit

_OHLCV = ("Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
          "2026-06-29,360.0,370.0,359.0,368.57,1,0.0,0.0\n"
          "2026-06-30,368.0,375.0,367.0,373.02,1,0.0,0.0\n")


def _seed(raw):
    (raw / "financials.json").write_text(json.dumps({"ticker": "MSFT", "trade_date": "2026-06-30"}), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps({"ohlcv": _OHLCV}), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps({"transactions": []}), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps({"AAPL": {}, "GOOGL": {}}), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps({"ticker": "MSFT", "trade_date": "2026-06-30", "reference_price": 373.02}), encoding="utf-8")


def test_reuse_skips_reproducible_fetches_but_refetches_news_social(tmp_path, monkeypatch):
    raw = tmp_path
    _seed(raw)
    counts = {k: 0 for k in ("fin", "news", "insider", "social", "prices", "ind")}
    monkeypatch.setattr(R, "_fetch_financials", lambda t, d: counts.__setitem__("fin", counts["fin"] + 1) or {"ticker": t, "trade_date": d})
    monkeypatch.setattr(R, "_fetch_news", lambda t, d: counts.__setitem__("news", counts["news"] + 1) or {"n": 1})
    monkeypatch.setattr(R, "_fetch_insider", lambda t, d: counts.__setitem__("insider", counts["insider"] + 1) or {"transactions": []})
    monkeypatch.setattr(R, "_fetch_social", lambda t, d: counts.__setitem__("social", counts["social"] + 1) or {"s": 1})
    monkeypatch.setattr(R, "_fetch_prices", lambda t, d: counts.__setitem__("prices", counts["prices"] + 1) or {"ohlcv": _OHLCV})
    monkeypatch.setattr(R, "_fetch_indicators", lambda t, d: counts.__setitem__("ind", counts["ind"] + 1) or {})

    bundle, reused = R._gather_raw("MSFT", "2026-06-30", ["AAPL", "GOOGL"], raw, reuse=True)

    # reproducible fetches skipped
    assert counts["fin"] == 0 and counts["prices"] == 0 and counts["insider"] == 0 and counts["ind"] == 0
    # peers reused (no per-peer financials fetch beyond the 0 above)
    assert reused["financials"] and reused["prices"] and reused["insider"] and reused["reference"] and reused["peers"]
    # news/social always fresh
    assert counts["news"] == 1 and counts["social"] == 1
    assert bundle["reference"]["reference_price"] == 373.02


def test_reuse_off_fetches_everything(tmp_path, monkeypatch):
    raw = tmp_path
    _seed(raw)  # files present, but reuse off => ignored
    counts = {k: 0 for k in ("fin", "prices", "ind", "insider", "news", "social")}
    monkeypatch.setattr(R, "_fetch_financials", lambda t, d: counts.__setitem__("fin", counts["fin"] + 1) or {"ticker": t, "trade_date": d})
    monkeypatch.setattr(R, "_fetch_news", lambda t, d: counts.__setitem__("news", counts["news"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_insider", lambda t, d: counts.__setitem__("insider", counts["insider"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_social", lambda t, d: counts.__setitem__("social", counts["social"] + 1) or {})
    monkeypatch.setattr(R, "_fetch_prices", lambda t, d: counts.__setitem__("prices", counts["prices"] + 1) or {"ohlcv": _OHLCV})
    monkeypatch.setattr(R, "_fetch_indicators", lambda t, d: counts.__setitem__("ind", counts["ind"] + 1) or {})

    bundle, reused = R._gather_raw("MSFT", "2026-06-30", ["AAPL", "GOOGL"], raw, reuse=False)

    assert counts["fin"] >= 1 and counts["prices"] == 1 and counts["ind"] == 1
    assert not any(reused.values())
