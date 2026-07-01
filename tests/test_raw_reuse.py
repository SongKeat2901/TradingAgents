import json
import pytest
from tradingagents.agents.utils.raw_reuse import reuse_or_fetch, reuse_or_fetch_peers

pytestmark = pytest.mark.unit


def _write(dirpath, name, obj):
    (dirpath / name).write_text(json.dumps(obj), encoding="utf-8")


def test_reuse_hit_skips_fetch(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "MSFT", "trade_date": "2026-06-30", "x": 1})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True,
                                  sanity=lambda d: d.get("ticker") == "MSFT" and d.get("trade_date") == "2026-06-30")
    assert reused is True and data["x"] == 1 and calls["n"] == 0


def test_reuse_miss_fetches(tmp_path):
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True)
    assert reused is False and data == {"fetched": True} and calls["n"] == 1


def test_reuse_sanity_fail_fetches(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "AAPL", "trade_date": "2026-06-30"})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True,
                                  sanity=lambda d: d.get("ticker") == "MSFT")
    assert reused is False and calls["n"] == 1  # wrong ticker -> fetch


def test_reuse_garbled_json_fetches(tmp_path):
    (tmp_path / "financials.json").write_text("{not json", encoding="utf-8")
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=True)
    assert reused is False and calls["n"] == 1  # no exception, fetched


def test_reuse_off_always_fetches(tmp_path):
    _write(tmp_path, "financials.json", {"ticker": "MSFT", "trade_date": "2026-06-30"})
    calls = {"n": 0}
    def fetch():
        calls["n"] += 1
        return {"fetched": True}
    data, reused = reuse_or_fetch(tmp_path, "financials.json", fetch, reuse=False)
    assert reused is False and calls["n"] == 1


def test_peers_reuse_hit(tmp_path):
    _write(tmp_path, "peers.json", {"AAPL": {}, "GOOGL": {}})
    calls = {"n": 0}
    def fetch_all():
        calls["n"] += 1
        return {"AAPL": {"fresh": 1}, "GOOGL": {"fresh": 1}}
    data, reused = reuse_or_fetch_peers(tmp_path, ["AAPL", "GOOGL"], fetch_all, reuse=True)
    assert reused is True and set(data.keys()) == {"AAPL", "GOOGL"} and calls["n"] == 0


def test_peers_keyset_mismatch_fetches(tmp_path):
    _write(tmp_path, "peers.json", {"AAPL": {}, "GOOGL": {}})
    calls = {"n": 0}
    def fetch_all():
        calls["n"] += 1
        return {"AAPL": {}, "META": {}}
    data, reused = reuse_or_fetch_peers(tmp_path, ["AAPL", "META"], fetch_all, reuse=True)
    assert reused is False and calls["n"] == 1  # peer set changed -> refetch
