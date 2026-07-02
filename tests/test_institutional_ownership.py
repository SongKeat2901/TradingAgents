import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.institutional_ownership import (
    normalize_institutional_ownership, format_ownership_block,
)

_RAW = {
    "pct_institutions": 0.75727, "pct_insiders": 0.00078, "institutions_count": 8031,
    "holders": [
        {"holder": "Blackrock Inc.", "pct_held": 0.071, "value": 228004302539,
         "pct_change": -0.0142, "date": "2026-03-31"},
        {"holder": "Vanguard", "pct_held": 0.058, "value": 185437420699,
         "pct_change": 1.0, "date": "2026-03-31"},
    ],
}


def test_normalize_percents_and_holders():
    r = normalize_institutional_ownership(_RAW)
    assert r["pct_institutions"] == 75.73
    assert r["pct_insiders"] == 0.08
    assert r["institutions_count"] == 8031
    assert r["top_holders"][0]["holder"] == "Blackrock Inc."
    assert r["top_holders"][0]["pct_held"] == 7.1
    assert r["top_holders"][0]["pct_change"] == -1.42


def test_normalize_caps_top_n():
    raw = {"holders": [{"holder": f"H{i}", "pct_held": 0.01} for i in range(20)]}
    r = normalize_institutional_ownership(raw)
    assert len(r["top_holders"]) == 10


def test_missing_data_normalizes_to_none():
    r = normalize_institutional_ownership({})
    assert r["pct_institutions"] is None and r["top_holders"] == []


def test_block_renders_values_and_verbatim():
    block = format_ownership_block(normalize_institutional_ownership(_RAW))
    assert "## Institutional & insider ownership" in block
    assert "75.73%" in block and "Blackrock Inc." in block
    assert "$228.0B" in block and "verbatim" in block


def test_block_unavailable_when_empty():
    block = format_ownership_block(normalize_institutional_ownership({}))
    assert "n/a (data unavailable)" in block
    assert "Do not cite ownership figures" in block
