# tests/test_recent_closes.py
import pytest
from tradingagents.agents.utils.recent_closes import (
    compute_recent_closes,
    format_recent_closes_block,
)

pytestmark = pytest.mark.unit

# Mirrors raw/prices.json ohlcv exactly: #-comment + header + date-ascending rows,
# Close at column index 4 (Date,Open,High,Low,Close,Volume,...). Note 2026-06-26
# close is 359.90 and 2026-06-29 close is 368.57 — the real MSFT values the LLM
# confused (it hallucinated "Jun 29 close $359.90").
_OHLCV = (
    "# Stock data for MSFT\n"
    "# Total records: 12\n\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-06-15,300.0,305.0,299.0,304.10,10000000,0.0,0.0\n"
    "2026-06-16,304.0,309.0,303.0,308.20,10000000,0.0,0.0\n"
    "2026-06-17,308.0,312.0,307.0,311.30,10000000,0.0,0.0\n"
    "2026-06-18,311.0,315.0,310.0,314.40,10000000,0.0,0.0\n"
    "2026-06-19,314.0,318.0,313.0,317.50,10000000,0.0,0.0\n"
    "2026-06-22,317.0,321.0,316.0,320.60,10000000,0.0,0.0\n"
    "2026-06-23,320.0,324.0,319.0,323.70,10000000,0.0,0.0\n"
    "2026-06-24,323.0,327.0,322.0,326.80,10000000,0.0,0.0\n"
    "2026-06-25,326.0,331.0,325.0,328.77,10000000,0.0,0.0\n"
    "2026-06-26,328.0,362.0,327.0,359.90,10000000,0.0,0.0\n"
    "2026-06-29,360.0,370.0,359.0,368.57,10000000,0.0,0.0\n"
    "2026-06-30,368.0,375.0,367.0,373.02,10000000,0.0,0.0\n"
)
_PRICES = {"ohlcv": _OHLCV}


def test_returns_last_10_most_recent_first():
    rc = compute_recent_closes(_PRICES, "2026-06-30", n=10)
    assert rc["unavailable"] is False
    assert len(rc["rows"]) == 10
    assert rc["rows"][0]["date"] == "2026-06-30"      # most-recent-first
    assert rc["rows"][0]["close"] == 373.02
    assert rc["rows"][1]["date"] == "2026-06-29"
    assert rc["rows"][1]["close"] == 368.57
    dates = [r["date"] for r in rc["rows"]]
    assert "2026-06-15" not in dates and "2026-06-16" not in dates  # oldest 2 dropped by n=10


def test_on_or_before_trade_date_boundary():
    rc = compute_recent_closes(_PRICES, "2026-06-25", n=10)
    dates = [r["date"] for r in rc["rows"]]
    assert "2026-06-26" not in dates and "2026-06-29" not in dates and "2026-06-30" not in dates
    assert rc["rows"][0]["date"] == "2026-06-25"


def test_fewer_than_n_rows():
    rc = compute_recent_closes(_PRICES, "2026-06-17", n=10)
    assert len(rc["rows"]) == 3
    assert rc["rows"][0]["date"] == "2026-06-17"


def test_empty_prices_unavailable():
    rc = compute_recent_closes({"ohlcv": ""}, "2026-06-30")
    assert rc["unavailable"] is True and rc["rows"] == []


def test_block_renders_table_and_mandate():
    block = format_recent_closes_block(compute_recent_closes(_PRICES, "2026-06-30"))
    assert "## Recent closes" in block
    assert "| 2026-06-29 | $368.57 |" in block
    assert "verbatim" in block and "validator" in block


def test_block_unavailable_note():
    block = format_recent_closes_block({"unavailable": True, "reason": "x", "rows": []})
    assert "## Recent closes — unavailable" in block
    assert "Do not cite" in block


def test_source_alignment_with_validator_parse():
    """Lock the block's Close to the validator's own parse (ohlcv col 4)."""
    rc = compute_recent_closes(_PRICES, "2026-06-30", n=10)
    block_by_date = {r["date"]: f"{r['close']:.2f}" for r in rc["rows"]}
    validator_by_date = {}
    for line in _OHLCV.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) >= 5:
            validator_by_date[parts[0].strip()] = f"{float(parts[4]):.2f}"
    for d, c in block_by_date.items():
        assert validator_by_date[d] == c
