import pytest
pytestmark = pytest.mark.unit

_OHLCV = (
    "# Stock data for X\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-05-20,10,12,9,11,1000,0.0,0.0\n"
    "2026-05-21,11,13,10,12,2000,0.0,0.0\n"
)

def test_parse_ohlcv_rows():
    from tradingagents.agents.utils.volume_profile import parse_ohlcv
    rows = parse_ohlcv(_OHLCV)
    assert len(rows) == 2
    assert rows[0] == ("2026-05-20", 10.0, 12.0, 9.0, 11.0, 1000.0)
    assert rows[1][2] == 13.0

def test_select_window_takes_trailing_rows():
    from tradingagents.agents.utils.volume_profile import parse_ohlcv, select_window
    rows = parse_ohlcv(_OHLCV)
    assert select_window(rows, months=36) == rows
    assert select_window(rows, months=0) == []

def test_histogram_poc_and_value_area():
    from tradingagents.agents.utils.volume_profile import (
        parse_ohlcv, build_histogram, point_of_control, value_area,
    )
    ohlcv = (
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2026-05-20,10,12,10,11,9000,0.0,0.0\n"
        "2026-05-21,18,20,18,19,1000,0.0,0.0\n"
    )
    rows = parse_ohlcv(ohlcv)
    bins = build_histogram(rows, n_bins=20)
    poc = point_of_control(bins)
    assert 10.0 <= poc <= 12.0
    val, vah = value_area(bins, pct=0.70)
    assert val <= poc <= vah
    assert vah < 18.0
