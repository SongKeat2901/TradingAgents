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
