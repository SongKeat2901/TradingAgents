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

def test_hvn_lvn_extraction():
    from tradingagents.agents.utils.volume_profile import Bin, high_volume_nodes, low_volume_nodes
    bins = [
        Bin(9, 10, 9.5, 100), Bin(10, 11, 10.5, 900), Bin(11, 12, 11.5, 100),
        Bin(12, 13, 12.5, 50),  Bin(13, 14, 13.5, 80),  Bin(14, 15, 14.5, 800),
        Bin(15, 16, 15.5, 90),
    ]
    hvn = high_volume_nodes(bins, max_nodes=3)
    lvn = low_volume_nodes(bins, max_nodes=3)
    assert 10.5 in [round(p, 1) for p in hvn]
    assert 14.5 in [round(p, 1) for p in hvn]
    assert any(12.0 <= p <= 13.0 for p in lvn)

def test_compute_volume_profile_dual_window():
    from tradingagents.agents.utils.volume_profile import compute_volume_profile
    ohlcv = "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n" + "\n".join(
        f"2026-{(i//28)+1:02d}-{(i%28)+1:02d},10,12,10,11,{1000+i}" for i in range(60)
    )
    vp = compute_volume_profile(ohlcv, n_bins=20)
    for win in ("structural_36mo", "tactical_6mo"):
        assert win in vp
        assert vp[win]["poc"] is not None
        assert vp[win]["vah"] >= vp[win]["val"]
        assert isinstance(vp[win]["hvn"], list)
        assert isinstance(vp[win]["lvn"], list)
    assert vp["n_bins"] == 20
