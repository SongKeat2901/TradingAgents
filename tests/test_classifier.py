"""Tests for the deterministic technical classifier (Phase-6 stochasticity mitigation)."""
import pytest

pytestmark = pytest.mark.unit


def _ohlcv(rows):
    """Build a get_stock_data-style CSV string from (date, open, high, low, close, volume) rows."""
    header = "# Stock data\nDate,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    body = "\n".join(
        f"{d},{o},{h},{l},{c},{v},0.0,0.0" for (d, o, h, l, c, v) in rows
    )
    return header + body + "\n"


def _ref(reference_price, spot_50dma, spot_200dma, ytd_high, ytd_low, atr_14, trade_date="2026-05-01"):
    return {
        "ticker": "MSFT",
        "trade_date": trade_date,
        "reference_price": reference_price,
        "spot_50dma": spot_50dma,
        "spot_200dma": spot_200dma,
        "ytd_high": ytd_high,
        "ytd_low": ytd_low,
        "atr_14": atr_14,
    }


def _flat_history(spot, days=100, vol=20_000_000):
    """100 days of low-vol flat trading at `spot`. Used as a base for tests that
    want one specific day to stand out."""
    from datetime import date, timedelta
    end = date(2026, 5, 1)
    rows = []
    for i in range(days):
        d = (end - timedelta(days=days - 1 - i)).isoformat()
        rows.append((d, spot, spot + 0.5, spot - 0.5, spot, vol))
    return rows


def test_capitulation_top_decile_volume_with_large_move(tmp_path):
    """CAPITULATION: top-decile volume, >1.5σ move, bear MA alignment."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=25_000_000)
    rows[-1] = ("2026-05-01", 410.0, 411.0, 385.0, 385.5, 100_000_000)
    ref = _ref(
        reference_price=385.5,
        spot_50dma=405.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CAPITULATION"
    assert out["recent_volume_signal"] == "capitulation"
    assert out["upside_target"] == 460.0
    assert out["downside_target"] == 380.0
    assert "rationale" in out
    assert "100" in out["rationale"] or "top decile" in out["rationale"].lower()


def test_breakdown_below_50dma_with_volume_spike():
    """BREAKDOWN: spot < 50-DMA, 50-DMA < 200-DMA, gap > 8%, vol > 1.5× 50d avg."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    rows[-1] = ("2026-05-01", 410.0, 411.0, 405.0, 408.0, 35_000_000)
    ref = _ref(
        reference_price=408.0,
        spot_50dma=420.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "BREAKDOWN"
    assert out["upside_target"] == 420.0


def test_downtrend_catchall_when_bear_alignment_no_other_trigger():
    """DOWNTREND: spot < 200-DMA + 50<200, no capitulation/breakdown trigger."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=410.0,
        spot_50dma=400.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "DOWNTREND"
    assert out["upside_target"] == 460.0
    assert out["downside_target"] == 380.0


def test_consolidation_when_near_both_mas_and_tight_range():
    """CONSOLIDATION: |gap_to_50dma| < 3%, |gap_to_200dma| < 8%, range < 1.5× ATR."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=460.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=460.0,
        spot_50dma=458.0,
        spot_200dma=465.0,
        ytd_high=475.0,
        ytd_low=445.0,
        atr_14=4.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CONSOLIDATION"


def test_uptrend_when_bull_alignment_above_200dma():
    """UPTREND: spot > 200-DMA, 50-DMA > 200-DMA."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=480.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=480.0,
        spot_50dma=470.0,
        spot_200dma=440.0,
        ytd_high=485.0,
        ytd_low=400.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "UPTREND"


def test_breakout_recent_50over200_cross_with_volume():
    """BREAKOUT: recent 50/200 golden cross + 5d vol > 90d median."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=470.0, days=100, vol=20_000_000)
    for i in range(95, 100):
        d, o, h, l, c, _ = rows[i]
        rows[i] = (d, o, h, l, c, 30_000_000)
    ref = _ref(
        reference_price=470.0,
        spot_50dma=465.0,
        spot_200dma=464.0,
        ytd_high=475.0,
        ytd_low=420.0,
        atr_14=6.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "BREAKOUT"


def test_indeterminate_when_reference_has_nulls():
    from tradingagents.agents.utils.classifier import compute_classification

    ref = _ref(
        reference_price=None,
        spot_50dma=400.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(_flat_history(410.0)))
    assert out["setup_class"] == "INDETERMINATE"


def test_first_match_wins_capitulation_over_breakdown():
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    rows[-1] = ("2026-05-01", 410.0, 412.0, 375.0, 377.0, 80_000_000)
    ref = _ref(
        reference_price=377.0,
        spot_50dma=420.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=370.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CAPITULATION"


def test_asymmetry_math_basic():
    from tradingagents.agents.utils.classifier import compute_classification

    ref = _ref(
        reference_price=400.0,
        spot_50dma=395.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(_flat_history(400.0)))
    assert out["setup_class"] == "DOWNTREND"
    assert abs(out["upside_pct"] - 15.0) < 0.01
    assert abs(out["downside_pct"] - (-5.0)) < 0.01
    assert abs(out["reward_risk_ratio"] - 3.0) < 0.01
