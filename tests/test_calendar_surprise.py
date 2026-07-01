"""Tests for earnings-surprise history capture (Task 7, WP1b)."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from tradingagents.agents.utils.calendar import compute_calendar

pytestmark = pytest.mark.unit


def _earnings_df(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"EPS Estimate": [r[1] for r in rows],
         "Reported EPS": [r[2] for r in rows],
         "Surprise(%)": [r[3] for r in rows]},
        index=idx,
    )


def test_surprise_history_captured():
    df = _earnings_df([
        ("2025-07-30", 2.90, 2.95, 1.72),
        ("2025-10-29", 3.05, 3.10, 1.64),
        ("2026-01-29", 3.20, 3.22, 0.63),
        ("2026-04-29", 3.40, 3.45, 1.47),
        ("2026-07-25", 3.55, None, None),
    ])
    fake = MagicMock()
    fake.earnings_dates = df
    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake):
        out = compute_calendar("2026-05-01", ["MSFT"])
    sur = out["MSFT"]["surprises"]
    assert len(sur) == 4                       # only past, reported rows
    assert sur[0]["surprise_pct"] == 1.47      # most recent first
    assert sur[0]["reported"] == 3.45


def test_surprise_history_most_recent_first_full_order():
    df = _earnings_df([
        ("2025-07-30", 2.90, 2.95, 1.72),
        ("2025-10-29", 3.05, 3.10, 1.64),
        ("2026-01-29", 3.20, 3.22, 0.63),
        ("2026-04-29", 3.40, 3.45, 1.47),
    ])
    fake = MagicMock()
    fake.earnings_dates = df
    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake):
        out = compute_calendar("2026-05-01", ["MSFT"])
    sur = out["MSFT"]["surprises"]
    dates = [s["date"] for s in sur]
    assert dates == ["2026-04-29", "2026-01-29", "2025-10-29", "2025-07-30"]
    assert sur[-1]["estimate"] == 2.90


def test_surprise_history_capped_at_eight():
    rows = [
        (f"202{2 + i // 4}-{(i % 4) * 3 + 1:02d}-15", 1.0 + i * 0.01, 1.05 + i * 0.01, 5.0)
        for i in range(10)
    ]
    df = _earnings_df(rows)
    fake = MagicMock()
    fake.earnings_dates = df
    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake):
        out = compute_calendar("2026-05-01", ["MSFT"])
    sur = out["MSFT"]["surprises"]
    assert len(sur) <= 8


def test_surprise_history_missing_columns_does_not_crash():
    """Fixture without EPS Estimate / Surprise(%) columns (existing
    test_calendar.py shape) must not crash. The reported-EPS row still
    yields one surprise entry, but with estimate/surprise_pct as None
    rather than fabricated — row.get(...) tolerates the missing columns."""
    idx = pd.to_datetime(["2026-04-29"])
    df = pd.DataFrame({"Reported EPS": [3.45]}, index=idx)
    fake = MagicMock()
    fake.earnings_dates = df
    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake):
        out = compute_calendar("2026-05-01", ["MSFT"])
    sur = out["MSFT"]["surprises"]
    assert len(sur) == 1
    assert sur[0]["reported"] == 3.45
    assert sur[0]["estimate"] is None
    assert sur[0]["surprise_pct"] is None
