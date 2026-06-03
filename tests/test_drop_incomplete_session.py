"""Tests for drop_incomplete_session — the intraday-bar guard.

Regression cover for the 2026-06-02 cadence bug: runs that fetched price
data while the US session was still open captured yfinance's in-progress
intraday bar as the trade_date 'close', corrupting the reference price
(AAPL reported $308.85 @10am ET vs the $315.20 settlement close).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingagents.dataflows.stockstats_utils import drop_incomplete_session

_NY = ZoneInfo("America/New_York")


def _df_index(dates, closes):
    return pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))


def _df_col(dates, closes):
    return pd.DataFrame({"Date": pd.to_datetime(dates), "Close": closes})


pytestmark = pytest.mark.unit


def test_drops_today_bar_when_session_open_index():
    df = _df_index(["2026-06-01", "2026-06-02"], [306.31, 308.85])
    now = datetime(2026, 6, 2, 10, 9, tzinfo=_NY)  # 10:09am ET, market open
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 1
    assert out.index[-1].date().isoformat() == "2026-06-01"


def test_keeps_today_bar_after_close_index():
    df = _df_index(["2026-06-01", "2026-06-02"], [306.31, 315.20])
    now = datetime(2026, 6, 2, 16, 30, tzinfo=_NY)  # after 4pm ET close
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 2
    assert float(out["Close"].iloc[-1]) == 315.20


def test_keeps_prior_session_bar_next_morning():
    # Running pre-open the next day: last bar is yesterday's settled close.
    df = _df_index(["2026-06-01", "2026-06-02"], [306.31, 315.20])
    now = datetime(2026, 6, 3, 8, 0, tzinfo=_NY)
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 2


def test_historical_fetch_unaffected():
    # trade_date far in the past — last bar is long settled, always kept.
    df = _df_index(["2026-05-28", "2026-05-29"], [420.0, 450.24])
    now = datetime(2026, 6, 2, 10, 0, tzinfo=_NY)
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 2


def test_supports_date_column_form():
    df = _df_col(["2026-06-01", "2026-06-02"], [306.31, 308.85])
    now = datetime(2026, 6, 2, 11, 0, tzinfo=_NY)
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 1
    assert out["Date"].iloc[-1].date().isoformat() == "2026-06-01"


def test_close_boundary_exactly_4pm_keeps_bar():
    df = _df_index(["2026-06-01", "2026-06-02"], [306.31, 315.20])
    now = datetime(2026, 6, 2, 16, 0, tzinfo=_NY)  # exactly 16:00 ET
    out = drop_incomplete_session(df, now=now)
    assert len(out) == 2


def test_empty_df_noop():
    df = _df_index([], [])
    assert len(drop_incomplete_session(df, now=datetime(2026, 6, 2, 10, 0, tzinfo=_NY))) == 0
