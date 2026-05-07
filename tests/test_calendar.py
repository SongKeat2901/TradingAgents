"""Tests for the deterministic earnings calendar (Phase-6.2 temporal-anchor)."""
from datetime import date as _date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _earnings_df(rows):
    """Build a yfinance.Ticker.earnings_dates-style DataFrame.

    rows: list of (date_str, reported_eps_or_None) tuples. None → NaN
    (i.e., not yet reported).
    """
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {"Reported EPS": [r[1] for r in rows]},
        index=idx,
    )


def test_compute_calendar_splits_past_and_future():
    """Past = Reported EPS not NaN AND date < trade_date.
    Future = date > trade_date.
    """
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2025-07-30", 2.95),
        ("2025-10-29", 3.10),
        ("2026-01-29", 3.22),
        ("2026-04-29", 3.45),
        ("2026-07-25", None),
        ("2026-10-28", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["trade_date"] == "2026-05-01"
    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["MSFT"]["next_expected"] == "2026-07-25"
    assert out["MSFT"]["source"] == "yfinance"
    assert "_unavailable" in out
    assert out["_unavailable"] == []


def test_compute_calendar_handles_empty_yfinance_return():
    from tradingagents.agents.utils.calendar import compute_calendar

    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = pd.DataFrame()

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["TICKERX"])

    assert out["TICKERX"]["unavailable"] is True
    assert "yfinance" in out["TICKERX"]["reason"].lower()
    assert "TICKERX" in out["_unavailable"]


def test_compute_calendar_handles_yfinance_exception():
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings_msft = _earnings_df([
        ("2026-04-29", 3.45),
        ("2026-07-25", None),
    ])

    def _ticker_factory(symbol):
        if symbol == "BAD":
            raise RuntimeError("yfinance internal error")
        m = MagicMock()
        m.earnings_dates = earnings_msft
        return m

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", side_effect=_ticker_factory):
        out = compute_calendar("2026-05-01", ["MSFT", "BAD"])

    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["BAD"]["unavailable"] is True
    assert "BAD" in out["_unavailable"]
    assert "MSFT" not in out["_unavailable"]


def test_fiscal_period_derivation_msft_non_calendar_fy():
    """MSFT FY runs Jul-Jun. 2026-04-29 reports the quarter ending 2026-03-31,
    which is FY26 Q3 (Jan-Mar 2026 = 9 months into FY26 that started Jul-2025)."""
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-29", 3.45),
        ("2026-07-25", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["MSFT"]["fiscal_period"] == "FY26 Q3"


def test_fiscal_period_derivation_calendar_year_company():
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-22", 1.85),
        ("2026-07-23", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["GOOGL"])

    assert out["GOOGL"]["fiscal_period"] == "Q1 2026"


def test_etf_returns_structural_unavailable():
    """Phase 6.8: ETFs / mutual funds / indices don't have earnings, so the
    calendar must short-circuit with a structural N/A label rather than
    fetching earnings_dates and stamping `(yfinance unavailable)`. Surfaced
    by the COIN 2026-05-06 run where IBIT (iShares Bitcoin Trust) appeared
    in the calendar table as `IBIT (yfinance unavailable) unknown
    (yfinance unavailable)` — misleading because it wasn't a transient data
    failure, it was a structural truth (ETFs don't report earnings)."""
    from tradingagents.agents.utils.calendar import compute_calendar

    fake_etf = MagicMock()
    fake_etf.info = {"quoteType": "ETF", "shortName": "iShares Bitcoin Trust"}
    # earnings_dates should not be fetched on an ETF — set a tripwire.
    type(fake_etf).earnings_dates = property(
        lambda self: (_ for _ in ()).throw(
            AssertionError("earnings_dates should not be queried for ETFs")
        )
    )

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_etf):
        out = compute_calendar("2026-05-06", ["IBIT"])

    assert out["IBIT"]["unavailable"] is True
    assert out["IBIT"].get("structural") is True
    assert out["IBIT"].get("instrument_type") == "ETF"
    assert "no earnings reporting" in out["IBIT"]["reason"].lower()
    assert "IBIT" in out["_unavailable"]


def test_mutualfund_and_index_treated_as_structural_unavailable():
    """Same path for mutual funds and indices."""
    from tradingagents.agents.utils.calendar import compute_calendar

    def _ticker_factory(symbol):
        m = MagicMock()
        if symbol == "FXAIX":
            m.info = {"quoteType": "MUTUALFUND"}
        elif symbol == "^GSPC":
            m.info = {"quoteType": "INDEX"}
        return m

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", side_effect=_ticker_factory):
        out = compute_calendar("2026-05-06", ["FXAIX", "^GSPC"])

    assert out["FXAIX"]["structural"] is True
    assert out["FXAIX"]["instrument_type"] == "MUTUALFUND"
    assert out["^GSPC"]["structural"] is True
    assert out["^GSPC"]["instrument_type"] == "INDEX"


def test_no_past_earnings_returns_unavailable():
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-07-25", None),
        ("2026-10-28", None),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["NEWCO"])

    assert out["NEWCO"]["unavailable"] is True
    assert "NEWCO" in out["_unavailable"]


def test_no_future_earnings_still_returns_last_reported():
    from tradingagents.agents.utils.calendar import compute_calendar

    earnings = _earnings_df([
        ("2026-04-29", 3.45),
    ])
    fake_ticker = MagicMock()
    fake_ticker.earnings_dates = earnings

    with patch("tradingagents.agents.utils.calendar.yf.Ticker", return_value=fake_ticker):
        out = compute_calendar("2026-05-01", ["MSFT"])

    assert out["MSFT"]["last_reported"] == "2026-04-29"
    assert out["MSFT"]["next_expected"] is None
    assert "MSFT" not in out["_unavailable"]
