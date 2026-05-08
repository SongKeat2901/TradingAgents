"""Tests for Fix #13: auto-resolve trade_date to latest yfinance close."""
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _fake_history(latest_close_date: str, n_rows: int = 5):
    """Build a yfinance-style history DataFrame ending at `latest_close_date`."""
    dates = pd.date_range(end=latest_close_date, periods=n_rows, freq="B")
    return pd.DataFrame(
        {"Open": [100.0] * n_rows, "High": [105.0] * n_rows,
         "Low": [95.0] * n_rows, "Close": [102.0] * n_rows,
         "Volume": [1_000_000] * n_rows},
        index=dates,
    )


def test_no_adjustment_when_requested_date_at_or_before_latest():
    from cli.auto_resolve_date import auto_resolve_trade_date

    with patch("cli.auto_resolve_date._fetch_latest_close_date", return_value="2026-05-08"):
        # Requested date == latest close
        d, o, adj = auto_resolve_trade_date("MSFT", "2026-05-08", "/data/2026-05-08-MSFT")
        assert d == "2026-05-08"
        assert o == "/data/2026-05-08-MSFT"
        assert adj is False

        # Requested date BEFORE latest close
        d, o, adj = auto_resolve_trade_date("MSFT", "2026-05-05", "/data/2026-05-05-MSFT")
        assert d == "2026-05-05"
        assert adj is False


def test_adjusts_when_requested_date_after_latest_close():
    """The COIN/MSFT scenario: user passes 2026-05-08, yfinance only has
    through 2026-05-07. trade_date and output_dir both adjusted."""
    from cli.auto_resolve_date import auto_resolve_trade_date

    with patch("cli.auto_resolve_date._fetch_latest_close_date", return_value="2026-05-07"):
        d, o, adj = auto_resolve_trade_date(
            "MSFT", "2026-05-08",
            "/Users/trueknot/.openclaw/data/research/2026-05-08-MSFT",
        )
        assert d == "2026-05-07"
        assert o == "/Users/trueknot/.openclaw/data/research/2026-05-07-MSFT"
        assert adj is True


def test_no_adjustment_to_output_dir_when_basename_lacks_date():
    """If the output dir basename doesn't contain the date string, leave
    it alone (e.g., custom dir names like 'msft-test-run')."""
    from cli.auto_resolve_date import auto_resolve_trade_date

    with patch("cli.auto_resolve_date._fetch_latest_close_date", return_value="2026-05-07"):
        d, o, adj = auto_resolve_trade_date(
            "MSFT", "2026-05-08", "/data/custom-msft-run",
        )
        assert d == "2026-05-07"
        assert o == "/data/custom-msft-run"  # unchanged
        assert adj is True


def test_graceful_degradation_when_yfinance_unavailable():
    """yfinance error → return arguments unchanged. Pipeline can still
    proceed; Phase 7 validators will catch any drift."""
    from cli.auto_resolve_date import auto_resolve_trade_date

    with patch("cli.auto_resolve_date._fetch_latest_close_date", return_value=None):
        d, o, adj = auto_resolve_trade_date("MSFT", "2026-05-08", "/data/2026-05-08-MSFT")
        assert d == "2026-05-08"
        assert o == "/data/2026-05-08-MSFT"
        assert adj is False


def test_graceful_degradation_on_malformed_requested_date():
    from cli.auto_resolve_date import auto_resolve_trade_date

    with patch("cli.auto_resolve_date._fetch_latest_close_date", return_value="2026-05-07"):
        d, o, adj = auto_resolve_trade_date("MSFT", "not-a-date", "/data/dir")
        # Returns unchanged — caller will fail downstream with a clearer error
        assert d == "not-a-date"
        assert adj is False


def test_fetch_latest_close_uses_yfinance_history_correctly():
    """Integration: real call shape — yf.Ticker(symbol).history() last-row index."""
    from cli.auto_resolve_date import _fetch_latest_close_date

    fake_hist = _fake_history("2026-05-07")

    class _FakeTicker:
        def __init__(self, sym):
            self._sym = sym

        def history(self, **kwargs):
            return fake_hist

    with patch("yfinance.Ticker", _FakeTicker):
        result = _fetch_latest_close_date("MSFT")
        assert result == "2026-05-07"


def test_fetch_latest_close_returns_none_on_empty_history():
    from cli.auto_resolve_date import _fetch_latest_close_date

    class _FakeTicker:
        def __init__(self, sym):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    with patch("yfinance.Ticker", _FakeTicker):
        assert _fetch_latest_close_date("DELISTED") is None


def test_fetch_latest_close_returns_none_on_yfinance_exception():
    from cli.auto_resolve_date import _fetch_latest_close_date

    class _FakeTicker:
        def __init__(self, sym):
            raise RuntimeError("yfinance unreachable")

    with patch("yfinance.Ticker", _FakeTicker):
        assert _fetch_latest_close_date("MSFT") is None
