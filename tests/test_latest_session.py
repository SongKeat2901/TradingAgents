"""Tests for the Phase-6.9 deterministic latest-session block."""
import pytest

pytestmark = pytest.mark.unit


_COIN_OHLCV = (
    "# Stock data for COIN\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-04-28,190.2,195.94,188.75,194.1,6316100,0.0,0.0\n"
    "2026-04-29,187.27,187.27,177.62,181.73,12609600,0.0,0.0\n"
    "2026-04-30,181.6,189.56,179.89,187.77,7786700,0.0,0.0\n"
    "2026-05-01,191.88,194.51,189.86,191.25,6771100,0.0,0.0\n"
    "2026-05-04,199.41,206.71,197.85,202.99,11243400,0.0,0.0\n"
    "2026-05-05,208.88,208.88,194.4,197.75,10074200,0.0,0.0\n"
    "2026-05-06,195.78,198.5,193.25,197.96,7764900,0.0,0.0\n"
    "2026-05-07,196.24,198.15,190.32,192.96,8641932,0.0,0.0\n"
)


def test_compute_latest_session_identifies_most_recent_row():
    """Happy path: the most-recent-date row in OHLCV is the latest session."""
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-08")

    assert out["unavailable"] is False
    assert out["latest_session_date"] == "2026-05-07"
    assert out["close"] == 192.96
    assert out["volume"] == 8_641_932
    assert out["open"] == 196.24
    assert out["high"] == 198.15
    assert out["low"] == 190.32


def test_compute_latest_session_flags_trade_date_after_latest():
    """Phase 6.9 load-bearing case: trade_date is later than latest indexed
    session → trade_date_has_closed = False, gap > 0. This is the COIN
    2026-05-08 scenario where the LLM must be forbidden from inventing a
    trade-date close."""
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-08")

    assert out["trade_date_requested"] == "2026-05-08"
    assert out["latest_session_date"] == "2026-05-07"
    assert out["gap_calendar_days"] == 1
    assert out["trade_date_has_closed"] is False


def test_compute_latest_session_when_trade_date_matches_latest():
    """When trade_date equals the latest session date, gap = 0 and
    trade_date_has_closed = True (the typical case for next-day analysis)."""
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-07")

    assert out["gap_calendar_days"] == 0
    assert out["trade_date_has_closed"] is True


def test_compute_latest_session_when_trade_date_is_earlier_than_latest():
    """Backtest / historical-replay case: trade_date earlier than latest
    indexed session. gap negative, has_closed True."""
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-01")

    assert out["gap_calendar_days"] == -6
    assert out["trade_date_has_closed"] is True


def test_compute_latest_session_handles_empty_ohlcv():
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": ""}, "2026-05-08")

    assert out["unavailable"] is True
    assert "no parseable" in out["reason"].lower()
    assert "ohlcv" in out["reason"].lower()


def test_compute_latest_session_handles_non_dict_input():
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session(None, "2026-05-08")  # type: ignore[arg-type]
    assert out["unavailable"] is True


def test_compute_latest_session_handles_malformed_trade_date():
    """Malformed trade_date string → gap is None but the cell extraction
    still works (graceful degradation; the cells are still useful)."""
    from tradingagents.agents.utils.latest_session import compute_latest_session

    out = compute_latest_session({"ohlcv": _COIN_OHLCV}, "not-a-date")

    assert out["unavailable"] is False
    assert out["close"] == 192.96
    assert out["gap_calendar_days"] is None


def test_format_latest_session_block_includes_authoritative_spot_anchor():
    """The block must include the authoritative-spot line + the COIN 2026-05-08
    incident reference, since this is the load-bearing instruction that
    forbids trade-date-close fabrication."""
    from tradingagents.agents.utils.latest_session import (
        compute_latest_session,
        format_latest_session_block,
    )

    session = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-08")
    block = format_latest_session_block(session)

    # Cells visible to the LLM
    assert "## Latest available session" in block
    assert "**close=$192.96**" in block
    assert "8,641,932" in block
    # Authoritative spot line
    assert "Authoritative spot for this report: **$192.96**" in block
    # Forbidden behaviour explicitly named
    assert "DO NOT cite a \"trade-date close\"" in block
    assert "$206.50" in block  # the COIN-2026-05-08 incident reference
    assert "14.39M" in block


def test_format_latest_session_block_loud_note_when_trade_date_after_latest():
    """When trade_date is after the latest session (the COIN scenario), the
    block must include a prominent note flagging the gap."""
    from tradingagents.agents.utils.latest_session import (
        compute_latest_session,
        format_latest_session_block,
    )

    session = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-08")
    block = format_latest_session_block(session)

    assert "**Note: trade_date 2026-05-08 is after the latest available session" in block
    assert "yfinance has NOT yet indexed" in block
    assert "Trade-date session has closed in yfinance? | NO" in block


def test_format_latest_session_block_no_loud_note_when_trade_date_matches():
    """When trade_date == latest session (the typical next-day case), the
    loud-warning note is omitted — the table still renders for context."""
    from tradingagents.agents.utils.latest_session import (
        compute_latest_session,
        format_latest_session_block,
    )

    session = compute_latest_session({"ohlcv": _COIN_OHLCV}, "2026-05-07")
    block = format_latest_session_block(session)

    assert "Trade-date session has closed in yfinance? | YES" in block
    # No "is after" warning
    assert "is after the latest available" not in block


def test_format_latest_session_block_returns_empty_for_unavailable():
    """When prices.json has no rows, the formatter returns "" so the caller
    can decide what to render in its place."""
    from tradingagents.agents.utils.latest_session import (
        compute_latest_session,
        format_latest_session_block,
    )

    session = compute_latest_session({"ohlcv": ""}, "2026-05-08")
    assert format_latest_session_block(session) == ""
