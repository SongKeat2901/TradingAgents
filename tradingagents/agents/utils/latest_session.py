"""Deterministic latest-session block (Phase-6.9).

The 2026-05-08 COIN run surfaced a forward-projection failure mode that
sat outside Phase 6.4/6.5 cell coverage. The Market Analyst LLM emitted a
"Revision 1" entry in technicals_v2.md claiming::

    "COIN closed the session at $206.50 on 14.39M shares — roughly
    1.8–2x the trailing daily average — after the 10-Q filing on May 7."

But yfinance had only indexed through 2026-05-07's close ($192.96 on
8.64M shares) at run time — the 2026-05-08 session hadn't completed in
NYC yet. The fabricated `$206.50 close` then propagated into decision.md
and decision_executive.md, anchoring the entire trading plan (no-new-
longs hard rule at $206.50, R/R 0.8:1 math, "trim into intraday strength
toward $210-$215") to a price that didn't exist.

This module computes the latest CLOSED session from raw/prices.json and
appends a deterministic block to pm_brief.md with explicit instruction:
DO NOT cite a "trade-date close" for any date later than the latest
indexed session. Mirrors the Phase 6.4 peer-ratios + Phase 6.5 net-debt
pattern — compute authoritatively in Python, render verbatim, instruct
the LLM to anchor on the cells.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_ohlcv(ohlcv_csv: str) -> list[dict[str, Any]]:
    """Parse the OHLCV CSV that the Researcher writes to raw/prices.json.

    Format (one row per session, date-ascending):
      Date,Open,High,Low,Close,Volume,Dividends,Stock Splits

    Returns a list of {date, open, high, low, close, volume} dicts. Skips
    header / comment / malformed lines silently.
    """
    rows: list[dict[str, Any]] = []
    for line in ohlcv_csv.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "date": parts[0].strip(),
                "open": float(parts[1]),
                "high": float(parts[2]),
                "low": float(parts[3]),
                "close": float(parts[4]),
                "volume": float(parts[5]),
            })
        except ValueError:
            continue
    return rows


def compute_latest_session(prices_data: dict[str, Any], trade_date: str) -> dict[str, Any]:
    """Identify the most-recent completed session in raw/prices.json and
    flag whether the requested trade_date is later (i.e., that session
    hasn't closed yet in yfinance).

    Returns:
        Empty/unavailable case::
            {"unavailable": True, "reason": str}

        Populated case::
            {
              "trade_date_requested": "YYYY-MM-DD",
              "latest_session_date": "YYYY-MM-DD",
              "open" / "high" / "low" / "close" / "volume": float,
              "gap_calendar_days": int | None,
              "trade_date_has_closed": bool,  # True iff gap <= 0
              "unavailable": False,
            }
    """
    ohlcv = ""
    if isinstance(prices_data, dict):
        ohlcv = prices_data.get("ohlcv", "") or ""

    rows = _parse_ohlcv(ohlcv)
    if not rows:
        return {
            "unavailable": True,
            "reason": "raw/prices.json has no parseable OHLCV rows",
        }

    latest = rows[-1]  # date-ascending; last row is most recent

    gap: int | None = None
    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d").date()
        ld = datetime.strptime(latest["date"], "%Y-%m-%d").date()
        gap = (td - ld).days
    except (ValueError, TypeError):
        gap = None

    return {
        "trade_date_requested": trade_date,
        "latest_session_date": latest["date"],
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "close": latest["close"],
        "volume": latest["volume"],
        "gap_calendar_days": gap,
        "trade_date_has_closed": (gap is not None and gap <= 0),
        "unavailable": False,
    }


def format_latest_session_block(session: dict[str, Any]) -> str:
    """Render the latest-session cells as a Markdown block for pm_brief.md.

    Returns "" when session is unavailable; the caller decides whether to
    surface an explicit warning instead.
    """
    if session.get("unavailable"):
        return ""

    td = session["trade_date_requested"]
    ld = session["latest_session_date"]
    gap = session.get("gap_calendar_days")
    has_closed = session.get("trade_date_has_closed")

    closed_label = "YES" if has_closed else "NO"

    # When the trade_date is later than the latest indexed session, prepend
    # a loud note. This is the high-leverage case (the COIN 2026-05-08
    # failure mode) — the LLM must not invent a "trade-date close" for an
    # in-progress or future session.
    gap_note = ""
    if not has_closed and gap is not None:
        gap_note = (
            f"\n\n**Note: trade_date {td} is after the latest available "
            f"session ({ld}, gap of {gap} calendar days). yfinance has "
            f"NOT yet indexed that session's close.**\n"
        )

    return (
        f"\n\n## Latest available session (from raw/prices.json)\n\n"
        f"| Cell | Value |\n"
        f"|---|---|\n"
        f"| Latest session date | {ld} |\n"
        f"| Latest session OHLC | open=${session['open']:.2f}, "
        f"high=${session['high']:.2f}, low=${session['low']:.2f}, "
        f"**close=${session['close']:.2f}** |\n"
        f"| Latest session volume | {int(session['volume']):,} |\n"
        f"| Trade date requested | {td} |\n"
        f"| Gap (calendar days) | {gap if gap is not None else '?'} |\n"
        f"| Trade-date session has closed in yfinance? | {closed_label} |\n"
        f"{gap_note}"
        f"\n*Authoritative spot for this report: **${session['close']:.2f}** "
        f"(latest yfinance close, {ld}). DO NOT cite a \"trade-date close\" "
        f"for any date later than {ld}. DO NOT cite intraday volumes, opens, "
        f"or closes for sessions that have not completed in yfinance. The "
        f"Market Analyst on the prior 2026-05-08 COIN run hallucinated "
        f"`$206.50 on 14.39M shares` as the May 8 close when actual most-"
        f"recent close was $192.96 on 8.64M shares — exactly the forward-"
        f"projection failure mode this block is designed to catch. If the "
        f"report needs to discuss post-event reaction, frame it as \"options-"
        f"implied move\" or \"extended-hours indication\", never as a "
        f"realized close.*\n"
    )
