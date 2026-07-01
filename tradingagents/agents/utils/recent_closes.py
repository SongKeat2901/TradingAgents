# tradingagents/agents/utils/recent_closes.py
"""Deterministic recent-closes block (rerun-reduction Phase C).

Pins the last N trailing daily closes from raw/prices.json so the LLM quotes
specific-date closes verbatim instead of hallucinating them (e.g. the MSFT
2026-06-30 run's "Jun 29 close $359.90" vs the real $368.57, which the
phase_7_1_price_date validator blocks). Built from the SAME prices.json Close
column (index 4) the validator reads — via latest_session._parse_ohlcv — so the
two agree by construction and stay inside the validator's $0.50 tolerance.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tradingagents.agents.utils.latest_session import _parse_ohlcv


def compute_recent_closes(
    prices_data: dict[str, Any], trade_date: str, n: int = 10
) -> dict[str, Any]:
    ohlcv = ""
    if isinstance(prices_data, dict):
        ohlcv = prices_data.get("ohlcv", "") or ""
    rows = _parse_ohlcv(ohlcv)  # date-ascending; close = float(parts[4])
    if not rows:
        return {"unavailable": True, "reason": "raw/prices.json has no parseable OHLCV rows", "rows": []}

    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d").date()
        eligible = [r for r in rows if datetime.strptime(r["date"], "%Y-%m-%d").date() <= td]
    except (ValueError, TypeError):
        eligible = rows  # if trade_date unparseable, fall back to all rows

    if not eligible:
        return {"unavailable": True, "reason": f"no sessions on or before {trade_date}", "rows": []}

    recent = list(reversed(eligible[-n:]))  # last n, then most-recent-first
    return {
        "trade_date": trade_date,
        "as_of": recent[0]["date"],
        "rows": [{"date": r["date"], "close": r["close"]} for r in recent],
        "source": "raw/prices.json ohlcv (Close, col 4)",
        "unavailable": False,
    }


def format_recent_closes_block(rc: dict[str, Any]) -> str:
    if rc.get("unavailable") or not rc.get("rows"):
        reason = rc.get("reason", "no data")
        return (
            f"\n\n## Recent closes — unavailable ({reason})\n\n"
            "*Do not cite a closing price for any specific date; none are pinned here.*\n"
        )
    n = len(rc["rows"])
    body = "\n".join(f"| {r['date']} | ${r['close']:.2f} |" for r in rc["rows"])
    return (
        f"\n\n## Recent closes (last {n} sessions, verbatim from raw/prices.json)\n\n"
        f"| Date | Close |\n|---|---|\n{body}\n\n"
        "*Any closing price you cite for a specific date MUST be quoted verbatim "
        "from this table (source: raw/prices.json Close). Do not state a close for "
        "a date not listed here — if you need an older close, say 'not in the "
        "recent-closes table' rather than estimating. This is the same source the "
        "validator checks, so a paraphrased or rounded price will be flagged.*\n"
    )
