"""Deterministic earnings calendar (Phase-6.2 temporal-anchor mitigation).

Run #6 of the deterministic-classifier validation (2026-05-04, trade date
2026-05-01) showed the LLM writing "GOOGL/AMZN earnings (late April 2026,
data to follow)" for events that had already happened ~7 days earlier.
The cause: no explicit "today is 2026-05-01" anchor in the prompt.

This module pulls per-ticker earnings dates from yfinance and produces a
deterministic dict that the Researcher writes to raw/calendar.json. PM
Pre-flight Python-appends a formatted Markdown block to pm_brief.md
after its LLM call so the dates can never be paraphrased.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


# Companies with non-calendar fiscal years.
_NON_CALENDAR_FISCAL_YEARS: dict[str, dict[str, int]] = {
    "MSFT": {"fy_start_month": 7},
    "ORCL": {"fy_start_month": 6},
    "ADBE": {"fy_start_month": 12},
    "CRM":  {"fy_start_month": 2},
}


def _quarter_end_for_report_date(report_date: datetime) -> datetime:
    """Map a report date to the quarter-end it most likely covers.
    Earnings are typically reported within ~30 days of quarter-end."""
    year = report_date.year
    candidates = [
        datetime(year, 3, 31),
        datetime(year, 6, 30),
        datetime(year, 9, 30),
        datetime(year, 12, 31),
        datetime(year - 1, 12, 31),
    ]
    past = [c for c in candidates if c <= report_date]
    return max(past) if past else candidates[-1]


def _fiscal_period(ticker: str, report_date_str: str) -> str:
    """Return a "FY{YY} Q{n}" or "Q{n} {YYYY}" string for the report date."""
    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    quarter_end = _quarter_end_for_report_date(report_date)

    if ticker in _NON_CALENDAR_FISCAL_YEARS:
        fy_start_month = _NON_CALENDAR_FISCAL_YEARS[ticker]["fy_start_month"]
        # MSFT FY=Jul-Jun: Jul-Sep is FY Q1, Oct-Dec is Q2, Jan-Mar is Q3, Apr-Jun is Q4.
        offset = (quarter_end.month - fy_start_month) % 12
        q_index = offset // 3 + 1
        # Fiscal year = year in which the FY ENDS.
        # MSFT FY26 = Jul-2025 → Jun-2026. A quarter ending 2026-03-31 is FY26 Q3.
        # Logic: if quarter_end.month < fy_start_month, FY year = quarter_end.year.
        # Else FY year = quarter_end.year + 1.
        if quarter_end.month < fy_start_month:
            fy_year = quarter_end.year
        else:
            fy_year = quarter_end.year + 1
        return f"FY{fy_year % 100:02d} Q{q_index}"
    else:
        cal_q = (quarter_end.month - 1) // 3 + 1
        return f"Q{cal_q} {quarter_end.year}"


def _compute_one_ticker(symbol: str, trade_date: str) -> dict[str, Any]:
    """Pull earnings_dates for one ticker; split past/future."""
    try:
        ticker = yf.Ticker(symbol)
        # Phase 6.8: detect ETFs / mutual funds / indices upfront. They
        # don't report earnings, so attempting an earnings_dates call and
        # rendering "(yfinance unavailable)" misleads the reader into
        # thinking data was supposed to exist. Surface the structural
        # truth instead. yfinance .info is a dict; quoteType is the
        # canonical field. Some ETFs return None / missing — fall through
        # to the earnings path in that case (rare; safer to try).
        info = getattr(ticker, "info", None) or {}
        quote_type = (info.get("quoteType") or "").upper()
        if quote_type in ("ETF", "MUTUALFUND", "INDEX"):
            return {
                "unavailable": True,
                "structural": True,
                "instrument_type": quote_type,
                "reason": f"{quote_type.lower()} — no earnings reporting",
            }
        earnings = ticker.earnings_dates
    except Exception as exc:
        return {"unavailable": True, "reason": f"yfinance error: {exc}"}

    if earnings is None or earnings.empty:
        return {"unavailable": True, "reason": "yfinance returned no earnings dates"}

    trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")

    rows = earnings.reset_index()
    if "Earnings Date" in rows.columns:
        rows = rows.rename(columns={"Earnings Date": "earnings_date"})
    elif "index" in rows.columns:
        rows = rows.rename(columns={"index": "earnings_date"})

    past_rows = []
    future_rows = []
    surprises = []
    for _, row in rows.iterrows():
        d = row["earnings_date"]
        if hasattr(d, "to_pydatetime"):
            d = d.to_pydatetime()
        d_naive = d.replace(tzinfo=None) if d.tzinfo else d
        date_str = d_naive.strftime("%Y-%m-%d")
        reported = row.get("Reported EPS")
        estimate = row.get("EPS Estimate")
        surprise_pct = row.get("Surprise(%)")
        has_estimate = estimate is not None and not pd.isna(estimate)
        has_surprise_pct = surprise_pct is not None and not pd.isna(surprise_pct)
        if reported is not None and not pd.isna(reported) and d_naive < trade_dt:
            past_rows.append((date_str, d_naive))
            # Only capture a surprise entry when there's real surprise
            # signal (an estimate or a surprise %) — a Reported-EPS-only
            # row with no estimate/surprise columns/values carries no
            # surprise information and would just be reported-only noise.
            if has_estimate or has_surprise_pct:
                surprises.append({
                    "date": date_str,
                    "reported": float(reported),
                    "estimate": float(estimate) if has_estimate else None,
                    "surprise_pct": float(surprise_pct) if has_surprise_pct else None,
                })
        elif d_naive > trade_dt:
            future_rows.append((date_str, d_naive))

    if not past_rows:
        return {"unavailable": True, "reason": "no past earnings before trade_date"}

    last_reported, _ = max(past_rows, key=lambda r: r[1])
    next_expected = None
    if future_rows:
        next_expected, _ = min(future_rows, key=lambda r: r[1])

    surprises.sort(key=lambda s: s["date"], reverse=True)
    surprises = surprises[:8]

    return {
        "last_reported": last_reported,
        "fiscal_period": _fiscal_period(symbol, last_reported),
        "next_expected": next_expected,
        "source": "yfinance",
        "surprises": surprises,
    }


def compute_calendar(trade_date: str, tickers: list[str]) -> dict[str, Any]:
    """Build the per-ticker earnings calendar.

    trade_date: "YYYY-MM-DD" string. The "today" anchor for past/future split.
    tickers: list of ticker symbols (typically [main_ticker] + peers).

    Returns:
        {
          "trade_date": str,
          "<TICKER>": {
            "last_reported": "YYYY-MM-DD",
            "fiscal_period": "FY26 Q3" or "Q1 2026",
            "next_expected": "YYYY-MM-DD" or None,
            "source": "yfinance",
            "surprises": [
              {"date": "YYYY-MM-DD", "reported": float,
               "estimate": float or None, "surprise_pct": float or None},
              ...  # most-recent-first, at most the last 8 PAST rows with a
                   # non-NaN Reported EPS AND real surprise signal (a
                   # non-NaN EPS Estimate and/or Surprise(%)). A row whose
                   # DataFrame lacks those columns, or has them but NaN,
                   # is skipped entirely — it carries no surprise
                   # information, so it's excluded rather than emitted as
                   # a reported-only, estimate=None/surprise_pct=None
                   # noise entry. When emitted, estimate/surprise_pct are
                   # each independently float or None (never fabricated).
            ],
          } OR {"unavailable": True, "reason": "..."},
          ...
          "_unavailable": list of ticker symbols that returned unavailable
        }
    """
    out: dict[str, Any] = {"trade_date": trade_date, "_unavailable": []}
    for symbol in tickers:
        ticker_data = _compute_one_ticker(symbol, trade_date)
        out[symbol] = ticker_data
        if ticker_data.get("unavailable"):
            out["_unavailable"].append(symbol)
    return out
