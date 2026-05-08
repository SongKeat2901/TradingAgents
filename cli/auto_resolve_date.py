"""Phase 7.5 (Fix #13): auto-resolve trade_date to latest closed yfinance session.

The COIN 2026-05-08 + MSFT 2026-05-08 runs surfaced a consistent failure
mode: when the user passes `--date 2026-05-08` but US markets haven't
closed for that session yet (or yfinance hasn't ingested it), the LLM
invents a plausible "post-print" close. Phase 7.4 catches it post-hoc
but the pipeline still ran 30+ minutes of LLM work to produce a flawed
report.

This module resolves the trade_date BEFORE the pipeline starts:

1. Fetch the latest yfinance close for the ticker (one API call, ~1s).
2. If args.date > latest_close: override args.date = latest_close.
3. If args.output_dir contains the original date string: optionally
   adjust to the new date so artifacts land in a consistent dir.
4. Log loudly so the operator sees the adjustment.

Default behaviour: always auto-adjust. Operator can disable with
`--no-auto-adjust-date` for backtests / historical replays where the
pipeline must run on a specific past date even if newer closes exist.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _fetch_latest_close_date(ticker: str) -> str | None:
    """Return the latest indexed close date for `ticker` from yfinance,
    formatted as YYYY-MM-DD. Returns None on any error (network, no data,
    delisted, etc.) — the pipeline should fall back to the user's date."""
    try:
        import yfinance as yf
        # Pull the last 14 days; we only need the most-recent row
        t = yf.Ticker(ticker)
        hist = t.history(period="14d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        latest_idx = hist.index[-1]
        if hasattr(latest_idx, "to_pydatetime"):
            latest_idx = latest_idx.to_pydatetime()
        return latest_idx.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 - yfinance is best-effort
        return None


def auto_resolve_trade_date(
    ticker: str,
    requested_date: str,
    requested_output_dir: str,
) -> tuple[str, str, bool]:
    """Resolve the actual trade_date and output_dir to use.

    Returns ``(trade_date, output_dir, was_adjusted)``:
    - If the requested date is ≤ the latest yfinance close, returns the
      arguments unchanged (was_adjusted=False).
    - If the requested date is > the latest yfinance close, returns the
      latest close as `trade_date`. If the output_dir basename contains
      the original date string, adjusts the basename to use the latest
      close date.
    - On any yfinance error, returns the requested arguments unchanged
      (graceful degradation; the caller may still hit Phase 7 validators).
    """
    latest = _fetch_latest_close_date(ticker)
    if latest is None:
        return requested_date, requested_output_dir, False

    try:
        req = datetime.strptime(requested_date, "%Y-%m-%d").date()
        lat = datetime.strptime(latest, "%Y-%m-%d").date()
    except ValueError:
        return requested_date, requested_output_dir, False

    if req <= lat:
        # Requested date already at or before latest close — no adjustment.
        return requested_date, requested_output_dir, False

    # Adjust output_dir basename if it contains the requested date string.
    out = Path(requested_output_dir)
    new_basename = out.name.replace(requested_date, latest)
    new_output_dir = str(out.parent / new_basename) if new_basename != out.name else requested_output_dir

    print(
        f"[auto-resolve] trade_date adjusted: {requested_date} → {latest} "
        f"(yfinance latest close for {ticker} is {latest}, requested date "
        f"is in the future or session not yet closed). "
        f"Output dir: {requested_output_dir} → {new_output_dir}",
        file=sys.stderr,
    )
    return latest, new_output_dir, True
