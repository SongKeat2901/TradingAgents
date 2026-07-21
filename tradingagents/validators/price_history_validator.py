"""Price-history sufficiency validator (wk29 SATS silent-corruption class).

When a ticker is renamed or delisted, yfinance returns a stub series (SATS
2026-07-17 returned a single session despite a 5-year request). stockstats
then computes DEGENERATE moving averages from that one bar — the "50-DMA"
equals the single close, gap-to-MA 0.0% — and the deterministic classifier
emits a confident-but-wrong setup class (SATS: CONSOLIDATION) with no
violation raised. The whole technical layer is built on sand, yet
validation_report.json reads 0 violations because every OTHER validator
compares claims to raw/ and here raw/ itself is the corruption.

This validator closes that hole with a single hard floor: no real,
research-eligible equity has fewer than a trading month (~20 sessions) of
history, so a bar count below the floor is a data-fetch failure, not a
young listing. It is a BLOCKING (MATERIAL) violation so promotion/QC treat
the run as unusable rather than silently publishing degenerate technicals.

Deliberately NOT a "bars < 200 while a 200-DMA is reported" contradiction
check: stockstats reports a degenerate MA for genuinely-young names too, so
that check would false-positive on legitimate recent IPOs. The hard floor
catches the observed corruption class (a ~1-bar stub) with zero risk to any
real listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# One trading month. Below this, a research-eligible equity's price series is
# a fetch stub (renamed/delisted ticker), not a real short history.
_MIN_BARS_FLOOR = 20


@dataclass(frozen=True)
class PriceHistoryViolation:
    severity: Literal["MATERIAL"]
    type: Literal["insufficient_price_history"]
    ticker: str
    bars: int
    floor: int
    match_text: str


def _count_bars(prices_json_path: Path) -> int | None:
    """Number of OHLCV data rows in raw/prices.json. None if unreadable."""
    try:
        data = json.loads(prices_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ohlcv = data.get("ohlcv") if isinstance(data, dict) else None
    if not isinstance(ohlcv, str):
        return None
    n = 0
    for line in ohlcv.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        # a data row starts with a date digit (YYYY-...)
        if line[0].isdigit():
            n += 1
    return n


def validate_price_history(
    prices_json_path: Path, ticker: str | None = None,
) -> list[PriceHistoryViolation]:
    """Flag a grossly-insufficient price series (fetch stub) as BLOCKING.

    Silent when prices.json is absent/unreadable (a different phase owns
    the "missing data" case) or when bar count is at/above the floor.
    """
    bars = _count_bars(Path(prices_json_path))
    if bars is None or bars >= _MIN_BARS_FLOOR:
        return []
    tkr = ticker or "?"
    return [PriceHistoryViolation(
        severity="MATERIAL",
        type="insufficient_price_history",
        ticker=tkr,
        bars=bars,
        floor=_MIN_BARS_FLOOR,
        match_text=(
            f"raw/prices.json for {tkr} has {bars} bar(s) (floor {_MIN_BARS_FLOOR}); "
            f"moving averages and the setup class are degenerate. Likely a "
            f"renamed/delisted ticker returning a stub series (e.g. SATS→ECHO)."
        ),
    )]
