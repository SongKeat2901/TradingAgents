"""Deterministic volume-by-price profile (liquidity levels).

Replaces the LLM-eyeballed "Volume profile zones" section with a computed
volume-by-price histogram and the levels traders care about: Point of
Control, Value Area (70%), High/Low Volume Nodes. stdlib only.
"""
from __future__ import annotations

Bar = tuple[str, float, float, float, float, float]  # date, o, h, l, c, volume

_TRADING_DAYS_PER_MONTH = 21


def parse_ohlcv(ohlcv_csv: str) -> list[Bar]:
    rows: list[Bar] = []
    for line in ohlcv_csv.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append((parts[0], float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), float(parts[5])))
        except ValueError:
            continue
    return rows


def select_window(rows: list[Bar], months: int) -> list[Bar]:
    if months <= 0:
        return []
    n = months * _TRADING_DAYS_PER_MONTH
    return rows[-n:] if len(rows) > n else list(rows)
