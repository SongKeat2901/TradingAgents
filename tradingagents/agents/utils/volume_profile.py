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


from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    mid: float
    volume: float


def build_histogram(rows: list[Bar], n_bins: int = 50) -> list[Bin]:
    if not rows:
        return []
    lo = min(r[3] for r in rows)
    hi = max(r[2] for r in rows)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / n_bins
    edges = [lo + i * width for i in range(n_bins + 1)]
    vols = [0.0] * n_bins
    for _d, _o, h, low, _c, vol in rows:
        if h <= low:
            h = low + 1e-9
        span = h - low
        for i in range(n_bins):
            overlap = min(h, edges[i + 1]) - max(low, edges[i])
            if overlap > 0:
                vols[i] += vol * (overlap / span)
    return [Bin(edges[i], edges[i + 1], (edges[i] + edges[i + 1]) / 2, vols[i])
            for i in range(n_bins)]


def point_of_control(bins: list[Bin]) -> float | None:
    if not bins:
        return None
    return max(bins, key=lambda b: b.volume).mid


def value_area(bins: list[Bin], pct: float = 0.70) -> tuple[float | None, float | None]:
    if not bins:
        return (None, None)
    total = sum(b.volume for b in bins)
    if total <= 0:
        return (None, None)
    poc_idx = max(range(len(bins)), key=lambda i: bins[i].volume)
    captured = bins[poc_idx].volume
    lo_idx = hi_idx = poc_idx
    target = pct * total
    while captured < target and (lo_idx > 0 or hi_idx < len(bins) - 1):
        below = bins[lo_idx - 1].volume if lo_idx > 0 else -1.0
        above = bins[hi_idx + 1].volume if hi_idx < len(bins) - 1 else -1.0
        if above >= below:
            hi_idx += 1
            captured += bins[hi_idx].volume
        else:
            lo_idx -= 1
            captured += bins[lo_idx].volume
    return (bins[lo_idx].low, bins[hi_idx].high)
