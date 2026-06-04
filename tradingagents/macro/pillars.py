"""Score each macro pillar from raw series. Pure — no I/O.

Each indicator → a z-score of its latest value vs a trailing window, blended
with the sign of its recent trend, squashed to [-1,+1]. `invert` flips the
sign for indicators where HIGHER = risk-off (VIX, spreads, inflation). Pillar
score = weighted mean of its indicators; status is R/A/G by threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    IndicatorSpec, INDICATORS, PILLARS, PILLAR_GREEN_AT, PILLAR_RED_AT,
)


@dataclass
class PillarScore:
    name: str
    score: float                       # [-1, +1]
    status: str                        # "R" | "A" | "G"
    contributors: dict[str, float] = field(default_factory=dict)


def zscore_latest(s: pd.Series, window: int) -> float:
    """Z-score of the most recent value vs the trailing `window`."""
    s = s.dropna()
    if len(s) < 5:
        return 0.0
    tail = s.iloc[-window:]
    mu, sd = float(tail.mean()), float(tail.std(ddof=0))
    if sd == 0:
        return 0.0
    return (float(s.iloc[-1]) - mu) / sd


def _trend_sign(s: pd.Series, lookback: int = 20) -> float:
    s = s.dropna()
    if len(s) < lookback + 1:
        return 0.0
    return float(np.sign(s.iloc[-1] - s.iloc[-1 - lookback]))


def indicator_score(spec: IndicatorSpec, s: pd.Series) -> float:
    """Single-indicator score in [-1,+1]: tanh(z) blended with trend sign,
    then inverted if the indicator is risk-off-when-high."""
    z = zscore_latest(s, spec.window_days)
    base = math.tanh(z)                          # squash to (-1,1)
    trend = _trend_sign(s)
    score = 0.7 * base + 0.3 * trend
    score = max(-1.0, min(1.0, score))
    return -score if spec.invert else score


def _status(score: float) -> str:
    if score >= PILLAR_GREEN_AT:
        return "G"
    if score <= PILLAR_RED_AT:
        return "R"
    return "A"


def score_pillar(name: str, specs: list[IndicatorSpec],
                 series: dict[str, pd.Series]) -> PillarScore:
    contributors: dict[str, float] = {}
    num = den = 0.0
    for spec in specs:
        s = series.get(spec.name)
        if s is None or len(s.dropna()) < 5:
            continue
        sc = indicator_score(spec, s)
        contributors[spec.name] = round(sc, 4)
        num += spec.weight * sc
        den += spec.weight
    score = (num / den) if den else 0.0
    return PillarScore(name=name, score=round(score, 4),
                       status=_status(score), contributors=contributors)


def score_all(series: dict[str, pd.Series]) -> list[PillarScore]:
    out = []
    for pillar in PILLARS:
        specs = [sp for sp in INDICATORS if sp.pillar == pillar]
        out.append(score_pillar(pillar, specs, series))
    return out
