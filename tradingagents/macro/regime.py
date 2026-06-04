"""Aggregate pillar scores into a regime label, a Growth×Inflation quadrant,
and a trade gate. Pure — operates on the PillarScore list only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import (
    PILLAR_WEIGHTS, GATE_RED_BREADTH, GATE_SCORE_FLOOR, GATE_CAUTION_AT,
)
from .pillars import PillarScore


@dataclass
class Regime:
    score: float
    label: str
    quadrant: str
    gate: str                          # "GO" | "CAUTION" | "STAND_DOWN"
    pillars: list[PillarScore] = field(default_factory=list)
    red_count: int = 0


def _aggregate(pillars: list[PillarScore]) -> float:
    num = den = 0.0
    for p in pillars:
        w = PILLAR_WEIGHTS.get(p.name, 1.0)
        num += w * p.score
        den += w
    return round(num / den, 4) if den else 0.0


def _quadrant(by_name: dict[str, float]) -> str:
    growth_up = by_name.get("growth", 0.0) >= 0
    # inflation pillar is inverted (high score = disinflation), so falling
    # inflation == positive pillar score.
    inflation_falling = by_name.get("inflation", 0.0) >= 0
    if growth_up and inflation_falling:
        return "Goldilocks"
    if growth_up and not inflation_falling:
        return "Reflation"
    if not growth_up and inflation_falling:
        return "Deflation"
    return "Stagflation"


def _gate(score: float, red_count: int) -> str:
    if red_count >= GATE_RED_BREADTH or score <= GATE_SCORE_FLOOR:
        return "STAND_DOWN"
    if score <= GATE_CAUTION_AT:
        return "CAUTION"
    return "GO"


def _label(score: float, quadrant: str, gate: str) -> str:
    tone = ("Risk-On" if score > abs(GATE_CAUTION_AT)
            else "Risk-Off" if score < GATE_CAUTION_AT else "Neutral")
    gate_tag = f" [{gate}]" if gate != "GO" else ""
    return f"{tone} · {quadrant}{gate_tag}"


def build(pillars: list[PillarScore]) -> Regime:
    score = _aggregate(pillars)
    red_count = sum(1 for p in pillars if p.status == "R")
    by_name = {p.name: p.score for p in pillars}
    quadrant = _quadrant(by_name)
    gate = _gate(score, red_count)
    return Regime(score=score, label=_label(score, quadrant, gate),
                  quadrant=quadrant, gate=gate, pillars=pillars,
                  red_count=red_count)
