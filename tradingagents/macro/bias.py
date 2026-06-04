"""The three-layer macro overlay (pure):

1. Tilt — adjust EV by Σ(beta · expected_factor_move), scaled + capped.
2. Conviction/size — regime quality × beta alignment × beta confidence.
3. Gate — STAND_DOWN zeroes conviction and overrides the action.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import (
    FACTORS, FACTOR_REGIME_MAP, EV_TILT_CAP, MACRO_RETURN_SCALE,
    BIAS_GREEN_AT, BIAS_RED_AT, ACTION_ADD_AT, ACTION_TRIM_AT,
    CONVICTION_HEADWIND_MULT, CONVICTION_LOW_CONF_MULT, CONVICTION_CAUTION_MULT,
)
from .betas import Betas
from .regime import Regime


@dataclass
class StockBias:
    ticker: str
    rating: str
    driver: str
    macro_bias: str                    # "R" | "A" | "G"
    research_ev_pct: float | None
    macro_delta_pct: float
    adjusted_ev_pct: float | None
    conviction: float                  # 0..1
    action: str


def expected_factor_moves(regime: Regime) -> dict[str, float]:
    by_pillar = {p.name: p.score for p in regime.pillars}
    moves: dict[str, float] = {}
    for factor in FACTORS:
        weights = FACTOR_REGIME_MAP.get(factor, {})
        m = sum(w * by_pillar.get(pillar, 0.0) for pillar, w in weights.items())
        moves[factor] = max(-1.0, min(1.0, m))
    return moves


def _macro_contribution(betas: Betas, moves: dict[str, float]) -> float:
    raw = sum(betas.betas.get(f, 0.0) * moves[f] for f in FACTORS)
    delta = raw * MACRO_RETURN_SCALE
    return max(-EV_TILT_CAP, min(EV_TILT_CAP, delta))


def _bias_status(delta: float) -> str:
    if delta >= BIAS_GREEN_AT:
        return "G"
    if delta <= BIAS_RED_AT:
        return "R"
    return "A"


def describe_driver(betas: Betas) -> str:
    ranked = sorted(betas.betas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = [f"{name}({val:+.2f})" for name, val in ranked[:2] if val != 0.0]
    return ", ".join(top) if top else "—"


def _conviction(regime: Regime, delta: float, betas: Betas) -> float:
    if regime.gate == "STAND_DOWN":
        return 0.0
    base = 0.5 + 0.5 * max(0.0, regime.score)      # regime quality
    align = 1.0 if delta >= 0 else CONVICTION_HEADWIND_MULT
    conf = 1.0 if betas.confidence == "high" else CONVICTION_LOW_CONF_MULT
    haircut = CONVICTION_CAUTION_MULT if regime.gate == "CAUTION" else 1.0
    return round(max(0.0, min(1.0, base * align * conf * haircut)), 3)


def _action(regime: Regime, rating: str, adjusted_ev_pct: float | None) -> str:
    if regime.gate == "STAND_DOWN":
        return "STAND DOWN — no new risk (macro red)"
    if regime.gate == "CAUTION":
        return f"Caution — half size; {rating}"
    if adjusted_ev_pct is None:
        return rating
    if adjusted_ev_pct > ACTION_ADD_AT:
        return f"{rating} — add/hold"
    if adjusted_ev_pct < ACTION_TRIM_AT:
        return f"{rating} — trim/avoid"
    return f"{rating} — hold"


def bias_stock(ticker: str, rating: str, regime: Regime, betas: Betas,
               research_ev_pct: float | None) -> StockBias:
    moves = expected_factor_moves(regime)
    delta = _macro_contribution(betas, moves)
    adjusted = None if research_ev_pct is None else round(research_ev_pct + delta, 4)
    conviction = _conviction(regime, delta, betas)
    return StockBias(
        ticker=ticker, rating=rating, driver=describe_driver(betas),
        macro_bias=_bias_status(delta), research_ev_pct=research_ev_pct,
        macro_delta_pct=round(delta, 4), adjusted_ev_pct=adjusted,
        conviction=conviction, action=_action(regime, rating, adjusted),
    )
