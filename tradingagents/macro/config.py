"""Static configuration for the macro regime engine.

Indicator definitions, pillar membership/weights, regime/gate thresholds, and
the factor→pillar map that turns a regime into expected factor moves. All
tunable; defaults chosen for a sensible v1 (weights/thresholds are post-v1
backtest items per the spec).
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical factor order — used by betas.py, bias.py, and tests. Do not reorder.
FACTORS: list[str] = ["d_10y", "d_dxy", "d_hy_spread", "oil_ret", "mkt", "growth_value"]

PILLARS: list[str] = [
    "growth", "inflation", "liquidity", "financial_conditions",
    "risk_appetite", "positioning",
]


@dataclass(frozen=True)
class IndicatorSpec:
    name: str            # unique key, e.g. "vix"
    source: str          # "yfinance" | "fred"
    code: str            # yfinance ticker or FRED series id
    pillar: str          # one of PILLARS
    weight: float = 1.0  # weight within its pillar
    invert: bool = False # True when a HIGHER reading is risk-OFF (e.g. VIX, spreads)
    window_days: int = 504  # trailing window for z-scoring (~2yr)


# v1 indicator set. Hard-data (FRED) + market-priced (yfinance). Positioning is
# intentionally thin (weak free data) — low weight, upgrade post-v1.
INDICATORS: list[IndicatorSpec] = [
    # Growth
    IndicatorSpec("indpro", "fred", "INDPRO", "growth", 1.0),  # Industrial Production (ISM PMI not free on FRED)
    IndicatorSpec("jobless_claims", "fred", "ICSA", "growth", 1.0, invert=True),
    IndicatorSpec("curve_10y2y", "fred", "T10Y2Y", "growth", 1.0),
    IndicatorSpec("curve_10y3m", "fred", "T10Y3M", "growth", 1.0),
    IndicatorSpec("copper_gold", "yfinance", "HG=F", "growth", 0.5),
    # Inflation (invert=True: rising inflation is a headwind for the regime score)
    IndicatorSpec("cpi_yoy", "fred", "CPIAUCSL", "inflation", 1.0, invert=True),
    IndicatorSpec("breakeven_10y", "fred", "T10YIE", "inflation", 1.0, invert=True),
    IndicatorSpec("oil", "yfinance", "CL=F", "inflation", 0.5, invert=True),
    IndicatorSpec("commodities", "yfinance", "DBC", "inflation", 0.5, invert=True),
    # Liquidity / policy
    IndicatorSpec("fed_funds", "fred", "DFF", "liquidity", 1.0, invert=True),
    IndicatorSpec("real_10y", "fred", "DFII10", "liquidity", 1.0, invert=True),
    IndicatorSpec("m2", "fred", "M2SL", "liquidity", 1.0),
    # Financial conditions (invert=True: tighter = risk-off)
    IndicatorSpec("dxy", "yfinance", "DX-Y.NYB", "financial_conditions", 1.0, invert=True),
    IndicatorSpec("ig_spread", "fred", "BAMLC0A0CM", "financial_conditions", 1.0, invert=True),
    IndicatorSpec("hy_spread", "fred", "BAMLH0A0HYM2", "financial_conditions", 1.0, invert=True),
    IndicatorSpec("move", "yfinance", "^MOVE", "financial_conditions", 1.0, invert=True),
    IndicatorSpec("nfci", "fred", "NFCI", "financial_conditions", 1.0, invert=True),
    # Risk appetite
    IndicatorSpec("vix", "yfinance", "^VIX", "risk_appetite", 1.0, invert=True),
    IndicatorSpec("hibeta_lowvol", "yfinance", "SPHB", "risk_appetite", 0.5),
    IndicatorSpec("cyc_def", "yfinance", "XLY", "risk_appetite", 0.5),
    IndicatorSpec("btc", "yfinance", "BTC-USD", "risk_appetite", 0.5),
    # Positioning (thin — low weight)
    IndicatorSpec("aaii_proxy", "fred", "UMCSENT", "positioning", 0.5),
]

# Pillar weights in the regime aggregate.
PILLAR_WEIGHTS: dict[str, float] = {
    "growth": 1.0,
    "inflation": 1.0,
    "liquidity": 1.0,
    "financial_conditions": 1.0,
    "risk_appetite": 1.0,
    "positioning": 0.5,
}

# Pillar status thresholds on the [-1,+1] pillar score.
PILLAR_GREEN_AT = 0.2    # score >= → "G"
PILLAR_RED_AT = -0.2     # score <= → "R"; between → "A"

# Gate: STAND_DOWN when this many pillars are red, OR regime score below floor.
GATE_RED_BREADTH = 4
GATE_SCORE_FLOOR = -0.4
GATE_CAUTION_AT = -0.1   # regime score below this (but above floor / breadth) → CAUTION

# Quadrant labels keyed by (sign(growth) >= 0, sign(inflation_raw) >= 0).
# inflation_raw is the *non-inverted* inflation direction (rising = True).

# EV bias.
EV_TILT_CAP = 0.15       # adjusted EV may move at most ±15% from research EV
MACRO_RETURN_SCALE = 0.10  # converts (Σ beta·expected_move) into a 12-mo return delta

# Bias / action thresholds (bias.py) — tunable post-v1 backtest.
BIAS_GREEN_AT = 0.02              # macro_delta_pct at/above which macro_bias = "G"
BIAS_RED_AT = -0.02              # at/below which macro_bias = "R"
ACTION_ADD_AT = 0.05             # adjusted_ev_pct above which action = add/hold
ACTION_TRIM_AT = -0.05          # adjusted_ev_pct below which action = trim/avoid
CONVICTION_HEADWIND_MULT = 0.5  # conviction penalty when macro tilt is a headwind (delta < 0)
CONVICTION_LOW_CONF_MULT = 0.5  # conviction penalty for "low"-confidence betas
CONVICTION_CAUTION_MULT = 0.5   # conviction haircut under the CAUTION gate

# Maps each factor to the pillars that drive its expected move, with signs.
# expected_move[factor] = clip(Σ weight · pillar_score, -1, +1).
# Sign convention: positive expected_move = factor RISES.
# IMPORTANT: pillar scores use the "+ = risk-supportive" convention, so the
# inflation pillar is HIGH when inflation is FALLING (invert=True) and the
# liquidity pillar is HIGH when policy is EASING. Coefficients below are written
# against those *scores*, not the raw macro variable.
#   d_10y (rates) rise with strong growth, RISING inflation (low infl-score) and
#     TIGHTENING liquidity (low liq-score) → +growth -inflation -liquidity
#   d_dxy rises with tight financial conditions (low fc-score) & risk-off → -fc -risk_appetite
#   d_hy_spread widens when risk-off / tight → -risk_appetite -financial_conditions
#   oil rises with growth & RISING inflation (low infl-score) → +growth -inflation
#   mkt (equities) rise with risk-on / easing / growth → +risk_appetite +liquidity +growth
#   growth_value (growth − value) favors growth when inflation FALLS (high infl-score)
#     and liquidity EASES (high liq-score) → +liquidity +inflation
FACTOR_REGIME_MAP: dict[str, dict[str, float]] = {
    "d_10y": {"growth": 0.5, "inflation": -0.5, "liquidity": -0.5},
    "d_dxy": {"financial_conditions": -0.5, "risk_appetite": -0.5},
    "d_hy_spread": {"risk_appetite": -0.7, "financial_conditions": -0.3},
    "oil_ret": {"growth": 0.5, "inflation": -0.5},
    "mkt": {"risk_appetite": 0.5, "liquidity": 0.3, "growth": 0.2},
    "growth_value": {"liquidity": 0.5, "inflation": 0.3},
}

# Shrinkage for short-history betas: blend toward 0 below this many observations.
BETA_MIN_OBS = 252       # full-confidence window
BETA_SHRINK_FLOOR = 60   # below this, beta is heavily shrunk; flagged "low"

SHEET_MAX_ROWS = 100  # to_grid pads to this height so a shorter run can't leave stale trailing rows (no-dupes rule)
