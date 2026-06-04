# Macro Regime Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone daily "macro regime engine" that scores the macro tape across six pillars, computes each researched stock's statistical sensitivity to macro factors, and biases each name's EV (tilt + conviction + global gate) — writing the result to the Trading Plan Google Sheet.

**Architecture:** A new self-contained package `tradingagents/macro/` with single-responsibility units chained left→right: data fetch (yfinance + FRED, cached) → pillar scoring (pure) → regime aggregation (pure) → per-stock betas (pure OLS) → three-layer bias (pure) → idempotent sheet write. An orchestrator CLI (`tradingmacro`) runs the chain daily after the US close. Fully decoupled from the research pipeline — it only *reads* finished report dirs.

**Tech Stack:** Python 3.10+, pandas + numpy (numpy.linalg.lstsq for OLS — no statsmodels), `requests` for FRED, `yfinance` for market data, `gog` CLI for the Sheets write. Tests via pytest `-m unit`, mocking all network I/O.

---

## File Structure

```
tradingagents/macro/
  __init__.py        # package marker
  config.py          # indicator specs, pillar weights, thresholds, factor map, constants
  macro_data.py      # fetch + cache raw series (yfinance + FRED); load stock prices
  pillars.py         # score each pillar from series → PillarScore (pure)
  regime.py          # aggregate pillars → Regime (label, quadrant, gate) (pure)
  betas.py           # build factor returns + rolling OLS per ticker → Betas (pure)
  reports.py         # read base EV/rating/scenarios from research run dirs (reuses daily_followup)
  bias.py            # regime × betas × base-EV → StockBias (tilt/conviction/gate) (pure)
  plan_writer.py     # build sheet payload (pure) + idempotent gog write (I/O)
  macro_daily.py     # orchestrator + CLI entry point

tests/macro/
  test_macro_config.py
  test_macro_data.py
  test_pillars.py
  test_regime.py
  test_betas.py
  test_reports.py
  test_bias.py
  test_plan_writer.py
  test_macro_daily.py

ops/com.trueknot.macrodaily.plist   # launchd schedule for the mini
```

**Data types passed between units** (all defined in `config.py` or their owning module):
- `IndicatorSpec(name, source, code, pillar, weight, invert, window_days)` — config.
- raw series → `dict[str, pandas.Series]` keyed by indicator `name`.
- `PillarScore(name, score: float, status: str, contributors: dict[str, float])` — pillars.py. `score` ∈ [−1,+1]; `status` ∈ {"R","A","G"}.
- `Regime(score, label, quadrant, gate, pillars: list[PillarScore], red_count)` — regime.py. `gate` ∈ {"GO","CAUTION","STAND_DOWN"}.
- `Betas(ticker, betas: dict[str,float], r2: float, confidence: str, n_obs: int)` — betas.py. `confidence` ∈ {"high","low"}.
- `BaseEV(ticker, research_date, rating, reference_price, ev, scenarios, hard_stop)` — reports.py.
- `StockBias(ticker, rating, driver, macro_bias, research_ev_pct, macro_delta_pct, adjusted_ev_pct, conviction, action)` — bias.py.

**Factor set** (canonical order, used everywhere): `["d_10y", "d_dxy", "d_hy_spread", "oil_ret", "mkt", "growth_value"]`.

---

## Task 1: Package scaffold + config

**Files:**
- Create: `tradingagents/macro/__init__.py`
- Create: `tradingagents/macro/config.py`
- Test: `tests/macro/__init__.py`, `tests/macro/test_macro_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/__init__.py` (empty file) and `tests/macro/test_macro_config.py`:

```python
import pytest
from tradingagents.macro import config

pytestmark = pytest.mark.unit


def test_factor_order_is_canonical():
    assert config.FACTORS == ["d_10y", "d_dxy", "d_hy_spread", "oil_ret", "mkt", "growth_value"]


def test_every_indicator_has_a_known_pillar():
    for spec in config.INDICATORS:
        assert spec.pillar in config.PILLARS, f"{spec.name} has unknown pillar {spec.pillar}"


def test_pillar_weights_cover_all_pillars():
    assert set(config.PILLAR_WEIGHTS) == set(config.PILLARS)


def test_factor_regime_map_uses_known_factors_and_pillars():
    for factor, weights in config.FACTOR_REGIME_MAP.items():
        assert factor in config.FACTORS
        for pillar in weights:
            assert pillar in config.PILLARS
    assert set(config.FACTOR_REGIME_MAP) == set(config.FACTORS)


def test_gate_thresholds_present():
    assert config.GATE_RED_BREADTH == 4
    assert 0.0 < config.EV_TILT_CAP <= 1.0
    assert -1.0 <= config.GATE_SCORE_FLOOR < 0.0
    assert -1.0 <= config.GATE_CAUTION_AT < 0.0
    assert config.GATE_CAUTION_AT > config.GATE_SCORE_FLOOR        # ordering sanity
    assert -1.0 <= config.PILLAR_RED_AT < config.PILLAR_GREEN_AT <= 1.0
    assert 0.0 < config.MACRO_RETURN_SCALE <= 1.0
    assert config.BETA_SHRINK_FLOOR < config.BETA_MIN_OBS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradingagents.macro'`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/__init__.py`:

```python
"""Standalone daily macro regime engine.

Scores the macro tape across six pillars, computes each researched stock's
statistical macro betas, and biases each name's EV (tilt + conviction +
global gate). Decoupled from the research pipeline — reads finished report
dirs only. See docs/superpowers/specs/2026-06-04-macro-regime-engine-design.md
"""
```

Create `tradingagents/macro/config.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_config.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/__init__.py tradingagents/macro/config.py tests/macro/__init__.py tests/macro/test_macro_config.py
git commit -m "feat(macro): package scaffold + indicator/pillar/factor config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: macro_data — fetch + cache raw series

**Files:**
- Create: `tradingagents/macro/macro_data.py`
- Test: `tests/macro/test_macro_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_macro_data.py`:

```python
import json
import pandas as pd
import pytest

from tradingagents.macro import macro_data
from tradingagents.macro.config import IndicatorSpec

pytestmark = pytest.mark.unit


def test_parse_fred_observations_to_series():
    payload = {"observations": [
        {"date": "2026-05-01", "value": "4.5"},
        {"date": "2026-05-02", "value": "."},      # FRED missing marker
        {"date": "2026-05-03", "value": "4.7"},
    ]}
    s = macro_data._parse_fred(payload)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-05-01", "2026-05-03"]
    assert s.iloc[-1] == 4.7


def test_load_series_caches_after_first_fetch(tmp_path, monkeypatch):
    spec = IndicatorSpec("vix", "yfinance", "^VIX", "risk_appetite")
    calls = {"n": 0}

    def fake_fetch(spec_):
        calls["n"] += 1
        return pd.Series([10.0, 11.0], index=pd.to_datetime(["2026-05-01", "2026-05-02"]))

    monkeypatch.setattr(macro_data, "_fetch_yfinance", fake_fetch)
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)

    s1 = macro_data.load_series(spec, as_of="2026-05-02")
    s2 = macro_data.load_series(spec, as_of="2026-05-02")
    assert calls["n"] == 1            # second call served from cache
    assert s1.equals(s2)


def test_load_series_routes_fred(tmp_path, monkeypatch):
    spec = IndicatorSpec("cpi", "fred", "CPIAUCSL", "inflation")
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(macro_data, "_fetch_fred",
                        lambda s: pd.Series([1.0], index=pd.to_datetime(["2026-05-01"])))
    out = macro_data.load_series(spec, as_of="2026-05-01")
    assert out.iloc[-1] == 1.0


def test_load_all_skips_failed_series(monkeypatch):
    from tradingagents.macro.config import IndicatorSpec
    good = IndicatorSpec("good", "yfinance", "G", "growth")
    bad = IndicatorSpec("bad", "fred", "B", "growth")

    def fake_load(spec, as_of):
        if spec.name == "bad":
            raise RuntimeError("boom")
        return pd.Series([1.0], index=pd.to_datetime(["2026-05-01"]))

    monkeypatch.setattr(macro_data, "load_series", fake_load)
    out = macro_data.load_all([good, bad], as_of="2026-06-02")
    assert "good" in out and "bad" not in out


def test_load_prices_drops_incomplete_and_caches(tmp_path, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(macro_data, "CACHE_DIR", tmp_path)
    idx = pd.to_datetime(["2026-05-29", "2026-06-01"]).tz_localize("America/New_York")
    fake_hist = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    fake_hist.index.name = "Date"

    class FakeTicker:
        def __init__(self, code):
            pass

        def history(self, period="2y", auto_adjust=True):
            return fake_hist.copy()

    # Import stockstats_utils BEFORE patching yfinance, since it imports yf at
    # module level.  Then stub drop_incomplete_session so the test stays hermetic.
    import tradingagents.dataflows.stockstats_utils as _stu
    monkeypatch.setattr(_stu, "drop_incomplete_session", lambda df, now=None: df)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    s = macro_data.load_prices("AAA", as_of="2026-06-02")
    assert [round(v, 2) for v in s.values] == [100.0, 101.0]
    assert s.index[-1].strftime("%Y-%m-%d") == "2026-06-01"
    # second call served from cache (no yfinance import needed)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert macro_data.load_prices("AAA", as_of="2026-06-02").equals(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradingagents.macro.macro_data'`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/macro_data.py`:

```python
"""Fetch + cache the raw series the engine needs.

Two sources: yfinance (market-priced) and FRED (hard macro data). One fetch
per series per `as_of` date, cached to CSV under CACHE_DIR so a re-run on the
same day is free and offline-stable. Network errors propagate to the caller
(the orchestrator decides whether a missing series is fatal).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests

from .config import IndicatorSpec

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get(
    "MACRO_CACHE_DIR", str(Path.home() / ".cache" / "tradingagents-macro")
))
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _parse_fred(payload: dict) -> pd.Series:
    """Turn a FRED observations JSON payload into a float Series, dropping the
    '.' missing-value markers FRED uses."""
    dates, vals = [], []
    for obs in payload.get("observations", []):
        v = obs.get("value", ".")
        if v in (".", "", None):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
        dates.append(obs["date"])
    return pd.Series(vals, index=pd.to_datetime(dates), name="value").sort_index()


def _fetch_fred(spec: IndicatorSpec) -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set — required for FRED indicators")
    resp = requests.get(FRED_BASE, params={
        "series_id": spec.code, "api_key": key, "file_type": "json",
    }, timeout=30)
    resp.raise_for_status()
    return _parse_fred(resp.json())


def _fetch_yfinance(spec: IndicatorSpec) -> pd.Series:
    import yfinance as yf
    df = yf.Ticker(spec.code).history(period="3y", auto_adjust=True)  # adjusted Close (consistent w/ load_prices)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {spec.code}")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = "value"
    return s


def load_series(spec: IndicatorSpec, as_of: str) -> pd.Series:
    """Return the series for `spec`, cached per (name, as_of). `as_of` is the
    YYYY-MM-DD run date so each day's snapshot is reproducible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{spec.name}_{as_of}.csv"
    if cache_path.exists():
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True)["value"]
        return s
    fetch = _fetch_fred if spec.source == "fred" else _fetch_yfinance
    s = fetch(spec)
    s.to_frame("value").to_csv(cache_path)
    return s


def load_all(specs: list[IndicatorSpec], as_of: str) -> dict[str, pd.Series]:
    """Load every spec; skip (with a warning) any that fail so one dead series
    doesn't sink the whole run."""
    out: dict[str, pd.Series] = {}
    for spec in specs:
        try:
            out[spec.name] = load_series(spec, as_of)
        except Exception as exc:  # noqa: BLE001 — best-effort per series
            logger.warning("macro series %s (%s) failed: %s", spec.name, spec.code, exc)
    return out


def load_prices(ticker: str, as_of: str, period: str = "2y") -> pd.Series:
    """Daily settled closes for a stock/factor ticker, cached per (ticker, as_of)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"px_{ticker}_{as_of}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)["value"]
    import yfinance as yf
    from tradingagents.dataflows.stockstats_utils import drop_incomplete_session
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no price history for {ticker}")
    df = df.reset_index()  # yfinance names the daily index "Date"
    df = drop_incomplete_session(df)          # drop the in-progress US bar
    s = pd.Series(df["Close"].values,
                  index=pd.to_datetime(df["Date"]).dt.tz_localize(None), name="value")
    s.to_frame("value").to_csv(cache_path)
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/macro_data.py tests/macro/test_macro_data.py
git commit -m "feat(macro): cached yfinance + FRED data layer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: pillars — score each pillar (pure)

**Files:**
- Create: `tradingagents/macro/pillars.py`
- Test: `tests/macro/test_pillars.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_pillars.py`:

```python
import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import pillars
from tradingagents.macro.config import IndicatorSpec

pytestmark = pytest.mark.unit


def _rising_series(n=300, start=100.0, step=0.5):
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(start + step * np.arange(n), index=idx)


def test_zscore_latest_is_positive_for_rising_series():
    z = pillars.zscore_latest(_rising_series(), window=200)
    assert z > 0


def test_indicator_score_inverts_when_flagged():
    spec_plain = IndicatorSpec("a", "fred", "A", "growth", invert=False)
    spec_inv = IndicatorSpec("b", "fred", "B", "growth", invert=True)
    s = _rising_series()
    assert pillars.indicator_score(spec_plain, s) > 0
    assert pillars.indicator_score(spec_inv, s) < 0


def test_score_pillar_aggregates_and_statuses_green():
    specs = [IndicatorSpec("a", "fred", "A", "growth"),
             IndicatorSpec("b", "fred", "B", "growth")]
    series = {"a": _rising_series(), "b": _rising_series()}
    ps = pillars.score_pillar("growth", specs, series)
    assert ps.name == "growth"
    assert ps.score > 0.2 and ps.status == "G"


def test_score_pillar_handles_missing_series_gracefully():
    specs = [IndicatorSpec("a", "fred", "A", "growth")]
    ps = pillars.score_pillar("growth", specs, series={})  # nothing loaded
    assert ps.status == "A" and ps.score == 0.0


def test_score_all_returns_one_per_pillar():
    from tradingagents.macro.config import INDICATORS, PILLARS
    series = {sp.name: _rising_series() for sp in INDICATORS}
    out = pillars.score_all(series)
    assert {p.name for p in out} == set(PILLARS)


def test_zscore_latest_returns_zero_for_flat_series():
    flat = pd.Series([5.0] * 300, index=pd.date_range("2025-01-01", periods=300, freq="D"))
    assert pillars.zscore_latest(flat, window=200) == 0.0


def test_zscore_latest_returns_zero_for_short_series():
    short = pd.Series([1.0, 2.0, 3.0],
                      index=pd.date_range("2025-01-01", periods=3, freq="D"))
    assert pillars.zscore_latest(short, window=200) == 0.0


def test_score_pillar_skips_too_short_series():
    from tradingagents.macro.config import IndicatorSpec
    specs = [IndicatorSpec("a", "fred", "A", "growth")]
    series = {"a": pd.Series([1.0, 2.0, 3.0],
                             index=pd.date_range("2025-01-01", periods=3, freq="D"))}
    ps = pillars.score_pillar("growth", specs, series)   # <5 non-na → skipped
    assert ps.score == 0.0 and ps.status == "A"
    assert "a" not in ps.contributors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_pillars.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/pillars.py`:

```python
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

_TREND_LOOKBACK = 20  # single short-term trend window across all indicators (v1; per-frequency tuning deferred)


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
    # NOTE (v1 tech-debt): the latest point is included in the window stats, a
    # negligible in-sample bias at the 504d default; strict out-of-sample would
    # use s.iloc[-window:-1]. Tuning deferred per spec.
    tail = s.iloc[-window:]
    mu, sd = float(tail.mean()), float(tail.std(ddof=0))
    if sd == 0:
        return 0.0
    return (float(s.iloc[-1]) - mu) / sd


def _trend_sign(s: pd.Series, lookback: int = _TREND_LOOKBACK) -> float:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_pillars.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/pillars.py tests/macro/test_pillars.py
git commit -m "feat(macro): six-pillar scoring (z-score + trend, invert-aware)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: regime — aggregate pillars into a regime + quadrant + gate (pure)

**Files:**
- Create: `tradingagents/macro/regime.py`
- Test: `tests/macro/test_regime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_regime.py`:

```python
import pytest

from tradingagents.macro import regime
from tradingagents.macro.pillars import PillarScore

pytestmark = pytest.mark.unit


def _pillars(score_by_name):
    out = []
    for name, sc in score_by_name.items():
        status = "G" if sc >= 0.2 else "R" if sc <= -0.2 else "A"
        out.append(PillarScore(name=name, score=sc, status=status))
    return out


_ALL = ["growth", "inflation", "liquidity", "financial_conditions",
        "risk_appetite", "positioning"]


def test_gate_stand_down_when_breadth_of_red():
    ps = _pillars({n: -0.5 for n in _ALL})       # all red
    r = regime.build(ps)
    assert r.gate == "STAND_DOWN"
    assert r.red_count == 6


def test_gate_go_when_broadly_green():
    ps = _pillars({n: 0.5 for n in _ALL})
    r = regime.build(ps)
    assert r.gate == "GO"
    assert r.score > 0


def test_gate_caution_in_the_middle():
    # Two red pillars (breadth < 4) pulling the aggregate to ~-0.18 — below the
    # CAUTION threshold (-0.1) but above the STAND_DOWN floor (-0.4).
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -0.5
    scores["financial_conditions"] = -0.5
    r = regime.build(_pillars(scores))
    assert r.gate == "CAUTION"


def test_quadrant_goldilocks_growth_up_inflation_down():
    # inflation pillar score is HIGH when inflation is FALLING (invert=True),
    # so a positive inflation pillar score == disinflation == quadrant "down".
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = 0.5
    scores["inflation"] = 0.5
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Goldilocks"


def test_quadrant_stagflation_growth_down_inflation_up():
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -0.5
    scores["inflation"] = -0.5     # low pillar score == rising inflation
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Stagflation"


def test_quadrant_reflation_growth_up_inflation_up():
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = 0.5
    scores["inflation"] = -0.5     # low pillar score == rising inflation
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Reflation"


def test_quadrant_deflation_growth_down_inflation_down():
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -0.5
    scores["inflation"] = 0.5      # high pillar score == disinflation
    r = regime.build(_pillars(scores))
    assert r.quadrant == "Deflation"


def test_label_contains_tone_and_quadrant():
    r = regime.build(_pillars({n: 0.5 for n in _ALL}))
    assert "Risk-On" in r.label
    assert "Goldilocks" in r.label


def test_gate_stand_down_by_score_alone():
    # Three red pillars (still < breadth threshold of 4) but the weighted
    # aggregate (-3.0/5.5 ≈ -0.55) is below GATE_SCORE_FLOOR (-0.4).
    scores = {n: 0.0 for n in _ALL}
    scores["growth"] = -1.0
    scores["inflation"] = -1.0
    scores["liquidity"] = -1.0
    r = regime.build(_pillars(scores))
    assert r.gate == "STAND_DOWN"
    assert r.red_count < 4         # proves the score-floor path, not breadth
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_regime.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/regime.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_regime.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/regime.py tests/macro/test_regime.py
git commit -m "feat(macro): regime aggregation, G×I quadrant, trade gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: betas — factor returns + rolling OLS per ticker (pure)

**Files:**
- Create: `tradingagents/macro/betas.py`
- Test: `tests/macro/test_betas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_betas.py`:

```python
import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import betas
from tradingagents.macro.config import FACTORS

pytestmark = pytest.mark.unit


def _factor_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    data = {f: rng.normal(0, 0.01, n) for f in FACTORS}
    return pd.DataFrame(data, index=idx)


def test_recovers_known_betas_on_synthetic_data():
    fac = _factor_frame()
    true_b = {f: float(v) for f, v in zip(FACTORS, [1.5, -0.8, -2.0, 0.3, 1.0, 0.5])}
    noise = np.random.default_rng(1).normal(0, 1e-6, len(fac))
    stock_ret = sum(true_b[f] * fac[f] for f in FACTORS) + noise
    out = betas.compute_betas("TEST", stock_ret, fac)
    for f in FACTORS:
        assert abs(out.betas[f] - true_b[f]) < 0.05
    assert out.r2 > 0.99
    assert out.confidence == "high"


def test_short_history_is_shrunk_and_flagged_low():
    fac = _factor_frame(n=40)
    stock_ret = 2.0 * fac["mkt"] + np.random.default_rng(2).normal(0, 0.001, len(fac))
    out = betas.compute_betas("SHORT", stock_ret, fac)
    assert out.confidence == "low"
    assert abs(out.betas["mkt"]) < 2.0          # shrunk toward zero
    assert out.n_obs == 40


def test_build_factor_returns_shapes_and_columns():
    idx = pd.bdate_range("2025-01-01", periods=10)
    raw = {
        "tnx": pd.Series(np.linspace(4.0, 4.2, 10), index=idx),     # level → diff
        "dxy": pd.Series(np.linspace(100, 102, 10), index=idx),     # level → ret
        "hy": pd.Series(np.linspace(3.0, 3.1, 10), index=idx),
        "oil": pd.Series(np.linspace(70, 72, 10), index=idx),
        "spy": pd.Series(np.linspace(500, 510, 10), index=idx),
        "iwf": pd.Series(np.linspace(300, 305, 10), index=idx),
        "iwd": pd.Series(np.linspace(180, 181, 10), index=idx),
    }
    fac = betas.build_factor_returns(raw)
    assert list(fac.columns) == FACTORS
    assert len(fac) == 9                          # one row lost to differencing


def test_linear_shrink_zone_ramps_between_floor_and_full():
    # n=156 is mid-ramp: t=(156-60)/192=0.5 → k=0.25+0.75*0.5=0.625
    fac = _factor_frame(n=156)
    stock_ret = 2.0 * fac["mkt"] + np.random.default_rng(3).normal(0, 1e-6, len(fac))
    out = betas.compute_betas("MID", stock_ret, fac)
    assert out.confidence == "low"
    assert out.n_obs == 156
    implied_k = out.betas["mkt"] / 2.0
    assert 0.55 < implied_k < 0.70   # ~0.625
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_betas.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/betas.py`:

```python
"""Per-stock macro betas via rolling OLS. Pure — operates on provided frames.

`build_factor_returns` constructs the standardized factor-return matrix from
raw factor series (yields → daily change; prices → daily return; growth−value
spread). `compute_betas` regresses a stock's daily returns on the factors via
numpy.linalg.lstsq and applies shrinkage for short samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FACTORS, BETA_MIN_OBS, BETA_SHRINK_FLOOR


@dataclass
class Betas:
    ticker: str
    betas: dict[str, float]
    r2: float
    confidence: str                    # "high" | "low"
    n_obs: int


def build_factor_returns(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """raw keys: tnx, dxy, hy, oil, spy, iwf, iwd. Returns a DataFrame whose
    columns are exactly FACTORS, aligned on common dates, NaNs dropped."""
    cols = {
        "d_10y": raw["tnx"].diff(),
        "d_dxy": raw["dxy"].pct_change(),
        "d_hy_spread": raw["hy"].diff(),
        "oil_ret": raw["oil"].pct_change(),
        "mkt": raw["spy"].pct_change(),
        "growth_value": raw["iwf"].pct_change() - raw["iwd"].pct_change(),
    }
    f = pd.DataFrame(cols)[FACTORS].dropna()
    return f


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_betas(ticker: str, stock_ret: pd.Series, factors: pd.DataFrame) -> Betas:
    df = pd.concat([stock_ret.rename("y"), factors], axis=1).dropna()
    n = len(df)
    if n < 5:
        return Betas(ticker, {f: 0.0 for f in FACTORS}, 0.0, "low", n)
    y = df["y"].to_numpy()
    X = df[FACTORS].to_numpy()
    Xc = np.column_stack([np.ones(n), X])         # intercept
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    beta_vec = coef[1:]
    yhat = Xc @ coef
    r2 = _r2(y, yhat)

    # Shrinkage: full weight at/above BETA_MIN_OBS, linearly toward 0 down to
    # BETA_SHRINK_FLOOR, heavily shrunk below.
    if n >= BETA_MIN_OBS:
        k, confidence = 1.0, "high"
    elif n >= BETA_SHRINK_FLOOR:
        # ramp from the 0.25 floor at BETA_SHRINK_FLOOR up to 1.0 at BETA_MIN_OBS
        t = (n - BETA_SHRINK_FLOOR) / (BETA_MIN_OBS - BETA_SHRINK_FLOOR)
        k = 0.25 + 0.75 * t
        confidence = "low"
    else:
        k, confidence = 0.25, "low"
    betas = {f: round(float(b * k), 4) for f, b in zip(FACTORS, beta_vec)}
    return Betas(ticker, betas, round(r2, 4), confidence, n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_betas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/betas.py tests/macro/test_betas.py
git commit -m "feat(macro): factor-return matrix + rolling-OLS betas with shrinkage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: reports — read base EV from research run dirs

**Files:**
- Create: `tradingagents/macro/reports.py`
- Test: `tests/macro/test_reports.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_reports.py`:

```python
import json
import pytest

from tradingagents.macro import reports

pytestmark = pytest.mark.unit

_DECISION = """\
# Decision

Reference price: **$100.00**

**Rating: BUY**

## 12-Month Scenario Analysis

| Scenario | Prob | Target |
|---|---|---|
| Bull | 30% | $140.00 |
| Base | 50% | $120.00 |
| Bear | 20% | $80.00 |

EV = **$116.00**
"""


def _run_dir(tmp_path, ticker="TEST", date="2026-06-01", body=_DECISION):
    d = tmp_path / f"{date}-{ticker}"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": ticker, "trade_date": date}))
    (d / "decision.md").write_text(body)
    return d


def test_load_base_ev_reads_ev_and_pct(tmp_path):
    be = reports.load_base_ev(_run_dir(tmp_path))
    assert be.ticker == "TEST"
    assert be.reference_price == 100.0
    assert be.ev == 116.0
    assert round(reports.ev_pct(be), 4) == 0.16     # +16%


def test_ev_pct_derives_from_scenarios_when_ev_absent(tmp_path):
    body = _DECISION.replace("EV = **$116.00**", "")
    be = reports.load_base_ev(_run_dir(tmp_path, body=body))
    assert be.ev is None
    # derived = Σ prob·target = .3*140 + .5*120 + .2*80 = 118 → +18%
    assert round(reports.ev_pct(be), 4) == 0.18


def test_returns_none_for_incomplete_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert reports.load_base_ev(d) is None


def test_latest_run_per_ticker_picks_newest_date(tmp_path):
    _run_dir(tmp_path, "AAA", "2026-05-01")
    _run_dir(tmp_path, "AAA", "2026-06-01")
    _run_dir(tmp_path, "BBB", "2026-05-15")
    latest = reports.latest_runs(tmp_path)
    assert latest["AAA"].research_date == "2026-06-01"
    assert set(latest) == {"AAA", "BBB"}


def test_load_base_ev_survives_unreadable_decision(tmp_path):
    import json
    d = tmp_path / "2026-06-01-ERR"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "ERR", "trade_date": "2026-06-01"}))
    (d / "decision.md").mkdir()   # a dir where a file is expected → OSError on read_text
    assert reports.load_base_ev(d) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_reports.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/reports.py`:

```python
"""Read each report's base EV from a research run dir.

Reuses cli.daily_followup.parse_research (the existing, battle-tested
decision.md parser) so we have a single source of truth for the regexes. Adds
a percentage view and a scenario-weighted fallback when no explicit EV line
exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import logging

from cli.daily_followup import parse_research, Scenario

logger = logging.getLogger(__name__)


@dataclass
class BaseEV:
    ticker: str
    research_date: str
    rating: str
    reference_price: float
    ev: float | None
    scenarios: list[Scenario]
    hard_stop: float | None


def load_base_ev(run_dir: Path) -> BaseEV | None:
    try:
        parsed = parse_research(Path(run_dir))
    except OSError:
        return None
    if not parsed:
        return None
    return BaseEV(
        ticker=parsed["ticker"],
        research_date=parsed["research_date"],
        rating=parsed["rating"],
        reference_price=parsed["reference_price"],
        ev=parsed["ev"],
        scenarios=parsed["scenarios"],
        hard_stop=parsed["hard_stop"],
    )


def _scenario_weighted_target(be: BaseEV) -> float | None:
    num = den = 0.0
    for sc in be.scenarios:
        if sc.probability is None or sc.target is None:
            continue
        num += sc.probability * sc.target
        den += sc.probability
    return (num / den) if den else None


def ev_pct(be: BaseEV) -> float | None:
    """12-mo EV as a fraction of reference price. Uses the explicit EV line if
    present, else the scenario-probability-weighted target."""
    ev_abs = be.ev if be.ev is not None else _scenario_weighted_target(be)
    if ev_abs is None:
        logger.warning("ev_pct: no EV or usable scenarios for %s (%s)",
                       be.ticker, be.research_date)
        return None
    if not be.reference_price:
        return None
    return (ev_abs - be.reference_price) / be.reference_price


def latest_runs(base_dir: Path) -> dict[str, BaseEV]:
    """Newest BaseEV per ticker across all run dirs under base_dir."""
    out: dict[str, BaseEV] = {}
    for child in Path(base_dir).iterdir():
        if not child.is_dir():
            continue
        be = load_base_ev(child)
        if not be:
            continue
        prev = out.get(be.ticker)
        if prev is None or be.research_date > prev.research_date:
            out[be.ticker] = be
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_reports.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/reports.py tests/macro/test_reports.py docs/superpowers/plans/2026-06-04-macro-regime-engine.md
git commit -m "fix(macro): anchor Scenario type, guard reports against unreadable decision.md, log missing EV

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: bias — the three-layer overlay (pure)

**Files:**
- Create: `tradingagents/macro/bias.py`
- Test: `tests/macro/test_bias.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_bias.py`:

```python
import pytest

from tradingagents.macro import bias, regime as regime_mod
from tradingagents.macro.betas import Betas
from tradingagents.macro.pillars import PillarScore

pytestmark = pytest.mark.unit

_ALL = ["growth", "inflation", "liquidity", "financial_conditions",
        "risk_appetite", "positioning"]


def _regime(score_by_name):
    ps = []
    for n in _ALL:
        sc = score_by_name.get(n, 0.0)
        status = "G" if sc >= 0.2 else "R" if sc <= -0.2 else "A"
        ps.append(PillarScore(name=n, score=sc, status=status))
    return regime_mod.build(ps)


def test_expected_factor_moves_rates_rise_on_growth_inflation():
    r = _regime({"growth": 1.0, "inflation": -1.0, "liquidity": 0.0})
    # inflation pillar -1.0 == rising inflation; growth +1.0 → rates rise
    moves = bias.expected_factor_moves(r)
    assert moves["d_10y"] > 0


def test_positive_tilt_when_betas_align_with_regime():
    r = _regime({n: 0.6 for n in _ALL})           # risk-on, GO
    b = Betas("X", {"d_10y": 0, "d_dxy": 0, "d_hy_spread": 0,
                    "oil_ret": 0, "mkt": 1.5, "growth_value": 0}, 0.8, "high", 300)
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.10)
    assert sb.macro_delta_pct > 0
    assert sb.adjusted_ev_pct > 0.10
    assert sb.conviction > 0


def test_tilt_capped_at_ev_tilt_cap():
    r = _regime({n: 1.0 for n in _ALL})
    b = Betas("X", {f: 50.0 for f in ["d_10y", "d_dxy", "d_hy_spread",
              "oil_ret", "mkt", "growth_value"]}, 0.9, "high", 300)
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.10)
    assert abs(sb.macro_delta_pct) <= bias.EV_TILT_CAP + 1e-9


def test_gate_stand_down_zeroes_conviction_and_flags_action():
    r = _regime({n: -0.6 for n in _ALL})          # all red → STAND_DOWN
    b = Betas("X", {"d_10y": 0.0, "d_dxy": 0.0, "d_hy_spread": 0.0,
                    "oil_ret": 0.0, "mkt": 1.5, "growth_value": 0.0},
              0.8, "high", 300)                     # non-zero beta would normally give conviction
    sb = bias.bias_stock("X", "BUY", r, b, research_ev_pct=0.20)
    assert sb.conviction == 0.0
    assert "STAND DOWN" in sb.action.upper() or "NO NEW RISK" in sb.action.upper()


def test_driver_names_top_two_betas_by_abs():
    b = Betas("X", {"d_10y": 0.1, "d_dxy": -2.0, "d_hy_spread": 1.5,
                    "oil_ret": 0.0, "mkt": 0.2, "growth_value": 0.0}, 0.7, "high", 300)
    drv = bias.describe_driver(b)
    assert "d_dxy" in drv and "d_hy_spread" in drv


def test_caution_path_haircuts_conviction_and_labels_action():
    r = _regime({"growth": -0.5, "financial_conditions": -0.5})   # → CAUTION
    b = Betas("X", {"d_10y": 0.0, "d_dxy": 0.0, "d_hy_spread": 0.0,
                    "oil_ret": 0.0, "mkt": 0.0, "growth_value": 0.0}, 0.7, "high", 300)
    sb = bias.bias_stock("X", "HOLD", r, b, research_ev_pct=0.05)
    assert r.gate == "CAUTION"
    assert "Caution" in sb.action
    assert 0.0 < sb.conviction <= 0.5


def test_none_research_ev_yields_none_adjusted_and_bare_rating():
    r = _regime({n: 0.5 for n in _ALL})            # GO
    b = Betas("X", {"d_10y": 0.0, "d_dxy": 0.0, "d_hy_spread": 0.0,
                    "oil_ret": 0.0, "mkt": 0.0, "growth_value": 0.0}, 0.8, "high", 300)
    sb = bias.bias_stock("X", "HOLD", r, b, research_ev_pct=None)
    assert sb.adjusted_ev_pct is None
    assert sb.action == "HOLD"


def test_driver_dash_when_all_betas_zero():
    b = Betas("X", {"d_10y": 0.0, "d_dxy": 0.0, "d_hy_spread": 0.0,
                    "oil_ret": 0.0, "mkt": 0.0, "growth_value": 0.0}, 0.0, "low", 0)
    assert bias.describe_driver(b) == "—"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_bias.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/bias.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_bias.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/bias.py tests/macro/test_bias.py
git commit -m "feat(macro): three-layer EV bias (tilt/conviction/gate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: plan_writer — build payload (pure) + idempotent gog write

**Files:**
- Create: `tradingagents/macro/plan_writer.py`
- Test: `tests/macro/test_plan_writer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_plan_writer.py`:

```python
import pytest

from tradingagents.macro import plan_writer
from tradingagents.macro.bias import StockBias
from tradingagents.macro.pillars import PillarScore
from tradingagents.macro.regime import Regime

pytestmark = pytest.mark.unit


def _regime():
    ps = [PillarScore("growth", 0.5, "G"), PillarScore("inflation", 0.1, "A")]
    return Regime(score=0.3, label="Risk-On · Goldilocks", quadrant="Goldilocks",
                  gate="GO", pillars=ps, red_count=0)


def _bias(ticker="AAPL"):
    return StockBias(ticker, "BUY", "d_dxy(-1.20)", "G", 0.12, 0.03, 0.15,
                     0.8, "BUY — add/hold")


def test_build_payload_has_regime_board_and_rows():
    payload = plan_writer.build_payload(_regime(), [_bias()],
                                        pdf_links={"AAPL": "http://x/AAPL.pdf"})
    assert payload["regime"]["gate"] == "GO"
    assert payload["regime"]["quadrant"] == "Goldilocks"
    assert any(p["name"] == "growth" for p in payload["pillars"])
    row = payload["rows"][0]
    assert row["ticker"] == "AAPL"
    assert row["adjusted_ev_pct"] == 0.15
    assert row["pdf_link"] == "http://x/AAPL.pdf"


def test_rows_sorted_by_adjusted_ev_desc():
    a = StockBias("AAA", "BUY", "", "G", 0.05, 0.0, 0.05, 0.5, "")
    b = StockBias("BBB", "BUY", "", "G", 0.20, 0.0, 0.20, 0.5, "")
    payload = plan_writer.build_payload(_regime(), [a, b], pdf_links={})
    assert [r["ticker"] for r in payload["rows"]] == ["BBB", "AAA"]


def test_load_manifest_parses_tab_separated(tmp_path):
    m = tmp_path / "pdf_ids.tsv"
    m.write_text("AAPL\tfileid_aapl\nMSFT\tfileid_msft\n")
    out = plan_writer.load_manifest(m)
    assert out == {"AAPL": "fileid_aapl", "MSFT": "fileid_msft"}


def test_pdf_links_from_manifest_build_drive_urls(tmp_path):
    m = tmp_path / "pdf_ids.tsv"
    m.write_text("AAPL\tabc123\n")
    links = plan_writer.pdf_links_from_manifest(m)
    assert links["AAPL"] == "https://drive.google.com/file/d/abc123/view"


def test_to_grid_pads_to_constant_height_with_header_and_data():
    from tradingagents.macro.config import SHEET_MAX_ROWS
    payload = plan_writer.build_payload(_regime(), [_bias()],
                                        pdf_links={"AAPL": "http://x/AAPL.pdf"})
    grid = plan_writer.to_grid(payload)
    assert len(grid) == SHEET_MAX_ROWS            # constant height → overwrite covers prior runs
    header = grid[4]
    assert header[0] == "Ticker" and header[-1] == "Research"
    data_row = grid[5]
    assert data_row[0] == "AAPL"
    assert data_row[6] == "+15.0%"                # adjusted_ev_pct 0.15 formatted
    assert grid[-1] == [""] * 10                  # trailing padding row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_plan_writer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/plan_writer.py`:

```python
"""Build the Trading Plan sheet payload (pure) and write it idempotently.

The payload builder is pure and fully tested. The actual Sheets write goes
through the `gog` CLI on the mini and replaces cells in a known sheet by ID —
never name-search, per the no-duplicates rule. PDF hyperlinks are resolved
from the existing pdf_ids.tsv manifest (ticker<TAB>driveFileId).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .bias import StockBias
from .config import SHEET_MAX_ROWS
from .regime import Regime


def load_manifest(path: Path) -> dict[str, str]:
    """Parse ticker<TAB>fileId rows. Parsed in Python (never `IFS=$"\\t"` /
    `grep -P`, which are broken on macOS)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pdf_ids manifest not found: {p}")
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        ticker, file_id = line.split("\t", 1)
        out[ticker.strip()] = file_id.strip()
    return out


def pdf_links_from_manifest(path: Path) -> dict[str, str]:
    return {t: f"https://drive.google.com/file/d/{fid}/view"
            for t, fid in load_manifest(path).items()}


def build_payload(regime: Regime, biases: list[StockBias],
                  pdf_links: dict[str, str]) -> dict:
    """Pure: assemble the regime board + per-ticker rows, rows sorted by
    adjusted EV descending (best-positioned first)."""
    rows = []
    for sb in sorted(biases,
                     key=lambda b: (b.adjusted_ev_pct is None, -(b.adjusted_ev_pct or 0))):
        rows.append({
            "ticker": sb.ticker,
            "rating": sb.rating,
            "driver": sb.driver,
            "macro_bias": sb.macro_bias,
            "research_ev_pct": sb.research_ev_pct,
            "macro_delta_pct": sb.macro_delta_pct,
            "adjusted_ev_pct": sb.adjusted_ev_pct,
            "conviction": sb.conviction,
            "action": sb.action,
            "pdf_link": pdf_links.get(sb.ticker, ""),
        })
    return {
        "regime": {
            "score": regime.score, "label": regime.label,
            "quadrant": regime.quadrant, "gate": regime.gate,
            "red_count": regime.red_count,
        },
        "pillars": [{"name": p.name, "score": p.score, "status": p.status}
                    for p in regime.pillars],
        "rows": rows,
    }


def to_grid(payload: dict) -> list[list]:
    """Flatten the payload into a 2-D cell grid for a full-range overwrite
    (idempotent: same range, replaced in place — no appends, no dupes)."""
    grid: list[list] = []
    r = payload["regime"]
    grid.append(["MACRO REGIME", r["label"], "Gate:", r["gate"],
                 "Score:", r["score"], "Red pillars:", r["red_count"]])
    grid.append(["Pillar"] + [p["name"] for p in payload["pillars"]])
    grid.append(["Status"] + [f'{p["status"]} ({p["score"]:+.2f})'
                              for p in payload["pillars"]])
    grid.append([])
    grid.append(["Ticker", "Rating", "Macro Driver", "Bias", "Research EV%",
                 "Macro Δ%", "Adjusted EV%", "Conviction", "Action", "Research"])
    for row in payload["rows"]:
        grid.append([
            row["ticker"], row["rating"], row["driver"], row["macro_bias"],
            _pct(row["research_ev_pct"]), _pct(row["macro_delta_pct"]),
            _pct(row["adjusted_ev_pct"]), row["conviction"], row["action"],
            row["pdf_link"],
        ])
    while len(grid) < SHEET_MAX_ROWS:
        grid.append([""] * 10)
    return grid


def _pct(v) -> str:
    return "" if v is None else f"{v*100:+.1f}%"


def write_to_sheet(grid: list[list], sheet_id: str, tab: str = "Macro",
                   runner=subprocess.run) -> None:
    """Overwrite the tab's range with `grid` via gog (replace-in-place →
    idempotent). `runner` is injectable for tests. Requires the mini's gog
    auth (7-day token; re-auth per the update-summary skill on invalid_grant).

    Note for the implementer: the exact `gog sheets update` flags must be
    verified against the installed `gog` version on the mini
    (`gog sheets update --help`)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(grid, fh)
        payload_path = fh.name
    try:
        runner(["gog", "sheets", "update", sheet_id, "--tab", tab,
                "--range", "A1", "--values-json", payload_path], check=True)
    finally:
        os.unlink(payload_path)
```

> The `write_to_sheet` I/O path is exercised in the Task 9 smoke test with an
> injected `runner`; only the pure `build_payload`/`to_grid`/manifest functions
> are unit-tested here.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_plan_writer.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/macro/plan_writer.py tests/macro/test_plan_writer.py
git commit -m "feat(macro): Trading Plan payload builder + idempotent gog writer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: macro_daily — orchestrator + CLI entry

**Files:**
- Create: `tradingagents/macro/macro_daily.py`
- Modify: `pyproject.toml` (add `tradingmacro` console script)
- Test: `tests/macro/test_macro_daily.py`

- [ ] **Step 1: Write the failing test**

Create `tests/macro/test_macro_daily.py`:

```python
import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import macro_daily

pytestmark = pytest.mark.unit


def _series(n=400, start=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(start + 0.1 * np.arange(n), index=idx)


def test_run_assembles_payload_and_calls_writer(tmp_path, monkeypatch):
    # one fake report dir
    import json
    d = tmp_path / "2026-06-01-AAPL"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "AAPL", "trade_date": "2026-06-01"}))
    (d / "decision.md").write_text(
        "Reference price: **$100.00**\n**Rating: BUY**\nEV = **$112.00**\n")

    # stub every network boundary
    monkeypatch.setattr(macro_daily.macro_data, "load_all",
                        lambda specs, as_of: {sp.name: _series() for sp in specs})
    monkeypatch.setattr(macro_daily.macro_data, "load_series",
                        lambda spec, as_of: _series())   # factor-source fetches
    monkeypatch.setattr(macro_daily.macro_data, "load_prices",
                        lambda t, as_of, period="2y": _series())
    captured = {}
    monkeypatch.setattr(macro_daily.plan_writer, "write_to_sheet",
                        lambda grid, sheet_id, **kw: captured.update(grid=grid, sheet=sheet_id))

    payload = macro_daily.run(reports_dir=tmp_path, sheet_id="SHEET1",
                              manifest_path=None, as_of="2026-06-02", write=True)
    assert payload["regime"]["gate"] in {"GO", "CAUTION", "STAND_DOWN"}
    assert any(r["ticker"] == "AAPL" for r in payload["rows"])
    assert captured["sheet"] == "SHEET1"


def test_run_no_write_skips_writer(tmp_path, monkeypatch):
    import json
    d = tmp_path / "2026-06-01-AAPL"
    d.mkdir()
    (d / "state.json").write_text(json.dumps(
        {"company_of_interest": "AAPL", "trade_date": "2026-06-01"}))
    (d / "decision.md").write_text(
        "Reference price: **$100.00**\n**Rating: BUY**\nEV = **$112.00**\n")
    monkeypatch.setattr(macro_daily.macro_data, "load_all",
                        lambda specs, as_of: {sp.name: _series() for sp in specs})
    monkeypatch.setattr(macro_daily.macro_data, "load_series",
                        lambda spec, as_of: _series())
    monkeypatch.setattr(macro_daily.macro_data, "load_prices",
                        lambda t, as_of, period="2y": _series())
    called = {"n": 0}
    monkeypatch.setattr(macro_daily.plan_writer, "write_to_sheet",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    payload = macro_daily.run(reports_dir=tmp_path, sheet_id="S", manifest_path=None,
                              as_of="2026-06-02", write=False)
    assert any(r["ticker"] == "AAPL" for r in payload["rows"])   # non-empty run
    assert called["n"] == 0                                        # writer still skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_daily.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `tradingagents/macro/macro_daily.py`:

```python
"""Orchestrator + CLI for the daily macro regime engine.

Chains: data → pillars → regime → (per ticker) prices → betas → bias →
payload → sheet. Every network boundary lives in macro_data/plan_writer so
this module stays thin and testable with stubs.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from . import macro_data, pillars, regime as regime_mod, betas as betas_mod
from . import reports as reports_mod, bias as bias_mod, plan_writer
from .config import INDICATORS, IndicatorSpec

logger = logging.getLogger(__name__)

# Factor source tickers/series consumed by betas.build_factor_returns.
_FACTOR_SOURCES = {
    "tnx": ("yfinance", "^TNX"), "dxy": ("yfinance", "DX-Y.NYB"),
    "hy": ("fred", "BAMLH0A0HYM2"), "oil": ("yfinance", "CL=F"),
    "spy": ("yfinance", "SPY"), "iwf": ("yfinance", "IWF"),
    "iwd": ("yfinance", "IWD"),
}


def _load_factor_returns(as_of: str):
    raw = {}
    for key, (src, code) in _FACTOR_SOURCES.items():
        raw[key] = macro_data.load_series(
            IndicatorSpec(f"factor_{key}", src, code, "financial_conditions"), as_of)
    return betas_mod.build_factor_returns(raw)


def run(reports_dir, sheet_id, manifest_path, as_of=None, write=True) -> dict:
    as_of = as_of or datetime.now().strftime("%Y-%m-%d")

    # 1. Regime (stock-independent)
    series = macro_data.load_all(INDICATORS, as_of)
    pillar_scores = pillars.score_all(series)
    regime = regime_mod.build(pillar_scores)
    logger.info("regime: %s gate=%s score=%.3f", regime.label, regime.gate, regime.score)

    # 2. Per-stock overlay
    factor_returns = _load_factor_returns(as_of)
    base_evs = reports_mod.latest_runs(Path(reports_dir))
    biases = []
    for ticker, be in base_evs.items():
        try:
            px = macro_data.load_prices(ticker, as_of)
            stock_ret = px.pct_change().dropna()
            b = betas_mod.compute_betas(ticker, stock_ret, factor_returns)
        except Exception as exc:  # noqa: BLE001 — one bad ticker shouldn't sink the run
            logger.warning("betas failed for %s: %s", ticker, exc)
            b = betas_mod.Betas(ticker, {f: 0.0 for f in betas_mod.FACTORS}, 0.0, "low", 0)
        biases.append(bias_mod.bias_stock(
            ticker, be.rating, regime, b, reports_mod.ev_pct(be)))

    # 3. Payload + write
    pdf_links = plan_writer.pdf_links_from_manifest(manifest_path) if manifest_path else {}
    payload = plan_writer.build_payload(regime, biases, pdf_links)
    if write:
        plan_writer.write_to_sheet(plan_writer.to_grid(payload), sheet_id)
    return payload


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Daily macro regime engine → Trading Plan sheet")
    p.add_argument("--reports-dir", required=True, help="dir of research run dirs")
    p.add_argument("--sheet-id", required=True, help="Trading Plan Google Sheet ID")
    p.add_argument("--manifest", default=None, help="pdf_ids.tsv for PDF hyperlinks")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today, host local date)")
    p.add_argument("--no-write", action="store_true", help="compute only, don't touch the sheet")
    args = p.parse_args(argv)
    try:
        payload = run(args.reports_dir, args.sheet_id, args.manifest,
                      as_of=args.as_of, write=not args.no_write)
    except Exception:
        logger.exception("macro daily run failed")
        return 1
    print(f"Regime: {payload['regime']['label']} | gate={payload['regime']['gate']} "
          f"| {len(payload['rows'])} names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Modify `pyproject.toml` — add the console script under `[project.scripts]`:

```toml
[project.scripts]
tradingagents = "cli.main:app"
tradingresearch = "cli.research:main"
tradingmacro = "tradingagents.macro.macro_daily:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/macro/test_macro_daily.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the whole macro suite + reinstall the entry point**

Run: `.venv/bin/python -m pytest tests/macro -q -m unit --tb=line`
Expected: all macro tests PASS.
Run: `.venv/bin/pip install -e . --quiet && .venv/bin/tradingmacro --help`
Expected: argparse help prints (confirms the `tradingmacro` entry point resolves).

- [ ] **Step 6: Commit**

```bash
git add tradingagents/macro/macro_daily.py pyproject.toml tests/macro/test_macro_daily.py
git commit -m "feat(macro): daily orchestrator + tradingmacro CLI entry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Scheduling (launchd on the mini) + docs

**Files:**
- Create: `ops/com.trueknot.macrodaily.plist`
- Modify: `CLAUDE.md` (add a "Macro engine" pointer section)

- [ ] **Step 1: Create the launchd plist**

Create `ops/com.trueknot.macrodaily.plist` (loaded on `macmini-trueknot`; runs daily at 05:10 SGT = after the 16:00 ET US close):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.trueknot.macrodaily</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/trueknot/local/bin/tradingmacro</string>
    <string>--reports-dir</string>
    <string>/Users/trueknot/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/final</string>
    <string>--sheet-id</string>
    <string>REPLACE_WITH_TRADING_PLAN_SHEET_ID</string>
    <string>--manifest</string>
    <string>/Users/trueknot/gsheet-tool/pdf_ids.tsv</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>FRED_API_KEY</key><string>REPLACE_WITH_FRED_KEY</string>
    <key>MACRO_CACHE_DIR</key><string>/Users/trueknot/.cache/tradingagents-macro</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>5</integer><key>Minute</key><integer>10</integer></dict>
  <key>StandardOutPath</key><string>/Users/trueknot/.macrodaily.log</string>
  <key>StandardErrorPath</key><string>/Users/trueknot/.macrodaily.err</string>
</dict>
</plist>
```

- [ ] **Step 2: Document install steps in CLAUDE.md**

Add this section to `CLAUDE.md` (after the "Pointers" section):

```markdown
## Macro regime engine (daily)

- Package: `tradingagents/macro/` — standalone daily engine; CLI `tradingmacro`.
  Spec: `docs/superpowers/specs/2026-06-04-macro-regime-engine-design.md`.
- Manual run on the mini:
  `FRED_API_KEY=… tradingmacro --reports-dir "$TK/final" --sheet-id <id> --manifest ~/gsheet-tool/pdf_ids.tsv`
  (add `--no-write` to compute without touching the sheet).
- Scheduled via `ops/com.trueknot.macrodaily.plist` →
  `cp ops/com.trueknot.macrodaily.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.trueknot.macrodaily.plist`
  (fill in the FRED key + Trading Plan sheet ID first). Runs 05:10 SGT (post US close).
- Needs a free FRED API key (Growth/Inflation/Liquidity hard data); yfinance covers
  the market-priced pillars. Sheet write uses `gog` (7-day token; re-auth per the
  update-summary skill on invalid_grant).
```

- [ ] **Step 3: Commit**

```bash
git add ops/com.trueknot.macrodaily.plist CLAUDE.md
git commit -m "ops(macro): launchd daily schedule + CLAUDE.md pointer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Full suite green + deploy

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: the prior 216 tests + the new macro tests all PASS.

- [ ] **Step 2: Deploy to the mini (per CLAUDE.md E2E flow)**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main --quiet && .venv/bin/pip install -e . --quiet'
```

- [ ] **Step 3: One-off live dry run on the mini (no sheet write)**

Run:
```bash
ssh macmini-trueknot 'FRED_API_KEY=<key> ~/local/bin/tradingmacro \
  --reports-dir "$HOME/Library/CloudStorage/GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/final" \
  --sheet-id <TRADING_PLAN_SHEET_ID> --no-write'
```
Expected: prints `Regime: … | gate=… | N names` with no traceback (validates live yfinance + FRED fetch and the full chain end to end).

- [ ] **Step 4: Verify the FRED series codes returned data**

Run: `ssh macmini-trueknot 'ls -1 ~/.cache/tradingagents-macro/ | wc -l'`
Expected: one cache file per indicator + factor source for today's date (≈29). If any indicator is missing, check `~/.macrodaily.err` for the warning and confirm the FRED series id in `config.py`.

---

## Self-Review notes

- **Spec coverage:** Architecture/units (Tasks 1–9), six pillars + sources (Task 1 config + Task 3), scoring methodology z-score+trend (Task 3), regime/quadrant/gate (Task 4), statistical betas + shrinkage (Task 5), base-EV source reuse (Task 6), three-layer bias incl. ±15% cap + gate override (Task 7), idempotent sheet write + manifest hyperlinks (Task 8), daily schedule (Task 10), FRED prerequisite + positioning-pillar-thin (Task 1 config comment + Task 10 docs). All spec sections map to a task.
- **Positioning pillar:** shipped thin (single low-weight proxy `UMCSENT`) per the spec's v1 decision; upgrade is a post-v1 open item.
- **gog flags caveat:** `gog sheets update` flags in `write_to_sheet` are marked for verification against the installed gog version before the live write (the pure payload path is fully tested regardless).
