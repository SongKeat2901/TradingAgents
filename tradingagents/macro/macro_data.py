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
    return pd.Series(vals, index=pd.to_datetime(dates), name="value")


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
    df = yf.Ticker(spec.code).history(period="3y", auto_adjust=False)
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
    df = df.reset_index().rename(columns={"index": "Date"})
    df = drop_incomplete_session(df)          # drop the in-progress US bar
    s = pd.Series(df["Close"].values,
                  index=pd.to_datetime(df["Date"]).dt.tz_localize(None), name="value")
    s.to_frame("value").to_csv(cache_path)
    return s
