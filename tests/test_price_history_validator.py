"""Tests for the price-history sufficiency validator (wk29 SATS silent-corruption class).

When a ticker is renamed/delisted, yfinance returns a stub (SATS 2026-07-17
returned a single session for a 5-year request). stockstats then computes
DEGENERATE moving averages from that one bar (the "50-DMA" equals the single
close, gap 0.0%), and the classifier emits a confident-but-wrong setup class
with no violation raised. This validator flags the grossly-insufficient
history as a BLOCKING violation so promotion cannot pick up the corrupt run.
"""
import json

import pytest

pytestmark = pytest.mark.unit


def _write_prices(tmp_path, n_bars, start_price=90.0):
    """Write raw/prices.json with n_bars daily rows (ohlcv CSV-string shape)."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lines = ["# Stock data", "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"]
    for i in range(n_bars):
        d = f"2026-07-{(i % 28) + 1:02d}"
        p = start_price + i
        lines.append(f"{d},{p},{p + 1},{p - 1},{p},1000000,0.0,0.0")
    p = raw / "prices.json"
    p.write_text(json.dumps({"ohlcv": "\n".join(lines) + "\n"}), encoding="utf-8")
    return p


def test_flags_single_bar_stub_as_blocking(tmp_path):
    """SATS 2026-07-17: 1 bar returned for a multi-year request → BLOCKING."""
    from tradingagents.validators import validate_price_history
    prices = _write_prices(tmp_path, n_bars=1)
    viols = validate_price_history(prices, ticker="SATS")
    assert len(viols) == 1
    v = viols[0]
    assert v.severity == "MATERIAL"          # blocking (not MINOR)
    assert v.type == "insufficient_price_history"
    assert v.bars == 1


def test_passes_full_history(tmp_path):
    """A normal ~5y run (1255 bars) raises nothing."""
    from tradingagents.validators import validate_price_history
    prices = _write_prices(tmp_path, n_bars=1255)
    assert validate_price_history(prices, ticker="ECHO") == []


def test_passes_at_floor_boundary(tmp_path):
    """Exactly at the 20-bar floor is acceptable (only < floor flags)."""
    from tradingagents.validators import validate_price_history
    prices = _write_prices(tmp_path, n_bars=20)
    assert validate_price_history(prices, ticker="X") == []


def test_flags_just_below_floor(tmp_path):
    """19 bars (< floor) → BLOCKING."""
    from tradingagents.validators import validate_price_history
    prices = _write_prices(tmp_path, n_bars=19)
    viols = validate_price_history(prices, ticker="X")
    assert len(viols) == 1 and viols[0].bars == 19


def test_missing_prices_file_is_silent(tmp_path):
    """No prices.json → no violation (a different phase owns that)."""
    from tradingagents.validators import validate_price_history
    assert validate_price_history(tmp_path / "raw" / "prices.json", ticker="X") == []


def _min_run_dir(tmp_path, n_bars):
    """Minimal run dir: raw/prices.json + state.json (ticker)."""
    _write_prices(tmp_path, n_bars=n_bars)
    (tmp_path / "state.json").write_text(
        json.dumps({"company_of_interest": "SATS"}), encoding="utf-8")
    return tmp_path


def test_wired_report_counts_stub_as_blocking(tmp_path):
    """End-to-end: a 1-bar run must surface in run_phase_7_validators as a
    blocking violation, not a silent 0/0."""
    from cli.research_validation import run_phase_7_validators
    _min_run_dir(tmp_path, n_bars=1)
    report = run_phase_7_validators(tmp_path)
    assert report["blocking_violations"] >= 1
    ph = report["phase_7_6_price_history"]["violations"]
    assert len(ph) == 1 and ph[0]["type"] == "insufficient_price_history"


def test_wired_report_clean_on_full_history(tmp_path):
    """A full-history run adds nothing to the blocking count from this phase."""
    from cli.research_validation import run_phase_7_validators
    _min_run_dir(tmp_path, n_bars=1255)
    report = run_phase_7_validators(tmp_path)
    assert report["phase_7_6_price_history"]["violations"] == []
