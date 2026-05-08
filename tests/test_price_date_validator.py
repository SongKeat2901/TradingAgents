"""Tests for the Phase-7.1 price/date validator."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# COIN-style OHLCV with the most-recent close at 2026-05-07 / $192.96
_COIN_PRICES_JSON = {
    "ohlcv": (
        "# Stock data for COIN\n"
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2026-05-04,199.41,206.71,197.85,202.99,11243400,0.0,0.0\n"
        "2026-05-05,208.88,208.88,194.4,197.75,10074200,0.0,0.0\n"
        "2026-05-06,195.78,198.5,193.25,197.96,7764900,0.0,0.0\n"
        "2026-05-07,196.24,198.15,190.32,192.96,8641932,0.0,0.0\n"
    )
}


def _write_prices(tmp_path, data=_COIN_PRICES_JSON):
    p = tmp_path / "raw" / "prices.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_passes_when_all_claims_match_actual_close(tmp_path):
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    prices = _write_prices(tmp_path)

    text = (
        "On 2026-05-05 COIN closed at $197.75. "
        "On 2026-05-07 the close was $192.96."
    )
    claims = extract_date_close_claims(text)
    violations = validate_date_close_claims(claims, prices)
    assert violations == []


def test_catches_coin_2026_05_08_fabrication(tmp_path):
    """The exact failure mode: claims a close for a date later than the
    latest indexed session in raw/prices.json."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    prices = _write_prices(tmp_path)

    text = (
        "the May 8 session subsequently closed at $206.50 per prices.json"
    )
    claims = extract_date_close_claims(text, anchor_year=2026)
    violations = validate_date_close_claims(claims, prices)

    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "MATERIAL"
    assert v.type == "fabricated_future_close"
    assert v.claimed_date == "2026-05-08"
    assert v.claimed_price == 206.50
    assert v.latest_indexed_date == "2026-05-07"


def test_catches_wrong_close_within_indexed_window(tmp_path):
    """Date IS in prices.json but the cited dollar amount differs >$0.50
    from the actual close — cell-match drift, MATERIAL violation."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    prices = _write_prices(tmp_path)

    # Actual 2026-05-07 close is $192.96; claim says $200.00 → $7.04 delta
    text = "On 2026-05-07 COIN closed at $200.00 — first volume confirmed."
    claims = extract_date_close_claims(text)
    violations = validate_date_close_claims(claims, prices)

    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "MATERIAL"
    assert v.type == "wrong_close"
    assert v.claimed_price == 200.00
    assert v.actual_close == 192.96
    assert abs(v.delta - 7.04) < 0.01


def test_within_tolerance_passes(tmp_path):
    """Claim within $0.50 of actual close — rounding drift, not fabrication."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    prices = _write_prices(tmp_path)

    text = "On 2026-05-07 close was $192.50."  # 46¢ off from actual $192.96
    claims = extract_date_close_claims(text)
    violations = validate_date_close_claims(claims, prices)
    assert violations == []  # within tolerance


def test_emits_no_prices_data_when_prices_json_missing(tmp_path):
    """When prices.json is missing, return a single informational violation
    rather than silently passing all claims."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    # No prices.json written

    text = "On 2026-05-08 COIN closed at $206.50"
    claims = extract_date_close_claims(text)
    # Manually set the file field for the violation report
    claims = [
        type(c)(date_raw=c.date_raw, date_iso=c.date_iso, price=c.price,
               match_text=c.match_text, line_no=c.line_no, file="decision.md")
        for c in claims
    ]
    violations = validate_date_close_claims(claims, tmp_path / "raw" / "prices.json")
    assert len(violations) == 1
    assert violations[0].type == "no_prices_data"
    assert violations[0].severity == "MINOR"


def test_skips_dates_before_window(tmp_path):
    """Claims older than the earliest indexed date are out-of-scope; skip
    silently. (Historical price references are common and not fabrication.)"""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    prices = _write_prices(tmp_path)

    # Window is 2026-05-04 to 2026-05-07; this is older
    text = "On 2026-04-29 closed at $181.73 — that bottom held."
    claims = extract_date_close_claims(text)
    violations = validate_date_close_claims(claims, prices)
    assert violations == []  # silently ignored, not validated


def test_skips_unresolvable_dates(tmp_path):
    """date_iso None → skip. Don't crash, don't emit false violations."""
    from tradingagents.validators import validate_date_close_claims
    from tradingagents.validators.claim_extractor import DateCloseClaim
    prices = _write_prices(tmp_path)

    claims = [DateCloseClaim(
        date_raw="last Tuesday",
        date_iso=None,
        price=200.00,
        match_text="last Tuesday close $200",
        line_no=1,
        file="decision.md",
    )]
    violations = validate_date_close_claims(claims, prices)
    assert violations == []


def test_render_violations_text_pass_message():
    from tradingagents.validators.price_date_validator import render_violations_text
    out = render_violations_text([])
    assert "VALIDATION PASS" in out
    assert "0 violations" in out


def test_render_violations_text_includes_actionable_detail():
    """Violation report must show claimed date, claimed price, and either
    the latest indexed date (forward-projection) or actual close (drift)."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    from tradingagents.validators.price_date_validator import render_violations_text
    from tradingagents.validators.claim_extractor import DateCloseClaim

    # Need a fresh temp prices.json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "raw").mkdir()
        (td_path / "raw" / "prices.json").write_text(
            json.dumps(_COIN_PRICES_JSON), encoding="utf-8"
        )

        text = "the May 8 session subsequently closed at $206.50"
        claims = extract_date_close_claims(text, anchor_year=2026)
        # Set file field
        claims = [DateCloseClaim(
            date_raw=c.date_raw, date_iso=c.date_iso, price=c.price,
            match_text=c.match_text, line_no=c.line_no, file="decision.md",
        ) for c in claims]
        violations = validate_date_close_claims(claims, td_path / "raw" / "prices.json")

        out = render_violations_text(violations)
        assert "VALIDATION FAIL" in out
        assert "MATERIAL" in out
        assert "fabricated_future_close" in out
        assert "2026-05-08" in out
        assert "$206.50" in out
        assert "2026-05-07" in out  # latest indexed
        assert "decision.md" in out


def test_full_pipeline_against_coin_decision_excerpt(tmp_path):
    """Integration: feed the actual COIN 2026-05-08 decision.md prose through
    extract → validate → confirm we'd flag the $206.50 fabrication."""
    from tradingagents.validators import (
        extract_date_close_claims,
        validate_date_close_claims,
    )
    from tradingagents.validators.claim_extractor import DateCloseClaim
    prices = _write_prices(tmp_path)

    # Verbatim from the failed COIN 2026-05-08 decision.md (lines 7, 17, 24)
    text = (
        "- **Reference price:** $192.96 (yfinance close on or before "
        "2026-05-08; canonical reference snapshot). The May 8 session "
        "subsequently closed at $206.50 per prices.json.\n"
        "\n"
        "  - \"Tie trim sizing to the *first confirmed negative print*, not "
        "the *absence of a positive print*\" → the May 8 failed breakout "
        "(intraday $210.47, close $206.50 on 14.39M shares).\n"
        "\n"
        "- **Data freshness:** 10-Q for Q1 2026 filed 2026-05-07; "
        "technicals through the 2026-05-08 close (per prices.json: open "
        "$205.31, high $210.47, low $202.81, close $206.50, volume "
        "14,390,000); news through 2026-05-07.\n"
    )
    claims = extract_date_close_claims(text, anchor_year=2026)
    # Set file
    claims = [DateCloseClaim(
        date_raw=c.date_raw, date_iso=c.date_iso, price=c.price,
        match_text=c.match_text, line_no=c.line_no, file="decision.md",
    ) for c in claims]
    violations = validate_date_close_claims(claims, prices)

    # We expect at least one MATERIAL fabricated_future_close violation
    fabrications = [v for v in violations
                    if v.type == "fabricated_future_close"
                    and v.claimed_date == "2026-05-08"
                    and v.claimed_price == 206.50]
    assert len(fabrications) >= 1, (
        f"validator must catch the COIN 2026-05-08 $206.50 fabrication; "
        f"got violations: {violations}"
    )
