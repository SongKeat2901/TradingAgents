"""Validate `<DATE> close $X.XX` claims against raw/prices.json.

Two violation types:

1. **Forward-projection fabrication** (the COIN 2026-05-08 case): the
   claim cites a date later than the latest indexed session in
   raw/prices.json. yfinance hasn't seen that close yet, so any claimed
   price is invented. Severity: MATERIAL.

2. **Wrong close** (cell-match drift): the claim cites a date that IS in
   raw/prices.json, but the dollar amount doesn't match the actual close
   within the tolerance. Severity depends on the magnitude of the delta.

The validator is deterministic: same input → same violations. It runs
in milliseconds against any LLM-authored markdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tradingagents.validators.claim_extractor import DateCloseClaim


# Tolerance for cell-match validation. yfinance close prices are reported
# to 2 decimal places; if a claim is within $0.50 of the actual close,
# treat it as a rounding-fidelity issue, not fabrication. Anything larger
# is structurally wrong.
_CLOSE_MATCH_TOLERANCE_USD = 0.50


@dataclass(frozen=True)
class Violation:
    """A claim that failed verification against raw/prices.json."""

    severity: Literal["MATERIAL", "MINOR"]
    type: Literal["fabricated_future_close", "wrong_close", "no_prices_data"]
    file: str | None
    line_no: int
    match_text: str
    claimed_date: str | None
    claimed_price: float
    # Populated when known (not always for `no_prices_data` violations)
    actual_close: float | None = None
    delta: float | None = None
    latest_indexed_date: str | None = None


def _parse_prices_json(prices_json_path: Path) -> dict[str, float]:
    """Load raw/prices.json and return {date: close} for every row.

    Returns empty dict on any parse error — caller decides how to surface.
    """
    if not prices_json_path.exists():
        return {}
    try:
        d = json.loads(prices_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    ohlcv = d.get("ohlcv", "") if isinstance(d, dict) else ""
    if not isinstance(ohlcv, str):
        return {}

    by_date: dict[str, float] = {}
    for line in ohlcv.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            by_date[parts[0].strip()] = float(parts[4])
        except ValueError:
            continue
    return by_date


def validate_date_close_claims(
    claims: list[DateCloseClaim],
    prices_json_path: Path,
) -> list[Violation]:
    """Verify each `<date> close $X.XX` claim against raw/prices.json.

    Returns a list of violations (empty if all claims verify or are
    unverifiable due to date-resolution ambiguity).

    Three logical paths per claim:
    - `date_iso` couldn't be resolved → skip (not a violation; ambiguous date)
    - `date_iso > latest_indexed_date` → MATERIAL `fabricated_future_close`
    - `date_iso` is in by_date and price within tolerance → no violation
    - `date_iso` is in by_date and price differs > tolerance → MATERIAL `wrong_close`
    - `date_iso` is BEFORE the earliest indexed date → skip (out of scope; the
      OHLCV window is bounded; older claims are common and not load-bearing)
    """
    by_date = _parse_prices_json(prices_json_path)

    if not by_date:
        # No price data → can't validate anything; return a single
        # informational violation so the caller knows verification was
        # impossible (rather than falsely passing all claims).
        if claims:
            return [Violation(
                severity="MINOR",
                type="no_prices_data",
                file=claims[0].file,
                line_no=0,
                match_text=f"{prices_json_path} missing or unparseable",
                claimed_date=None,
                claimed_price=0.0,
            )]
        return []

    latest_indexed = max(by_date.keys())
    earliest_indexed = min(by_date.keys())

    violations: list[Violation] = []
    for claim in claims:
        if claim.date_iso is None:
            # Date string couldn't be resolved (rare; e.g., "Q1 close" without
            # a specific date). Skip — not actionable as a fabrication signal.
            continue

        if claim.date_iso > latest_indexed:
            violations.append(Violation(
                severity="MATERIAL",
                type="fabricated_future_close",
                file=claim.file,
                line_no=claim.line_no,
                match_text=claim.match_text,
                claimed_date=claim.date_iso,
                claimed_price=claim.price,
                latest_indexed_date=latest_indexed,
            ))
            continue

        if claim.date_iso < earliest_indexed:
            # Older than our window — not actionable. Skip silently.
            continue

        if claim.date_iso in by_date:
            actual = by_date[claim.date_iso]
            delta = abs(claim.price - actual)
            if delta > _CLOSE_MATCH_TOLERANCE_USD:
                violations.append(Violation(
                    severity="MATERIAL",
                    type="wrong_close",
                    file=claim.file,
                    line_no=claim.line_no,
                    match_text=claim.match_text,
                    claimed_date=claim.date_iso,
                    claimed_price=claim.price,
                    actual_close=actual,
                    delta=delta,
                ))
        # else: date is within the window but the specific date isn't a
        # trading day (weekend/holiday). Skip — ambiguous what the LLM meant.

    return violations


def render_violations_text(violations: list[Violation]) -> str:
    """Human-readable violation report for terminal / CI output."""
    if not violations:
        return "VALIDATION PASS: 0 violations"

    lines = [f"VALIDATION FAIL: {len(violations)} violation(s)"]
    for v in violations:
        loc = f"{v.file or '?'}:{v.line_no}" if v.line_no else (v.file or "?")
        lines.append(f"  [{v.severity}] {loc}  {v.type}")
        if v.type == "fabricated_future_close":
            lines.append(f"    claimed: {v.claimed_date} close ${v.claimed_price:,.2f}")
            lines.append(f"    latest indexed in prices.json: {v.latest_indexed_date}")
        elif v.type == "wrong_close" and v.actual_close is not None:
            lines.append(
                f"    claimed: {v.claimed_date} close ${v.claimed_price:,.2f} → "
                f"actual ${v.actual_close:,.2f} (delta ${v.delta:,.2f})"
            )
        elif v.type == "no_prices_data":
            lines.append(f"    {v.match_text}")
        lines.append(f"    text: {v.match_text[:120]}")
    return "\n".join(lines)
