"""CLI: validate a research run output for fabricated date+close claims.

Usage:
    python -m cli.validate_research <run_dir>

    # Strict mode: exit non-zero if any violation found (for CI gating)
    python -m cli.validate_research <run_dir> --strict

    # JSON output for downstream tooling
    python -m cli.validate_research <run_dir> --json

The validator scans these files (when present) for `<DATE> close $X.XX`
patterns:

    decision.md
    decision_executive.md
    debate_bull_bear.md
    debate_risk.md
    raw/technicals.md
    raw/technicals_v2.md
    analyst_market.md
    analyst_news.md
    analyst_social.md
    analyst_fundamentals.md

Each claim is validated against raw/prices.json:
- date later than latest indexed session → MATERIAL fabrication
- date matches a row but price differs > $0.50 → MATERIAL drift
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tradingagents.validators import (
    extract_date_close_claims,
    validate_date_close_claims,
)
from tradingagents.validators.price_date_validator import render_violations_text


_FILES_TO_SCAN = (
    "decision.md",
    "decision_executive.md",
    "debate_bull_bear.md",
    "debate_risk.md",
    "raw/technicals.md",
    "raw/technicals_v2.md",
    "analyst_market.md",
    "analyst_news.md",
    "analyst_social.md",
    "analyst_fundamentals.md",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_research",
        description="Validate a research run output for fabricated price/date claims.",
    )
    parser.add_argument("run_dir", help="path to research run output directory")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any violation is found (for CI / Telegram gate)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human-readable text",
    )
    parser.add_argument(
        "--anchor-year",
        type=int,
        default=2026,
        help="resolve `Month DAY` strings using this year (default 2026)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    prices_json = run_dir / "raw" / "prices.json"

    all_claims = []
    for fname in _FILES_TO_SCAN:
        path = run_dir / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in extract_date_close_claims(text, anchor_year=args.anchor_year):
            # Convert frozen dataclass to a mutable copy with `file` set
            from tradingagents.validators.claim_extractor import DateCloseClaim
            all_claims.append(DateCloseClaim(
                date_raw=claim.date_raw,
                date_iso=claim.date_iso,
                price=claim.price,
                match_text=claim.match_text,
                line_no=claim.line_no,
                file=fname,
            ))

    violations = validate_date_close_claims(all_claims, prices_json)

    if args.json:
        out = {
            "run_dir": str(run_dir),
            "files_scanned": [
                f for f in _FILES_TO_SCAN if (run_dir / f).exists()
            ],
            "claims_extracted": len(all_claims),
            "violations": [asdict(v) for v in violations],
        }
        print(json.dumps(out, indent=2))
    else:
        print(render_violations_text(violations))
        if all_claims and not violations:
            print(f"\n({len(all_claims)} claims extracted, all verified)")

    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
