"""CLI: validate a research run output for fabricated date+close claims.

Usage:
    python -m cli.validate_research <run_dir>

    # Strict mode: exit non-zero if any blocking violation found
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

Phase 7.9: this CLI is now a thin wrapper around the shared
`_collect_violations` helper in `cli/research_validation`. Both the CLI
and the daemon now use the same orchestration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cli.research_validation import (
    _collect_violations,
    format_validation_full_text,
    run_phase_7_validators,
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
        help="exit non-zero if any blocking violation is found (for CI / Telegram gate)",
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

    if args.json:
        results = run_phase_7_validators(run_dir, anchor_year=args.anchor_year)
        print(json.dumps(results, indent=2, default=str))
        blocking = results.get("blocking_violations", 0)
    else:
        raw = _collect_violations(run_dir, anchor_year=args.anchor_year)
        print(format_validation_full_text(raw))
        # Compute blocking count for --strict exit code
        blocking = sum(
            1 for vs in (
                raw["price_date_violations"],
                raw["quote_violations"],
                raw["peer_violations"],
                raw["net_debt_violations"],
            )
            for v in vs
            if getattr(v, "severity", None) != "MINOR"
        )

    if args.strict and blocking > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
