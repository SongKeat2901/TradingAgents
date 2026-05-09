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
    extract_attributed_quotes,
    extract_date_close_claims,
    extract_net_debt_claims,
    validate_attributed_quotes,
    validate_date_close_claims,
    validate_net_debt_claims,
    validate_peer_metrics,
)
from tradingagents.validators.net_debt_validator import render_net_debt_violations_text
from tradingagents.validators.peer_metric_validator import render_peer_violations_text
from tradingagents.validators.price_date_validator import render_violations_text
from tradingagents.validators.quote_attribution_validator import render_quote_violations_text


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
    peer_ratios_json = run_dir / "raw" / "peer_ratios.json"
    peers_json = run_dir / "raw" / "peers.json"
    net_debt_json = run_dir / "raw" / "net_debt.json"

    # ----- Phase 7.1: price/date validator -----
    all_claims = []
    for fname in _FILES_TO_SCAN:
        path = run_dir / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in extract_date_close_claims(text, anchor_year=args.anchor_year):
            from tradingagents.validators.claim_extractor import DateCloseClaim
            all_claims.append(DateCloseClaim(
                date_raw=claim.date_raw,
                date_iso=claim.date_iso,
                price=claim.price,
                match_text=claim.match_text,
                line_no=claim.line_no,
                file=fname,
            ))

    price_date_violations = validate_date_close_claims(all_claims, prices_json)

    # ----- Phase 7.2: quote attribution validator -----
    all_quotes = []
    for fname in _FILES_TO_SCAN:
        path = run_dir / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for q in extract_attributed_quotes(text):
            from tradingagents.validators.quote_attribution_validator import AttributedQuote
            all_quotes.append(AttributedQuote(
                quote_text=q.quote_text,
                agent_name=q.agent_name,
                file=fname,
                line_no=q.line_no,
                expected_source_file=q.expected_source_file,
            ))
    quote_violations = validate_attributed_quotes(all_quotes, run_dir)

    # ----- Phase 7.3: peer-metric validator -----
    peer_violations = []
    for fname in _FILES_TO_SCAN:
        path = run_dir / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        peer_violations.extend(validate_peer_metrics(
            text, fname, peer_ratios_json, peers_json,
        ))

    # ----- Phase 7.5: net-debt definitional consistency validator -----
    all_net_debt_claims = []
    for fname in _FILES_TO_SCAN:
        path = run_dir / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for n in extract_net_debt_claims(text):
            from tradingagents.validators.net_debt_validator import NetDebtClaim
            all_net_debt_claims.append(NetDebtClaim(
                label=n.label, is_cash=n.is_cash,
                value_raw=n.value_raw, value_dollars=n.value_dollars,
                file=fname, line_no=n.line_no, match_text=n.match_text,
            ))
    # Resolve main ticker from state.json so Phase 7.5 skips peer-attributed claims
    main_ticker: str | None = None
    state_path = run_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            main_ticker = state.get("company_of_interest")
        except (OSError, json.JSONDecodeError):
            pass
    net_debt_violations = validate_net_debt_claims(
        all_net_debt_claims, net_debt_json, main_ticker=main_ticker,
    )

    total_violations = (
        len(price_date_violations) + len(quote_violations)
        + len(peer_violations) + len(net_debt_violations)
    )
    # Phase 7.5 v1.3: count MATERIAL/blocking violations separately.
    # MINOR notices (skipped_non_usd_reporter) are informational and must
    # not suppress delivery; --strict gates on blocking only.
    def _blocking(violations: list) -> int:
        return sum(1 for v in violations if getattr(v, "severity", None) != "MINOR")
    blocking_violations = (
        _blocking(price_date_violations) + _blocking(quote_violations)
        + _blocking(peer_violations) + _blocking(net_debt_violations)
    )

    if args.json:
        out = {
            "run_dir": str(run_dir),
            "files_scanned": [
                f for f in _FILES_TO_SCAN if (run_dir / f).exists()
            ],
            "phase_7_1_price_date": {
                "claims_extracted": len(all_claims),
                "violations": [asdict(v) for v in price_date_violations],
            },
            "phase_7_2_quote_attribution": {
                "quotes_extracted": len(all_quotes),
                "violations": [asdict(v) for v in quote_violations],
            },
            "phase_7_3_peer_metric": {
                "violations": [asdict(v) for v in peer_violations],
            },
            "phase_7_5_net_debt": {
                "claims_extracted": len(all_net_debt_claims),
                "violations": [asdict(v) for v in net_debt_violations],
            },
            "total_violations": total_violations,
        }
        print(json.dumps(out, indent=2))
    else:
        print("=" * 70)
        print("PHASE 7.1 — Price / date claims")
        print("=" * 70)
        print(render_violations_text(price_date_violations))
        if all_claims and not price_date_violations:
            print(f"\n({len(all_claims)} date+close claims extracted, all verified)")

        print()
        print("=" * 70)
        print("PHASE 7.2 — Quote attribution")
        print("=" * 70)
        print(render_quote_violations_text(quote_violations))
        if all_quotes and not quote_violations:
            print(f"\n({len(all_quotes)} attributed quotes extracted, all verified)")

        print()
        print("=" * 70)
        print("PHASE 7.3 — Peer metrics")
        print("=" * 70)
        print(render_peer_violations_text(peer_violations))

        print()
        print("=" * 70)
        print("PHASE 7.5 — Net-debt definitional consistency")
        print("=" * 70)
        print(render_net_debt_violations_text(net_debt_violations))
        if all_net_debt_claims and not net_debt_violations:
            print(f"\n({len(all_net_debt_claims)} net-debt/cash claims extracted, all verified)")

        print()
        print("=" * 70)
        if blocking_violations == 0:
            if total_violations == 0:
                print(f"OVERALL: VALIDATION PASS (0 violations across 4 validators)")
            else:
                # MINOR-only — pass with notice
                print(
                    f"OVERALL: VALIDATION PASS "
                    f"({total_violations - blocking_violations} MINOR notice(s); 0 blocking)"
                )
        else:
            print(f"OVERALL: VALIDATION FAIL ({blocking_violations} blocking violations: "
                  f"{_blocking(price_date_violations)} price/date, "
                  f"{_blocking(quote_violations)} quote, "
                  f"{_blocking(peer_violations)} peer, "
                  f"{_blocking(net_debt_violations)} net-debt)")

    if args.strict and blocking_violations > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
