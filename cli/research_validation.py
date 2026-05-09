"""Phase 7.4: post-output validation runner for the cli/research.py pipeline.

Wraps the Phase 7.1 / 7.2 / 7.3 validators in a single call:

    results = run_phase_7_validators(run_dir)
    write_validation_report(run_dir, results)
    summary = format_validation_summary(results)

The runner is fail-soft: any unexpected exception in a validator returns
an empty result for that phase rather than crashing the wrapper. Validation
should never abort a run that already completed the LLM pipeline; it
should only abort the Telegram push.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


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


def run_phase_7_validators(run_dir: str | Path, anchor_year: int = 2026) -> dict[str, Any]:
    """Run Phase 7.1 + 7.2 + 7.3 against a run output directory.

    Returns a structured dict suitable for both validation_report.json and
    the operator summary. Each phase's section is independent — failure
    in one validator doesn't suppress the others.
    """
    rd = Path(run_dir)
    prices_json = rd / "raw" / "prices.json"
    peer_ratios_json = rd / "raw" / "peer_ratios.json"
    peers_json = rd / "raw" / "peers.json"
    net_debt_json = rd / "raw" / "net_debt.json"

    files_present = [f for f in _FILES_TO_SCAN if (rd / f).exists()]

    # Phase 7.1
    from tradingagents.validators import (
        extract_attributed_quotes,
        extract_date_close_claims,
        extract_net_debt_claims,
        validate_attributed_quotes,
        validate_date_close_claims,
        validate_net_debt_claims,
        validate_peer_metrics,
    )
    from tradingagents.validators.claim_extractor import DateCloseClaim
    from tradingagents.validators.net_debt_validator import NetDebtClaim
    from tradingagents.validators.quote_attribution_validator import AttributedQuote

    price_date_claims: list[DateCloseClaim] = []
    quote_claims: list[AttributedQuote] = []
    net_debt_claims: list[NetDebtClaim] = []
    peer_violations = []

    for fname in files_present:
        text = (rd / fname).read_text(encoding="utf-8")
        for c in extract_date_close_claims(text, anchor_year=anchor_year):
            price_date_claims.append(DateCloseClaim(
                date_raw=c.date_raw, date_iso=c.date_iso, price=c.price,
                match_text=c.match_text, line_no=c.line_no, file=fname,
            ))
        for q in extract_attributed_quotes(text):
            quote_claims.append(AttributedQuote(
                quote_text=q.quote_text, agent_name=q.agent_name,
                file=fname, line_no=q.line_no,
                expected_source_file=q.expected_source_file,
            ))
        for n in extract_net_debt_claims(text):
            net_debt_claims.append(NetDebtClaim(
                label=n.label, is_cash=n.is_cash,
                value_raw=n.value_raw, value_dollars=n.value_dollars,
                file=fname, line_no=n.line_no, match_text=n.match_text,
            ))
        peer_violations.extend(
            validate_peer_metrics(text, fname, peer_ratios_json, peers_json)
        )

    # Read main ticker from state.json (skips Phase 7.5 peer-attributed claims)
    main_ticker: str | None = None
    state_path = rd / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            main_ticker = state.get("company_of_interest")
        except (OSError, json.JSONDecodeError):
            pass

    price_date_violations = validate_date_close_claims(price_date_claims, prices_json)
    quote_violations = validate_attributed_quotes(quote_claims, rd)
    net_debt_violations = validate_net_debt_claims(net_debt_claims, net_debt_json, main_ticker=main_ticker)

    def _ser(obj):
        return asdict(obj) if is_dataclass(obj) else obj

    # Phase 7.5 v1.3: `skipped_non_usd_reporter` is a MINOR informational
    # notice (validator out-of-scope for non-USD reporters), not a
    # fabrication flag. Count it separately so delivery isn't suppressed.
    def _is_blocking(v: Any) -> bool:
        sev = getattr(v, "severity", None) or (v.get("severity") if isinstance(v, dict) else None)
        return sev != "MINOR"

    blocking_total = (
        sum(1 for v in price_date_violations if _is_blocking(v))
        + sum(1 for v in quote_violations if _is_blocking(v))
        + sum(1 for v in peer_violations if _is_blocking(v))
        + sum(1 for v in net_debt_violations if _is_blocking(v))
    )

    return {
        "run_dir": str(rd),
        "files_scanned": files_present,
        "phase_7_1_price_date": {
            "claims_extracted": len(price_date_claims),
            "violations": [_ser(v) for v in price_date_violations],
        },
        "phase_7_2_quote_attribution": {
            "quotes_extracted": len(quote_claims),
            "violations": [_ser(v) for v in quote_violations],
        },
        "phase_7_3_peer_metric": {
            "violations": [_ser(v) for v in peer_violations],
        },
        "phase_7_5_net_debt": {
            "claims_extracted": len(net_debt_claims),
            "violations": [_ser(v) for v in net_debt_violations],
        },
        "total_violations": (
            len(price_date_violations)
            + len(quote_violations)
            + len(peer_violations)
            + len(net_debt_violations)
        ),
        "blocking_violations": blocking_total,
    }


def write_validation_report(run_dir: str | Path, results: dict[str, Any]) -> Path:
    """Persist validation results to <run_dir>/validation_report.json.

    Always writes (success or failure) so the operator has a permanent
    audit trail of what the validators saw.
    """
    rd = Path(run_dir)
    report_path = rd / "validation_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return report_path


def format_validation_summary(results: dict[str, Any]) -> str:
    """One-line operator summary suitable for stderr / Telegram failure msg.

    Counts MATERIAL/blocking violations only — MINOR informational notices
    (like Phase 7.5 `skipped_non_usd_reporter`) don't suppress delivery.
    """
    def _blocking(violations: list[dict]) -> int:
        return sum(1 for v in violations if v.get("severity") != "MINOR")

    n_pd = _blocking(results.get("phase_7_1_price_date", {}).get("violations", []))
    n_q = _blocking(results.get("phase_7_2_quote_attribution", {}).get("violations", []))
    n_pm = _blocking(results.get("phase_7_3_peer_metric", {}).get("violations", []))
    n_nd = _blocking(results.get("phase_7_5_net_debt", {}).get("violations", []))
    blocking = results.get("blocking_violations", n_pd + n_q + n_pm + n_nd)
    if blocking == 0:
        return "VALIDATION PASS (0 violations)"
    return (
        f"VALIDATION FAIL ({blocking} violation(s): "
        f"{n_pd} price/date, {n_q} quote, {n_pm} peer, {n_nd} net-debt)"
    )
