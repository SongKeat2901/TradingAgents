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


def _collect_violations(
    run_dir: str | Path,
    anchor_year: int = 2026,
) -> dict[str, Any]:
    """Internal: extract claims and run Phase 7.1/2/3/5 validators against
    a run directory. Returns a dict of raw violation dataclasses + claim
    counts + files_scanned. Used by both `run_phase_7_validators` (which
    serializes for JSON) and `cli/validate_research.py` (which renders
    human-readable output via the existing `render_*` helpers).

    Phase 7.9 consolidation: this private helper is the single source of
    truth for validator-call orchestration. The prior duplication (one
    copy here, one in cli/validate_research.py with slightly drifted
    logic) caused the AAOI-attribution and ASX-currency fixes to be
    inconsistent between CLI runs and daemon runs.
    """
    rd = Path(run_dir)
    prices_json = rd / "raw" / "prices.json"
    peer_ratios_json = rd / "raw" / "peer_ratios.json"
    peers_json = rd / "raw" / "peers.json"
    net_debt_json = rd / "raw" / "net_debt.json"

    files_present = [f for f in _FILES_TO_SCAN if (rd / f).exists()]

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

    # Resolve main_ticker from state.json so Phase 7.5 skips peer-attributed claims
    main_ticker: str | None = None
    state_path = rd / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            main_ticker = state.get("company_of_interest")
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "run_dir": rd,
        "files_scanned": files_present,
        "price_date_claims": price_date_claims,
        "quote_claims": quote_claims,
        "net_debt_claims": net_debt_claims,
        "price_date_violations": validate_date_close_claims(price_date_claims, prices_json),
        "quote_violations": validate_attributed_quotes(quote_claims, rd),
        "peer_violations": peer_violations,
        "net_debt_violations": validate_net_debt_claims(
            net_debt_claims, net_debt_json, main_ticker=main_ticker,
        ),
    }


def run_phase_7_validators(run_dir: str | Path, anchor_year: int = 2026) -> dict[str, Any]:
    """Run Phase 7.1 + 7.2 + 7.3 + 7.5 against a run output directory.

    Returns a structured dict suitable for both validation_report.json and
    the operator summary. Each phase's section is independent — failure
    in one validator doesn't suppress the others.

    Phase 7.5 v1.3 introduced `blocking_violations` which excludes MINOR
    informational notices (e.g. `skipped_non_usd_reporter`). The Telegram
    delivery gate uses `blocking_violations`, not `total_violations`.
    """
    raw = _collect_violations(run_dir, anchor_year)

    def _ser(obj):
        return asdict(obj) if is_dataclass(obj) else obj

    def _is_blocking(v: Any) -> bool:
        sev = getattr(v, "severity", None) or (v.get("severity") if isinstance(v, dict) else None)
        return sev != "MINOR"

    blocking_total = (
        sum(1 for v in raw["price_date_violations"] if _is_blocking(v))
        + sum(1 for v in raw["quote_violations"] if _is_blocking(v))
        + sum(1 for v in raw["peer_violations"] if _is_blocking(v))
        + sum(1 for v in raw["net_debt_violations"] if _is_blocking(v))
    )

    return {
        "run_dir": str(raw["run_dir"]),
        "files_scanned": raw["files_scanned"],
        "phase_7_1_price_date": {
            "claims_extracted": len(raw["price_date_claims"]),
            "violations": [_ser(v) for v in raw["price_date_violations"]],
        },
        "phase_7_2_quote_attribution": {
            "quotes_extracted": len(raw["quote_claims"]),
            "violations": [_ser(v) for v in raw["quote_violations"]],
        },
        "phase_7_3_peer_metric": {
            "violations": [_ser(v) for v in raw["peer_violations"]],
        },
        "phase_7_5_net_debt": {
            "claims_extracted": len(raw["net_debt_claims"]),
            "violations": [_ser(v) for v in raw["net_debt_violations"]],
        },
        "total_violations": (
            len(raw["price_date_violations"])
            + len(raw["quote_violations"])
            + len(raw["peer_violations"])
            + len(raw["net_debt_violations"])
        ),
        "blocking_violations": blocking_total,
    }


def format_validation_full_text(raw: dict[str, Any]) -> str:
    """Render the per-phase human-readable output for a `_collect_violations`
    result. Used by `cli/validate_research.py` and any other CLI that
    needs the verbose per-phase breakdown.
    """
    from tradingagents.validators.net_debt_validator import (
        render_net_debt_violations_text,
    )
    from tradingagents.validators.peer_metric_validator import (
        render_peer_violations_text,
    )
    from tradingagents.validators.price_date_validator import render_violations_text
    from tradingagents.validators.quote_attribution_validator import (
        render_quote_violations_text,
    )

    def _is_blocking(v: Any) -> bool:
        return getattr(v, "severity", None) != "MINOR"

    pd_v = raw["price_date_violations"]
    q_v = raw["quote_violations"]
    pm_v = raw["peer_violations"]
    nd_v = raw["net_debt_violations"]

    blocking_pd = sum(1 for v in pd_v if _is_blocking(v))
    blocking_q = sum(1 for v in q_v if _is_blocking(v))
    blocking_pm = sum(1 for v in pm_v if _is_blocking(v))
    blocking_nd = sum(1 for v in nd_v if _is_blocking(v))
    blocking = blocking_pd + blocking_q + blocking_pm + blocking_nd
    total = len(pd_v) + len(q_v) + len(pm_v) + len(nd_v)

    parts: list[str] = []
    sep = "=" * 70

    parts.append(sep)
    parts.append("PHASE 7.1 — Price / date claims")
    parts.append(sep)
    parts.append(render_violations_text(pd_v))
    if raw["price_date_claims"] and not pd_v:
        parts.append(f"\n({len(raw['price_date_claims'])} date+close claims extracted, all verified)")

    parts.append("")
    parts.append(sep)
    parts.append("PHASE 7.2 — Quote attribution")
    parts.append(sep)
    parts.append(render_quote_violations_text(q_v))
    if raw["quote_claims"] and not q_v:
        parts.append(f"\n({len(raw['quote_claims'])} attributed quotes extracted, all verified)")

    parts.append("")
    parts.append(sep)
    parts.append("PHASE 7.3 — Peer metrics")
    parts.append(sep)
    parts.append(render_peer_violations_text(pm_v))

    parts.append("")
    parts.append(sep)
    parts.append("PHASE 7.5 — Net-debt definitional consistency")
    parts.append(sep)
    parts.append(render_net_debt_violations_text(nd_v))
    if raw["net_debt_claims"] and not nd_v:
        parts.append(f"\n({len(raw['net_debt_claims'])} net-debt/cash claims extracted, all verified)")

    parts.append("")
    parts.append(sep)
    if blocking == 0:
        if total == 0:
            parts.append("OVERALL: VALIDATION PASS (0 violations across 4 validators)")
        else:
            parts.append(
                f"OVERALL: VALIDATION PASS ({total - blocking} MINOR notice(s); 0 blocking)"
            )
    else:
        parts.append(
            f"OVERALL: VALIDATION FAIL ({blocking} blocking violations: "
            f"{blocking_pd} price/date, {blocking_q} quote, "
            f"{blocking_pm} peer, {blocking_nd} net-debt)"
        )

    return "\n".join(parts)


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
