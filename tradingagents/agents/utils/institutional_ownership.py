"""Deterministic institutional & insider ownership block (FA-101 Phase 2b, §8).

Consumes yfinance's 13F-derived holdings (institutional %, insider %, holder
count, top institutional holders with quarterly stake change) — free, no SEC
parsing. `normalize_institutional_ownership` is pure/testable; the network
fetch lives in `dataflows/y_finance.get_institutional_ownership`. Missing ->
None -> "n/a"; never fabricated. 13F holdings are quarterly with a ~45-day lag.
"""
from __future__ import annotations

from typing import Any

_TOP_N = 10


def _pct(x, nd=2):
    return None if x is None else round(x * 100, nd)


def normalize_institutional_ownership(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw fetch dict (fractional percents, holder rows) into the
    block's rounded shape. Tolerates missing keys/rows -> None / []."""
    raw = raw or {}
    holders = []
    for h in (raw.get("holders") or [])[:_TOP_N]:
        holders.append({
            "holder": h.get("holder"),
            "pct_held": _pct(h.get("pct_held")),
            "value": h.get("value"),
            "pct_change": _pct(h.get("pct_change")),
            "date": h.get("date"),
        })
    return {
        "pct_institutions": _pct(raw.get("pct_institutions")),
        "pct_insiders": _pct(raw.get("pct_insiders")),
        "institutions_count": raw.get("institutions_count"),
        "top_holders": holders,
    }


def _na(v, suffix=""):
    return "n/a (data unavailable)" if v is None else f"{v}{suffix}"


def _usd(v):
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:.1f}B"
    if a >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def format_ownership_block(result: dict[str, Any]) -> str:
    r = result or {}
    holders = r.get("top_holders") or []
    if (r.get("pct_institutions") is None and r.get("pct_insiders") is None
            and not holders):
        return ("\n\n## Institutional & insider ownership — n/a (data unavailable)\n\n"
                "*No institutional-holdings data from yfinance for this name "
                "(common for foreign ADRs / thin coverage). Do not cite ownership figures.*\n")
    head = (
        "\n\n## Institutional & insider ownership (yfinance, 13F-derived)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Institutional ownership | {_na(r.get('pct_institutions'), '%')} |\n"
        f"| Insider ownership | {_na(r.get('pct_insiders'), '%')} |\n"
        f"| # institutional holders | {_na(r.get('institutions_count'))} |\n"
    )
    if holders:
        rows = "".join(
            f"| {h.get('holder') or 'n/a'} | {_na(h.get('pct_held'), '%')} | "
            f"{_usd(h.get('value'))} | {_na(h.get('pct_change'), '%')} |\n"
            for h in holders
        )
        date = holders[0].get("date") or "latest 13F"
        head += (
            f"\nTop institutional holders (as of {date}):\n\n"
            "| Holder | % held | Stake | QoQ change |\n|---|---|---|---|\n" + rows
        )
    return head + (
        "\n*Use these ownership figures verbatim; do not recompute. 13F holdings "
        "are quarterly with a ~45-day filing lag, not real-time.*\n"
    )
