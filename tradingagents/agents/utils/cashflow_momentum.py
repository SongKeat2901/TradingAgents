# tradingagents/agents/utils/cashflow_momentum.py
"""Cash-flow momentum (QoQ) block (pro-deck technique D; deck p70).

The pro deck's 'earnings call takeaways' quantitative half — OCF +100% QoQ,
capex −11% QoQ, FCF −1.8B vs −11.5B prior quarter — is exactly reproducible
from the quarterly cashflow columns already in raw/financials.json. Making it
a deterministic block means the role LLM cites the deltas verbatim instead of
re-deriving (or hallucinating) them.

Conventions:
- Quarters are rendered chronologically (oldest → newest).
- Capex QoQ % is computed on SPEND MAGNITUDE (|capex|): "capex −11%" means
  the company spent 11% less, matching how analysts talk about it.
- A % change against a non-positive base is meaningless → None (the $ delta
  is still reported).
- FCF prefers the explicit `Free Cash Flow` row, else OCF + capex (capex is
  negative in yfinance).
"""
from __future__ import annotations

from typing import Any, Optional

from tradingagents.agents.utils.net_debt import _parse_quarterly_csv


def _pct(latest: Optional[float], prev: Optional[float]) -> Optional[float]:
    if latest is None or prev is None or prev <= 0:
        return None
    return latest / prev - 1.0


def compute_cashflow_momentum(financials: Any) -> dict[str, Any]:
    """Quarterly OCF / capex / FCF series + latest-quarter QoQ deltas.
    Pure function over the financials.json bundle — no network."""
    d = financials if isinstance(financials, dict) else {}
    cols, cf = _parse_quarterly_csv(d.get("cashflow", ""))
    out: dict[str, Any] = {"available": False, "reason": None, "quarters": [],
                           "latest_qoq": None}

    def _series(*aliases):
        for a in aliases:
            vals = cf.get(a)
            if vals:
                return vals
        return []

    ocf = _series("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _series("Capital Expenditure")
    fcf = _series("Free Cash Flow")

    quarters = []
    for i, col in enumerate(cols):
        o = ocf[i] if i < len(ocf) else None
        c = capex[i] if i < len(capex) else None
        f = fcf[i] if i < len(fcf) else None
        if f is None and o is not None and c is not None:
            f = o + c  # capex negative in yfinance
        if o is None and c is None and f is None:
            continue
        quarters.append({"end": col, "ocf": o, "capex": c, "fcf": f})
    quarters.reverse()  # yfinance columns are most-recent-first -> chronological

    if len(quarters) < 2:
        out["reason"] = ("fewer than 2 quarters of cashflow columns available — "
                         "QoQ momentum not computable")
        return out

    latest, prev = quarters[-1], quarters[-2]
    capex_latest = abs(latest["capex"]) if latest["capex"] is not None else None
    capex_prev = abs(prev["capex"]) if prev["capex"] is not None else None
    out.update({
        "available": True,
        "quarters": quarters[-5:],
        "latest_qoq": {
            "ocf_latest": latest["ocf"], "ocf_prev": prev["ocf"],
            "ocf_pct": _pct(latest["ocf"], prev["ocf"]),
            "capex_latest": latest["capex"], "capex_prev": prev["capex"],
            "capex_pct": _pct(capex_latest, capex_prev),
            "fcf_latest": latest["fcf"], "fcf_prev": prev["fcf"],
            "fcf_delta": (latest["fcf"] - prev["fcf"]
                          if latest["fcf"] is not None and prev["fcf"] is not None
                          else None),
        },
    })
    return out


def _b(v) -> str:
    if v is None:
        return "n/a"
    sign = "−" if v < 0 else ""
    return f"{sign}${abs(v) / 1e9:,.1f}B"


def _pc(v) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.1%}"


def format_cashflow_momentum_block(m: dict[str, Any], trade_date: str | None) -> str:
    header = (f"\n\n## Cash-flow momentum (QoQ) (computed from raw/financials.json "
              f"quarterly cashflow, trade_date {trade_date})\n\n")
    if not m.get("available"):
        reason = m.get("reason") or "quarterly cashflow unavailable"
        return (header +
                f"**Cash-flow momentum unavailable** — {reason}. **Do not cite "
                "quarter-over-quarter cash-flow deltas in this report.** If cash-flow "
                "trajectory is essential to the thesis, flag it as `(quarterly cashflow "
                "unavailable)` and do not invent figures from memory.\n")

    lines = [header.rstrip("\n"), ""]
    lines.append("| Quarter end | Operating cash flow | Capex | Free cash flow |")
    lines.append("|---|---|---|---|")
    for q in m["quarters"]:
        lines.append(f"| {q['end']} | {_b(q['ocf'])} | {_b(q['capex'])} | {_b(q['fcf'])} |")
    qoq = m["latest_qoq"]
    lines.append("")
    lines.append(
        f"**Latest QoQ:** OCF {_pc(qoq['ocf_pct'])} ({_b(qoq['ocf_prev'])} → "
        f"{_b(qoq['ocf_latest'])}) · capex spend {_pc(qoq['capex_pct'])} "
        f"({_b(qoq['capex_prev'])} → {_b(qoq['capex_latest'])}) · FCF "
        f"{'improved' if (qoq['fcf_delta'] or 0) >= 0 else 'deteriorated'} "
        f"{_b(qoq['fcf_delta'])} ({_b(qoq['fcf_prev'])} → {_b(qoq['fcf_latest'])}).")
    lines.append("")
    lines.append(
        "*Use these values verbatim — do not recompute QoQ deltas or percentages. "
        "A % vs a non-positive base renders n/a by design; cite the $ change "
        "instead. This is the quantitative half of the latest-quarter takeaways; "
        "management commentary must come from news.json and be labeled as "
        "news-sourced.*")
    lines.append("")
    return "\n".join(lines)
