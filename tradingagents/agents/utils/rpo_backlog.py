# tradingagents/agents/utils/rpo_backlog.py
"""RPO / backlog deep-dive block (pro-deck technique B; deck pp. 67-68).

Deterministic, free-data: SEC XBRL companyconcept
`us-gaap/RevenueRemainingPerformanceObligation` gives total RPO per period
(the deck's centerpiece number — ORCL $638B — verbatim). From the series we
compute QoQ additions, RPO ÷ market-cap ("contracted revenue vs the market's
price") and RPO ÷ TTM revenue, plus the same for peers.

Honesty rules:
- Self-gating: names without a fresh dimensionless RPO fact (missing concept,
  or latest fact older than STALE_DAYS) render "not applicable" — never
  fabricated, never estimated. (AMZN stopped tagging the total in 2020 →
  honest n/a.)
- The conversion/duration waterfall (next-12mo %, 13-36mo % …) is tagged with
  dimensions the companyconcept API does not expose — the block explicitly
  defers those to the filing's own RPO paragraph (targeted excerpt in
  raw/sec_filing.md) and forbids inventing buckets.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

STALE_DAYS = 400  # annual filers must survive; 2 missed quarters must not

_CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
                "CIK{cik:010d}/us-gaap/RevenueRemainingPerformanceObligation.json")


def fetch_rpo_facts(ticker: str) -> list[dict[str, Any]]:
    """Raw USD facts for the RPO concept from SEC XBRL. [] on any failure
    (fail-open; absence of the concept is a legitimate 'not applicable')."""
    import json as _json

    from tradingagents.agents.utils.sec_edgar import _http_get, _resolve_cik

    try:
        cik = _resolve_cik(ticker.upper())
        if cik is None:
            return []
        body = _http_get(_CONCEPT_URL.format(cik=cik))
        if body is None:
            return []
        data = _json.loads(body)
        facts = (data.get("units") or {}).get("USD") or []
        return [
            {"end": f.get("end"), "val": f.get("val"),
             "form": f.get("form"), "filed": f.get("filed")}
            for f in facts
            if f.get("end") and f.get("val") is not None
        ]
    except Exception:  # noqa: BLE001 — never crash the run over an optional block
        return []


def dedupe_rpo_facts(facts: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
    """Drop facts after trade_date; per period-end keep the latest-filed fact;
    return sorted ascending by end date."""
    by_end: dict[str, dict[str, Any]] = {}
    for f in facts or []:
        end = f.get("end")
        if not end or f.get("val") is None or end > trade_date:
            continue
        prior = by_end.get(end)
        if prior is None or (f.get("filed") or "") > (prior.get("filed") or ""):
            by_end[end] = f
    return [by_end[k] for k in sorted(by_end)]


def _staleness(latest_end: str, trade_date: str) -> Optional[int]:
    try:
        d1 = datetime.strptime(latest_end, "%Y-%m-%d")
        d2 = datetime.strptime(trade_date, "%Y-%m-%d")
        return (d2 - d1).days
    except ValueError:
        return None


def _peer_row(peer: dict[str, Any], trade_date: str) -> dict[str, Any]:
    ticker = peer.get("ticker")
    facts = dedupe_rpo_facts(peer.get("facts") or [], trade_date)
    if not facts:
        return {"ticker": ticker, "applicable": False,
                "reason": "no RPO tagged in SEC XBRL on/before trade_date"}
    latest = facts[-1]
    age = _staleness(latest["end"], trade_date)
    if age is None or age > STALE_DAYS:
        return {"ticker": ticker, "applicable": False,
                "reason": f"latest RPO fact is stale ({latest['end']})"}
    mc = peer.get("market_cap")
    return {
        "ticker": ticker, "applicable": True,
        "rpo": latest["val"], "as_of": latest["end"], "form": latest.get("form"),
        "market_cap": mc,
        "rpo_to_market_cap": round(latest["val"] / mc, 2) if mc else None,
    }


def compute_rpo_backlog(
    ticker: str,
    facts: list[dict[str, Any]],
    trade_date: str,
    market_cap: Optional[float] = None,
    revenue_ttm: Optional[float] = None,
    peers: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build the RPO deep-dive dict from deduped facts (see dedupe_rpo_facts).
    Pure function — no network."""
    out: dict[str, Any] = {"ticker": ticker, "trade_date": trade_date,
                           "applicable": False, "reason": None}
    if not facts:
        out["reason"] = ("does not report / tag a dimensionless total RPO in SEC "
                         "XBRL — not a contracted-backlog reporter")
        return out
    latest = facts[-1]
    age = _staleness(latest["end"], trade_date)
    if age is None or age > STALE_DAYS:
        out["reason"] = (f"latest RPO fact is stale (period end {latest['end']}, "
                         f"{age} days before trade_date) — treat as not applicable")
        return out

    history = []
    prev_val: Optional[float] = None
    for f in facts[-9:]:
        row = {"end": f["end"], "rpo": f["val"], "form": f.get("form"),
               "qoq_add": (f["val"] - prev_val) if prev_val is not None else None}
        history.append(row)
        prev_val = f["val"]
    # first row's qoq_add needs the fact just before the 9-quarter window
    if len(facts) > 9 and facts[-10].get("val") is not None:
        history[0]["qoq_add"] = history[0]["rpo"] - facts[-10]["val"]

    # YoY: a fact whose end is ~1 year before the latest (330..430-day window)
    yoy = None
    latest_dt = datetime.strptime(latest["end"], "%Y-%m-%d")
    for f in facts[:-1]:
        try:
            gap = (latest_dt - datetime.strptime(f["end"], "%Y-%m-%d")).days
        except ValueError:
            continue
        if 330 <= gap <= 430 and f["val"]:
            yoy = latest["val"] / f["val"] - 1.0
    out.update({
        "applicable": True,
        "rpo_total": latest["val"],
        "as_of": latest["end"],
        "form": latest.get("form"),
        "history": history,
        "yoy_growth": round(yoy, 4) if yoy is not None else None,
        "market_cap": market_cap,
        "revenue_ttm": revenue_ttm,
        "rpo_to_market_cap": round(latest["val"] / market_cap, 2) if market_cap else None,
        "rpo_to_revenue_ttm": round(latest["val"] / revenue_ttm, 2) if revenue_ttm else None,
        "peers": [_peer_row(p, trade_date) for p in (peers or [])],
        "source": "SEC XBRL companyconcept us-gaap/RevenueRemainingPerformanceObligation",
    })
    return out


def _b(v) -> str:
    """$ billions, 1dp."""
    return f"${v / 1e9:,.1f}B" if v is not None else "n/a"


def format_rpo_block(rpo: dict[str, Any], trade_date: str | None) -> str:
    header = (f"\n\n## RPO / backlog deep-dive (computed from SEC XBRL "
              f"companyconcept, trade_date {trade_date})\n\n")
    if not rpo.get("applicable"):
        reason = rpo.get("reason") or "no RPO data"
        return (header +
                f"**Not applicable** — {rpo.get('ticker', 'this name')} {reason}. "
                "For names without contracted-backlog disclosure, write \"RPO: not "
                "applicable\" — **do not fabricate** backlog, bookings, or RPO "
                "figures from memory or news paraphrase.\n")

    lines = [header.rstrip("\n"), ""]
    ratio_mc = rpo.get("rpo_to_market_cap")
    ratio_rev = rpo.get("rpo_to_revenue_ttm")
    summary = (f"**Total RPO {_b(rpo['rpo_total'])}** as of {rpo['as_of']} "
               f"({rpo.get('form')})")
    if ratio_mc is not None:
        summary += (f" · **RPO ÷ market cap = {ratio_mc:.2f}x** "
                    f"(market cap {_b(rpo.get('market_cap'))}) — contracted revenue "
                    "vs the market's current price")
    if ratio_rev is not None:
        summary += f" · RPO ÷ TTM revenue = {ratio_rev:.2f}x ({_b(rpo.get('revenue_ttm'))} TTM)"
    if rpo.get("yoy_growth") is not None:
        summary += f" · RPO YoY {rpo['yoy_growth']:+.0%}"
    lines.append(summary + ".")
    lines.append("")
    lines.append("| Period end | Total RPO | QoQ addition | Form |")
    lines.append("|---|---|---|---|")
    for h in rpo.get("history", []):
        add = h.get("qoq_add")
        add_s = f"{'+' if add >= 0 else '−'}{abs(add) / 1e9:,.1f}B" if add is not None else "—"
        lines.append(f"| {h['end']} | {_b(h['rpo'])} | {add_s} | {h.get('form') or ''} |")

    peers = rpo.get("peers") or []
    if peers:
        lines.append("")
        lines.append("Peer backlog comparison (same SEC XBRL concept):")
        lines.append("")
        lines.append("| Ticker | Total RPO | As of | Market cap | RPO ÷ market cap |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| **{rpo['ticker']}** | {_b(rpo['rpo_total'])} | {rpo['as_of']} | "
                     f"{_b(rpo.get('market_cap'))} | "
                     f"{f'{ratio_mc:.2f}x' if ratio_mc is not None else 'n/a'} |")
        for p in peers:
            if p.get("applicable"):
                r = p.get("rpo_to_market_cap")
                lines.append(f"| {p['ticker']} | {_b(p['rpo'])} | {p['as_of']} | "
                             f"{_b(p.get('market_cap'))} | "
                             f"{f'{r:.2f}x' if r is not None else 'n/a'} |")
            else:
                lines.append(f"| {p['ticker']} | n/a ({p.get('reason')}) | — | — | — |")

    lines.append("")
    lines.append(
        "*Use these values verbatim — do not recompute, re-derive, or substitute "
        "backlog figures from memory. The conversion/duration waterfall (next-12-month "
        "%, months 13-36, 37-60, thereafter) is NOT exposed in structured XBRL: quote "
        "the timing percentages from the \"remaining performance obligation\" targeted "
        "excerpt in raw/sec_filing.md if disclosed there, else write \"conversion "
        "timing not disclosed\".*")
    lines.append("")
    return "\n".join(lines)
