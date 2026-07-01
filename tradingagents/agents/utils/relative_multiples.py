"""Deterministic relative-valuation-multiples block for pm_brief.md.

Subject vs peer-median. EV = market cap + net debt (same net-debt figure the
net-debt block uses, so the two blocks tie out). Missing inputs -> None -> n/a.

Forward P/E basis: `market_cap / (forward_eps * diluted_shares)`. This is the
concrete basis chosen for Task 4 (the brief's placeholder expression was
self-cancelling and explicitly flagged as needing a real formula). It requires
`forward_eps`, `market_cap`, and `fin["diluted_shares"]` (> 0) all present;
otherwise -> None (free-data honesty; never approximate).

Peer inputs come from Task 3's `compute_peer_ratios` output. A peer flagged
`unavailable` (or otherwise missing `market_cap` / `net_debt` / `ttm_ebitda`)
contributes `None` to the peer-median EV/EBITDA computation via `.get(...)`
lookups -- it never raises and never fabricates a peer multiple.
"""
from __future__ import annotations

from statistics import median
from typing import Any


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def _median(vals):
    xs = [v for v in vals if v is not None]
    return round(median(xs), 2) if xs else None


def _peer_ev(p: dict[str, Any]) -> float | None:
    """Peer EV = market_cap + net_debt; None if either input is absent
    (covers Task-3 `{"unavailable": True, ...}` peers and any peer missing
    a leverage cell -- both simply lack the keys, so `.get` returns None)."""
    mc = p.get("market_cap")
    nd = p.get("net_debt")
    if mc is None or nd is None:
        return None
    return mc + nd


def compute_relative_multiples(
    fin: dict[str, Any],
    market_cap: float | None,
    net_debt: float | None,
    peers: dict[str, Any],
    forward_eps: float | None = None,
) -> dict[str, Any]:
    fin = fin or {}
    ev = market_cap + net_debt if (market_cap is not None and net_debt is not None) else None

    diluted_shares = fin.get("diluted_shares")
    p_e_fwd = None
    if (
        market_cap is not None
        and forward_eps is not None
        and diluted_shares is not None
        and diluted_shares > 0
    ):
        p_e_fwd = _r(market_cap / (forward_eps * diluted_shares))

    subject = {
        "market_cap": market_cap,
        "ev": ev,
        "ev_ebitda": _r(_div(ev, fin.get("ebitda"))),
        "ev_ebit": _r(_div(ev, fin.get("ebit"))),
        "ev_sales": _r(_div(ev, fin.get("revenue_ttm"))),
        "p_e_ttm": _r(_div(market_cap, fin.get("net_income"))),
        "p_e_fwd": p_e_fwd,
        "p_b": _r(_div(market_cap, fin.get("total_equity"))),
        "p_s": _r(_div(market_cap, fin.get("revenue_ttm"))),
        "p_fcf": _r(_div(market_cap, fin.get("fcf"))),
    }

    peers = peers or {}
    peer_list = [p for p in peers.values() if isinstance(p, dict)]
    peer_median = {
        "ev_ebitda": _median(
            [_r(_div(_peer_ev(p), p.get("ttm_ebitda"))) for p in peer_list]
        ),
        "p_e_ttm": _median([p.get("ttm_pe") for p in peer_list]),
        "p_e_fwd": _median([p.get("forward_pe") for p in peer_list]),
    }
    return {"subject": subject, "peer_median": peer_median}


_NA = "n/a (data unavailable)"


def _c(v, suf="x"):
    return _NA if v is None else f"{v}{suf}"


def format_relative_multiples_block(mult: dict[str, Any], trade_date: str | None) -> str:
    mult = mult or {}
    s = mult.get("subject") or {}
    pm = mult.get("peer_median") or {}
    rows = [
        ("EV/EBITDA", _c(s.get("ev_ebitda")), _c(pm.get("ev_ebitda"))),
        ("EV/EBIT", _c(s.get("ev_ebit")), _NA),
        ("EV/Sales", _c(s.get("ev_sales")), _NA),
        ("P/E (TTM)", _c(s.get("p_e_ttm")), _c(pm.get("p_e_ttm"))),
        ("P/E (fwd)", _c(s.get("p_e_fwd")), _c(pm.get("p_e_fwd"))),
        ("P/B", _c(s.get("p_b")), _NA),
        ("P/S", _c(s.get("p_s")), _NA),
        ("P/FCF", _c(s.get("p_fcf")), _NA),
    ]
    body = "\n".join(f"| {k} | {sub} | {peer} |" for k, sub, peer in rows)
    ev_disp = _NA if s.get("ev") is None else f"${s['ev']:,.0f}"
    return (
        f"\n\n## Relative valuation multiples (computed from raw/financials.json + "
        f"raw/peer_ratios.json, trade_date {trade_date})\n\n"
        f"Subject EV = market cap + net debt = {ev_disp} (ties to the net-debt block).\n\n"
        "| Multiple | Subject | Peer median |\n|---|---|---|\n"
        f"{body}\n\n"
        "*Use these values verbatim. EV = market cap + net debt, using the same "
        "net-debt figure as the net-debt block. Forward P/E = market cap / "
        "(forward EPS × diluted shares). Any `n/a (data unavailable)` means an "
        "input was absent -- do NOT fabricate a peer multiple or an EV "
        "component.*\n"
    )
