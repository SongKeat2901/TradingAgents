"""Deterministic peer-ratio computation (Phase-6.4 caveat-wrapping closure).

Phase 6.3 audits showed the PM citing fabricated peer capex intensities
(GOOGL 4.9%, AMZN 5.1%; actual 32.5%, 24.4%) under "inherited from prior
debate, not revalidated" caveats that the QC LLM accepted because it
doesn't have access to raw/peers.json. This module computes authoritative
per-peer ratios from peers.json data; PM Pre-flight Python-appends them
to pm_brief.md verbatim, removing the LLM's chance to paraphrase.

Mirrors the Phase 6.2 deterministic earnings calendar pattern.
"""

from __future__ import annotations

import re
from typing import Any


def _parse_quarterly_csv(text: str) -> dict[str, list[float]]:
    """Parse the comma-table format yfinance writes for income_statement /
    cashflow / balance_sheet. Returns {row_name: [col0, col1, ...]} where
    column 0 is the most-recent quarter."""
    rows: dict[str, list[float]] = {}
    if not text:
        return rows
    for line in text.split("\n"):
        if not line or line.startswith("#") or "," not in line:
            continue
        parts = line.split(",")
        name = parts[0].strip()
        if not name:
            continue
        vals: list[float] = []
        for p in parts[1:]:
            p = p.strip()
            if not p:
                continue
            try:
                vals.append(float(p))
            except ValueError:
                pass
        if vals:
            rows[name] = vals
    return rows


def _parse_pe_from_fundamentals(text: str) -> tuple[float | None, float | None]:
    """Extract (TTM PE, Forward PE) from the yfinance fundamentals text block.
    Returns (None, None) if either is missing or unparseable."""
    if not text:
        return None, None

    def _find(pattern: str) -> float | None:
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except (ValueError, IndexError):
            return None

    ttm = _find(r"PE Ratio \(TTM\):\s*([0-9.]+)")
    fwd = _find(r"Forward PE:\s*([0-9.]+)")
    return ttm, fwd


def _compute_one_peer(peer_data: dict[str, Any]) -> dict[str, Any]:
    """Per-peer ratio computation; returns either a populated dict or
    {"unavailable": True, "reason": "..."} on any failure."""
    inc = _parse_quarterly_csv(peer_data.get("income_statement", ""))
    cf = _parse_quarterly_csv(peer_data.get("cashflow", ""))

    rev_col = inc.get("Total Revenue")
    opi_col = inc.get("Operating Income")
    capex_col = cf.get("Capital Expenditure")

    if not rev_col or not opi_col or not capex_col:
        missing = []
        if not rev_col:
            missing.append("Total Revenue")
        if not opi_col:
            missing.append("Operating Income")
        if not capex_col:
            missing.append("Capital Expenditure")
        return {"unavailable": True, "reason": f"missing rows: {', '.join(missing)}"}

    revenue = rev_col[0]
    if revenue <= 0:
        return {"unavailable": True, "reason": f"degenerate revenue: {revenue}"}

    op_income = opi_col[0]
    capex = abs(capex_col[0])

    ttm_pe, forward_pe = _parse_pe_from_fundamentals(peer_data.get("fundamentals", ""))

    return {
        "latest_quarter_capex_to_revenue": round(capex / revenue * 100, 2),
        "latest_quarter_op_margin": round(op_income / revenue * 100, 2),
        "ttm_pe": round(ttm_pe, 2) if ttm_pe is not None else None,
        "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
        "source": "peers.json (yfinance via Q1 capex/revenue)",
    }


def compute_peer_ratios(peers_data: dict[str, Any], trade_date: str) -> dict[str, Any]:
    """Compute authoritative peer ratios from raw/peers.json data.

    peers_data: dict mapping ticker → peer-data dict (the structure yfinance
        writes via the researcher: ticker, trade_date, fundamentals,
        balance_sheet, cashflow, income_statement).
    trade_date: "YYYY-MM-DD" string; passed through to the output.

    Returns:
        {
          "trade_date": "2026-05-01",
          "_unavailable": list of ticker symbols where computation failed,
          "<TICKER>": {
            "latest_quarter_capex_to_revenue": <float, in %>,
            "latest_quarter_op_margin": <float, in %>,
            "ttm_pe": <float or None>,
            "forward_pe": <float or None>,
            "source": "peers.json (...)",
          } OR {"unavailable": True, "reason": "..."},
        }
    """
    out: dict[str, Any] = {"trade_date": trade_date, "_unavailable": []}
    for ticker, peer_data in peers_data.items():
        if not isinstance(peer_data, dict):
            continue
        result = _compute_one_peer(peer_data)
        out[ticker] = result
        if result.get("unavailable"):
            out["_unavailable"].append(ticker)
    return out


def format_peer_ratios_block(ratios: dict[str, Any]) -> str:
    """Render peer_ratios.json content as a Markdown table for appending to
    pm_brief.md after the PM Pre-flight LLM call.

    Returns "" if no peer rows can be rendered (all unavailable or empty).
    """
    if not ratios:
        return ""
    trade_date = ratios.get("trade_date", "?")
    unavailable_set = set(ratios.get("_unavailable", []))

    rows: list[str] = []
    for key, val in ratios.items():
        if key in ("trade_date", "_unavailable"):
            continue
        if not isinstance(val, dict):
            continue
        if key in unavailable_set or val.get("unavailable"):
            rows.append(f"| {key} | (unavailable) | (unavailable) | (unavailable) | (unavailable) |")
            continue

        def _pct(v):
            return f"{v:.1f}%" if v is not None else "(n/a)"

        def _x(v):
            return f"{v:.2f}x" if v is not None else "(n/a)"

        rows.append(
            f"| {key} | {_pct(val.get('latest_quarter_capex_to_revenue'))} | "
            f"{_pct(val.get('latest_quarter_op_margin'))} | "
            f"{_x(val.get('ttm_pe'))} | "
            f"{_x(val.get('forward_pe'))} |"
        )

    if not rows:
        return ""

    table = "\n".join(rows)
    return (
        f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {trade_date})\n\n"
        "| Ticker | Q1 capex/revenue | Q1 op margin | TTM P/E | Forward P/E |\n"
        "|---|---|---|---|---|\n"
        f"{table}\n\n"
        "*Use these values verbatim. Do NOT cite \"approximate\" or "
        "\"inherited from prior debate\" alternatives — these are the "
        "authoritative current-quarter figures derived from yfinance data on "
        "the trade date. If you need to make a peer-comparison claim, "
        "recompute deltas from this table, not from memory.*\n"
    )
