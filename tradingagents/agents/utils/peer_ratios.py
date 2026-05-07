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


def _parse_ttm_ebitda(text: str) -> float | None:
    """Extract TTM EBITDA from the yfinance fundamentals block.

    Format is `EBITDA: <signed_number>` on its own line. Negative values
    are legitimate (loss-making peers like Bitcoin miners; the caller
    decides whether ND/EBITDA is computable from the result).
    """
    if not text:
        return None
    m = re.search(r"^EBITDA:\s*(-?[0-9.]+)", text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _parse_balance_sheet_leverage(text: str) -> tuple[float | None, float | None, float | None]:
    """Extract (Net Debt, Total Debt, Cash+STI) from the balance_sheet CSV.

    Uses yfinance's Net Debt row directly when present; falls back to
    `Total Debt - Cash+STI` arithmetic when the row is absent (some
    tickers — e.g., legacy MARA quarters — only populate the components).
    Returns the column-0 (most-recent quarter) values for each cell.
    """
    if not text:
        return None, None, None

    def _col0(row_name: str) -> float | None:
        for line in text.split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if not parts or parts[0].strip() != row_name:
                continue
            for cell in parts[1:]:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    return float(cell)
                except ValueError:
                    return None
        return None

    net_debt = _col0("Net Debt")
    total_debt = _col0("Total Debt")
    cash_sti = _col0("Cash Cash Equivalents And Short Term Investments")
    if cash_sti is None:
        cash_sti = _col0("Cash And Cash Equivalents")

    if net_debt is None and total_debt is not None and cash_sti is not None:
        net_debt = total_debt - cash_sti

    return net_debt, total_debt, cash_sti


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

    # Phase-6.4 leverage extension: append Net Debt + TTM EBITDA + ND/EBITDA
    # cells so the analyst can cite leverage without fabricating ratios from
    # memory (the 2026-05-06 MARA decision invented "RIOT EV/EBITDA ~12×,
    # CIFR ND/EBITDA ~1.5×, CLSK op margin ~5%" — actual CLSK op margin was
    # −37.83%, sign-flipped). ND/EBITDA only computed when EBITDA > 0;
    # negative-EBITDA peers (typical for crypto miners) get a None placeholder
    # that the formatter renders as (n/m).
    net_debt, _total_debt, _cash_sti = _parse_balance_sheet_leverage(
        peer_data.get("balance_sheet", "")
    )
    ttm_ebitda = _parse_ttm_ebitda(peer_data.get("fundamentals", ""))
    if net_debt is not None and ttm_ebitda is not None and ttm_ebitda > 0:
        nd_ebitda: float | None = net_debt / ttm_ebitda
    else:
        nd_ebitda = None

    return {
        "latest_quarter_capex_to_revenue": round(capex / revenue * 100, 2),
        "latest_quarter_op_margin": round(op_income / revenue * 100, 2),
        "ttm_pe": round(ttm_pe, 2) if ttm_pe is not None else None,
        "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
        "net_debt": net_debt,
        "ttm_ebitda": ttm_ebitda,
        "nd_ebitda": round(nd_ebitda, 2) if nd_ebitda is not None else None,
        "source": "peers.json (yfinance: capex/revenue/op-margin from quarterly income+cashflow; PE/EBITDA from fundamentals; net-debt from balance_sheet col 0)",
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
            rows.append(
                f"| {key} | (unavailable) | (unavailable) | (unavailable) | "
                f"(unavailable) | (unavailable) | (unavailable) | (unavailable) |"
            )
            continue

        def _pct(v):
            return f"{v:.1f}%" if v is not None else "(n/a)"

        def _x(v):
            return f"{v:.2f}x" if v is not None else "(n/a)"

        def _b(v):
            if v is None:
                return "(n/a)"
            if abs(v) >= 1_000_000_000:
                return f"${v / 1_000_000_000:.2f}B"
            if abs(v) >= 1_000_000:
                return f"${v / 1_000_000:.0f}M"
            return f"${v:,.0f}"

        def _ratio(v):
            return f"{v:.2f}x" if v is not None else "(n/m)"

        rows.append(
            f"| {key} | {_pct(val.get('latest_quarter_capex_to_revenue'))} | "
            f"{_pct(val.get('latest_quarter_op_margin'))} | "
            f"{_x(val.get('ttm_pe'))} | "
            f"{_x(val.get('forward_pe'))} | "
            f"{_b(val.get('net_debt'))} | "
            f"{_b(val.get('ttm_ebitda'))} | "
            f"{_ratio(val.get('nd_ebitda'))} |"
        )

    if not rows:
        return ""

    table = "\n".join(rows)
    return (
        f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {trade_date})\n\n"
        "| Ticker | Q1 capex/revenue | Q1 op margin | TTM P/E | Forward P/E | "
        "Net Debt | TTM EBITDA | ND/EBITDA |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"{table}\n\n"
        "*Use these values verbatim. Do NOT cite \"approximate\" or "
        "\"inherited from prior debate\" alternatives — these are the "
        "authoritative current-quarter figures derived from yfinance data on "
        "the trade date. If you need to make a peer-comparison claim, "
        "recompute deltas from this table, not from memory. **ND/EBITDA = "
        "(n/m)** when TTM EBITDA is ≤ 0 (negative-EBITDA peers — common for "
        "crypto miners, biotech pre-revenue — make the leverage ratio "
        "uninterpretable; cite the Net Debt and TTM EBITDA cells separately "
        "rather than inventing a substitute multiple).*\n"
    )
