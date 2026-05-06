"""Deterministic net-debt block (Phase-6.5).

Mirrors the Phase-6.4 peer-ratios pattern. The 2026-05-06 cadence audit
found two material net-debt issues that prompt-only QC item 16(b) failed
to catch:

- APA: decision.md cited "Total Debt $6.0B" against actual $4.59B from
  raw/financials.json. The QC retry "fixed" the original flag by adding
  inline arithmetic that used a fabricated $6.0B cell — QC accepted
  because the structural form looked right.
- ORCL: decision.md cited "Net Debt $96.2B" matching yfinance's Net Debt
  row, but the bear case framed it as `Total Debt − Cash` which equals
  $114B — definition drift that the LLM glossed and QC didn't surface.

This module reads raw/financials.json's balance_sheet CSV, extracts the
cells the LLM needs, and formats them as a verbatim ground-truth block
the Researcher Python-appends to pm_brief.md after the Phase-6.4 peer
ratios block. Downstream agents inherit authoritative cells; QC item
16(b) becomes a verbatim-match check rather than a structural one.
"""

from __future__ import annotations

from typing import Any


# yfinance balance_sheet rows we care about. Names must match the row
# labels yfinance writes (which themselves match the Yahoo Finance UI).
_NET_DEBT_ROW = "Net Debt"
_TOTAL_DEBT_ROW = "Total Debt"
_LT_DEBT_ROW = "Long Term Debt"
_CURRENT_DEBT_ROW = "Current Debt"
_CAP_LEASE_ROW = "Capital Lease Obligations"
_CASH_ROW = "Cash And Cash Equivalents"
_CASH_PLUS_STI_ROW = "Cash Cash Equivalents And Short Term Investments"
_STI_ROW = "Short Term Investments"
_OTHER_STI_ROW = "Other Short Term Investments"


def _parse_quarterly_csv(text: str) -> tuple[list[str], dict[str, list[float | None]]]:
    """Parse the comma-table format yfinance writes for balance_sheet.

    Returns (column_dates, {row_name: [col0_value, col1_value, ...]}).
    Column 0 is the most-recent quarter. Empty cells become None so we
    can distinguish "no data" from "zero".
    """
    if not text:
        return [], {}

    rows: dict[str, list[float | None]] = {}
    columns: list[str] = []
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            continue
        parts = line.split(",")
        # The header row starts with an empty first cell, then column dates.
        if parts[0].strip() == "" and not columns:
            columns = [p.strip() for p in parts[1:] if p.strip()]
            continue
        name = parts[0].strip()
        if not name:
            continue
        vals: list[float | None] = []
        for p in parts[1:]:
            p = p.strip()
            if not p:
                vals.append(None)
                continue
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(None)
        if vals:
            rows[name] = vals
    return columns, rows


def _col0(rows: dict[str, list[float | None]], name: str) -> float | None:
    """Return the column-0 (most-recent quarter) value for `name`, or None."""
    vals = rows.get(name)
    if not vals:
        return None
    return vals[0]


def compute_net_debt(financials_data: dict[str, Any]) -> dict[str, Any]:
    """Compute the net-debt block from raw/financials.json data.

    Returns:
        {
          "trade_date": str | None,             # passed-through from input
          "as_of_quarter": "YYYY-MM-DD" | None, # col 0 column header
          "net_debt": float | None,             # yfinance Net Debt row, col 0
          "net_debt_source": "yfinance" | "computed" | None,
          "total_debt": float | None,
          "long_term_debt": float | None,
          "current_debt": float | None,
          "capital_lease_obligations": float | None,
          "cash_and_equivalents": float | None,
          "cash_plus_short_term_investments": float | None,
          "unavailable": bool,                  # True if Total Debt missing
          "unavailable_reason": str | None,
        }
    """
    bs = financials_data.get("balance_sheet", "") if isinstance(financials_data, dict) else ""
    trade_date = financials_data.get("trade_date") if isinstance(financials_data, dict) else None
    columns, rows = _parse_quarterly_csv(bs)
    as_of = columns[0] if columns else None

    total_debt = _col0(rows, _TOTAL_DEBT_ROW)
    cash = _col0(rows, _CASH_ROW)
    cash_plus_sti = _col0(rows, _CASH_PLUS_STI_ROW)
    if cash_plus_sti is None and cash is not None:
        # Some tickers have only Cash And Cash Equivalents (no Cash+STI
        # composite) — surface cash alone in that slot.
        cash_plus_sti = cash

    net_debt_yf = _col0(rows, _NET_DEBT_ROW)

    # If yfinance didn't compute Net Debt for this ticker, fall back to
    # Total Debt − (Cash + STI). Surface the source so the LLM can cite
    # appropriately.
    net_debt: float | None
    net_debt_source: str | None
    if net_debt_yf is not None:
        net_debt = net_debt_yf
        net_debt_source = "yfinance"
    elif total_debt is not None and cash_plus_sti is not None:
        net_debt = total_debt - cash_plus_sti
        net_debt_source = "computed"
    else:
        net_debt = None
        net_debt_source = None

    unavailable = total_debt is None
    unavailable_reason: str | None = None
    if unavailable:
        missing = [r for r in (_TOTAL_DEBT_ROW, _CASH_ROW)
                   if _col0(rows, r) is None]
        unavailable_reason = f"missing balance-sheet rows: {', '.join(missing)}"

    return {
        "trade_date": trade_date,
        "as_of_quarter": as_of,
        "net_debt": net_debt,
        "net_debt_source": net_debt_source,
        "total_debt": total_debt,
        "long_term_debt": _col0(rows, _LT_DEBT_ROW),
        "current_debt": _col0(rows, _CURRENT_DEBT_ROW),
        "capital_lease_obligations": _col0(rows, _CAP_LEASE_ROW),
        "cash_and_equivalents": cash,
        "cash_plus_short_term_investments": cash_plus_sti,
        "unavailable": unavailable,
        "unavailable_reason": unavailable_reason,
    }


def _fmt_b(value: float | None) -> str:
    """Format a USD value as $X.XXB (or $X.XXM if < $1B)."""
    if value is None:
        return "(n/a)"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    return f"${value:,.0f}"


def format_net_debt_block(net_debt: dict[str, Any]) -> str:
    """Render the net-debt cells as a markdown block for pm_brief.md.

    Returns "" when Total Debt cell is missing (unavailable case is
    handled by the caller, which appends an explicit warning instead).
    """
    if net_debt.get("unavailable"):
        return ""

    trade_date = net_debt.get("trade_date") or "?"
    as_of = net_debt.get("as_of_quarter") or "?"
    nd = net_debt.get("net_debt")
    nd_source = net_debt.get("net_debt_source") or "?"

    nd_line = (
        f"**Authoritative Net Debt: {_fmt_b(nd)}**"
        f" (source: {nd_source}, col 0 of raw/financials.json balance_sheet)"
        if nd is not None
        else "**Net Debt: (n/a)** — yfinance Net Debt row absent and inputs incomplete."
    )

    return (
        f"\n\n## Net debt (computed from raw/financials.json balance_sheet, "
        f"trade_date {trade_date}, col 0 = quarter ending {as_of})\n\n"
        "| Cell | Value (col 0) |\n"
        "|---|---|\n"
        f"| Net Debt (yfinance row) | {_fmt_b(net_debt.get('net_debt') if net_debt.get('net_debt_source') == 'yfinance' else None)} |\n"
        f"| Total Debt | {_fmt_b(net_debt.get('total_debt'))} |\n"
        f"| Long Term Debt | {_fmt_b(net_debt.get('long_term_debt'))} |\n"
        f"| Current Debt | {_fmt_b(net_debt.get('current_debt'))} |\n"
        f"| Capital Lease Obligations | {_fmt_b(net_debt.get('capital_lease_obligations'))} |\n"
        f"| Cash And Cash Equivalents | {_fmt_b(net_debt.get('cash_and_equivalents'))} |\n"
        f"| Cash + Short Term Investments | {_fmt_b(net_debt.get('cash_plus_short_term_investments'))} |\n\n"
        f"{nd_line}\n\n"
        "*Use the cells above verbatim. If you cite inline net-debt arithmetic, "
        "use these exact values and state the formula explicitly (e.g., "
        "`Total Debt − (Cash + STI)`). Note that yfinance's Net Debt row may "
        "differ from `Total Debt − Cash` because of capital-lease and "
        "short-term-investment definitions — surface the discrepancy if both "
        "appear in the report. **Do not introduce cells not in this table.***\n"
    )
