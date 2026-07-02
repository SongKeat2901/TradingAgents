"""Deterministic short-interest + analyst-consensus block (FA-101 WP5a).

Parses fields already present in the get_fundamentals text blob; missing -> None
-> "n/a". Nothing fabricated."""
from __future__ import annotations

import re
from typing import Any


def _num(t: str, label: str):
    m = re.search(rf"^{re.escape(label)}:\s*(-?[0-9.]+)", t, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _text(t: str, label: str):
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", t, re.MULTILINE)
    return m.group(1).strip() if m else None


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def compute_sentiment_consensus(
    financials: dict[str, Any], reference_price: float | None = None
) -> dict[str, Any]:
    t = financials.get("fundamentals", "") if isinstance(financials, dict) else ""
    shares_short = _num(t, "Shares Short")
    prior = _num(t, "Shares Short Prior Month")
    pct_float = _num(t, "Short Percent Of Float")
    n_analysts = _num(t, "Number Of Analyst Opinions")
    target_mean = _num(t, "Target Mean Price")
    current = _num(t, "Current Price")
    if current is None and reference_price is not None and reference_price > 0:
        current = reference_price
    return {
        "short_pct_float": _r(pct_float * 100) if pct_float is not None else None,
        "days_to_cover": _r(_num(t, "Short Ratio Days To Cover")),
        "short_mom_change_pct": _r((shares_short - prior) / prior * 100) if (shares_short is not None and prior) else None,
        "rating": _text(t, "Analyst Recommendation"),
        "rating_mean": _r(_num(t, "Analyst Recommendation Mean")),
        "n_analysts": int(n_analysts) if n_analysts is not None else None,
        "target_mean": _r(target_mean),
        "target_median": _r(_num(t, "Target Median Price")),
        "target_upside_pct": _r((target_mean / current - 1) * 100) if (target_mean is not None and current) else None,
        "target_low": _r(_num(t, "Target Low Price")),
        "target_high": _r(_num(t, "Target High Price")),
    }


_NA = "n/a (data unavailable)"


def _c(v, suf=""):
    return _NA if v is None else f"{v}{suf}"


def format_sentiment_block(r: dict[str, Any]) -> str:
    r = r or {}
    if not any(v is not None for v in r.values()):
        return ("\n\n## Sentiment & consensus (short interest + analyst view) — unavailable\n\n"
                "*No short-interest / consensus data in the free feed; do not cite figures.*\n")
    rows = [
        ("Short % of float", _c(r.get("short_pct_float"), "%")),
        ("Days to cover", _c(r.get("days_to_cover"), "x")),
        ("Short interest MoM change", _c(r.get("short_mom_change_pct"), "%")),
        ("Analyst rating", _c(r.get("rating"))),
        ("Rating mean (1=buy..5=sell)", _c(r.get("rating_mean"))),
        ("# analysts", _c(r.get("n_analysts"))),
        ("Target mean / median", f"{_c(r.get('target_mean'))} / {_c(r.get('target_median'))}"),
        ("Target implied upside", _c(r.get("target_upside_pct"), "%")),
        ("Target low–high", f"{_c(r.get('target_low'))} – {_c(r.get('target_high'))}"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "\n\n## Sentiment & consensus (short interest + analyst view)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"{body}\n\n"
        "*Use these values verbatim; do not recompute. Short interest and price "
        "targets are point-in-time yfinance snapshots (bi-monthly settlement / "
        "sell-side aggregate), not real-time.*\n"
    )
