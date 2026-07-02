# tradingagents/agents/utils/eps_scenario.py
"""Forward-EPS × exit-multiple price-target grid (pro-deck technique A).

Reference: Tiger Brokers 30-Jun FA Outlook p69 (Oracle case study) — project
EPS along the consensus growth path, price each year at exit multiples
(20x/25x/30x + the current TTM multiple), and show the implied P/E if the
price stays flat (the "compression" view: growth de-rates the multiple for
you even at a flat price).

Free-data honesty rules:
- EPS base + growth come from yfinance consensus (`earnings_estimate` avg /
  growth, `forwardEps` fallback). No guidance is invented; when consensus is
  absent or non-positive the grid renders an honest n/a.
- Out-years (+2..+4y) EXTRAPOLATE the +1y consensus at the labeled growth
  rate — an assumption, not guidance; the block says so explicitly.
- Plausibility gate: |growth| outside (−50%, +60%)/yr → n/a (a 4-year
  extrapolation at hypergrowth or collapse rates is not meaningful).
"""
from __future__ import annotations

from typing import Any, Optional

MULTIPLES = (20.0, 25.0, 30.0)
PROJECTION_YEARS = 4
GROWTH_LO, GROWTH_HI = -0.50, 0.60
CURRENT_PE_LO, CURRENT_PE_HI = 3.0, 100.0


def _pos(x) -> Optional[float]:
    """x as float if a positive finite number, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 and v == v and v not in (float("inf"),) else None


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def compute_eps_scenario(
    price: Optional[float],
    estimates: dict[str, Any],
    trailing_eps: Optional[float] = None,
    forward_eps: Optional[float] = None,
) -> dict[str, Any]:
    """Build the price-target grid. Pure function — no network.

    `estimates`: {"eps_0y", "eps_1y", "growth_1y", "n_analysts_0y",
    "n_analysts_1y", "source"} (all optional / None-able) from
    fetch_eps_estimates. `forward_eps` (yfinance info) backstops eps_1y.
    """
    est = estimates or {}
    out: dict[str, Any] = {"available": False, "reason": None, "inputs": {}, "years": []}

    px = _pos(price)
    if px is None:
        out["reason"] = "reference price unavailable"
        return out

    eps_0y = _num(est.get("eps_0y"))
    eps_1y = _num(est.get("eps_1y"))
    if eps_1y is None:
        eps_1y = _num(forward_eps)
    if eps_1y is None and eps_0y is None:
        out["reason"] = "consensus forward EPS unavailable"
        return out
    if (eps_1y is not None and eps_1y <= 0) or (eps_1y is None and eps_0y <= 0):
        out["reason"] = "consensus forward EPS non-positive — a P/E grid is not meaningful"
        return out

    growth = _num(est.get("growth_1y"))
    growth_source = "analyst consensus growth (+1y)"
    if growth is None and eps_0y and eps_0y > 0 and eps_1y is not None:
        growth = eps_1y / eps_0y - 1.0
        growth_source = "derived from consensus 0y→+1y EPS path"
    if growth is None:
        out["reason"] = "no consensus EPS growth available to extrapolate out-years"
        return out
    if not (GROWTH_LO < growth < GROWTH_HI):
        out["reason"] = (f"consensus growth {growth:+.0%}/yr outside plausibility bounds "
                         f"({GROWTH_LO:+.0%}..{GROWTH_HI:+.0%}) — multi-year extrapolation "
                         "not meaningful")
        return out

    if eps_1y is None:
        eps_1y = eps_0y * (1.0 + growth)

    t_eps = _pos(trailing_eps)
    current_pe = px / t_eps if t_eps else None
    if current_pe is not None and not (CURRENT_PE_LO <= current_pe <= CURRENT_PE_HI):
        current_pe = None

    years: list[dict[str, Any]] = []
    eps = eps_1y
    for i in range(1, PROJECTION_YEARS + 1):
        if i > 1:
            eps = eps * (1.0 + growth)
        row: dict[str, Any] = {
            "year": f"+{i}y",
            "eps": round(eps, 2),
            "implied_pe_flat_price": round(px / eps, 2),
        }
        for m in MULTIPLES:
            row[f"price_at_{int(m)}x"] = round(eps * m, 2)
        if current_pe is not None:
            row["price_at_current_pe"] = round(eps * current_pe, 2)
        years.append(row)

    out.update({
        "available": True,
        "inputs": {
            "price": px,
            "eps_0y": eps_0y,
            "eps_1y": round(eps_1y, 4),
            "growth": round(growth, 4),
            "growth_source": growth_source,
            "current_pe_ttm": round(current_pe, 2) if current_pe is not None else None,
            "n_analysts_0y": est.get("n_analysts_0y"),
            "n_analysts_1y": est.get("n_analysts_1y"),
            "source": est.get("source") or "yfinance consensus",
        },
        "years": years,
    })
    return out


def fetch_eps_estimates(ticker: str) -> dict[str, Any]:
    """Pull consensus EPS estimates from yfinance. Fail-open: every field
    None-able; never raises."""
    result: dict[str, Any] = {"eps_0y": None, "eps_1y": None, "growth_1y": None,
                              "n_analysts_0y": None, "n_analysts_1y": None,
                              "source": "yfinance earnings_estimate"}
    try:
        import yfinance as yf
        from tradingagents.dataflows.stockstats_utils import yf_retry
        df = yf_retry(lambda: yf.Ticker(ticker.upper()).earnings_estimate)
        if df is None or getattr(df, "empty", True):
            return result

        def _cell(row, col):
            try:
                v = df.loc[row, col]
                return float(v) if v == v else None
            except Exception:  # noqa: BLE001 — missing row/col is expected
                return None

        result["eps_0y"] = _cell("0y", "avg")
        result["eps_1y"] = _cell("+1y", "avg")
        result["growth_1y"] = _cell("+1y", "growth")
        n0, n1 = _cell("0y", "numberOfAnalysts"), _cell("+1y", "numberOfAnalysts")
        result["n_analysts_0y"] = int(n0) if n0 else None
        result["n_analysts_1y"] = int(n1) if n1 else None
    except Exception:  # noqa: BLE001 — estimates are optional; never crash the run
        pass
    return result


def _fmt_px(v) -> str:
    return f"${v:,.0f}" if v is not None else "n/a"


def format_eps_scenario_block(grid: dict[str, Any], trade_date: str | None) -> str:
    """Render the pm_brief.md block. Mirrors the deck's p69 layout."""
    header = (f"\n\n## Forward-EPS price-target grid (computed from yfinance "
              f"consensus estimates, trade_date {trade_date})\n\n")
    if not grid.get("available"):
        reason = grid.get("reason") or "inputs missing"
        return (header +
                f"**Price-target grid unavailable** — {reason}. **Do not cite forward "
                "price targets or multi-year EPS projections in this report.** If a "
                "price-path view is essential to the thesis, flag it as `(consensus "
                "estimates unavailable)` and do not invent projections from memory.\n")

    inp = grid["inputs"]
    years = grid["years"]
    n0 = inp.get("n_analysts_0y")
    n1 = inp.get("n_analysts_1y")
    lines = [header.rstrip("\n"), ""]
    lines.append(
        f"Reference price ${inp['price']:,.2f} · consensus current-FY EPS "
        f"{('$%.2f' % inp['eps_0y']) if inp.get('eps_0y') is not None else 'n/a'}"
        f"{f' ({n0:.0f} analysts)' if n0 else ''} · +1y consensus EPS ${inp['eps_1y']:,.2f}"
        f"{f' ({n1:.0f} analysts)' if n1 else ''} · growth used {inp['growth']:+.1%}/yr "
        f"({inp['growth_source']}).")
    lines.append("")
    hdr = "| | " + " | ".join(y["year"] for y in years) + " |"
    sep = "|---|" + "---|" * len(years)
    lines.append(hdr)
    lines.append(sep)
    lines.append("| Projected EPS | " + " | ".join(f"${y['eps']:,.2f}" for y in years) + " |")
    lines.append(f"| Implied P/E if price stays ${inp['price']:,.2f} | "
                 + " | ".join(f"{y['implied_pe_flat_price']:.1f}x" for y in years) + " |")
    for m in MULTIPLES:
        key = f"price_at_{int(m)}x"
        lines.append(f"| Price at {int(m)}x P/E | "
                     + " | ".join(_fmt_px(y.get(key)) for y in years) + " |")
    if inp.get("current_pe_ttm") is not None:
        lines.append(f"| Price at current {inp['current_pe_ttm']:.1f}x TTM P/E | "
                     + " | ".join(_fmt_px(y.get("price_at_current_pe")) for y in years) + " |")
    lines.append("")
    lines.append(
        "*+1y EPS is the analyst consensus verbatim; +2y..+4y EXTRAPOLATE it at the "
        "growth rate above — a labeled assumption, NOT company guidance. Use these "
        "values verbatim; do not recompute or extend the grid. The `Implied P/E if "
        "price stays flat` row is the compression view: if the consensus EPS path "
        "plays out and the price does not move, the multiple de-rates to that level — "
        "cite it when weighing whether growth is already priced in.*")
    lines.append("")
    return "\n".join(lines)
