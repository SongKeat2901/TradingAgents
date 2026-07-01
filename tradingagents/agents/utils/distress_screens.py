"""Deterministic financial-distress screens for pm_brief.md (FA-101 WP4).

Altman Z'' (4-variable, non-manufacturer) — sector-robust, uses book equity
(no market cap). Missing inputs -> None -> rendered "n/a"; financials skipped.
Beneish M-score (WP4b) will be added here once the annual-statement data layer
exists.
"""
from __future__ import annotations

from typing import Any


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_altman_z(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    sector = fin.get("sector") or ""
    if "financial" in sector.lower():
        return {"model": "Altman Z''", "applicable": False,
                "skip_reason": f"Altman Z not meaningful for financials (sector: {sector})"}

    ta = fin.get("total_assets")
    eq = fin.get("total_equity")
    ca = fin.get("current_assets")
    cl = fin.get("current_liabilities")
    re = fin.get("retained_earnings")
    ebit = fin.get("ebit_ttm")

    tl = None if (ta is None or eq is None) else ta - eq
    wc = None if (ca is None or cl is None) else ca - cl
    x1 = _div(wc, ta)
    x2 = _div(re, ta)
    x3 = _div(ebit, ta)
    x4 = _div(eq, tl) if (tl is not None and tl > 0) else None

    if None in (x1, x2, x3, x4):
        missing = [n for n, v in (("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4)) if v is None]
        return {"model": "Altman Z''", "applicable": True, "z_score": None, "zone": None,
                "unavailable_reason": f"missing/undefined inputs for {', '.join(missing)}",
                "x1": _r(x1), "x2": _r(x2), "x3": _r(x3), "x4": _r(x4)}

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    zone = "Safe" if z > 2.6 else ("Distress" if z < 1.1 else "Grey")
    return {"model": "Altman Z''", "applicable": True, "z_score": round(z, 2), "zone": zone,
            "x1": _r(x1), "x2": _r(x2), "x3": _r(x3), "x4": _r(x4)}


def _r(x):
    return None if x is None else round(x, 3)


def format_distress_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable", True):
        return (f"\n\n## Distress screen (Altman Z″) — not applicable "
                f"({r.get('skip_reason', 'financials')})\n\n"
                "*Altman Z is not meaningful for this sector; do not cite a Z-score for it.*\n")
    if r.get("z_score") is None:
        return (f"\n\n## Distress screen (Altman Z″) — n/a (data unavailable: "
                f"{r.get('unavailable_reason', 'missing inputs')})\n\n"
                "*Do not cite a Z-score; required inputs were missing.*\n")
    return (
        f"\n\n## Distress screen (Altman Z″) (computed from raw/financials.json)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Altman Z″** | **{r['z_score']}** |\n"
        f"| **Zone** | **{r['zone']}** (>2.6 Safe · 1.1-2.6 Grey · <1.1 Distress) |\n"
        f"| X1 working capital / total assets | {r['x1']} |\n"
        f"| X2 retained earnings / total assets | {r['x2']} |\n"
        f"| X3 EBIT(TTM) / total assets | {r['x3']} |\n"
        f"| X4 book equity / total liabilities | {r['x4']} |\n\n"
        "*Use the Z″ score and zone verbatim; do not recompute. Z″ (4-variable, "
        "non-manufacturer) is a relative distress indicator, not a default prediction.*\n"
    )
