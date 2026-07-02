"""Gordon-Growth Dividend Discount Model for stable payers (FA-101 §5).

DDM is the standard absolute-valuation cross-check for dividend payers. It is
notoriously sensitive to the growth assumption, so this block: (a) only applies
to genuine stable payers (positive earnings/equity/dividends, payout in a sane
band), (b) uses a *sustainable* growth g = retention × ROE (not a fetched trend),
(c) reports a ±1% g sensitivity band rather than a false-precision point, and
(d) renders honest n/a for non-payers / degenerate inputs. All inputs come from
already-parsed data + the intrinsic-value cost of equity. Nothing fabricated.
"""
from __future__ import annotations

from typing import Any

_MIN_SPREAD = 0.015  # require r - g >= this; below it Gordon Growth is unstable


def _gordon(d0_ps: float, r: float, g: float):
    if r - g < _MIN_SPREAD:
        return None
    return round(d0_ps * (1 + g) / (r - g), 2)


def compute_ddm(fin: dict[str, Any], cost_of_equity: float | None) -> dict[str, Any]:
    fin = fin or {}
    ni = fin.get("net_income")
    eq = fin.get("total_equity")
    divs = fin.get("dividends_paid_ttm")
    sh = fin.get("diluted_shares")
    r = cost_of_equity
    if not (ni and ni > 0 and eq and eq > 0 and divs and sh and sh > 0 and r and r > 0):
        return {"applicable": False, "reason": "not a stable dividend payer (missing profit/equity/dividend inputs)"}
    divs = abs(divs)
    payout = divs / ni
    if payout <= 0.05:
        return {"applicable": False, "reason": "negligible/zero dividend (not a payer)"}
    if payout > 1.0:
        return {"applicable": False, "reason": f"payout {round(payout*100,1)}% > 100% (unsustainable — DDM not meaningful)"}
    roe = ni / eq
    g = max((1 - payout) * roe, 0.0)
    # DDM (Gordon Growth) is only valid when sustainable growth < cost of equity.
    # High-ROE / low-payout compounders violate this — DDM is meaningless for them
    # (rely on the DCF), so decline honestly rather than emit a misleading number.
    if g > r - _MIN_SPREAD:
        return {"applicable": False,
                "reason": (f"sustainable growth {round(g*100,1)}% ≥ cost of equity "
                           f"{round(r*100,1)}% — Gordon Growth invalid for this growth "
                           f"profile; rely on the DCF")}
    d0_ps = divs / sh
    return {
        "applicable": True,
        "value": _gordon(d0_ps, r, g),
        "value_g_minus": _gordon(d0_ps, r, max(g - 0.01, 0.0)),
        "value_g_plus": _gordon(d0_ps, r, g + 0.01),  # may be None if g+1% crosses the spread
        "d0_ps": round(d0_ps, 2), "r": round(r, 4), "g": round(g, 4),
        "payout_pct": round(payout * 100, 1), "roe_pct": round(roe * 100, 1),
    }


def format_ddm_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable", False):
        return (f"\n\n## Dividend Discount Model (Gordon Growth) — n/a "
                f"({r.get('reason', 'not applicable')})\n\n"
                "*DDM applies only to stable dividend payers; do not cite a DDM value here.*\n")
    if r.get("value") is None:
        return (f"\n\n## Dividend Discount Model (Gordon Growth) — n/a "
                f"({r.get('reason', 'degenerate inputs')})\n\n"
                "*Do not cite a DDM value; the growth/discount spread was too tight.*\n")
    gm, gp = r.get("value_g_minus"), r.get("value_g_plus")
    if gm is not None and gp is not None:
        sens = f"${gm} – ${gp}"
    elif gm is not None:
        sens = f"≥ ${gm} (upside unstable as g approaches r)"
    else:
        sens = "n/a"
    return (
        "\n\n## Dividend Discount Model (Gordon Growth, stable-payer cross-check)\n\n"
        "| Input / output | Value |\n|---|---|\n"
        f"| Dividend / share (D0) | ${r['d0_ps']} |\n"
        f"| Cost of equity (r) | {round(r['r']*100, 2)}% |\n"
        f"| Sustainable growth g = (1−payout)×ROE | {round(r['g']*100, 2)}% "
        f"(payout {r['payout_pct']}%, ROE {r['roe_pct']}%) |\n"
        f"| **DDM fair value / share** | **${r['value']}** |\n"
        f"| Sensitivity (g ±1%) | {sens} |\n\n"
        "*Use these figures verbatim; do not recompute. DDM is highly sensitive to g "
        "— treat it as one cross-check alongside the DCF/EPV/multiples, weighing the "
        "±1% band, not a point target.*\n"
    )
