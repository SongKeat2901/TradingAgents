"""Deterministic financial-distress screens for pm_brief.md (FA-101 WP4).

Altman Z'' (4-variable, non-manufacturer) — sector-robust, uses book equity
(no market cap). Missing inputs -> None -> rendered "n/a"; financials skipped.
Beneish M-score (WP4b) — 8-ratio earnings-manipulation screen computed from
annual current/prior statement pairs (`beneish_inputs`); same missing-inputs
-> "n/a" and financials-skipped conventions as Altman Z.
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
    retained = fin.get("retained_earnings")  # not `re` — avoids shadowing stdlib re (Beneish WP4b lands here)
    ebit = fin.get("ebit_ttm")

    tl = None if (ta is None or eq is None) else ta - eq
    wc = None if (ca is None or cl is None) else ca - cl
    x1 = _div(wc, ta)
    x2 = _div(retained, ta)
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


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def _add(a, b):
    return None if (a is None or b is None) else a + b


def compute_beneish_m(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    sector = fin.get("sector") or ""
    if "financial" in sector.lower():
        return {"model": "Beneish M", "applicable": False,
                "skip_reason": f"Beneish not meaningful for financials (sector: {sector})"}
    bi = fin.get("beneish_inputs") or {}
    c = bi.get("current") or {}
    p = bi.get("prior") or {}

    def gm(s):   # gross margin
        return _div(_sub(s.get("sales"), s.get("cogs")), s.get("sales"))

    def aq(s):   # asset quality = 1 - (CA+PPE)/TA
        num = _add(s.get("current_assets"), s.get("ppe"))
        frac = _div(num, s.get("total_assets"))
        return None if frac is None else 1 - frac

    def dr(s):   # depreciation rate = dep/(dep+ppe)
        return _div(s.get("depreciation"), _add(s.get("depreciation"), s.get("ppe")))

    def lev(s):  # leverage = total_liabilities / total_assets
        return _div(_sub(s.get("total_assets"), s.get("total_equity")), s.get("total_assets"))

    dsri = _div(_div(c.get("receivables"), c.get("sales")), _div(p.get("receivables"), p.get("sales")))
    gmi = _div(gm(p), gm(c))
    aqi = _div(aq(c), aq(p))
    sgi = _div(c.get("sales"), p.get("sales"))
    depi = _div(dr(p), dr(c))
    sgai = _div(_div(c.get("sga"), c.get("sales")), _div(p.get("sga"), p.get("sales")))
    lvgi = _div(lev(c), lev(p))
    tata = _div(_sub(c.get("net_income"), c.get("cfo")), c.get("total_assets"))

    ratios = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi,
              "DEPI": depi, "SGAI": sgai, "LVGI": lvgi, "TATA": tata}
    if any(v is None for v in ratios.values()):
        missing = [k for k, v in ratios.items() if v is None]
        return {"model": "Beneish M", "applicable": True, "m_score": None, "flag": None,
                "unavailable_reason": f"missing ratios: {', '.join(missing)}",
                **{k: _r(v) for k, v in ratios.items()}}

    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    flag = "elevated" if m > -1.78 else "normal"
    return {"model": "Beneish M", "applicable": True, "m_score": round(m, 2), "flag": flag,
            **{k: _r(v) for k, v in ratios.items()}}


def format_beneish_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable", True):
        return (f"\n\n## Manipulation screen (Beneish M-score) — not applicable "
                f"({r.get('skip_reason', 'financials')})\n\n"
                "*Beneish is not meaningful for this sector; do not cite an M-score for it.*\n")
    if r.get("m_score") is None:
        return (f"\n\n## Manipulation screen (Beneish M-score) — n/a (data unavailable: "
                f"{r.get('unavailable_reason', 'missing annual inputs')})\n\n"
                "*Do not cite an M-score; required annual inputs were missing.*\n")
    rows = "".join(f"| {k} | {r.get(k)} |\n" for k in ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"))
    return (
        f"\n\n## Manipulation screen (Beneish M-score) (computed from annual statements)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Beneish M** | **{r['m_score']}** |\n"
        f"| **Flag** | **{r['flag']}** (M > -1.78 = elevated manipulation risk) |\n"
        f"{rows}\n"
        "*Use the M-score and flag verbatim; do not recompute. Beneish flags RISK of "
        "earnings manipulation, not proof; it is unreliable for financials, recent IPOs, "
        "and heavy-M&A years.*\n"
    )


# --- FA-101 Phase 5: goodwill-impairment risk flag (§7) ----------------------
# Goodwill applies across sectors (banks carry acquisition goodwill too), so no
# sector skip. A large goodwill cushion relative to equity is where an
# impairment most threatens book value. Names carrying no goodwill are NOT a
# red flag — that renders honestly as "no goodwill reported".

def compute_goodwill_flag(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    goodwill = fin.get("goodwill")
    if goodwill is None:
        return {"reported": False}
    ta = fin.get("total_assets")
    eq = fin.get("total_equity")
    pct_assets = _div(goodwill * 100, ta)
    pct_equity = _div(goodwill * 100, eq) if (eq is not None and eq > 0) else None
    elevated = ((pct_equity is not None and pct_equity >= 50)
                or (pct_assets is not None and pct_assets >= 30))
    return {
        "reported": True,
        "goodwill": goodwill,
        "pct_assets": None if pct_assets is None else round(pct_assets, 1),
        "pct_equity": None if pct_equity is None else round(pct_equity, 1),
        "flag": "elevated" if elevated else "normal",
    }


def compute_refinancing_pressure(fin: dict[str, Any]) -> dict[str, Any]:
    """Near-term refinancing / maturity-wall proxy (FA-101 §2/§7). Current debt =
    Total Debt − Long-Term Debt; its share of total + whether cash covers it is
    the key near-term signal. NOT the full year-by-year 10-K maturity ladder —
    labelled as a proxy. Missing inputs -> n/a; never fabricated."""
    fin = fin or {}
    td = fin.get("total_debt")
    ltd = fin.get("long_term_debt")
    cash = fin.get("cash_and_equivalents")
    if td is None or ltd is None or td <= 0 or ltd < 0 or ltd > td:
        return {"applicable": False, "reason": "current/long-term debt split not available"}
    current_debt = td - ltd
    pct_current = current_debt / td * 100
    cash_cover = (cash / current_debt) if (cash is not None and current_debt > 0) else None
    elevated = pct_current >= 40 and (cash_cover is not None and cash_cover < 1.0)
    moderate = (pct_current >= 40) or (cash_cover is not None and cash_cover < 1.0)
    flag = "elevated" if elevated else ("moderate" if moderate else "low")
    return {"applicable": True, "current_debt": current_debt,
            "pct_current_of_total": round(pct_current, 1),
            "cash_cover_current": None if cash_cover is None else round(cash_cover, 2),
            "flag": flag}


def format_refinancing_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable"):
        return (f"\n\n## Refinancing / maturity-wall proxy — n/a "
                f"({r.get('reason', 'unavailable')})\n\n"
                "*Current/long-term debt split unavailable; do not cite a maturity-wall figure.*\n")
    cc = r.get("cash_cover_current")
    cc_cell = "n/a" if cc is None else f"{cc}x"
    return (
        "\n\n## Refinancing / maturity-wall proxy (computed from raw/financials.json)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Current (near-term) debt | {r.get('current_debt')} |\n"
        f"| Current debt / total debt | {r.get('pct_current_of_total')}% |\n"
        f"| Cash coverage of current debt | {cc_cell} |\n"
        f"| **Refinancing-pressure flag** | **{r.get('flag')}** "
        "(elevated = ≥40% due near-term AND cash < it) |\n\n"
        "*Use verbatim; do not recompute. This is a near-term proxy (current vs "
        "long-term split), NOT the full year-by-year maturity ladder — that requires "
        "the 10-K debt note. A high near-term share with thin cash flags rollover risk.*\n"
    )


def format_goodwill_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("reported"):
        return ("\n\n## Goodwill / impairment screen — n/a (no goodwill reported)\n\n"
                "*This name carries no goodwill on the latest balance sheet — not a "
                "red flag. Do not cite a goodwill ratio for it.*\n")
    pa = "n/a (data unavailable)" if r.get("pct_assets") is None else f"{r['pct_assets']}%"
    pe = "n/a (data unavailable)" if r.get("pct_equity") is None else f"{r['pct_equity']}%"
    return (
        "\n\n## Goodwill / impairment screen (computed from raw/financials.json)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Goodwill (latest) | {r.get('goodwill')} |\n"
        f"| Goodwill / total assets | {pa} |\n"
        f"| Goodwill / book equity | {pe} |\n"
        f"| **Flag** | **{r.get('flag')}** (elevated when ≥50% of equity or ≥30% of assets) |\n\n"
        "*Use the flag and ratios verbatim; do not recompute. An elevated goodwill "
        "cushion signals impairment RISK to book value, not an impairment itself.*\n"
    )
