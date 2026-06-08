"""Deterministic intrinsic-value (IV) block (2026-06-01).

Triangulated fair-value range computed in Python from raw/ data, reconciled
against the Monte-Carlo scenario EV. Mirrors the peer_ratios / net_debt
deterministic-block pattern: the researcher writes raw/intrinsic_value.json and
appends a "## Intrinsic value" block to pm_brief.md; the LLM only interprets it.

Methods (per applicability profile): DCF (FCF), EPV (earnings power, no growth),
peer-multiples-implied, and reverse-DCF (growth the price implies). No method is
forced where its inputs don't support it — gaps are stated, never filled.
See docs/superpowers/specs/2026-06-01-intrinsic-value-design.md.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any

# --- transparent constants (echoed into the JSON `constants_note`) ---
ERP = 0.05                    # equity risk premium
TERMINAL_GROWTH = 0.025       # long-run terminal growth (~GDP)
HORIZON_YEARS = 5             # explicit DCF window
NEAR_TERM_GROWTH_CAP = 0.25   # cap on analyst-implied near-term growth
DISCOUNT_RATE_FLOOR = 0.08    # floor on cost of equity (high-beta guard)
COST_OF_DEBT_SPREAD = 0.02    # cost of debt = risk_free + spread
RISK_FREE_FALLBACK = 0.043    # used if ^TNX fetch fails
GROWTH_DELTA = 0.02           # bear/bull growth ±
DISCOUNT_DELTA = 0.01         # bear/bull discount ±
RECONCILE_TOLERANCE = 0.15    # |IV vs EV| within this → AGREE
NAV_PROXY_TICKERS = {"MSTR"}

# --- growth-tier methodology (2026-06-08): tier-weighted blend, EPV as stress ---
FCF_NORM_CONVERSION = 0.80    # through-cycle owner-earnings proxy (FCF ≈ NI × this)
COMPOUNDER_MIN_GROWTH = 0.12  # revenue-growth threshold for COMPOUNDER
COMPOUNDER_MIN_GROSS_MARGIN = 0.50
CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
CYCLICAL_INDUSTRY_KEYWORDS = ("semiconductor", "oil", "gas", "mining", "steel",
                              "metals", "coal", "copper", "aluminum")
# Blend weights per growth tier. EPV is a downside stress test — weighted into the
# base ONLY for MATURE; for COMPOUNDER/CYCLICAL it is computed and shown but not blended.
TIER_WEIGHTS = {
    "COMPOUNDER": {"dcf": 0.65, "multiples": 0.35, "epv": 0.0},
    "MATURE":     {"dcf": 0.40, "multiples": 0.30, "epv": 0.30},
    "CYCLICAL":   {"dcf": 0.30, "multiples": 0.70, "epv": 0.0},
}


# ---------------------------------------------------------------- parsing
def _num(text: str, label: str) -> float | None:
    m = re.search(rf"^{re.escape(label)}:\s*(-?[\d.]+)\s*$", text, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _text(text: str, label: str) -> str | None:
    m = re.search(rf"^{re.escape(label)}:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


def _col0(csv: str, row_name: str) -> float | None:
    for line in (csv or "").split("\n"):
        parts = line.split(",")
        if parts and parts[0].strip() == row_name:
            for cell in parts[1:]:
                cell = cell.strip()
                if cell:
                    try:
                        return float(cell)
                    except ValueError:
                        return None
    return None


def parse_fundamentals(financials: dict) -> dict:
    """Pull the IV inputs from raw/financials.json (fundamentals text block +
    income_statement CSV). Missing values come back as None."""
    fund = financials.get("fundamentals", "") or ""
    inc = financials.get("income_statement", "") or ""
    revenue = _num(fund, "Revenue (TTM)")
    gross_profit = _num(fund, "Gross Profit")
    gross_margin = (gross_profit / revenue) if (gross_profit is not None and revenue) else None
    return {
        "market_cap": _num(fund, "Market Cap"),
        "eps": _num(fund, "EPS (TTM)"),
        "forward_eps": _num(fund, "Forward EPS"),
        "beta": _num(fund, "Beta"),
        "ebitda": _num(fund, "EBITDA"),
        "net_income": _num(fund, "Net Income"),
        "fcf": _num(fund, "Free Cash Flow"),
        "revenue": revenue,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "revenue_growth": _num(fund, "Revenue Growth"),
        "sector": _text(fund, "Sector"),
        "industry": _text(fund, "Industry"),
        "diluted_shares": _col0(inc, "Diluted Average Shares"),
        "tax": _col0(inc, "Tax Rate For Calcs"),
        "ebit": _col0(inc, "EBIT"),
        "currency": (financials.get("financial_currency") or "USD"),
    }


def classify_valuation_profile(fund: dict, ticker: str | None) -> str:
    if ticker and ticker.upper() in NAV_PROXY_TICKERS:
        return "NAV_PROXY"
    sector = (fund.get("sector") or "")
    if sector.startswith("Financial") or "Bank" in sector:
        return "FINANCIAL"
    ni = fund.get("net_income")
    if ni is not None and ni <= 0:
        return "UNPROFITABLE"
    # Profitable but capex-heavy (negative FCF, e.g. AMKR) stays STANDARD: its DCF
    # base FCF is normalized (NI × FCF_NORM_CONVERSION) so the cash-flow leg holds.
    return "STANDARD"


def classify_growth_tier(fund: dict) -> str:
    """Within STANDARD: COMPOUNDER / MATURE / CYCLICAL, from free yfinance metrics.
    Cyclical (sector/industry) is checked first; then compounder (growth + margin);
    else mature — including when revenue growth is unknown (conservative fallback,
    so an unverified name keeps EPV's weight rather than over-crediting growth)."""
    sector = (fund.get("sector") or "")
    industry = (fund.get("industry") or "").lower()
    if sector in CYCLICAL_SECTORS or any(k in industry for k in CYCLICAL_INDUSTRY_KEYWORDS):
        return "CYCLICAL"
    g = fund.get("revenue_growth")
    gm = fund.get("gross_margin")
    if (g is not None and gm is not None
            and g >= COMPOUNDER_MIN_GROWTH and gm >= COMPOUNDER_MIN_GROSS_MARGIN):
        return "COMPOUNDER"
    return "MATURE"


def normalized_fcf_base(fund: dict) -> float | None:
    """Through-cycle FCF base for the DCF. When current FCF is capex-depressed
    (FCF/NI < FCF_NORM_CONVERSION, incl. negative) and NI is positive, substitute
    NI × FCF_NORM_CONVERSION so a capex peak doesn't crater the cash-flow leg.
    Otherwise use actual FCF (None/negative when NI ≤ 0 → DCF drops out)."""
    fcf, ni = fund.get("fcf"), fund.get("net_income")
    if ni is not None and ni > 0:
        if fcf is None or fcf <= 0 or (fcf / ni) < FCF_NORM_CONVERSION:
            return ni * FCF_NORM_CONVERSION
    return fcf


def _percentile(vals: list, q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    return s[lo] + (s[lo + 1] - s[lo]) * frac if lo + 1 < len(s) else s[-1]


def _weighted_blend(values: dict, weights: dict) -> float | None:
    """Weighted mean over available (non-None) methods; weights renormalize across
    whatever is present. Methods with weight 0 (e.g. EPV for compounders) are excluded."""
    pairs = [(weights.get(m, 0.0), v) for m, v in values.items()
             if isinstance(v, (int, float)) and weights.get(m, 0.0) > 0]
    tw = sum(w for w, _ in pairs)
    return (sum(w * v for w, v in pairs) / tw) if tw > 0 else None


# ------------------------------------------------------- cost of capital
def cost_of_capital(fund: dict, net_debt: dict, risk_free: float) -> dict:
    beta = fund.get("beta")
    coe = max((risk_free + (beta if beta is not None else 1.0) * ERP), DISCOUNT_RATE_FLOOR)
    tax = fund.get("tax") if fund.get("tax") is not None else 0.21
    cod = risk_free + COST_OF_DEBT_SPREAD
    nd = (net_debt or {}).get("net_debt")
    mktcap = fund.get("market_cap")
    weight_debt = 0.0
    if nd is not None and nd > 0 and mktcap:
        weight_debt = nd / (nd + mktcap)
    wacc = weight_debt * cod * (1 - tax) + (1 - weight_debt) * coe
    wacc = max(wacc, DISCOUNT_RATE_FLOOR)
    return {"cost_of_equity": coe, "cost_of_debt": cod, "wacc": wacc,
            "weight_debt": weight_debt, "tax": tax}


# -------------------------------------------------------------- methods
def dcf_value(fund: dict, wacc: float, near_g: float, term_g: float,
              horizon: int, net_debt: dict) -> float | None:
    fcf = fund.get("fcf")
    shares = fund.get("diluted_shares")
    if not fcf or fcf <= 0 or not shares or shares <= 0 or wacc <= term_g:
        return None
    pv = 0.0
    f = fcf
    for t in range(1, horizon + 1):
        g = near_g + (term_g - near_g) * (t - 1) / (horizon - 1) if horizon > 1 else term_g
        f = f * (1 + g)
        pv += f / (1 + wacc) ** t
    terminal = f * (1 + term_g) / (wacc - term_g)
    pv += terminal / (1 + wacc) ** horizon
    equity = pv - ((net_debt or {}).get("net_debt") or 0.0)
    return equity / shares


def dcf_value_phased(fund: dict, wacc: float, near_g: float, term_g: float,
                     horizon: int, net_debt: dict,
                     fcf_override: float | None = None) -> float | None:
    """3-phase DCF: years 1–3 at near_g, years 4–horizon linear-decay near_g → term_g,
    then terminal perpetuity at term_g. `fcf_override` lets the caller pass a normalized
    (through-cycle) FCF base instead of the noisy trailing FCF."""
    fcf = fcf_override if fcf_override is not None else fund.get("fcf")
    shares = fund.get("diluted_shares")
    if not fcf or fcf <= 0 or not shares or shares <= 0 or wacc <= term_g:
        return None
    high_phase = min(3, horizon)
    rem = horizon - high_phase
    pv = 0.0
    f = fcf
    for t in range(1, horizon + 1):
        if t <= high_phase:
            g = near_g
        else:
            g = near_g + (term_g - near_g) * ((t - high_phase) / rem) if rem > 0 else term_g
        f = f * (1 + g)
        pv += f / (1 + wacc) ** t
    terminal = f * (1 + term_g) / (wacc - term_g)
    pv += terminal / (1 + wacc) ** horizon
    equity = pv - ((net_debt or {}).get("net_debt") or 0.0)
    return equity / shares


def epv_value(fund: dict, cost_of_equity: float) -> float | None:
    """Earnings Power Value of equity = TTM net income capitalised at the cost
    of equity, no growth (the conservative floor). NI is TTM and post-interest,
    so this is an equity-level value — no net-debt adjustment, no quarterly/TTM
    mixing. Returns None if NI <= 0 or shares unknown."""
    ni = fund.get("net_income")
    shares = fund.get("diluted_shares")
    if not ni or ni <= 0 or not shares or shares <= 0 or cost_of_equity <= 0:
        return None
    return (ni / cost_of_equity) / shares


def multiples_value(fund: dict, peer_ratios: dict, net_debt: dict) -> dict:
    eps = fund.get("eps")
    pes = [v.get("ttm_pe") for v in (peer_ratios or {}).values()
           if isinstance(v, dict) and isinstance(v.get("ttm_pe"), (int, float)) and v.get("ttm_pe") > 0]
    pe_implied = (median(pes) * eps) if pes and eps and eps > 0 else None
    # Peer EV/EBITDA is not available in raw (no peer market cap in peer_ratios);
    # left None rather than fabricated.
    return {"pe_implied": pe_implied, "ev_ebitda_implied": None}


def reverse_dcf_growth(fund: dict, wacc: float, term_g: float, horizon: int,
                       net_debt: dict, price: float) -> float | None:
    if not price or price <= 0 or not fund.get("fcf") or fund["fcf"] <= 0:
        return None
    lo, hi = -0.5, NEAR_TERM_GROWTH_CAP
    flo = dcf_value(fund, wacc, lo, term_g, horizon, net_debt)
    fhi = dcf_value(fund, wacc, hi, term_g, horizon, net_debt)
    if flo is None or fhi is None or not (flo <= price <= fhi):
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        fmid = dcf_value(fund, wacc, mid, term_g, horizon, net_debt)
        if fmid is None:
            return None
        if abs(fmid - price) < 1e-4:
            return mid
        if fmid < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------- MC EV from forward dist
def mc_ev_from_forward(forward_probabilities: dict | None) -> float | None:
    if not isinstance(forward_probabilities, dict):
        return None
    rows = None
    sc = forward_probabilities.get("scenarios")
    if isinstance(sc, dict) and sc:           # real schema: {bull,base,bear} dicts
        rows = list(sc.values())
    elif isinstance(sc, list) and sc:
        rows = sc
    if rows is None:
        for key in ("terminal_zones", "zones", "bins", "partition"):
            v = forward_probabilities.get(key)
            if isinstance(v, list) and v:
                rows = v
                break
    if rows is None:
        return None
    ev = 0.0
    got = False
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = next((r[k] for k in ("probability", "prob", "p", "weight") if isinstance(r.get(k), (int, float))), None)
        tgt = next((r[k] for k in ("target", "price", "level", "value") if isinstance(r.get(k), (int, float))), None)
        if p is None or tgt is None:
            continue
        ev += p * tgt
        got = True
    return ev if got else None


# ----------------------------------------------------- orchestration
def compute_intrinsic_value(financials: dict, net_debt: dict, reference: dict,
                            peer_ratios: dict, risk_free: float,
                            forward_probabilities: dict | None = None,
                            ticker: str | None = None,
                            fx_rate: float | None = None) -> dict:
    fund = parse_fundamentals(financials)
    ticker = ticker or financials.get("ticker")
    profile = classify_valuation_profile(fund, ticker)
    cc = cost_of_capital(fund, net_debt, risk_free)
    wacc, coe = cc["wacc"], cc["cost_of_equity"]
    price = (reference or {}).get("reference_price")

    # near-term growth from analyst forward EPS, capped
    eps, feps = fund.get("eps"), fund.get("forward_eps")
    if eps and feps and eps > 0:
        near_g = min(NEAR_TERM_GROWTH_CAP, max(feps / eps - 1.0, -0.2))
    else:
        near_g = TERMINAL_GROWTH

    skipped: list[dict] = []
    methods: dict[str, Any] = {}
    fx_caveat = None

    # Foreign issuer: statements are in the reporting currency (e.g. TWD) while
    # eps and price are USD per ADR. Derive the FX rate from the data's own
    # internal consistency — fx = eps_USD × ADR_count / net_income_reporting — so
    # the reporting-ccy flows (EPV/DCF) convert to USD per ADR with NO external FX
    # feed and NO separate ADR-ratio lookup (both are already implicit in eps vs
    # the reporting-ccy NI). If that anchor is missing (no eps, or NI <= 0) we
    # skip rather than guess. eps-based methods (multiples) are already USD per
    # ADR and must NOT be re-converted — see the multiples lines below.
    if fund["currency"] != "USD" and fx_rate is None:
        ni_rep, eps_usd, shr = fund.get("net_income"), fund.get("eps"), fund.get("diluted_shares")
        # Plausibility gate: the eps-anchored derivation is only valid when the
        # USD eps and the USD ADR price are on the SAME ADR basis. yfinance is
        # inconsistent here across ADRs — when eps is mis-scaled vs the price the
        # implied no-growth P/E (price/eps) comes out absurd (e.g. >60). TSM is
        # clean (~38x); IFNNY/STM/NOK are not. Compute only when the basis is
        # self-consistent; otherwise skip honestly rather than emit a wrong number.
        implied_pe = (price / eps_usd) if (price and eps_usd and eps_usd > 0) else None
        anchored = bool(ni_rep and ni_rep > 0 and eps_usd and shr and shr > 0)
        if anchored and implied_pe is not None and 3.0 <= implied_pe <= 60.0:
            fx_rate = eps_usd * shr / ni_rep
        else:
            why = ("no USD EPS anchor" if not anchored else
                   f"USD eps vs price implies P/E≈{implied_pe:.0f} — the ADR/FX basis "
                   f"is inconsistent, so a USD fair value can't be trusted")
            fx_caveat = (f"Foreign issuer ({fund['currency']} statements vs USD ADR price): "
                         f"{why}. Intrinsic value not computable — rely on the scenario EV.")
            skipped.append({"method": "all", "reason": fx_caveat})

    def conv(v):  # convert reporting-currency per-share to price currency
        if v is None:
            return None
        return v * fx_rate if (fx_rate and fund["currency"] != "USD") else v

    growth_tier = None
    scenario_drivers = None
    fair_value = {"bear": None, "base": None, "bull": None}

    if fx_caveat is None and profile == "STANDARD":
        growth_tier = classify_growth_tier(fund)
        weights = TIER_WEIGHTS[growth_tier]
        eps = fund.get("eps")

        # EPV — always computed; surfaced as the no-growth downside-stress floor.
        # Blended into the base only for MATURE (weight 0 for COMPOUNDER/CYCLICAL).
        epv = conv(epv_value(fund, coe))
        methods["epv"] = {"value": epv, "role": "downside-stress floor"}

        # Peer multiples: median for base, 25th/75th pct for the band (eps already price-ccy).
        pes = [v.get("ttm_pe") for v in (peer_ratios or {}).values()
               if isinstance(v, dict) and isinstance(v.get("ttm_pe"), (int, float)) and v.get("ttm_pe") > 0]
        peer_base = (median(pes) * eps) if pes and eps and eps > 0 else None
        peer_bear = (_percentile(pes, 0.25) * eps) if pes and eps and eps > 0 else None
        peer_bull = (_percentile(pes, 0.75) * eps) if pes and eps and eps > 0 else None
        methods["multiples"] = {"pe_implied": peer_base, "ev_ebitda_implied": None}

        # DCF — 3-phase on a normalized (through-cycle) FCF base so a capex peak
        # doesn't exclude the cash-flow leg (the old FCF/NI<0.5 hard exclusion is gone).
        fcf_norm = normalized_fcf_base(fund)
        dcf_base = conv(dcf_value_phased(fund, wacc, near_g, TERMINAL_GROWTH, HORIZON_YEARS, net_debt, fcf_norm))
        dcf_bear = dcf_bull = None
        if dcf_base is not None:
            dcf_bear = conv(dcf_value_phased(fund, wacc + DISCOUNT_DELTA, near_g - GROWTH_DELTA, TERMINAL_GROWTH, HORIZON_YEARS, net_debt, fcf_norm))
            dcf_bull = conv(dcf_value_phased(fund, wacc - DISCOUNT_DELTA, near_g + GROWTH_DELTA, TERMINAL_GROWTH, HORIZON_YEARS, net_debt, fcf_norm))
            fcf_act, ni = fund.get("fcf"), fund.get("net_income")
            fcf_quality = (fcf_act / ni) if (fcf_act and ni and ni > 0) else None
            methods["dcf"] = {"bear": dcf_bear, "base": dcf_base, "bull": dcf_bull,
                              "fcf_to_ni": round(fcf_quality, 2) if fcf_quality is not None else None,
                              "normalized": fcf_norm != fcf_act}
            methods["reverse_dcf"] = {"implied_growth":
                reverse_dcf_growth(fund, wacc, TERMINAL_GROWTH, HORIZON_YEARS, net_debt, price) if price else None}
        else:
            skipped.append({"method": "dcf", "reason": "FCF/earnings base not positive — DCF not meaningful"})

        # Tier-weighted blend (EPV included only where its weight > 0; weights
        # renormalize across whatever methods are available).
        base = _weighted_blend({"dcf": dcf_base, "multiples": peer_base, "epv": epv}, weights)
        bear = _weighted_blend({"dcf": dcf_bear, "multiples": peer_bear, "epv": epv}, weights)
        bull = _weighted_blend({"dcf": dcf_bull, "multiples": peer_bull, "epv": epv}, weights)
        if base is not None and bear is not None and bull is not None:
            lo, hi = min(bear, bull), max(bear, bull)
            bear, bull = min(lo, base), max(hi, base)   # guarantee bear ≤ base ≤ bull
        fair_value = {"bear": bear, "base": base, "bull": bull}
        if base is not None:
            epv_note = f"EPV {weights['epv']:.0%}" if weights["epv"] > 0 else "EPV stress-only"
            scenario_drivers = {
                "bear": "FCF growth −2pp, WACC +1pp, peer P/E 25th pct",
                "base": f"{growth_tier}: DCF {weights['dcf']:.0%} / peer {weights['multiples']:.0%} / {epv_note}",
                "bull": "FCF growth +2pp, WACC −1pp, peer P/E 75th pct",
            }
    elif fx_caveat is None and profile == "UNPROFITABLE":
        skipped += [{"method": "dcf", "reason": "no positive FCF/earnings base"},
                    {"method": "epv", "reason": "no positive operating earnings"}]
        methods["multiples"] = dict(multiples_value(fund, peer_ratios, net_debt).items())  # eps-based, already price-ccy
        methods["reverse_dcf"] = {"implied_growth": None}
        methods["note"] = "path-to-profitability dependent; fair value not anchored to current cash flows"
    elif fx_caveat is None and profile == "FINANCIAL":
        skipped += [{"method": "dcf", "reason": "FCF-DCF not the right model for financials"},
                    {"method": "epv", "reason": "use P/B / P/E / dividend-discount for financials"}]
        methods["multiples"] = dict(multiples_value(fund, peer_ratios, net_debt).items())  # eps-based, already price-ccy
        if methods["multiples"].get("pe_implied") is not None:
            fair_value = {"bear": None, "base": methods["multiples"]["pe_implied"], "bull": None}
    elif fx_caveat is None and profile == "NAV_PROXY":
        skipped.append({"method": "all", "reason": "asset/NAV-driven proxy — DCF/EPS intrinsic value not meaningful"})

    # eps sanity (all profiles): EPV and peer-multiples both rest on TTM eps. If
    # price/eps is absurd the eps feed is mis-scaled (a per-ticker data glitch —
    # e.g. STM showed P/E ~117 from a 20x-off eps), so the fair value can't be
    # trusted. Suppress rather than emit a wrong number (user rule).
    eps_chk = fund.get("eps")
    if fair_value.get("base") is not None and price and eps_chk and eps_chk > 0:
        implied_pe_final = price / eps_chk
        if not (3.0 <= implied_pe_final <= 60.0):
            skipped.append({"method": "all", "reason":
                f"price/eps implies P/E≈{implied_pe_final:.0f} — eps feed looks mis-scaled; "
                f"fair value suppressed. Rely on the scenario EV."})
            fair_value = {"bear": None, "base": None, "bull": None}
            methods = {}

    # Output sanity (all profiles): even with clean eps, a fair value that diverges
    # wildly from the market price is a peer-multiple/model blow-up, not a credible
    # point estimate (e.g. TIGR's peer-implied $37.6 vs a $4.79 price = +650%).
    # Suppress beyond [0.2x, 3x] of price rather than emit a misleading number.
    fvb = fair_value.get("base")
    if fvb is not None and price and not (0.2 * price <= fvb <= 3.0 * price):
        skipped.append({"method": "all", "reason":
            f"fair value {fvb:.2f} diverges implausibly from price {price:.2f} "
            f"(outside 0.2x-3x) — not a credible estimate; suppressed. Rely on the scenario EV."})
        fair_value = {"bear": None, "base": None, "bull": None}
        methods = {}

    if fair_value.get("base") is None:
        scenario_drivers = None   # a guard suppressed the value — no drivers to report

    iv_base = fair_value.get("base")
    margin = ((iv_base - price) / price) if (iv_base is not None and price) else None

    mc_ev = mc_ev_from_forward(forward_probabilities)
    recon = {"mc_ev": mc_ev, "iv_base": iv_base, "price": price,
             "iv_vs_ev_pct": None, "iv_vs_price_pct": margin, "flag": "N/A"}
    if iv_base is not None and mc_ev and price:
        recon["iv_vs_ev_pct"] = (iv_base - mc_ev) / mc_ev
        same_side = (iv_base - price) * (mc_ev - price) >= 0
        recon["flag"] = "AGREE" if (same_side and abs(recon["iv_vs_ev_pct"]) <= RECONCILE_TOLERANCE) else "DIVERGE"

    applicable = [m for m in ("dcf", "epv", "multiples", "reverse_dcf")
                  if m in methods and methods[m] not in (None, {}, {"value": None})]

    return {
        "trade_date": financials.get("trade_date"),
        "ticker": ticker,
        "profile": profile,
        "growth_tier": growth_tier,
        "scenario_drivers": scenario_drivers,
        "currency": fund["currency"],
        "fx_rate": fx_rate,
        "fx_caveat": fx_caveat,
        "applicable_methods": applicable,
        "skipped_methods": skipped,
        "inputs": {
            "risk_free": risk_free, "erp": ERP, "beta": fund.get("beta"),
            "cost_of_equity": coe, "wacc": wacc, "tax_rate": cc["tax"],
            "fcf_ttm": fund.get("fcf"), "eps_ttm": fund.get("eps"),
            "forward_eps": fund.get("forward_eps"), "near_term_growth": near_g,
            "terminal_growth": TERMINAL_GROWTH, "horizon_years": HORIZON_YEARS,
            "diluted_shares": fund.get("diluted_shares"),
            "net_debt": (net_debt or {}).get("net_debt"), "ebit": fund.get("ebit"),
            "ebitda": fund.get("ebitda"), "market_cap": fund.get("market_cap"),
        },
        "methods": methods,
        "fair_value": fair_value,
        "margin_of_safety_pct": margin,
        "reconciliation": recon,
        "constants_note": (
            f"ERP={ERP:.1%}, terminal g={TERMINAL_GROWTH:.1%}, horizon={HORIZON_YEARS}y, "
            f"near-term-growth cap={NEAR_TERM_GROWTH_CAP:.0%}, discount floor={DISCOUNT_RATE_FLOOR:.0%}, "
            f"cost-of-debt spread={COST_OF_DEBT_SPREAD:.1%}, bear/bull = growth ±{GROWTH_DELTA:.0%} / "
            f"discount ±{DISCOUNT_DELTA:.0%}; risk-free from 10-Y Treasury. "
            f"Tier-weighted blend (compounder 65/35 DCF/peer, mature 40/30/30 DCF/peer/EPV, "
            f"cyclical 30/70 DCF/peer); DCF on 3-phase normalized FCF "
            f"(NI×{FCF_NORM_CONVERSION:.0%} when capex-depressed); EPV is a downside stress test."
        ),
    }


# ----------------------------------------------------------- formatter
def _money(v, cur="$"):
    return f"{cur}{v:,.2f}" if isinstance(v, (int, float)) else "(n/a)"


def _pct(v):
    return f"{v:+.1%}" if isinstance(v, (int, float)) else "(n/a)"


def format_intrinsic_value_block(iv: dict) -> str:
    td = iv.get("trade_date", "?")
    head = f"\n\n## Intrinsic value (computed from fundamentals & balance-sheet data, trade_date {td})\n\n"
    inp = iv["inputs"]
    fv = iv["fair_value"]
    recon = iv["reconciliation"]
    cur = "$" if iv.get("currency") == "USD" else f"{iv.get('currency')} "

    if iv.get("fx_caveat") or all(v is None for v in fv.values()):
        reason = iv.get("fx_caveat") or "; ".join(
            f"{s['method']}: {s['reason']}" for s in iv.get("skipped_methods", [])) or "no applicable method"
        return (head + f"**Intrinsic value not computable** for this profile "
                f"(`{iv['profile']}`) — {reason}. **Rely on the scenario EV.** "
                "Do not invent a fair value.\n")

    lines = [head]
    tier = iv.get("growth_tier")
    tier_str = f" · **Tier:** {tier}" if tier else ""
    lines.append(f"**Profile:** {iv['profile']}{tier_str} · **Methods:** {', '.join(iv['applicable_methods']) or '(none)'}\n\n")
    lines.append("| Fair value | Bear | Base | Bull |\n|---|---:|---:|---:|\n")
    lines.append(f"| per share | {_money(fv['bear'], cur)} | {_money(fv['base'], cur)} | {_money(fv['bull'], cur)} |\n\n")
    lines.append(f"**Margin of safety** (base vs reference {_money(recon['price'], cur)}): "
                 f"{_pct(iv['margin_of_safety_pct'])}.\n\n")
    sd = iv.get("scenario_drivers")
    if sd:
        lines.append(f"**Swing variables** — bear: {sd.get('bear')}; base: {sd.get('base')}; "
                     f"bull: {sd.get('bull')}.\n\n")
    lines.append(
        f"**Assumptions:** risk-free {inp['risk_free']:.2%}, beta {inp['beta']}, "
        f"cost of equity {inp['cost_of_equity']:.2%}, **WACC {inp['wacc']:.2%}**, "
        f"near-term growth {inp['near_term_growth']:+.1%} → terminal {inp['terminal_growth']:.1%} "
        f"over {inp['horizon_years']}y.\n\n")
    m = iv["methods"]
    if "epv" in m and m["epv"].get("value") is not None:
        lines.append(f"- EPV (no-growth downside-stress floor): {_money(m['epv']['value'], cur)}\n")
    if "multiples" in m and m["multiples"].get("pe_implied") is not None:
        lines.append(f"- Peer-P/E-implied: {_money(m['multiples']['pe_implied'], cur)}\n")
    if "reverse_dcf" in m and m["reverse_dcf"].get("implied_growth") is not None:
        lines.append(f"- Reverse-DCF: the current price implies ~{m['reverse_dcf']['implied_growth']:+.1%} FCF growth\n")
    lines.append(
        f"\n**Reconciliation:** IV base {_money(recon['iv_base'], cur)} vs 12-mo scenario EV "
        f"{_money(recon['mc_ev'], cur)} vs price {_money(recon['price'], cur)} → "
        f"IV-vs-EV {_pct(recon['iv_vs_ev_pct'])}, **{recon['flag']}**. "
        "If DIVERGE, the report must address why the valuation and the scenario distribution disagree.\n\n")
    if iv.get("skipped_methods"):
        lines.append("*Methods skipped:* " + "; ".join(
            f"{s['method']} ({s['reason']})" for s in iv["skipped_methods"]) + ".\n\n")
    lines.append(f"*{iv['constants_note']} Use these values verbatim; the rating still "
                 "derives from the scenario engine — intrinsic value is decision-support.*\n")
    return "".join(lines)


# ----------------------------------------------------------- risk-free
def fetch_risk_free() -> float:
    """10-Y Treasury yield via ^TNX (quoted ×1 in %, /100 → decimal). Falls back
    to RISK_FREE_FALLBACK on any failure."""
    try:
        import yfinance as yf
        from tradingagents.dataflows.stockstats_utils import yf_retry
        h = yf_retry(lambda: yf.Ticker("^TNX").history(period="5d", auto_adjust=False))
        if h is None or h.empty:
            return RISK_FREE_FALLBACK
        v = float(h["Close"].iloc[-1]) / 100.0
        return v if 0.0 < v < 0.15 else RISK_FREE_FALLBACK
    except Exception:
        return RISK_FREE_FALLBACK
