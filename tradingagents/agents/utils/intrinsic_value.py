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
    return {
        "market_cap": _num(fund, "Market Cap"),
        "eps": _num(fund, "EPS (TTM)"),
        "forward_eps": _num(fund, "Forward EPS"),
        "beta": _num(fund, "Beta"),
        "ebitda": _num(fund, "EBITDA"),
        "net_income": _num(fund, "Net Income"),
        "fcf": _num(fund, "Free Cash Flow"),
        "revenue": _num(fund, "Revenue (TTM)"),
        "sector": _text(fund, "Sector"),
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
    ni, fcf = fund.get("net_income"), fund.get("fcf")
    if (ni is not None and ni <= 0) or (fcf is not None and fcf <= 0):
        return "UNPROFITABLE"
    return "STANDARD"


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


def epv_value(fund: dict, wacc: float, net_debt: dict) -> float | None:
    ebit = fund.get("ebit")
    shares = fund.get("diluted_shares")
    tax = fund.get("tax") if fund.get("tax") is not None else 0.21
    if not ebit or ebit <= 0 or not shares or shares <= 0 or wacc <= 0:
        return None
    ev = ebit * (1 - tax) / wacc
    equity = ev - ((net_debt or {}).get("net_debt") or 0.0)
    return equity / shares


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
    for key in ("scenarios", "terminal_zones", "zones", "bins", "partition"):
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

    # currency mismatch without FX → cannot compare to a USD price
    if fund["currency"] != "USD" and fx_rate is None:
        fx_caveat = (f"Financials reported in {fund['currency']}; USD FX rate "
                     f"unavailable — per-share USD intrinsic value not computed.")
        skipped.append({"method": "all", "reason": fx_caveat})

    def conv(v):  # convert reporting-currency per-share to price currency
        if v is None:
            return None
        return v * fx_rate if (fx_rate and fund["currency"] != "USD") else v

    fair_value = {"bear": None, "base": None, "bull": None}

    if fx_caveat is None and profile == "STANDARD":
        base = dcf_value(fund, wacc, near_g, TERMINAL_GROWTH, HORIZON_YEARS, net_debt)
        bear = dcf_value(fund, wacc + DISCOUNT_DELTA, near_g - GROWTH_DELTA, TERMINAL_GROWTH, HORIZON_YEARS, net_debt)
        bull = dcf_value(fund, wacc - DISCOUNT_DELTA, near_g + GROWTH_DELTA, TERMINAL_GROWTH, HORIZON_YEARS, net_debt)
        methods["dcf"] = {"bear": conv(bear), "base": conv(base), "bull": conv(bull)}
        methods["epv"] = {"value": conv(epv_value(fund, wacc, net_debt))}
        methods["multiples"] = {k: conv(v) for k, v in multiples_value(fund, peer_ratios, net_debt).items()}
        rdg = reverse_dcf_growth(fund, wacc, TERMINAL_GROWTH, HORIZON_YEARS, net_debt, price) if price else None
        methods["reverse_dcf"] = {"implied_growth": rdg}
        if base is not None:
            fair_value = {"bear": conv(bear), "base": conv(base), "bull": conv(bull)}
    elif fx_caveat is None and profile == "UNPROFITABLE":
        skipped += [{"method": "dcf", "reason": "no positive FCF/earnings base"},
                    {"method": "epv", "reason": "no positive operating earnings"}]
        methods["multiples"] = {k: conv(v) for k, v in multiples_value(fund, peer_ratios, net_debt).items()}
        methods["reverse_dcf"] = {"implied_growth": None}
        methods["note"] = "path-to-profitability dependent; fair value not anchored to current cash flows"
    elif fx_caveat is None and profile == "FINANCIAL":
        skipped += [{"method": "dcf", "reason": "FCF-DCF not the right model for financials"},
                    {"method": "epv", "reason": "use P/B / P/E / dividend-discount for financials"}]
        methods["multiples"] = {k: conv(v) for k, v in multiples_value(fund, peer_ratios, net_debt).items()}
        if methods["multiples"].get("pe_implied") is not None:
            fair_value = {"bear": None, "base": methods["multiples"]["pe_implied"], "bull": None}
    elif fx_caveat is None and profile == "NAV_PROXY":
        skipped.append({"method": "all", "reason": "asset/NAV-driven proxy — DCF/EPS intrinsic value not meaningful"})

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
            f"discount ±{DISCOUNT_DELTA:.0%}; risk-free from 10-Y Treasury."
        ),
    }


# ----------------------------------------------------------- formatter
def _money(v, cur="$"):
    return f"{cur}{v:,.2f}" if isinstance(v, (int, float)) else "(n/a)"


def _pct(v):
    return f"{v:+.1%}" if isinstance(v, (int, float)) else "(n/a)"


def format_intrinsic_value_block(iv: dict) -> str:
    td = iv.get("trade_date", "?")
    head = f"\n\n## Intrinsic value (computed from raw/ data, trade_date {td})\n\n"
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
    lines.append(f"**Profile:** {iv['profile']} · **Methods:** {', '.join(iv['applicable_methods']) or '(none)'}\n\n")
    lines.append("| Fair value | Bear | Base | Bull |\n|---|---:|---:|---:|\n")
    lines.append(f"| per share | {_money(fv['bear'], cur)} | {_money(fv['base'], cur)} | {_money(fv['bull'], cur)} |\n\n")
    lines.append(f"**Margin of safety** (base vs reference {_money(recon['price'], cur)}): "
                 f"{_pct(iv['margin_of_safety_pct'])}.\n\n")
    lines.append(
        f"**Assumptions:** risk-free {inp['risk_free']:.2%}, beta {inp['beta']}, "
        f"cost of equity {inp['cost_of_equity']:.2%}, **WACC {inp['wacc']:.2%}**, "
        f"near-term growth {inp['near_term_growth']:+.1%} → terminal {inp['terminal_growth']:.1%} "
        f"over {inp['horizon_years']}y.\n\n")
    m = iv["methods"]
    if "epv" in m and m["epv"].get("value") is not None:
        lines.append(f"- EPV (no-growth floor): {_money(m['epv']['value'], cur)}\n")
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
