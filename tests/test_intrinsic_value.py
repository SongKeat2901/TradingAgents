import pytest

pytestmark = pytest.mark.unit

_FUND_TXT = (
    "Name: Acme\nSector: Technology\nMarket Cap: 1000000000\n"
    "PE Ratio (TTM): 20\nForward PE: 16\nEPS (TTM): 5.0\nForward EPS: 6.0\n"
    "Beta: 1.2\nEBITDA: 200000000\nNet Income: 100000000\nFree Cash Flow: 90000000\n"
    "Revenue (TTM): 800000000\n"
)
_INC = (
    "\nTax Rate For Calcs,0.15,0.15\nDiluted Average Shares,50000000,50000000\n"
    "EBIT,150000000,140000000\n"
)


def _fin(fund=_FUND_TXT, inc=_INC, ccy="USD"):
    return {"ticker": "ACME", "financial_currency": ccy, "fundamentals": fund,
            "income_statement": inc, "cashflow": "", "balance_sheet": ""}


# Compounder fixture: high revenue growth + high gross margin, capex-depressed FCF
# (FCF/NI = 27/90 = 0.30 < 0.8 → DCF normalized to NI*0.8). MSFT-shaped.
_FUND_CMP = (
    "Name: Compoundo\nSector: Technology\nIndustry: Software - Infrastructure\n"
    "Market Cap: 3000000000000\nPE Ratio (TTM): 36\nForward PE: 30\n"
    "EPS (TTM): 12.0\nForward EPS: 14.0\nBeta: 1.0\nEBITDA: 150000000000\n"
    "Net Income: 90000000000\nFree Cash Flow: 27000000000\nRevenue (TTM): 270000000000\n"
    "Gross Profit: 190000000000\nRevenue Growth: 0.18\n"
)
_INC_CMP = ("\nTax Rate For Calcs,0.15,0.15\nDiluted Average Shares,7500000000,7500000000\n"
            "EBIT,100000000000,95000000000\n")


# ---- Task 1: parser + classifier ----
def test_parse_fundamentals():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals
    f = parse_fundamentals(_fin())
    assert f["fcf"] == 90000000 and f["eps"] == 5.0 and f["forward_eps"] == 6.0
    assert f["beta"] == 1.2 and f["net_income"] == 100000000 and f["sector"] == "Technology"
    assert f["diluted_shares"] == 50000000 and abs(f["tax"] - 0.15) < 1e-9
    assert f["ebit"] == 150000000 and f["ebitda"] == 200000000 and f["currency"] == "USD"


def test_classify_profiles():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_valuation_profile
    assert classify_valuation_profile(parse_fundamentals(_fin()), "ACME") == "STANDARD"
    loss = _FUND_TXT.replace("Net Income: 100000000", "Net Income: -50000000")
    assert classify_valuation_profile(parse_fundamentals(_fin(loss)), "X") == "UNPROFITABLE"
    fin = _FUND_TXT.replace("Sector: Technology", "Sector: Financial Services")
    assert classify_valuation_profile(parse_fundamentals(_fin(fin)), "X") == "FINANCIAL"
    assert classify_valuation_profile(parse_fundamentals(_fin()), "MSTR") == "NAV_PROXY"


# ---- Task 2: cost of capital ----
def test_cost_of_capital():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, cost_of_capital
    cc = cost_of_capital(parse_fundamentals(_fin()), {"net_debt": -50000000}, risk_free=0.04)
    assert abs(cc["cost_of_equity"] - 0.10) < 1e-6   # 0.04 + 1.2*0.05
    assert cc["wacc"] >= 0.08                          # floor


# ---- Task 3: methods ----
def test_dcf_known():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, dcf_value
    f = parse_fundamentals(_fin())
    v = dcf_value(f, wacc=0.10, near_g=0.10, term_g=0.025, horizon=5, net_debt={"net_debt": -50000000})
    assert v is not None and 30 < v < 200


def test_epv_floor():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, epv_value
    # EPV of equity = TTM NI / cost_of_equity / shares = 100M / 0.10 / 50M = $20
    v = epv_value(parse_fundamentals(_fin()), cost_of_equity=0.10)
    assert abs(v - (100000000 / 0.10 / 50000000)) < 1e-6


def test_reverse_dcf_recovers_growth():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, dcf_value, reverse_dcf_growth
    f = parse_fundamentals(_fin())
    price = dcf_value(f, 0.10, 0.12, 0.025, 5, {"net_debt": 0})
    g = reverse_dcf_growth(f, 0.10, 0.025, 5, {"net_debt": 0}, price)
    assert g is not None and abs(g - 0.12) < 0.01


def test_multiples():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, multiples_value
    pr = {"PEERA": {"ttm_pe": 18}, "PEERB": {"ttm_pe": 22}}
    m = multiples_value(parse_fundamentals(_fin()), pr, {"net_debt": -50000000})
    assert abs(m["pe_implied"] - 20 * 5.0) < 1e-6   # median peer P/E (20) * EPS (5)


# ---- Task 4: orchestration ----
def test_compute_standard_assembles_range_and_reconciliation():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    fwd = {"scenarios": [{"probability": 0.3, "target": 140},
                         {"probability": 0.4, "target": 110},
                         {"probability": 0.3, "target": 80}]}
    iv = compute_intrinsic_value(_fin(), {"net_debt": -50000000}, {"reference_price": 100.0},
                                 {"PEERA": {"ttm_pe": 18}}, risk_free=0.04,
                                 forward_probabilities=fwd, ticker="ACME")
    assert iv["profile"] == "STANDARD"
    fv = iv["fair_value"]
    assert fv["bear"] <= fv["base"] <= fv["bull"]
    r = iv["reconciliation"]
    assert r["mc_ev"] == pytest.approx(0.3 * 140 + 0.4 * 110 + 0.3 * 80)
    assert r["flag"] in ("AGREE", "DIVERGE")


def test_unprofitable_skips_dcf_with_reason():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    loss = _FUND_TXT.replace("Net Income: 100000000", "Net Income: -50000000").replace(
        "Free Cash Flow: 90000000", "Free Cash Flow: -10000000")
    iv = compute_intrinsic_value(_fin(loss), {"net_debt": 0}, {"reference_price": 50.0},
                                 {}, risk_free=0.04, ticker="X")
    assert iv["profile"] == "UNPROFITABLE"
    assert any(s["method"] == "dcf" for s in iv["skipped_methods"])


def test_foreign_adr_computes_via_derived_fx():
    # Foreign ADR (TWD statements, USD eps/price): fx is derived from the data
    # (eps * shares / net_income) so the IV is computed in USD per ADR, no skip.
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(ccy="TWD"), {"net_debt": 0}, {"reference_price": 100.0},
                                 {"PEERA": {"ttm_pe": 18}}, risk_free=0.04, ticker="X", fx_rate=None)
    assert iv["currency"] == "TWD"
    assert iv.get("fx_caveat") is None                      # no longer skipped
    assert abs(iv["fx_rate"] - (5.0 * 50000000 / 100000000)) < 1e-9   # = 2.5
    # EPV (USD/ADR) == eps/coe; coe = max(0.04 + 1.2*0.05, floor 0.08) = 0.10
    assert abs(iv["methods"]["epv"]["value"] - 5.0 / 0.10) < 1.0       # ~$50
    # peer-multiple uses USD eps directly (NOT re-converted by fx): 18 * 5.0 = $90
    assert abs(iv["methods"]["multiples"]["pe_implied"] - 90.0) < 1e-6
    assert iv["fair_value"]["base"] is not None


def test_foreign_adr_skips_without_eps_anchor():
    # No USD EPS to anchor the FX derivation → honest skip, no fabricated value.
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    no_eps = _FUND_TXT.replace("EPS (TTM): 5.0", "EPS (TTM): ")
    iv = compute_intrinsic_value(_fin(no_eps, ccy="TWD"), {"net_debt": 0}, {"reference_price": 100.0},
                                 {}, risk_free=0.04, ticker="X", fx_rate=None)
    assert iv.get("fx_caveat") and iv["fair_value"]["base"] is None


def test_foreign_adr_skips_on_implausible_pe():
    # eps mis-scaled vs price → implied P/E absurd (1000/5 = 200) → skip, no wrong number.
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(ccy="TWD"), {"net_debt": 0}, {"reference_price": 1000.0},
                                 {}, risk_free=0.04, ticker="X", fx_rate=None)
    assert iv.get("fx_caveat") and iv["fair_value"]["base"] is None
    assert "P/E" in iv["fx_caveat"]


def test_usd_eps_mis_scale_suppresses_fair_value():
    # USD profile, but price/eps absurd (1000/5 = 200) → eps feed mis-scaled →
    # fair value suppressed rather than a wrong number (covers the STM case).
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(), {"net_debt": 0}, {"reference_price": 1000.0},
                                 {"PEERA": {"ttm_pe": 18}}, risk_free=0.04, ticker="ACME")
    assert iv["fair_value"]["base"] is None
    assert any("mis-scaled" in s["reason"] for s in iv["skipped_methods"])


def test_implausible_fair_value_vs_price_suppressed():
    # eps P/E is sane (20/5=4) but peer multiple blows the fair value far above
    # price (FINANCIAL → base = 30*5 = 150 vs price 20) → output-sanity suppress.
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    fin = _FUND_TXT.replace("Sector: Technology", "Sector: Financial Services")
    iv = compute_intrinsic_value(_fin(fin), {"net_debt": 0}, {"reference_price": 20.0},
                                 {"P": {"ttm_pe": 30}}, risk_free=0.04, ticker="BANK")
    assert iv["fair_value"]["base"] is None
    assert any("diverges" in s["reason"] for s in iv["skipped_methods"])


# ---- Task 5: formatter ----
def test_format_block_standard():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value, format_intrinsic_value_block
    iv = compute_intrinsic_value(_fin(), {"net_debt": -50000000}, {"reference_price": 100.0},
                                 {"PEERA": {"ttm_pe": 18}}, risk_free=0.04, ticker="ACME")
    b = format_intrinsic_value_block(iv)
    assert "## Intrinsic value" in b and "Margin of safety" in b and "WACC" in b


def test_format_block_not_computable():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value, format_intrinsic_value_block
    b = format_intrinsic_value_block(
        compute_intrinsic_value(_fin(), {"net_debt": 0}, {"reference_price": 100.0},
                                {}, risk_free=0.04, ticker="MSTR"))
    assert "not computable" in b.lower()


# ---- Task 6: risk-free fallback (no network) ----
def test_fetch_risk_free_fallback(monkeypatch):
    import tradingagents.agents.utils.intrinsic_value as iv
    import builtins
    real_import = builtins.__import__
    def boom(name, *a, **k):
        if name == "yfinance":
            raise ImportError("no yfinance")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", boom)
    rf = iv.fetch_risk_free()
    assert isinstance(rf, float) and 0.0 < rf < 0.10


def test_mc_ev_scenarios_dict():
    from tradingagents.agents.utils.intrinsic_value import mc_ev_from_forward
    fwd = {"scenarios": {"bull": {"probability": 0.5, "target": 120},
                         "base": {"probability": 0.3, "target": 100},
                         "bear": {"probability": 0.2, "target": 80}}}
    assert mc_ev_from_forward(fwd) == pytest.approx(0.5 * 120 + 0.3 * 100 + 0.2 * 80)


def test_capex_depressed_fcf_normalized_into_dcf():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    # NI > 0 but FCF < 0 (capex-heavy): the DCF is no longer excluded — its base FCF
    # is normalized to NI*0.8 so the cash-flow leg still contributes (new methodology).
    f = _FUND_TXT.replace("Free Cash Flow: 90000000", "Free Cash Flow: -20000000")
    iv = compute_intrinsic_value(_fin(f), {"net_debt": 100000000}, {"reference_price": 60.0},
                                 {"PEERA": {"ttm_pe": 18}}, risk_free=0.04, ticker="AMKR")
    assert iv["profile"] == "STANDARD"
    assert "dcf" in iv["methods"] and iv["methods"]["dcf"]["base"] is not None
    assert not any(s["method"] == "dcf" for s in iv["skipped_methods"])
    assert iv["fair_value"]["base"] is not None


# ---- Task 7: growth-tier classification ----
def test_classify_growth_tier_compounder():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_growth_tier
    assert classify_growth_tier(parse_fundamentals(_fin(_FUND_CMP, _INC_CMP))) == "COMPOUNDER"


def test_classify_growth_tier_mature_when_growth_unknown():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_growth_tier
    # _FUND_TXT has no Revenue Growth / Gross Profit → conservative MATURE fallback
    assert classify_growth_tier(parse_fundamentals(_fin())) == "MATURE"


def test_classify_growth_tier_mature_low_growth():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_growth_tier
    slow = _FUND_CMP.replace("Revenue Growth: 0.18", "Revenue Growth: 0.04")
    assert classify_growth_tier(parse_fundamentals(_fin(slow, _INC_CMP))) == "MATURE"


def test_classify_growth_tier_cyclical_semis():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, classify_growth_tier
    semi = _FUND_TXT.replace("Sector: Technology", "Sector: Technology\nIndustry: Semiconductors")
    assert classify_growth_tier(parse_fundamentals(_fin(semi))) == "CYCLICAL"


def test_parse_growth_and_gross_margin():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals
    f = parse_fundamentals(_fin(_FUND_CMP, _INC_CMP))
    assert f["revenue_growth"] == pytest.approx(0.18)
    assert f["gross_margin"] == pytest.approx(190000000000 / 270000000000)


# ---- Task 8: normalized forward-FCF DCF ----
def test_normalized_fcf_base():
    from tradingagents.agents.utils.intrinsic_value import parse_fundamentals, normalized_fcf_base
    f = parse_fundamentals(_fin(_FUND_CMP, _INC_CMP))  # FCF 27B, NI 90B → 0.30 < 0.8
    assert normalized_fcf_base(f) == pytest.approx(90000000000 * 0.80)
    healthy = parse_fundamentals(_fin())  # FCF 90M, NI 100M → 0.90 ≥ 0.8 → actual
    assert normalized_fcf_base(healthy) == pytest.approx(90000000)


# ---- Task 9: tier-weighted blend ----
def test_compounder_base_dcf_peer_weighted_no_epv():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(_FUND_CMP, _INC_CMP), {"net_debt": 0},
                                 {"reference_price": 441.0},
                                 {"A": {"ttm_pe": 30}, "B": {"ttm_pe": 34}},
                                 risk_free=0.04, ticker="CMP")
    assert iv["growth_tier"] == "COMPOUNDER"
    m = iv["methods"]
    assert m["epv"]["value"] is not None              # EPV still computed (stress)
    dcf, peer = m["dcf"]["base"], m["multiples"]["pe_implied"]
    assert iv["fair_value"]["base"] == pytest.approx(0.65 * dcf + 0.35 * peer, rel=1e-3)


def test_mature_base_includes_epv():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(), {"net_debt": -50000000}, {"reference_price": 100.0},
                                 {"A": {"ttm_pe": 18}}, risk_free=0.04, ticker="ACME")
    assert iv["growth_tier"] == "MATURE"
    m = iv["methods"]
    dcf, peer, epv = m["dcf"]["base"], m["multiples"]["pe_implied"], m["epv"]["value"]
    assert iv["fair_value"]["base"] == pytest.approx(0.40 * dcf + 0.30 * peer + 0.30 * epv, rel=1e-3)


def test_weights_renormalize_when_method_missing():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    # Compounder with no peers → peer leg drops → base is the DCF leg alone.
    iv = compute_intrinsic_value(_fin(_FUND_CMP, _INC_CMP), {"net_debt": 0},
                                 {"reference_price": 441.0}, {}, risk_free=0.04, ticker="CMP")
    assert iv["growth_tier"] == "COMPOUNDER"
    assert iv["fair_value"]["base"] == pytest.approx(iv["methods"]["dcf"]["base"], rel=1e-3)


def test_band_ordered_with_scenario_drivers():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(_FUND_CMP, _INC_CMP), {"net_debt": 0},
                                 {"reference_price": 441.0},
                                 {"A": {"ttm_pe": 28}, "B": {"ttm_pe": 30}, "C": {"ttm_pe": 34}},
                                 risk_free=0.04, ticker="CMP")
    fv = iv["fair_value"]
    assert fv["bear"] <= fv["base"] <= fv["bull"]
    assert set(iv["scenario_drivers"]) >= {"bear", "base", "bull"}


def test_compounder_base_above_old_epv_median():
    # Regression for the MSFT undervaluation: the old median(EPV, peer) blend would sit
    # at/below the EPV-peer midpoint; the new compounder blend (DCF+peer, EPV excluded)
    # must land strictly above EPV — i.e. EPV no longer drags the base down.
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value
    iv = compute_intrinsic_value(_fin(_FUND_CMP, _INC_CMP), {"net_debt": 0},
                                 {"reference_price": 441.0},
                                 {"A": {"ttm_pe": 30}, "B": {"ttm_pe": 34}},
                                 risk_free=0.04, ticker="CMP")
    epv = iv["methods"]["epv"]["value"]
    old_median = (epv + iv["methods"]["multiples"]["pe_implied"]) / 2
    assert iv["fair_value"]["base"] > old_median


def test_format_block_shows_tier_and_epv_stress():
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value, format_intrinsic_value_block
    iv = compute_intrinsic_value(_fin(_FUND_CMP, _INC_CMP), {"net_debt": 0},
                                 {"reference_price": 441.0},
                                 {"A": {"ttm_pe": 30}, "B": {"ttm_pe": 34}},
                                 risk_free=0.04, ticker="CMP")
    b = format_intrinsic_value_block(iv)
    assert "COMPOUNDER" in b and "stress" in b.lower()
