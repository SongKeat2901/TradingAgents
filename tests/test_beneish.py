import pytest

pytestmark = pytest.mark.unit


def test_fetch_financials_includes_annual_keys(monkeypatch):
    import tradingagents.agents.utils.agent_utils as au
    import tradingagents.agents.researcher as R

    class _FakeTool:
        def __init__(self, label): self.label = label
        def invoke(self, args):
            return f"{self.label}:{args.get('freq', 'quarterly')}"

    monkeypatch.setattr(au, "get_fundamentals", _FakeTool("F"))
    monkeypatch.setattr(au, "get_balance_sheet", _FakeTool("BS"))
    monkeypatch.setattr(au, "get_cashflow", _FakeTool("CF"))
    monkeypatch.setattr(au, "get_income_statement", _FakeTool("IS"))
    monkeypatch.setattr(R, "_fetch_financial_currency", lambda t: "USD")

    b = R._fetch_financials("MSFT", "2026-06-30")
    # quarterly (default) + annual variants present
    assert b["balance_sheet"] == "BS:quarterly"
    assert b["balance_sheet_annual"] == "BS:annual"
    assert b["income_statement_annual"] == "IS:annual"
    assert b["cashflow_annual"] == "CF:annual"


from tradingagents.agents.utils.distress_screens import compute_beneish_m, format_beneish_block


def _side(**kw):
    base = dict(receivables=100.0, sales=1000.0, cogs=600.0, current_assets=400.0,
                ppe=500.0, total_assets=1000.0, sga=100.0, depreciation=50.0, total_equity=600.0)
    base.update(kw)
    return base

# clean books: current == prior (all 8 ratios == 1), net_income==cfo (TATA=0) -> M = -2.48
_CLEAN = {"sector": "Technology", "beneish_inputs": {
    "current": dict(_side(), net_income=100.0, cfo=100.0),
    "prior": _side()}}


def test_beneish_clean_books_normal():
    r = compute_beneish_m(_CLEAN)
    assert r["applicable"] is True
    assert r["m_score"] == -2.48
    assert r["flag"] == "normal"


def test_beneish_manipulation_pattern_elevated():
    # spike receivables (DSRI), sales (SGI), and accruals (TATA: NI>>CFO)
    manip = {"sector": "Industrials", "beneish_inputs": {
        "current": dict(_side(sales=1500.0, receivables=300.0, cogs=900.0, total_assets=1200.0),
                        net_income=300.0, cfo=40.0),
        "prior": _side()}}
    r = compute_beneish_m(manip)
    assert r["flag"] == "elevated" and r["m_score"] > -1.78


def test_beneish_financials_skipped():
    r = compute_beneish_m(dict(_CLEAN, sector="Financial Services"))
    assert r["applicable"] is False


def test_beneish_missing_prior_na():
    r = compute_beneish_m({"sector": "Technology",
                           "beneish_inputs": {"current": dict(_side(), net_income=100.0, cfo=100.0),
                                              "prior": {k: None for k in _side()}}})
    assert r["m_score"] is None and r["flag"] is None


def test_beneish_block_render():
    block = format_beneish_block(compute_beneish_m(_CLEAN))
    assert "## Manipulation screen (Beneish M-score)" in block
    assert "normal" in block and "verbatim" in block
