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
