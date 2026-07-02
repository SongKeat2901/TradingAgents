import pytest
pytestmark = pytest.mark.unit
from tradingagents.agents.analysts import fundamentals_roles as fr

def test_financial_prompt_and_files():
    s = fr._SYSTEM_FINANCIAL
    assert "## Business-model framing" in s and "## Peer comparison matrix" in s
    assert "## Sanity check on reported numbers" in s
    assert "Revenue YoY" in s  # YoY pre-compute mandate moved here
    assert "restating a figure already shown" in s  # net-debt discipline preserved
    assert set(fr._FILES_FINANCIAL) == {"pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md"}

def test_risk_prompt_and_files():
    s = fr._SYSTEM_RISK
    assert "Altman Z" in s and "Beneish M-score" in s
    assert "sec_filing.md" in fr._FILES_RISK

def test_catalysts_prompt_and_files():
    s = fr._SYSTEM_CATALYSTS
    assert "## Deal math" in s and "## Insider transactions" in s
    assert "## What management needs to prove" in s
    assert "Sentiment & consensus" in s
    assert set(fr._FILES_CATALYSTS) == {"pm_brief.md", "reference.json", "news.json", "insider.json"}

def test_quality_prompt_and_files():
    s = fr._SYSTEM_QUALITY
    assert "## Competitive position" in s and "## Capital-allocation track record" in s
    assert "## Ownership & governance" in s
    assert "not determinable from" in s  # qualitative-claim discipline preserved

def test_shared_footer_on_all():
    for s in (fr._SYSTEM_FINANCIAL, fr._SYSTEM_RISK, fr._SYSTEM_CATALYSTS, fr._SYSTEM_QUALITY):
        assert "No invented numbers." in s

def test_factories_return_role_keys():
    # a stub llm whose invoke returns an object with .content
    class _Stub:
        def invoke(self, msgs):
            class R:  # min_chars=1200 -> pad
                content = "x" * 1500
            return R()
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-05", "raw_dir": "/nonexistent"}
    # format_for_prompt tolerates a missing dir (returns missing-markers); if it raises, use tmp
    import tradingagents.agents.analysts.fundamentals_roles as m
    for factory, key in [
        (m.create_financial_statement_analyst, "fundamentals_financial_report"),
        (m.create_risk_redflags_analyst, "fundamentals_riskflags_report"),
        (m.create_catalysts_ownership_analyst, "fundamentals_catalysts_report"),
        (m.create_competitive_quality_analyst, "fundamentals_quality_report"),
    ]:
        out = factory(_Stub())(state)
        assert key in out and len(out[key]) >= 1200
