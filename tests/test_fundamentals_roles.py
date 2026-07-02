import pytest
pytestmark = pytest.mark.unit
from tradingagents.agents.analysts import fundamentals_roles as fr

def test_financial_prompt_and_files():
    s = fr._SYSTEM_FINANCIAL
    assert "## Business-model framing" in s and "## Peer comparison matrix" in s
    assert "## Sanity check on reported numbers" in s
    assert "Revenue YoY" in s  # YoY pre-compute mandate moved here
    assert "restating a figure already shown" in s  # net-debt discipline preserved
    assert set(fr._FILES_FINANCIAL) == {"pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md", "earnings_release.md", "news.json"}

def test_financial_files_include_news_for_takeaways_directive():
    """ORCL 2026-07-01 wart: the Latest-quarter-takeaways directive says
    management commentary comes ONLY from news.json, but the role's file
    list didn't include news.json — the designed news-sourced call-color
    path could never trigger (the run honestly wrote 'news.json was not
    provided'). The directive and the file list must agree."""
    assert "news.json items" in fr._SYSTEM_FINANCIAL
    assert "news.json" in fr._FILES_FINANCIAL

def test_financial_prompt_yoy_preamble():
    """Migrated from test_fundamentals_analyst.py::test_fundamentals_prompt_includes_yoy_preamble
    (Phase-6.2 catch: fabricated '5.4% capex-to-revenue' for MSFT vs actual 37.3%)."""
    s = fr._SYSTEM_FINANCIAL
    assert "YoY computation from financials.json" in s
    assert "Revenue YoY" in s
    assert "Capex / revenue ratio" in s
    assert "DO NOT invent ratios" in s

def test_financial_prompt_sec_filing_read_step():
    """Migrated from test_fundamentals_analyst.py::test_fundamentals_prompt_includes_sec_filing_read_step."""
    s = fr._SYSTEM_FINANCIAL
    assert "raw/sec_filing.md" in s
    assert "Remaining Performance Obligations" in s
    assert "awaiting filing" in s
    assert "pending adjudication" in s
    assert "data to follow" in s
    assert "not yet disclosed" in s

def test_financial_reads_sec_filing_md_when_present(monkeypatch, tmp_path):
    """Migrated from test_fundamentals_analyst.py::test_fundamentals_reads_sec_filing_md_when_present
    — verify sec_filing.md is in the file list actually passed to format_for_prompt
    by the Financial-Statement node (not just present in the static _FILES_FINANCIAL list)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    captured = {}

    def fake_format(raw_dir, files):
        captured["files"] = list(files)
        return "stubbed context"

    monkeypatch.setattr(fr, "format_for_prompt", fake_format)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="report body. " * 200)

    node = fr.create_financial_statement_analyst(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path),
    }
    node(state)

    assert "sec_filing.md" in captured["files"]
    assert "pm_brief.md" in captured["files"]
    assert "financials.json" in captured["files"]

def test_financial_prompt_net_debt_restatement_discipline():
    """Migrated from test_fundamentals_prompt.py::test_net_debt_restatement_discipline_present."""
    s = fr._SYSTEM_FINANCIAL.lower()
    assert "net debt" in s
    assert "must not compute and cite a novel derived net-debt" in s

def test_risk_prompt_and_files():
    s = fr._SYSTEM_RISK
    assert "Altman Z" in s and "Beneish M-score" in s
    assert "sec_filing.md" in fr._FILES_RISK

def test_risk_prompt_distress_citation_mandated():
    """Migrated from test_fundamentals_prompt.py::test_distress_citation_mandated."""
    assert "do not compute your own z-score or invent a zone" in fr._SYSTEM_RISK.lower()

def test_risk_prompt_beneish_citation_mandated():
    """Migrated from test_fundamentals_prompt.py::test_beneish_citation_mandated."""
    s = fr._SYSTEM_RISK.lower()
    assert "beneish" in s or "manipulation screen" in s
    assert "do not compute your own m-score or invent a flag" in s

def test_catalysts_prompt_and_files():
    s = fr._SYSTEM_CATALYSTS
    assert "## Deal math" in s and "## Insider transactions" in s
    assert "## What management needs to prove" in s
    assert "Sentiment & consensus" in s
    assert set(fr._FILES_CATALYSTS) == {"pm_brief.md", "reference.json", "news.json", "insider.json", "earnings_release.md"}

def test_catalysts_prompt_insider_citation_mandate():
    """Migrated from test_fundamentals_prompt.py::test_insider_section_and_citation_mandate
    (insider.json now lives in the Catalysts & Ownership role's files/footer,
    not the monolithic prompt)."""
    assert "Insider transactions" in fr._SYSTEM_CATALYSTS
    assert "reference.json, or insider.json" in fr._SYSTEM_CATALYSTS

def test_quality_prompt_and_files():
    s = fr._SYSTEM_QUALITY
    assert "## Competitive position" in s and "## Capital-allocation track record" in s
    assert "## Ownership & governance" in s
    assert "not determinable from" in s  # qualitative-claim discipline preserved

def test_quality_prompt_antifabrication_clause():
    """Migrated from test_fundamentals_prompt.py::test_qualitative_antifabrication_clause."""
    s = fr._SYSTEM_QUALITY.lower()
    assert "not determinable from available free filings" in s
    assert "do not invent" in s

def test_shared_footer_on_all():
    for s in (fr._SYSTEM_FINANCIAL, fr._SYSTEM_RISK, fr._SYSTEM_CATALYSTS, fr._SYSTEM_QUALITY):
        assert "No invented numbers." in s

def test_financial_prompt_earnings_release_guidance_discipline():
    """EARNINGS_RELEASE_GOAL step 3: the Financial-Statement role must cite
    forward guidance + the capex/financing funding structure verbatim from the
    8-K press release, or say 'not disclosed' — never invent them."""
    s = fr._SYSTEM_FINANCIAL
    assert "## Latest earnings release (SEC 8-K Ex-99.1)" in s
    low = s.lower()
    assert "not disclosed in the earnings release" in low
    assert "do not invent guidance" in low
    assert "earnings_release.md" in fr._FILES_FINANCIAL

def test_financial_capex_bridge_upgraded_to_release_sourced():
    """The capex funding bridge (pro-deck technique C) is no longer call-only:
    the bridge paragraph must name the earnings release as a funding source."""
    s = fr._SYSTEM_FINANCIAL.lower()
    bridge = s.split("capex funding bridge discipline")[1].split("fcf trajectory discipline")[0]
    assert "earnings release" in bridge

def test_catalysts_prompt_management_color_section():
    """EARNINGS_RELEASE_GOAL step 3: the Catalysts & Ownership role must quote
    CEO/CFO management color verbatim from the release block, else 'not
    disclosed' — and the section is structurally required (retry-checked)."""
    s = fr._SYSTEM_CATALYSTS
    assert "## Management color (earnings release)" in s
    low = s.lower()
    assert "cfo" in low
    assert "verbatim" in low
    assert "not disclosed in the earnings release" in low
    assert "earnings_release.md" in fr._FILES_CATALYSTS
    assert "## Management color (earnings release)" in fr._REQUIRED_CATALYSTS

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


def test_financial_covers_sotp_and_kpis():
    s = fr._SYSTEM_FINANCIAL
    assert "sum-of-the-parts" in s.lower()
    assert "NRR" in s and "ARPU" in s and "same-store" in s.lower()
    assert "not disclosed in the available filing" in s  # anti-fabrication


def test_quality_covers_material_esg():
    s = fr._SYSTEM_QUALITY
    assert "Material ESG risks" in s
    assert "no material ESG risk disclosed" in s
    assert "do NOT import an ESG rating" in s  # anti-fabrication
