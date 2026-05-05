"""Tests for the PM Pass-2 self-correction QC checklist."""
import pytest

pytestmark = pytest.mark.unit


def test_qc_checklist_in_pm_system_prompt():
    """The PM system prompt must include the 13 QC items."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST

    must_contain = [
        "sum to exactly 100%",          # item 1
        "specific dollar values",       # item 2
        "named, falsifiable catalyst",  # item 3
        "Rating logically derives",     # item 4
        "Execution triggers are falsifiable",  # item 5
        "reachable in at least one scenario",  # item 6 (Flaw 8)
        "reference_price",              # item 7 (Flaw 2)
        "verbatim",                     # item 8 (Flaw 3)
        "Cross-section numerical consistency",  # item 9 (Flaw 5)
        "Sanity-check flags",           # item 10 (Flaw 4)
        "Inputs section",               # item 11
        "Peer comparisons cite specific",  # item 12
        "trace back to",                # item 13
    ]
    for keyword in must_contain:
        assert keyword in _QC_CHECKLIST, f"QC checklist missing: {keyword}"


def test_qc_checklist_has_self_correction_directive():
    """The system prompt must instruct the PM to self-correct on failure."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST
    # The instruction to apply the checklist before final output
    assert "self-correct" in _QC_CHECKLIST.lower() or "revise" in _QC_CHECKLIST.lower()


def test_output_contract_forbids_summary_emission():
    """The PM must be told its response IS decision.md, not a pointer to one."""
    from tradingagents.agents.managers.portfolio_manager import _OUTPUT_CONTRACT
    assert "Your entire response IS decision.md" in _OUTPUT_CONTRACT
    assert "DO NOT write" in _OUTPUT_CONTRACT
    assert "DO NOT emit a summary" in _OUTPUT_CONTRACT


def test_load_reference_block_pulls_canonical_values(tmp_path):
    """The reference block must surface raw/reference.json values verbatim
    so the PM cites reference_price + trade_date from a single source.
    """
    import json
    from tradingagents.agents.managers.portfolio_manager import _load_reference_block

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(json.dumps({
        "ticker": "MSFT",
        "trade_date": "2026-05-01",
        "reference_price": 410.0,
        "reference_price_source": "yfinance close on or before 2026-05-01",
        "spot_50dma": 405.0,
        "spot_200dma": 380.0,
        "ytd_high": 460.0,
        "ytd_low": 379.0,
        "atr_14": 4.2,
    }), encoding="utf-8")

    block = _load_reference_block({"raw_dir": str(raw)})
    assert "$410.0" in block
    assert "2026-05-01" in block
    assert "yfinance close on or before 2026-05-01" in block
    assert "verbatim" in block.lower()


def test_load_reference_block_returns_empty_when_raw_dir_missing():
    from tradingagents.agents.managers.portfolio_manager import _load_reference_block
    assert _load_reference_block({}) == ""
    assert _load_reference_block({"raw_dir": "/nonexistent"}) == ""


def test_pm_preflight_prompt_documents_fiscal_calendar():
    """PM Pre-flight must require a Fiscal calendar context section so the PM
    Final doesn't mislabel earnings quarters (e.g., calling MSFT's late-July
    print 'Q3 FY26' when it's Q4 FY26).
    """
    from tradingagents.agents.managers.pm_preflight import _SYSTEM
    assert "## Fiscal calendar context" in _SYSTEM
    assert "fiscal-quarter label" in _SYSTEM
    # The example in the prompt names the specific past mistake
    assert "Q4 FY26" in _SYSTEM


def test_mandated_sections_includes_technical_setup_adopted():
    """The PM's _MANDATED_SECTIONS must require the new 'Technical setup adopted'
    subsection so the disagreement with TA v2 is transcribed in decision.md."""
    from tradingagents.agents.managers.portfolio_manager import _MANDATED_SECTIONS
    assert "Technical setup adopted" in _MANDATED_SECTIONS
    assert "TA Agent v2 classification" in _MANDATED_SECTIONS
    # The three required choices
    assert "adopt" in _MANDATED_SECTIONS
    assert "partially adopt" in _MANDATED_SECTIONS
    assert "reject" in _MANDATED_SECTIONS
    # Reasoning length floor
    assert "≤80 words" in _MANDATED_SECTIONS or "<= 80 words" in _MANDATED_SECTIONS


def test_portfolio_manager_includes_sec_block_when_sec_filing_md_present(tmp_path, monkeypatch):
    """When raw/sec_filing.md exists, the PM prompt must include a 'Most recent
    SEC filing' block with the file's verbatim content + the temporal-anchor
    instruction. Catches the Run-#2 failure mode where the PM framed the
    already-public 10-Q as 'pending adjudication'."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    (raw / "sec_filing.md").write_text(
        "# SEC Filing — MSFT 10-Q filed 2026-04-29\n\n"
        "Azure and other cloud services revenue increased 40%.\n",
        encoding="utf-8",
    )

    captured = {}
    fake_llm = MagicMock()
    # Force bind_structured to return None so invoke_structured_or_freetext
    # routes directly to plain_llm.invoke — the path we instrument.
    fake_llm.with_structured_output.side_effect = NotImplementedError("no structured output")

    def _capture_invoke(messages):
        if isinstance(messages, str):
            captured["prompt"] = messages
        else:
            captured["prompt"] = "\n".join(
                m.content if hasattr(m, "content") else str(m) for m in messages
            )
        return AIMessage(content="## Inputs\n... full PM doc ...")

    fake_llm.invoke.side_effect = _capture_invoke

    node = create_portfolio_manager(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "market_report": "stub",
        "sentiment_report": "stub",
        "news_report": "stub",
        "fundamentals_report": "stub",
        "technicals_report": "stub",
        "qc_feedback": "",
    }
    node(state)

    prompt = captured["prompt"]
    assert "Most recent SEC filing" in prompt
    assert "Azure and other cloud services revenue increased 40%" in prompt
    assert "treat as known data" in prompt.lower() or "treat as **known data**" in prompt.lower()


def test_portfolio_manager_omits_sec_block_when_sec_filing_md_missing(tmp_path):
    """When raw/sec_filing.md is absent, the PM prompt must NOT render a
    sec_block (no fabrication, no template residue)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    # NO sec_filing.md

    captured = {}
    fake_llm = MagicMock()
    # Force bind_structured to return None so invoke_structured_or_freetext
    # routes directly to plain_llm.invoke — the path we instrument.
    fake_llm.with_structured_output.side_effect = NotImplementedError("no structured output")

    def _capture_invoke(messages):
        if isinstance(messages, str):
            captured["prompt"] = messages
        else:
            captured["prompt"] = "\n".join(
                m.content if hasattr(m, "content") else str(m) for m in messages
            )
        return AIMessage(content="## Inputs\n... full PM doc ...")

    fake_llm.invoke.side_effect = _capture_invoke

    node = create_portfolio_manager(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "market_report": "stub",
        "sentiment_report": "stub",
        "news_report": "stub",
        "fundamentals_report": "stub",
        "technicals_report": "stub",
        "qc_feedback": "",
    }
    node(state)

    prompt = captured["prompt"]
    assert "Most recent SEC filing" not in prompt
