"""Tests for the Fundamentals analyst (Phase-6.3 numerical-discipline)."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


def test_fundamentals_prompt_includes_yoy_preamble():
    """The system prompt must require a YoY computation pre-write step from
    financials.json so the analyst doesn't paraphrase ratios. Run #2 of the
    Phase-6.2 validation caught a fabricated 'capex/revenue 5.4%' (actual 37.3%)."""
    from tradingagents.agents.analysts.fundamentals_analyst import _SYSTEM
    assert "YoY computation from financials.json" in _SYSTEM
    assert "Revenue YoY" in _SYSTEM
    assert "Capex / revenue ratio" in _SYSTEM
    assert "DO NOT invent ratios" in _SYSTEM


def test_fundamentals_prompt_includes_sec_filing_read_step():
    """The system prompt must require the analyst to read raw/sec_filing.md
    when present and quote specific filing numbers (RPO, segment OpInc, etc.)."""
    from tradingagents.agents.analysts.fundamentals_analyst import _SYSTEM
    assert "raw/sec_filing.md" in _SYSTEM
    assert "Remaining Performance Obligations" in _SYSTEM
    assert "awaiting filing" in _SYSTEM
    assert "pending adjudication" in _SYSTEM
    assert "data to follow" in _SYSTEM
    assert "not yet disclosed" in _SYSTEM


def test_fundamentals_reads_sec_filing_md_when_present(monkeypatch, tmp_path):
    """Verify sec_filing.md is in the file list passed to format_for_prompt."""
    from tradingagents.agents.analysts import fundamentals_analyst

    captured = {}

    def fake_format(raw_dir, files):
        captured["files"] = list(files)
        return "stubbed context"

    monkeypatch.setattr(fundamentals_analyst, "format_for_prompt", fake_format)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="report body")

    node = fundamentals_analyst.create_fundamentals_analyst(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path),
    }
    node(state)

    assert "sec_filing.md" in captured["files"]
    # Sanity: existing files still present so we don't regress the prompt context.
    assert "pm_brief.md" in captured["files"]
    assert "financials.json" in captured["files"]
