"""Tests for cli.research_writer.write_research_outputs."""

import json

import pytest

pytestmark = pytest.mark.unit


def _stub_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2024-05-10",
        "market_report": "## Market\n- price up 3%",
        "sentiment_report": "## Sentiment\n- bullish on social",
        "news_report": "## News\n- earnings beat",
        "fundamentals_report": "## Fundamentals\n- P/E elevated",
        "investment_debate_state": {
            "bull_history": "Bull: strong momentum",
            "bear_history": "Bear: valuation stretched",
            "judge_decision": "Manager: lean bull",
        },
        "risk_debate_state": {
            "aggressive_history": "Agg: max long",
            "neutral_history": "Neu: half size",
            "conservative_history": "Cons: skip",
            "judge_decision": "PM: BUY at half size — strong fundamentals, manageable risk",
        },
        "final_trade_decision": "BUY",
    }


def test_writes_all_expected_files(tmp_path):
    from cli.research_writer import write_research_outputs

    written = write_research_outputs(_stub_state(), str(tmp_path))

    expected = {
        "decision.md", "analyst_market.md", "analyst_social.md",
        "analyst_news.md", "analyst_fundamentals.md",
        "debate_bull_bear.md", "debate_risk.md", "state.json",
    }
    assert {p.name for p in written} == expected
    for p in written:
        assert p.read_text(), f"{p.name} is empty"


def test_decision_md_contains_action_and_pm_judgement(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    decision = (tmp_path / "decision.md").read_text()
    assert "BUY" in decision
    assert "PM:" in decision  # PM rationale carried forward
    assert "NVDA" in decision
    assert "2024-05-10" in decision


def test_state_json_round_trips(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert loaded["company_of_interest"] == "NVDA"
    assert loaded["final_trade_decision"] == "BUY"


def test_creates_output_dir_if_missing(tmp_path):
    from cli.research_writer import write_research_outputs

    out = tmp_path / "deep" / "nested" / "dir"
    write_research_outputs(_stub_state(), str(out))
    assert (out / "decision.md").exists()
