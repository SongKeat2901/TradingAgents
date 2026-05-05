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
        # In production, portfolio_manager sets both fields to the same string
        # (the full PM body). Stub them identically here so tests reflect that.
        "final_trade_decision": "PM: BUY at half size — strong fundamentals, manageable risk",
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


def test_decision_md_renders_pm_body_under_ticker_header(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    decision = (tmp_path / "decision.md").read_text()
    assert "# NVDA — 2024-05-10" in decision
    assert "PM: BUY at half size" in decision
    # The PM body must appear exactly once — no duplication wrapper.
    assert decision.count("PM: BUY at half size") == 1


def test_decision_md_falls_back_to_risk_judge_decision(tmp_path):
    """If final_trade_decision is empty, fall back to risk_debate_state.judge_decision."""
    from cli.research_writer import write_research_outputs

    state = _stub_state()
    state["final_trade_decision"] = ""
    write_research_outputs(state, str(tmp_path))
    decision = (tmp_path / "decision.md").read_text()
    assert "PM: BUY at half size" in decision


def test_debate_risk_md_does_not_duplicate_pm_decision(tmp_path):
    """The PM judge_decision is the canonical content of decision.md and must
    NOT also appear inside debate_risk.md — the PDF renderer renders both
    files separately, so any inclusion here produces a duplicated 9-page
    block. Regression test for the run-#2 PDF audit (2026-05-05)."""
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    risk = (tmp_path / "debate_risk.md").read_text()
    assert "## Aggressive" in risk
    assert "## Neutral" in risk
    assert "## Conservative" in risk
    # PM decision must not be re-rendered here.
    assert "## Portfolio Manager Decision" not in risk
    assert "PM: BUY at half size" not in risk


def test_state_json_round_trips(tmp_path):
    from cli.research_writer import write_research_outputs

    write_research_outputs(_stub_state(), str(tmp_path))
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert loaded["company_of_interest"] == "NVDA"
    assert "BUY" in loaded["final_trade_decision"]
    # _meta is only added when config is provided
    assert "_meta" not in loaded


def test_state_json_records_model_meta_when_config_provided(tmp_path):
    """When config is passed, state.json must capture deep_think_llm /
    quick_think_llm / llm_provider so the PDF cover-page label can render
    actual models instead of the historical hardcoded 'Opus 4.6 / Haiku 4.5'."""
    from cli.research_writer import write_research_outputs

    config = {
        "deep_think_llm": "claude-opus-4-7",
        "quick_think_llm": "claude-sonnet-4-6",
        "llm_provider": "claude_code",
        "deep_via_cli": True,
        "quick_via_cli": True,
    }
    write_research_outputs(_stub_state(), str(tmp_path), config=config)
    loaded = json.loads((tmp_path / "state.json").read_text())
    assert loaded["_meta"]["deep_think_llm"] == "claude-opus-4-7"
    assert loaded["_meta"]["quick_think_llm"] == "claude-sonnet-4-6"
    assert loaded["_meta"]["llm_provider"] == "claude_code"
    assert loaded["_meta"]["deep_via_cli"] is True
    assert loaded["_meta"]["quick_via_cli"] is True


def test_state_json_meta_does_not_mutate_caller_state(tmp_path):
    """Adding `_meta` must not leak into the caller's state dict — the writer
    should copy before mutating."""
    from cli.research_writer import write_research_outputs

    state = _stub_state()
    write_research_outputs(state, str(tmp_path), config={"deep_think_llm": "x"})
    assert "_meta" not in state


def test_creates_output_dir_if_missing(tmp_path):
    from cli.research_writer import write_research_outputs

    out = tmp_path / "deep" / "nested" / "dir"
    write_research_outputs(_stub_state(), str(out))
    assert (out / "decision.md").exists()
