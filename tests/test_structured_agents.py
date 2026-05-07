"""Tests for structured-output agents (Trader and Research Manager).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader and Research Manager so all three
decision-making agents share the same shape.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_research_plan,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(action=TraderAction.HOLD, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Action**: Hold" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # The trailing FINAL TRANSACTION PROPOSAL line is preserved for the
        # analyst stop-signal text and any external code that greps for it.
        assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in md

    def test_optional_fields_included_when_present(self):
        p = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong technicals + fundamentals.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        md = render_trader_proposal(p)
        assert "**Action**: Buy" in md
        assert "**Entry Price**: 189.5" in md
        assert "**Stop Loss**: 178.0" in md
        assert "**Position Sizing**: 6% of portfolio" in md
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in md

    def test_optional_fields_omitted_when_absent(self):
        p = TraderProposal(action=TraderAction.SELL, reasoning="Guidance cut.")
        md = render_trader_proposal(p)
        assert "Entry Price" not in md
        assert "Stop Loss" not in md
        assert "Position Sizing" not in md
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case carried; tailwinds intact.",
            strategic_actions="Build position over two weeks; cap at 5%.",
        )
        md = render_research_plan(p)
        assert "**Recommendation**: Overweight" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**Strategic Actions**: Build position" in md

    def test_all_5_tier_ratings_render(self):
        for rating in PortfolioRating:
            p = ResearchPlan(
                recommendation=rating,
                rationale="r",
                strategic_actions="s",
            )
            md = render_research_plan(p)
            assert f"**Recommendation**: {rating.value}" in md


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: ...\n**Strategic Actions**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.BUY,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            entry_price=189.5,
            stop_loss=178.0,
            position_sizing="6% of portfolio",
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Action**: Buy" in plan
        assert "**Entry Price**: 189.5" in plan
        assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_investment_plan(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The investment plan is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Proposed Investment Plan" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Action**: Sell\n\nGuidance cut hits margins.\n\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="Balanced view across both sides.",
            strategic_actions="Hold current position; reassess after earnings.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            recommendation=PortfolioRating.OVERWEIGHT,
            rationale="Bull case is stronger; AI tailwind intact.",
            strategic_actions="Build position gradually over two weeks.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Recommendation**: Overweight" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**Strategic Actions**: Build position" in ip

    def test_prompt_uses_5_tier_rating_scale(self):
        """The RM prompt must list all five tiers so the schema enum matches user expectations."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for tier in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f"**{tier}**" in prompt, f"missing {tier} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Recommendation**: Sell\n\n**Rationale**: ...\n\n**Strategic Actions**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


@pytest.mark.unit
class TestExtractLlmContent:
    """Phase 6.7: detect degenerate LLM responses (empty content) and raise.

    The 2026-05-06 COIN cadence shipped UNDERWEIGHT with the literal LangChain
    envelope `content='' additional_kwargs={} ... tool_calls=[]` written into
    `analyst_fundamentals.md` (165 chars). The pre-existing
    `report = raw_content if raw_content else str(result)` pattern silently
    fell back to `str(result)` on empty content — this helper raises instead."""

    def test_returns_content_when_substantive(self):
        from tradingagents.agents.utils.structured import extract_llm_content
        result = MagicMock(content="This is the analyst's full report. " * 10)
        out = extract_llm_content(result, "Test Analyst")
        assert out.startswith("This is the analyst's full report.")

    def test_raises_on_empty_string_content(self):
        from tradingagents.agents.utils.structured import extract_llm_content
        result = MagicMock(content="")
        with pytest.raises(RuntimeError, match="Test Analyst.*empty content"):
            extract_llm_content(result, "Test Analyst")

    def test_raises_on_whitespace_only_content(self):
        from tradingagents.agents.utils.structured import extract_llm_content
        result = MagicMock(content="   \n\t  \n  ")
        with pytest.raises(RuntimeError, match="empty content"):
            extract_llm_content(result, "Test Analyst")

    def test_raises_on_missing_content_attribute(self):
        from tradingagents.agents.utils.structured import extract_llm_content

        class _NoContent:
            pass

        with pytest.raises(RuntimeError, match="empty content"):
            extract_llm_content(_NoContent(), "Test Analyst")

    def test_error_message_names_the_agent(self):
        """The agent name is essential for debugging which node failed."""
        from tradingagents.agents.utils.structured import extract_llm_content
        result = MagicMock(content="")
        with pytest.raises(RuntimeError, match="Fundamentals Analyst"):
            extract_llm_content(result, "Fundamentals Analyst")


@pytest.mark.unit
class TestInvokeWithEmptyRetry:
    """Phase 6.7 v2: 3 empty-content occurrences across recent runs (SOFI/COIN
    ×2) showed that fail-loud-on-first-empty kills good runs that would
    succeed on a single retry. Single-shot retry squares the per-call failure
    probability without compromising the fail-loud guarantee."""

    def test_returns_first_call_when_substantive(self):
        from tradingagents.agents.utils.structured import invoke_with_empty_retry
        llm = MagicMock()
        substantive = MagicMock(content="A real analyst report. " * 20)
        llm.invoke.return_value = substantive

        result, content = invoke_with_empty_retry(llm, ["msg"], "Test")

        assert result is substantive
        assert content.startswith("A real analyst report.")
        assert llm.invoke.call_count == 1

    def test_retries_once_then_succeeds(self):
        """First call returns empty content; second call returns substantive
        content. Helper must not raise."""
        from tradingagents.agents.utils.structured import invoke_with_empty_retry
        empty = MagicMock(content="")
        substantive = MagicMock(content="Recovered on retry. " * 20)
        llm = MagicMock()
        llm.invoke.side_effect = [empty, substantive]

        result, content = invoke_with_empty_retry(llm, ["msg"], "News Analyst")

        assert result is substantive
        assert content.startswith("Recovered on retry.")
        assert llm.invoke.call_count == 2

    def test_raises_when_both_calls_return_empty(self):
        """Two empty-content calls in a row → raise (a sustained
        degenerate state, not a transient flake)."""
        from tradingagents.agents.utils.structured import invoke_with_empty_retry
        empty1 = MagicMock(content="")
        empty2 = MagicMock(content="")
        llm = MagicMock()
        llm.invoke.side_effect = [empty1, empty2]

        with pytest.raises(RuntimeError, match="News Analyst.*empty content"):
            invoke_with_empty_retry(llm, ["msg"], "News Analyst")
        assert llm.invoke.call_count == 2

    def test_treats_whitespace_only_first_call_as_empty(self):
        from tradingagents.agents.utils.structured import invoke_with_empty_retry
        whitespace = MagicMock(content="  \n\t  \n  ")
        substantive = MagicMock(content="OK on retry. " * 30)
        llm = MagicMock()
        llm.invoke.side_effect = [whitespace, substantive]

        _result, content = invoke_with_empty_retry(llm, ["msg"], "Test")

        assert content.startswith("OK on retry.")
        assert llm.invoke.call_count == 2

    def test_retry_uses_same_messages(self):
        """The retry call must pass the same messages — re-invoking with a
        different prompt would mask the transient flake we're guarding."""
        from tradingagents.agents.utils.structured import invoke_with_empty_retry
        empty = MagicMock(content="")
        substantive = MagicMock(content="Substantive. " * 50)
        llm = MagicMock()
        llm.invoke.side_effect = [empty, substantive]

        msgs = ["original prompt"]
        invoke_with_empty_retry(llm, msgs, "Test")

        first_args = llm.invoke.call_args_list[0][0]
        second_args = llm.invoke.call_args_list[1][0]
        assert first_args == second_args == (msgs,)
