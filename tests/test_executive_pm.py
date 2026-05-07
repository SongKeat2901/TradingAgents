"""Tests for the Phase-6.7 Executive PM agent (stakeholder-voice translation)."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


_WORKING_NOTES = """## Inputs to this decision

- **Reference price:** $197.75 (yfinance close on or before 2026-05-06)
- **Trade date:** 2026-05-06

## Synthesis of the Risk Debate

**Neutral wins on coherence; Conservative loses on sizing math.**
Neutral's strongest punch is: *"a 30M-account distribution channel
entering at 0.50%..."*. The Aggressive Analyst transcript's "$32/share
buffer" claim is wrong (basic common math yields ~$15/share).

## Trading Plan

| Action | Sizing | Trigger |
|---|---|---|
| Trim 1 | Sell 30% | pre-close 2026-05-06 |

## What I am rejecting

- **Conservative's 50-unit entry on a $27/share CLARITY haircut** —
  verbatim from Conservative Analyst transcript: *"the CLARITY Act's
  idle-reserve yield ban..."*. Rejected.
"""


_EXECUTIVE_REWRITE = """## Executive Summary

COIN at $197.75 in a confirmed downtrend; Underweight at 65% of baseline
ahead of the Q1 FY26 print on 2026-05-07.

## Thesis

### Bull case

A clean Q1 print plus a CLARITY Act transactional-rewards carve-out
unlocks a multi-quarter re-rating to $238.

### Bear case

Take-rate compression from new entrants and an idle-reserve ban
collapse subscription-services revenue, validating a path to $151.

## Rating and Trading Plan

**Underweight**

Size: 65% of baseline; trim 30% pre-print.

| Action | Sizing | Trigger | Timing | Status |
|---|---|---|---|---|
| Trim 1 | Sell 30% | pre-close 2026-05-06 | Before print | Execute Now |

## Key Risks

- **Q1 print miss** — implied move ±12-17% on iron-condor pricing.
- **Take-rate compression** — Morgan Stanley E*TRADE entry at 0.50%.

## Catalysts

- **2026-05-07** — COIN Q1 FY26 earnings: binary; ±12-17% implied move.
- **2026-08** — Circle Agreement renegotiation opens.

## Supporting Analysis

### Technical setup
Spot $197.75 sits 24.2% below the 200-DMA $260.88; death-cross alignment
with 50-DMA $189.64. Apr-10 higher-low at $163.13 holds.

### Fundamentals
TTM revenue $7.1B with -21.6% YoY in Q4; net cash $4.08B = $11.91B - $7.83B.
44.5x TTM P/E vs HOOD 38.4x and CME 24.6x.

### Peer comparison
COIN trades at the highest TTM P/E in the peer set on the worst recent
print, with peer Net Debt: HOOD $8.38B, MSTR $5.89B, CME $1.03B.

## Caveats

Deribit Q1 contribution unquantifiable; CLARITY Act final text unknown.
Sizing reflects this uncertainty rather than betting on either
interpretation.
"""


def _state(working_notes: str = _WORKING_NOTES) -> dict:
    return {
        "company_of_interest": "COIN",
        "trade_date": "2026-05-06",
        "final_trade_decision": working_notes,
    }


def test_executive_pm_writes_translation_to_state():
    """Happy path: working notes present → LLM invoked → translation
    stored in `final_trade_decision_executive`."""
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=_EXECUTIVE_REWRITE)

    node = create_executive_pm_node(llm)
    out = node(_state())

    assert "final_trade_decision_executive" in out
    assert out["final_trade_decision_executive"].startswith("## Executive Summary")
    assert "Underweight" in out["final_trade_decision_executive"]


def test_executive_pm_short_circuits_on_empty_working_notes():
    """If `final_trade_decision` is empty (degenerate prior PM), the
    Executive PM must short-circuit without calling the LLM. This avoids
    making a wasted API call and lets the writer skip
    decision_executive.md so the PDF falls back to decision.md."""
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    llm = MagicMock()
    node = create_executive_pm_node(llm)
    out = node(_state(working_notes=""))

    assert out["final_trade_decision_executive"] == ""
    llm.invoke.assert_not_called()


def test_executive_pm_short_circuits_on_whitespace_only_working_notes():
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    llm = MagicMock()
    node = create_executive_pm_node(llm)
    out = node(_state(working_notes="   \n\t  \n  "))

    assert out["final_trade_decision_executive"] == ""
    llm.invoke.assert_not_called()


def test_executive_pm_passes_ticker_and_date_into_prompt():
    """The system prompt template has $TICKER and $DATE placeholders that
    must be substituted before the LLM call."""
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        return AIMessage(content=_EXECUTIVE_REWRITE)

    llm = MagicMock()
    llm.invoke.side_effect = fake_invoke
    node = create_executive_pm_node(llm)
    node(_state())

    assert "messages" in captured
    system_msg = captured["messages"][0].content
    human_msg = captured["messages"][1].content
    # System prompt no longer has unsubstituted placeholders
    assert "$TICKER" not in system_msg
    assert "$DATE" not in system_msg
    # Human message names the ticker + date
    assert "COIN" in human_msg
    assert "2026-05-06" in human_msg
    # Human message includes the working notes verbatim
    assert "Synthesis of the Risk Debate" in human_msg


def test_executive_pm_uses_retry_helper_for_empty_first_call():
    """If the first LLM call returns empty content, the retry helper
    (Fix #9) must fire the second call. This is integration with
    invoke_with_empty_retry."""
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    llm = MagicMock()
    llm.invoke.side_effect = [
        AIMessage(content=""),  # first call: degenerate
        AIMessage(content=_EXECUTIVE_REWRITE),  # retry: substantive
    ]
    node = create_executive_pm_node(llm)
    out = node(_state())

    assert llm.invoke.call_count == 2
    assert out["final_trade_decision_executive"].startswith("## Executive Summary")


def test_executive_pm_raises_on_sustained_empty_content():
    """Two consecutive empty responses → raise (the same fail-loud
    behaviour as every other analyst node post-Fix #9)."""
    from tradingagents.agents.managers.executive_pm import create_executive_pm_node

    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="")]
    node = create_executive_pm_node(llm)

    with pytest.raises(RuntimeError, match="Executive PM.*empty content"):
        node(_state())


def test_executive_pm_system_prompt_forbids_multi_agent_attribution():
    """Sanity check on the prompt itself: the system prompt must
    explicitly forbid the multi-agent attribution language that motivated
    the two-stage design."""
    from tradingagents.agents.managers.executive_pm import _SYSTEM

    # Prompt must name the forbidden constructs so the LLM refuses them
    forbidden_constructs = [
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Research Manager",
        "Bear Researcher",
        "Bull Researcher",
        "verbatim from",
    ]
    for term in forbidden_constructs:
        assert term in _SYSTEM, f"prompt must mention forbidden construct: {term}"

    # Mandated section template
    for section in ("Executive Summary", "Thesis", "Rating and Trading Plan",
                    "Key Risks", "Catalysts", "Supporting Analysis", "Caveats"):
        assert section in _SYSTEM, f"prompt missing mandated section: {section}"
