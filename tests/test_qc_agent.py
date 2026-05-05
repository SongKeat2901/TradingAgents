"""Tests for the QC Agent — independent reviewer of the PM's draft."""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


def _stub_state(tmp_path, decision="## Inputs to this decision\n... full PM doc ...", retries=0):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "reference.json").write_text(json.dumps({
        "ticker": "MSFT",
        "trade_date": "2026-05-01",
        "reference_price": 410.0,
    }), encoding="utf-8")
    return {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "final_trade_decision": decision,
        "qc_retries": retries,
    }


def test_qc_agent_passes_when_verdict_is_pass(tmp_path):
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=(
        "Item 1: PASS — probabilities sum to 100%.\n"
        "...\n"
        "Item 13: PASS — all numbers traced.\n"
        "QC_VERDICT: {\"status\": \"PASS\"}"
    ))
    node = create_qc_agent_node(fake_llm)
    out = node(_stub_state(tmp_path))
    assert out == {"qc_passed": True}


def test_qc_agent_fails_with_feedback_when_verdict_is_fail(tmp_path):
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=(
        "Item 1: FAIL — probabilities sum to 105%.\n"
        "...\n"
        "QC_VERDICT: {\"status\": \"FAIL\", \"feedback\": "
        "\"Scenario probabilities sum to 105% (Bull 40 + Base 40 + Bear 25). Adjust to 100%.\"}"
    ))
    node = create_qc_agent_node(fake_llm)
    out = node(_stub_state(tmp_path))
    assert out["qc_passed"] is False
    assert "105%" in out["qc_feedback"]
    assert out["qc_retries"] == 1


def test_qc_agent_short_circuits_when_already_retried(tmp_path):
    """After 1 retry, the QC agent must not run again — graph proceeds to END."""
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="should not be called")
    node = create_qc_agent_node(fake_llm)
    out = node(_stub_state(tmp_path, retries=1))
    assert out == {"qc_passed": True}
    fake_llm.invoke.assert_not_called()


def test_qc_agent_treats_unparseable_verdict_as_pass(tmp_path):
    """If the LLM emits a malformed verdict line, log + treat as PASS so the
    pipeline doesn't deadlock on a parsing error."""
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="QC_VERDICT: not-json garbage")
    node = create_qc_agent_node(fake_llm)
    out = node(_stub_state(tmp_path))
    assert out == {"qc_passed": True}


def test_qc_agent_skips_audit_when_decision_is_empty(tmp_path):
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    node = create_qc_agent_node(fake_llm)
    out = node(_stub_state(tmp_path, decision=""))
    assert out == {"qc_passed": True}
    fake_llm.invoke.assert_not_called()


def test_qc_agent_includes_reference_snapshot_in_user_message(tmp_path):
    """The audit prompt must include raw/reference.json contents so the QC can
    verify reference_price citations."""
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="QC_VERDICT: {\"status\": \"PASS\"}")
    node = create_qc_agent_node(fake_llm)
    node(_stub_state(tmp_path))

    call_args = fake_llm.invoke.call_args
    messages = call_args.args[0]
    user_content = messages[1].content
    assert "410.0" in user_content  # reference_price from stub
    assert "2026-05-01" in user_content  # trade_date from stub


def test_qc_system_prompt_includes_original_14_items():
    """Verifies that the original 14 checklist items are still present after the Phase-6.3 expansion to 16 items. Lower-bound regression check; the new 16-item count is verified by `test_qc_checklist_has_16_items_and_filing_anchor_text`."""
    from tradingagents.agents.managers.qc_agent import _SYSTEM
    for n in range(1, 15):
        assert f"{n}." in _SYSTEM, f"QC system prompt missing item {n}"
    assert "QC_VERDICT:" in _SYSTEM
    # Item 14 specifics
    assert "Technical setup adopted" in _SYSTEM
    assert "verbatim" in _SYSTEM


def test_parse_verdict_handles_pass_and_fail():
    from tradingagents.agents.managers.qc_agent import _parse_verdict
    assert _parse_verdict("QC_VERDICT: {\"status\": \"PASS\"}") == {"status": "PASS"}
    out = _parse_verdict("blah\nQC_VERDICT: {\"status\": \"FAIL\", \"feedback\": \"fix x\"}")
    assert out == {"status": "FAIL", "feedback": "fix x"}
    # Status must be PASS or FAIL
    assert _parse_verdict("QC_VERDICT: {\"status\": \"MAYBE\"}") is None
    # Malformed JSON
    assert _parse_verdict("QC_VERDICT: not json") is None
    # No verdict line
    assert _parse_verdict("nothing here") is None


def test_qc_checklist_has_16_items_and_filing_anchor_text():
    """The QC system prompt must (a) declare a 16-item checklist (was 14 pre-Phase-6.3),
    (b) include item 15 with key filing-anchor phrasing, (c) include item 16 with
    all three sub-rules (verbatim/computed, sign+direction, peer-delta reconcile)."""
    from tradingagents.agents.managers.qc_agent import _SYSTEM

    assert "16-item checklist" in _SYSTEM
    # Item 15: filing-anchor temporal correctness
    assert "Filing-anchor temporal correctness" in _SYSTEM
    assert "raw/sec_filing.md" in _SYSTEM
    assert "awaiting filing" in _SYSTEM
    # Item 16: numerical claims trace to source
    assert "Multi-decimal numerical claims" in _SYSTEM
    assert "trace" in _SYSTEM.lower()
    assert "raw/financials.json" in _SYSTEM
    # Item 16's source-cell list must include raw/prices.json (TA-derived bps figures)
    assert "raw/prices.json" in _SYSTEM
    # Item 16 sub-rule (a) — verbatim or computed (catches the original 5.4% bug)
    assert "Verbatim or computed" in _SYSTEM
    assert "5.4%" in _SYSTEM and "37.3%" in _SYSTEM
    # Item 16 sub-rule (b) — sign + direction (catches the $8.2B net-cash sign inversion)
    assert "Sign + direction" in _SYSTEM
    assert "net cash" in _SYSTEM and "Net Debt" in _SYSTEM
    # Item 16 sub-rule (c) — peer-delta reconciliation (catches the 5.4% above peers claim)
    assert "Peer-comparison deltas reconcile" in _SYSTEM


def test_qc_fails_pm_draft_calling_filed_10q_pending(tmp_path):
    """When raw/sec_filing.md exists and the PM draft frames its content as
    'pending adjudication', the QC verdict must be FAIL with feedback that
    references item 15. Catches the exact Run-#2 failure mode."""
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    (raw / "sec_filing.md").write_text(
        "# SEC Filing — MSFT 10-Q filed 2026-04-29\n\n"
        "Azure revenue increased 40%.\n",
        encoding="utf-8",
    )

    pm_draft = (
        "## Inputs to this decision\n"
        "Reference price: $410.00 ...\n\n"
        "## Catalyst path\n"
        "The mid-May 10-Q is the binary catalyst pending adjudication ...\n"
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=(
        "Item 1: PASS\n"
        "...\n"
        "Item 15: FAIL — PM frames the already-filed 2026-04-29 10-Q as "
        "'pending adjudication' but raw/sec_filing.md contains its full text.\n"
        "Item 16: PASS\n"
        "QC_VERDICT: {\"status\": \"FAIL\", \"feedback\": "
        "\"Item 15 violation: the 10-Q referenced as 'pending' is already "
        "in raw/sec_filing.md (filed 2026-04-29). Rewrite the catalyst "
        "narrative around the NEXT 10-Q (~2026-07).\"}"
    ))

    node = create_qc_agent_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "final_trade_decision": pm_draft,
        "qc_retries": 0,
    }
    out = node(state)

    assert out.get("qc_passed") is False
    assert "Item 15" in out.get("qc_feedback", "") or "item 15" in out.get("qc_feedback", "").lower()
    assert "raw/sec_filing.md" in out.get("qc_feedback", "")
