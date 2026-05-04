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


def test_qc_system_prompt_lists_all_13_items():
    """The QC agent's system prompt must enumerate the same 13 checklist items
    as the PM's _QC_CHECKLIST so audits are calibrated to the same standard."""
    from tradingagents.agents.managers.qc_agent import _SYSTEM
    # Each item is on its own numbered line
    for n in range(1, 14):
        assert f"{n}." in _SYSTEM, f"QC system prompt missing item {n}"
    assert "QC_VERDICT:" in _SYSTEM


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
