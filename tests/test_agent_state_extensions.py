"""Tests for the new AgentState fields added by the quant-research rebuild."""
import pytest

pytestmark = pytest.mark.unit


def test_agent_state_has_quant_research_fields():
    from tradingagents.agents.utils.agent_states import AgentState
    annotations = AgentState.__annotations__
    for field in ("pm_brief", "peers", "raw_dir", "technicals_report",
                  "pm_feedback", "pm_retries"):
        assert field in annotations, f"AgentState missing field: {field}"
