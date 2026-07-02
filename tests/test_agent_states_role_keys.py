import pytest

pytestmark = pytest.mark.unit


def test_role_report_keys_present():
    from tradingagents.agents.utils.agent_states import AgentState
    ann = AgentState.__annotations__
    for k in ("fundamentals_financial_report", "fundamentals_riskflags_report",
              "fundamentals_catalysts_report", "fundamentals_quality_report",
              "fundamentals_report"):
        assert k in ann, f"missing state key: {k}"
