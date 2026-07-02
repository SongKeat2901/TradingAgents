import pytest

pytestmark = pytest.mark.unit

_ROLES = ["financial", "riskflags", "catalysts", "quality"]


def test_role_control_keys_present():
    from tradingagents.agents.utils.agent_states import AgentState
    ann = AgentState.__annotations__
    for r in _ROLES:
        for suf in ("retries", "feedback", "passed"):
            assert f"fundamentals_{r}_{suf}" in ann, f"missing fundamentals_{r}_{suf}"


def test_role_control_keys_initialized():
    from tradingagents.graph.propagation import Propagator
    init = Propagator().create_initial_state("MSFT", "2026-07-01")
    for r in _ROLES:
        assert init[f"fundamentals_{r}_retries"] == 0
        assert init[f"fundamentals_{r}_feedback"] == ""
        assert init[f"fundamentals_{r}_passed"] is False
