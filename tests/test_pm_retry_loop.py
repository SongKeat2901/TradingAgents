"""Tests for the PM Pass-3 push-back retry mechanism."""
import pytest

pytestmark = pytest.mark.unit


def test_pm_system_prompt_documents_retry_signal():
    """PM must know how to emit a structured retry decision."""
    from tradingagents.agents.managers.portfolio_manager import _RETRY_DIRECTIVE
    assert "retry" in _RETRY_DIRECTIVE.lower()
    assert "research_manager" in _RETRY_DIRECTIVE
    assert "risk_team" in _RETRY_DIRECTIVE
    # Cap rule
    assert "max" in _RETRY_DIRECTIVE.lower() or "1" in _RETRY_DIRECTIVE


def test_research_manager_handles_pm_feedback():
    """Research Manager prompt must reference pm_feedback when set."""
    from tradingagents.agents.managers.research_manager import _PM_FEEDBACK_HANDLER
    assert "pm_feedback" in _PM_FEEDBACK_HANDLER
    assert "address" in _PM_FEEDBACK_HANDLER.lower()


def test_risk_debators_handle_pm_feedback():
    """Each risk debator must reference pm_feedback in its prompt."""
    from tradingagents.agents.risk_mgmt.aggressive_debator import _PM_FEEDBACK_HANDLER as agg
    from tradingagents.agents.risk_mgmt.conservative_debator import _PM_FEEDBACK_HANDLER as con
    from tradingagents.agents.risk_mgmt.neutral_debator import _PM_FEEDBACK_HANDLER as neu
    for handler in (agg, con, neu):
        assert "pm_feedback" in handler
