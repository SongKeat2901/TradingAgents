"""Tests for the temperature=0 pin on the analyst-tier LLM (Phase-6 stochasticity experiment)."""
import pytest

pytestmark = pytest.mark.unit


def test_default_config_has_temperature_zero():
    """The pin's default value lives in DEFAULT_CONFIG so any pipeline run
    that doesn't explicitly override it gets temperature=0."""
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG.get("temperature") == 0.0


def test_oauth_chat_anthropic_accepts_temperature_kwarg():
    """_OAuthChatAnthropic must accept temperature in its constructor and
    expose it on the resulting model instance."""
    from tradingagents.llm_clients.claude_code_client import _OAuthChatAnthropic
    chat = _OAuthChatAnthropic(model="claude-sonnet-4-6", api_key="test", temperature=0.0)
    assert chat.temperature == 0.0


def test_passthrough_kwargs_includes_temperature():
    """The kwargs allowlist must include temperature so user-provided values
    flow into _OAuthChatAnthropic instead of being dropped."""
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient
    assert "temperature" in ClaudeCodeClient._PASSTHROUGH_KWARGS


def test_provider_kwargs_includes_temperature_for_claude_code():
    """trading_graph._get_provider_kwargs must surface the config's temperature
    value into the LLM client kwargs for the claude_code provider."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    config = {
        "llm_provider": "claude_code",
        "deep_think_llm": "claude-opus-4-6",
        "quick_think_llm": "claude-sonnet-4-6",
        "claude_code_token_source": "keychain",
        "temperature": 0.0,
    }
    # Use object.__new__ to avoid triggering full __init__ (which builds the graph)
    g = object.__new__(TradingAgentsGraph)
    g.config = config
    kwargs = g._get_provider_kwargs()
    assert kwargs.get("temperature") == 0.0
