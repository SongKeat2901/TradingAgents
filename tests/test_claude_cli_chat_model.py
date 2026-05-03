"""Tests for ClaudeCliChatModel — the claude-CLI subprocess wrapper used
by the deep judges (Research Manager, Portfolio Manager) to bypass the
tight rate limit on the direct-API OAuth path."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

pytestmark = pytest.mark.unit


def test_format_messages_renders_role_headers():
    from tradingagents.llm_clients.claude_cli_chat_model import _format_messages

    out = _format_messages([
        SystemMessage(content="Be terse."),
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4."),
        HumanMessage(content="Why?"),
    ])
    assert "# System" in out
    assert "Be terse." in out
    assert "# User" in out
    assert "What is 2+2?" in out
    assert "# Assistant" in out
    assert "4." in out
    # Both User turns appear in order
    assert out.index("What is 2+2?") < out.index("Why?")


def test_resolve_cli_model_name_maps_full_ids():
    from tradingagents.llm_clients.claude_cli_chat_model import _resolve_cli_model_name

    assert _resolve_cli_model_name("claude-opus-4-6") == "opus"
    assert _resolve_cli_model_name("claude-sonnet-4-5") == "sonnet"
    assert _resolve_cli_model_name("claude-haiku-4-5") == "haiku"
    assert _resolve_cli_model_name("opus") == "opus"  # already-short stays
    assert _resolve_cli_model_name("custom-model") == "custom-model"


def test_invoke_calls_claude_cli_and_returns_aimessage(monkeypatch):
    from tradingagents.llm_clients import claude_cli_chat_model as mod

    captured: dict = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, check=None, timeout=None):
        captured["cmd"] = cmd
        captured["input"] = input
        captured["timeout"] = timeout
        return MagicMock(stdout="**HOLD**\n\nReason here.\n", stderr="", returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    llm = mod.ClaudeCliChatModel(model="claude-opus-4-6", timeout_seconds=42.0)
    result = llm.invoke([
        SystemMessage(content="You judge research."),
        HumanMessage(content="Decide BUY/SELL/HOLD."),
    ])

    assert isinstance(result, AIMessage)
    assert result.content.startswith("**HOLD**")
    # Command line: claude -p --model opus (alias resolved)
    assert captured["cmd"] == ["claude", "-p", "--model", "opus"]
    # Prompt body has both messages, with role headers
    assert "# System" in captured["input"]
    assert "# User" in captured["input"]
    assert "Decide BUY/SELL/HOLD." in captured["input"]
    # Timeout passed through
    assert captured["timeout"] == 42.0


def test_invoke_raises_on_cli_failure(monkeypatch):
    from tradingagents.llm_clients import claude_cli_chat_model as mod

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr="auth: invalid token"
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    llm = mod.ClaudeCliChatModel()
    with pytest.raises(RuntimeError, match="exited 1"):
        llm.invoke([HumanMessage(content="hi")])


def test_invoke_raises_on_timeout(monkeypatch):
    from tradingagents.llm_clients import claude_cli_chat_model as mod

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5.0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    llm = mod.ClaudeCliChatModel(timeout_seconds=5.0)
    with pytest.raises(RuntimeError, match="timed out after 5"):
        llm.invoke([HumanMessage(content="hi")])


def test_invoke_raises_on_missing_cli(monkeypatch):
    from tradingagents.llm_clients import claude_cli_chat_model as mod

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError(2, "no such file")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    llm = mod.ClaudeCliChatModel(cli_path="/missing/claude")
    with pytest.raises(RuntimeError, match="not found"):
        llm.invoke([HumanMessage(content="hi")])


def test_custom_cli_path_used(monkeypatch):
    """For trueknot the CLI is at /Users/trueknot/.nvm/.../claude — verify the
    cli_path field is honored."""
    from tradingagents.llm_clients import claude_cli_chat_model as mod

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    llm = mod.ClaudeCliChatModel(
        cli_path="/Users/trueknot/.nvm/versions/node/v24.14.1/bin/claude",
        model="claude-opus-4-6",
    )
    llm.invoke([HumanMessage(content="hi")])

    assert captured["cmd"][0] == "/Users/trueknot/.nvm/versions/node/v24.14.1/bin/claude"
