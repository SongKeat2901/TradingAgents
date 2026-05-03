"""Tests for the openclaw_profile token source in claude_code_client."""

import json

import pytest

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    _read_openclaw_profile,
    get_oauth_token,
)


pytestmark = pytest.mark.unit


def _write_profile(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


def test_reads_token_from_profile(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {
            "anthropic:default": {
                "type": "token",
                "provider": "anthropic",
                "token": "sk-ant-oat01-abc123",
            },
        },
    })
    assert _read_openclaw_profile(str(p), "anthropic:default") == "sk-ant-oat01-abc123"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ClaudeCodeAuthError, match="not found"):
        _read_openclaw_profile(str(tmp_path / "nope.json"), "anthropic:default")


def test_missing_profile_name_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {"version": 1, "profiles": {}})
    with pytest.raises(ClaudeCodeAuthError, match="anthropic:default"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_malformed_token_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "not-an-oauth-token"},
        },
    })
    with pytest.raises(ClaudeCodeAuthError, match="sk-ant-oat01"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    p.write_text("{not json")
    with pytest.raises(ClaudeCodeAuthError):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_missing_token_key_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {"anthropic:default": {"type": "oauth"}},
    })
    with pytest.raises(ClaudeCodeAuthError, match="no 'token' field"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_empty_token_raises(tmp_path):
    p = tmp_path / "auth-profiles.json"
    _write_profile(p, {
        "version": 1,
        "profiles": {"anthropic:default": {"type": "token", "token": ""}},
    })
    with pytest.raises(ClaudeCodeAuthError, match="empty"):
        _read_openclaw_profile(str(p), "anthropic:default")


def test_get_oauth_token_openclaw_source(tmp_path):
    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "sk-ant-oat01-xyz"},
        },
    }), encoding="utf-8")
    token = get_oauth_token(
        source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
    )
    assert token == "sk-ant-oat01-xyz"


def test_get_oauth_token_unknown_source_raises():
    with pytest.raises(ClaudeCodeAuthError, match="Unknown token source"):
        get_oauth_token(source="bogus")


def test_get_oauth_token_missing_path_raises():
    with pytest.raises(ClaudeCodeAuthError, match="openclaw_profile_path"):
        get_oauth_token(
            source="openclaw_profile",
            openclaw_profile_path=None,
            openclaw_profile_name="anthropic:default",
        )


def test_get_oauth_token_missing_name_raises():
    with pytest.raises(ClaudeCodeAuthError, match="openclaw_profile_name"):
        get_oauth_token(
            source="openclaw_profile",
            openclaw_profile_path="/some/path",
            openclaw_profile_name=None,
        )


def test_keychain_falls_back_to_credentials_file_on_darwin(tmp_path, monkeypatch):
    """On macOS hosts where the keychain entry is missing but
    ~/.claude/.credentials.json exists (e.g. OpenClawOps Lesson #2 setups),
    the keychain source must fall through to the file rather than fail."""
    import tradingagents.llm_clients.claude_code_client as mod

    fake_creds = tmp_path / "credentials.json"
    fake_creds.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-fallback",
        },
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "_LINUX_CREDS_PATH", fake_creds)
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")

    def boom():
        raise ClaudeCodeAuthError("keychain entry not found")

    monkeypatch.setattr(mod, "_read_macos_keychain", boom)

    assert get_oauth_token(source="keychain") == "sk-ant-oat01-fallback"


def test_keychain_failure_with_no_fallback_file_reraises(tmp_path, monkeypatch):
    """If keychain fails AND the fallback file does not exist, the original
    keychain error must propagate (otherwise the user gets a misleading
    'file not found' instead of the real keychain reason)."""
    import tradingagents.llm_clients.claude_code_client as mod

    monkeypatch.setattr(mod, "_LINUX_CREDS_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")

    def boom():
        raise ClaudeCodeAuthError("keychain locked")

    monkeypatch.setattr(mod, "_read_macos_keychain", boom)

    with pytest.raises(ClaudeCodeAuthError, match="keychain locked"):
        get_oauth_token(source="keychain")


def test_client_uses_openclaw_source(tmp_path, monkeypatch):
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient

    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {"type": "token", "token": "sk-ant-oat01-zzz"},
        },
    }), encoding="utf-8")

    captured = {}

    def fake_anthropic(**kwargs):
        captured.update(kwargs)
        return object()  # don't actually network

    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.Anthropic", fake_anthropic)
    monkeypatch.setattr("tradingagents.llm_clients.claude_code_client.AsyncAnthropic", fake_anthropic)
    # _OAuthChatAnthropic does network on first call but not on construction;
    # we just need the constructor to succeed.
    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client._OAuthChatAnthropic",
        lambda **kw: type("Stub", (), {"_client": None, "_async_client": None, **kw})(),
    )

    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
    )
    client.get_llm()
    assert captured["auth_token"] == "sk-ant-oat01-zzz"


def test_client_forwards_rate_limiter_to_chat_anthropic(tmp_path, monkeypatch):
    """A rate_limiter passed via kwargs lands on the underlying ChatAnthropic."""
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeClient

    p = tmp_path / "auth-profiles.json"
    p.write_text(json.dumps({
        "version": 1,
        "profiles": {"anthropic:default": {"type": "token", "token": "sk-ant-oat01-z"}},
    }), encoding="utf-8")

    captured: dict = {}

    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client.Anthropic",
        lambda **kw: object(),
    )
    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client.AsyncAnthropic",
        lambda **kw: object(),
    )

    def fake_chat(**kw):
        captured.update(kw)
        return type("Stub", (), {"_client": None, "_async_client": None})()

    monkeypatch.setattr(
        "tradingagents.llm_clients.claude_code_client._OAuthChatAnthropic",
        fake_chat,
    )

    sentinel = object()
    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source="openclaw_profile",
        openclaw_profile_path=str(p),
        openclaw_profile_name="anthropic:default",
        rate_limiter=sentinel,
    )
    client.get_llm()
    assert captured.get("rate_limiter") is sentinel
