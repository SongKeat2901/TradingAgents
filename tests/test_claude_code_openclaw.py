"""Tests for the openclaw_profile token source in claude_code_client."""

import json

import pytest

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    _read_openclaw_profile,
)


pytestmark = pytest.mark.unit


def _write_profile(path, content):
    path.write_text(json.dumps(content))


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
