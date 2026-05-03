"""Claude Code OAuth client.

Uses the OAuth credentials that Claude Code stores locally (macOS keychain
entry `Claude Code-credentials`, or `~/.claude/.credentials.json` on Linux)
instead of an `ANTHROPIC_API_KEY`. Intended for solo researchers who want
to drive multi-agent runs from their personal Claude subscription rather
than provisioning a separate API key.

Token refresh is delegated to Claude Code itself: if the access token has
expired, this client raises a clear error telling the user to run any
Claude Code command (which refreshes the credential store).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from anthropic import Anthropic, AsyncAnthropic
from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


_KEYCHAIN_SERVICE = "Claude Code-credentials"
_LINUX_CREDS_PATH = Path.home() / ".claude" / ".credentials.json"
_OAUTH_BETA_HEADER = "oauth-2025-04-20"


class ClaudeCodeAuthError(RuntimeError):
    """Raised when Claude Code OAuth credentials are missing or unusable."""


def _read_macos_keychain() -> dict:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", _KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except subprocess.CalledProcessError as e:
        raise ClaudeCodeAuthError(
            f"Could not read '{_KEYCHAIN_SERVICE}' from macOS keychain. "
            f"Run `claude /login` first. Underlying error: {e.stderr.strip()}"
        ) from e
    except FileNotFoundError as e:
        raise ClaudeCodeAuthError(
            "`security` command not found — is this macOS?"
        ) from e
    return json.loads(result.stdout)


def _read_linux_creds() -> dict:
    if not _LINUX_CREDS_PATH.exists():
        raise ClaudeCodeAuthError(
            f"No credentials at {_LINUX_CREDS_PATH}. Run `claude /login` first."
        )
    return json.loads(_LINUX_CREDS_PATH.read_text(encoding="utf-8"))


def _read_openclaw_profile(path: str, profile_name: str) -> str:
    """Return the access token for an OpenClaw auth-profiles.json profile."""
    p = Path(path)
    if not p.exists():
        raise ClaudeCodeAuthError(
            f"OpenClaw auth-profiles.json not found at {path}. "
            f"Verify the path or run OpenClaw's update-tokens.sh from the MacBook."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ClaudeCodeAuthError(
            f"OpenClaw auth-profiles.json at {path} is not valid JSON: {e}"
        ) from e

    profiles = data.get("profiles", {})
    profile = profiles.get(profile_name)
    if not profile:
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' not in {path}. "
            f"Available: {sorted(profiles.keys())}"
        )

    token = profile.get("token")
    if token is None:
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' has no 'token' field in {path}."
        )
    if not token:
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' 'token' field is empty in {path}."
        )
    if not token.startswith("sk-ant-oat01-"):
        raise ClaudeCodeAuthError(
            f"Profile '{profile_name}' token does not look like an Anthropic "
            f"OAuth token (expected sk-ant-oat01- prefix). Rotate via OpenClaw."
        )
    return token


def get_oauth_token(
    source: str = "keychain",
    openclaw_profile_path: str | None = None,
    openclaw_profile_name: str | None = None,
) -> str:
    """Return a non-expired Claude Code OAuth access token, or raise.

    source:
      - "keychain"          (default) — macOS keychain via `security` (Linux
                            falls back to ~/.claude/.credentials.json). On
                            macOS, if the keychain read fails (e.g. headless
                            host with no GUI session, or Lesson-#2 setups
                            that write only to ~/.claude/.credentials.json),
                            we transparently fall back to the file.
      - "openclaw_profile"  — read from an OpenClaw auth-profiles.json file
                            on the same host. Requires openclaw_profile_path
                            and openclaw_profile_name.
    """
    if source == "keychain":
        if platform.system() == "Darwin":
            try:
                creds = _read_macos_keychain()
            except ClaudeCodeAuthError:
                if _LINUX_CREDS_PATH.exists():
                    creds = _read_linux_creds()
                else:
                    raise
        else:
            creds = _read_linux_creds()
        oauth = creds.get("claudeAiOauth") or creds
        access_token = oauth.get("accessToken")
        expires_at = oauth.get("expiresAt")

        if not access_token:
            raise ClaudeCodeAuthError(
                "No accessToken in Claude Code credentials. Run `claude /login`."
            )
        if expires_at is not None:
            now_ms = int(time.time() * 1000)
            if now_ms >= expires_at:
                raise ClaudeCodeAuthError(
                    "Claude Code access token expired. Run any Claude Code "
                    "command (e.g. `claude /status`) to refresh the keychain entry, then retry."
                )
        return access_token

    if source == "openclaw_profile":
        if not openclaw_profile_path:
            raise ClaudeCodeAuthError(
                "openclaw_profile source requires openclaw_profile_path."
            )
        if not openclaw_profile_name:
            raise ClaudeCodeAuthError(
                "openclaw_profile source requires openclaw_profile_name."
            )
        return _read_openclaw_profile(openclaw_profile_path, openclaw_profile_name)

    raise ClaudeCodeAuthError(
        f"Unknown token source: {source!r}. Use 'keychain' or 'openclaw_profile'."
    )


class _OAuthChatAnthropic(ChatAnthropic):
    """ChatAnthropic with response content normalized to a plain string."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class ClaudeCodeClient(BaseLLMClient):
    """LLM client that authenticates with Claude Code OAuth credentials."""

    provider = "anthropic"

    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "max_tokens",
        "callbacks", "http_client", "http_async_client", "effort",
        "rate_limiter",
    )

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        token = get_oauth_token(
            source=self.kwargs.get("token_source", "keychain"),
            openclaw_profile_path=self.kwargs.get("openclaw_profile_path"),
            openclaw_profile_name=self.kwargs.get("openclaw_profile_name"),
        )

        sdk_kwargs: dict[str, Any] = {
            "auth_token": token,
            "default_headers": {"anthropic-beta": _OAUTH_BETA_HEADER},
        }
        if self.base_url:
            sdk_kwargs["base_url"] = self.base_url

        sync_client = Anthropic(**sdk_kwargs)
        async_client = AsyncAnthropic(**sdk_kwargs)

        # ChatAnthropic still requires a non-empty api_key at init for its own
        # validation; the value is unused because we replace the underlying
        # SDK client below to authenticate via OAuth bearer token instead.
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": "claude-code-oauth",
            "max_tokens": 8192,
            # Phase 5: absorb transient 429s with the Anthropic SDK's built-in
            # exponential backoff. Callers can still override via kwargs.
            "max_retries": 3,
        }
        for key in self._PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        chat = _OAuthChatAnthropic(**llm_kwargs)
        chat._client = sync_client
        chat._async_client = async_client
        return chat

    def validate_model(self) -> bool:
        return validate_model("anthropic", self.model)
