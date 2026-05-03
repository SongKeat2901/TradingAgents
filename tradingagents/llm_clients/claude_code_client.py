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
    return json.loads(_LINUX_CREDS_PATH.read_text())


def get_oauth_token() -> str:
    """Return a non-expired Claude Code OAuth access token, or raise."""
    if platform.system() == "Darwin":
        creds = _read_macos_keychain()
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
                "Claude Code access token expired. Run any Claude Code command "
                "(e.g. `claude /status`) to refresh the keychain entry, then retry."
            )

    return access_token


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
    )

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        token = get_oauth_token()

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
            # Without this the Anthropic SDK enforces streaming for any call
            # whose worst-case duration exceeds 10 minutes. Trading agent
            # reports comfortably fit in 8K tokens.
            "max_tokens": 8192,
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
