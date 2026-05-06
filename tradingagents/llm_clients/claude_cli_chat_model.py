"""Claude CLI subprocess chat model.

Sidesteps the tight per-minute rate limit observed on the direct Anthropic
API + OAuth bearer auth path (langchain_anthropic.ChatAnthropic with the
`oauth-2025-04-20` beta header) by using the same `claude -p` CLI flow
that OpenClaw uses internally.

Empirical finding (Phase 5, 2026-05-03): a 10K+ token prompt to Opus 4.6
via `claude -p` succeeds in ~14s; the same prompt via ChatAnthropic +
OAuth bearer 429s with a generic rate_limit_error. Anthropic's per-minute
buckets for the two paths are different — claude-cli is the relaxed one.

This wrapper is used for the **deep judges** (Research Manager, Portfolio
Manager) only:
- They don't use tool calls (analysts do; analysts stay on ChatAnthropic).
- They produce text from accumulated state — no streaming, no structured
  output beyond what we ask the prompt to emit.
- They are where the 429s happen.

Subprocess overhead per call (~2-3s) is irrelevant for a 14s call.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


_MODEL_ALIASES = {
    "claude-opus-4-7": "opus",
    "claude-opus-4-6": "opus",
    "claude-opus-4-5": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-sonnet-4-5": "sonnet",
    "claude-haiku-4-5": "haiku",
}


def _format_messages(messages: List[BaseMessage]) -> str:
    """Concatenate LangChain messages into a single prompt string for claude -p.

    Uses Markdown-style role headers because they are unambiguous to the model
    and survive shell escaping cleanly when piped via stdin.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"# System\n\n{msg.content}\n")
        elif isinstance(msg, HumanMessage):
            parts.append(f"# User\n\n{msg.content}\n")
        elif isinstance(msg, AIMessage):
            parts.append(f"# Assistant\n\n{msg.content}\n")
        else:
            parts.append(f"# {(msg.type or 'message').title()}\n\n{msg.content}\n")
    return "\n".join(parts)


def _resolve_cli_model_name(model: str) -> str:
    """Map full model ids to claude CLI's short aliases when possible."""
    return _MODEL_ALIASES.get(model, model)


def _discover_claude_cli() -> Optional[str]:
    """Find the claude executable when it's not on PATH.

    Order of search:
    1. shutil.which('claude') — happy path on dev machines.
    2. ~/.nvm/versions/node/*/bin/claude — nvm-managed installs (common on
       OpenClaw hosts; the trueknot daemon's subprocess PATH is bare so PATH
       lookup fails even though the binary exists).
    3. /opt/homebrew/bin/claude, /usr/local/bin/claude — Homebrew defaults.
    """
    found = shutil.which("claude")
    if found:
        return found

    home = os.path.expanduser("~")
    nvm_glob = os.path.join(home, ".nvm", "versions", "node", "*", "bin", "claude")
    matches = sorted(glob.glob(nvm_glob), reverse=True)  # newest version first
    if matches:
        return matches[0]

    for fallback in ("/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
            return fallback

    return None


class ClaudeCliChatModel(BaseChatModel):
    """LangChain BaseChatModel that shells out to the local ``claude`` CLI.

    Uses the same OAuth-via-claude-CLI path that OpenClaw uses internally,
    which empirically clears Anthropic's stricter per-minute rate limits on
    the direct-API + OAuth-bearer path.
    """

    model: str = "opus"
    """Model name — either a CLI alias (``opus``/``sonnet``/``haiku``) or a
    full id like ``claude-opus-4-6``. Full ids are mapped to aliases."""

    cli_path: str = "claude"
    """Path to the claude executable. Defaults to PATH lookup. Override
    for hosts where claude is at a non-standard nvm path (e.g. trueknot
    has it at ``/Users/trueknot/.nvm/versions/node/v24.14.1/bin/claude``)."""

    timeout_seconds: float = 900.0
    """Subprocess timeout in seconds. 15 min default — 600s headroom proved
    insufficient on the 2026-05-06 cadence (TSCO Fundamentals Analyst hit
    the limit on a verbose retail-cohort peers prompt and crashed the run).
    900s adds margin without affecting cost or steady-state behavior."""

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _format_messages(messages)
        model_arg = _resolve_cli_model_name(self.model)

        # If cli_path is the default 'claude' bare name and that's not on
        # PATH (common on daemon-spawned subprocesses with bare PATH), try
        # to auto-discover the executable in common install locations.
        cli_path = self.cli_path
        if cli_path == "claude" and shutil.which("claude") is None:
            discovered = _discover_claude_cli()
            if discovered:
                cli_path = discovered

        cmd = [cli_path, "-p", "--model", model_arg]

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "")[:1000]
            raise RuntimeError(
                f"claude CLI exited {e.returncode}: {stderr.strip()}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"claude CLI timed out after {self.timeout_seconds}s"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"claude CLI not found at {cli_path!r}. Set cli_path on "
                f"ClaudeCliChatModel, set claude_code_cli_path in config, "
                f"or install claude such that it's on PATH or in a "
                f"recognised location (~/.nvm/versions/node/*/bin/claude, "
                f"/opt/homebrew/bin/claude, /usr/local/bin/claude)."
            ) from e

        text = result.stdout.strip()
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])
