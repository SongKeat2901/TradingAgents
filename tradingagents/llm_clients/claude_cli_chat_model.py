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

import subprocess
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


_MODEL_ALIASES = {
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

    timeout_seconds: float = 600.0
    """Subprocess timeout in seconds. 10 min default — accommodates the
    occasional Opus prompt-warmup latency on cold OAuth caches."""

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
        cmd = [self.cli_path, "-p", "--model", model_arg]

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
                f"claude CLI not found at {self.cli_path!r}. Set cli_path "
                f"or ensure claude is on PATH."
            ) from e

        text = result.stdout.strip()
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])
