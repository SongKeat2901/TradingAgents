"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content


def extract_llm_content(result: Any, agent_name: str) -> str:
    """Return ``result.content`` or raise if empty / non-substantive.

    The 2026-05-06 COIN cadence run surfaced a degenerate failure mode where
    the claude CLI subprocess returned an LLM result whose ``.content`` was
    the empty string. The pre-existing call-site pattern was::

        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

    The truthiness check ``if raw_content`` returns ``False`` on the empty
    string, so the report falls through to ``str(result)`` — which renders
    the LangChain envelope ``"content='' additional_kwargs={} ... tool_calls=[]"``.
    Downstream agents (debate, risk, decision) consumed that 140-char stub
    as if it were the analyst's actual report, and the 16-item QC accepted
    it. The COIN run shipped UNDERWEIGHT with zero fundamentals analysis.

    Raise instead of falling back. The caller should let the exception
    propagate so the run dies before debaters / PM / QC see garbage.
    """
    raw = result.content if hasattr(result, "content") else None
    if raw is None or not raw.strip():
        raise RuntimeError(
            f"{agent_name}: LLM returned empty content. "
            f"This is a degenerate response — falling back to str(result) "
            f"would write a LangChain envelope stub that downstream agents "
            f"would consume as a real report (observed COIN 2026-05-06). "
            f"Re-run the node; if the failure persists, check prompt size "
            f"and CLI subprocess behaviour."
        )
    return raw
