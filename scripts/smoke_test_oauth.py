"""Smoke test: confirm Claude Code OAuth credentials work end-to-end.

Reads the local Claude Code keychain entry, builds a ChatAnthropic via the
ClaudeCodeClient, and sends a single ping. Prints the response (or the
auth error with remediation hint).

Run from the repo root:
    .venv/bin/python scripts/smoke_test_oauth.py
"""

from __future__ import annotations

import sys

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    ClaudeCodeClient,
    get_oauth_token,
)


def main() -> int:
    try:
        token = get_oauth_token()
    except ClaudeCodeAuthError as e:
        print(f"[auth] {e}", file=sys.stderr)
        return 2
    print(f"[auth] keychain access_token loaded ({len(token)} chars)")

    client = ClaudeCodeClient(model="claude-haiku-4-5")
    llm = client.get_llm()
    print(f"[client] {type(llm).__name__} ready, model={client.model}")

    response = llm.invoke("Reply with just the single word: PONG")
    print(f"[response] content={response.content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
