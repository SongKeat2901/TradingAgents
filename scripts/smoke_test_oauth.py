"""Smoke test: confirm Claude Code OAuth credentials work end-to-end.

Default reads the macOS keychain. Pass --source openclaw_profile to instead
read an OpenClaw auth-profiles.json file (useful for verifying the
production path locally before deploying to mini).

Run from the repo root:
    .venv/bin/python scripts/smoke_test_oauth.py
    .venv/bin/python scripts/smoke_test_oauth.py \\
        --source openclaw_profile \\
        --path /tmp/auth-profiles.json --name anthropic:default
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.llm_clients.claude_code_client import (
    ClaudeCodeAuthError,
    ClaudeCodeClient,
    get_oauth_token,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("keychain", "openclaw_profile"), default="keychain")
    parser.add_argument("--path", help="auth-profiles.json path (openclaw_profile only)")
    parser.add_argument("--name", default="anthropic:default")
    args = parser.parse_args()

    try:
        token = get_oauth_token(
            source=args.source,
            openclaw_profile_path=args.path,
            openclaw_profile_name=args.name,
        )
    except ClaudeCodeAuthError as e:
        print(f"[auth] {e}", file=sys.stderr)
        return 2
    print(f"[auth] {args.source} access_token loaded ({len(token)} chars)")

    client = ClaudeCodeClient(
        model="claude-haiku-4-5",
        token_source=args.source,
        openclaw_profile_path=args.path,
        openclaw_profile_name=args.name,
    )
    llm = client.get_llm()
    print(f"[client] {type(llm).__name__} ready, model={client.model}")

    response = llm.invoke("Reply with just the single word: PONG")
    print(f"[response] content={response.content!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
