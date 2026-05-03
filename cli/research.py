"""Headless CLI: run TradingAgents end-to-end and emit decision JSON + report files.

Designed to be invoked by an OpenClaw skill (TrueKnot trading-research). For
interactive use, see cli/main.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tradingresearch",
        description="Run a multi-agent equity research workflow on a ticker for a date.",
    )
    p.add_argument("--ticker", required=True, help="US-listed ticker symbol, e.g. NVDA.")
    p.add_argument("--date", required=True, help="Trade date YYYY-MM-DD (historical).")
    p.add_argument("--output-dir", required=True, help="Directory to write report files into.")

    p.add_argument("--deep", default="claude-sonnet-4-6", help="Deep-think model id.")
    p.add_argument("--quick", default="claude-haiku-4-5", help="Quick-think model id.")
    p.add_argument("--debate-rounds", type=int, default=1)
    p.add_argument("--risk-rounds", type=int, default=1)

    p.add_argument(
        "--token-source", choices=("keychain", "openclaw_profile"), default="keychain",
        help="Where the claude_code provider reads the OAuth token from.",
    )
    p.add_argument(
        "--openclaw-profile-path",
        help="Path to OpenClaw auth-profiles.json (only when --token-source=openclaw_profile).",
    )
    p.add_argument(
        "--openclaw-profile-name", default="anthropic:default",
        help="Profile key inside auth-profiles.json (default: anthropic:default).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Behavior in subsequent tasks.
    return 0


if __name__ == "__main__":
    sys.exit(main())
