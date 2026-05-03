"""Headless CLI: run TradingAgents end-to-end and emit decision JSON + report files.

Optionally posts the result to Telegram on completion (`--telegram-notify
<chat_id>` + `TRADINGRESEARCH_BOT_TOKEN` env var). This lets the binary
be invoked fire-and-forget by an OpenClaw skill without blocking the
agent's session under the daemon's no-output watchdog.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.research_progress import ProgressCallback
from cli.research_telegram import (
    TelegramSendError,
    notify_failure,
    notify_success,
)
from cli.research_writer import write_research_outputs


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
    p.add_argument(
        "--debate-rounds", type=int, default=1,
        help="Number of bull-bear debate rounds.",
    )
    p.add_argument(
        "--risk-rounds", type=int, default=1,
        help="Number of risk discussion rounds.",
    )

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

    p.add_argument(
        "--telegram-notify",
        metavar="CHAT_ID",
        help=(
            "Post the decision (or error) to this Telegram chat on exit. "
            "Bot token is read from env TRADINGRESEARCH_BOT_TOKEN. "
            "If either is missing, no notification is sent."
        ),
    )
    return p


def _build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "claude_code"
    config["deep_think_llm"] = args.deep
    config["quick_think_llm"] = args.quick
    config["max_debate_rounds"] = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds
    config["claude_code_token_source"] = args.token_source
    config["claude_code_openclaw_profile_path"] = args.openclaw_profile_path
    config["claude_code_openclaw_profile_name"] = args.openclaw_profile_name
    return config


def _telegram_args(args: argparse.Namespace) -> tuple[str, str] | None:
    """Return (bot_token, chat_id) if both are available, else None."""
    chat_id = args.telegram_notify
    bot_token = os.environ.get("TRADINGRESEARCH_BOT_TOKEN")
    if chat_id and bot_token:
        return bot_token, chat_id
    return None


def _safe_notify_failure(
    tg: tuple[str, str] | None, ticker: str, date: str, summary: str
) -> None:
    if tg is None:
        return
    try:
        notify_failure(tg[0], tg[1], ticker, date, summary)
    except TelegramSendError as te:
        print(f"telegram notify failed: {te}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    progress = ProgressCallback()
    tg = _telegram_args(args)

    # Lazy import so test fixtures can monkeypatch ClaudeCodeAuthError if needed.
    from tradingagents.llm_clients.claude_code_client import ClaudeCodeAuthError

    progress.on_node_start("research")
    started = time.monotonic()
    try:
        graph = TradingAgentsGraph(debug=False, config=config)
        final_state, _decision = graph.propagate(args.ticker, args.date)
        write_research_outputs(final_state, args.output_dir)
    except ClaudeCodeAuthError as e:
        msg = f"auth error: {e}"
        print(msg, file=sys.stderr)
        _safe_notify_failure(tg, args.ticker, args.date, msg)
        return 1
    except Exception as e:  # noqa: BLE001 - top-level CLI catch
        traceback.print_exc(file=sys.stderr)
        _safe_notify_failure(tg, args.ticker, args.date, f"runtime error: {e}")
        return 2

    duration = time.monotonic() - started
    progress.on_node_done("research", duration_s=duration)
    decision = final_state.get("final_trade_decision", "UNKNOWN")
    payload = {
        "decision": decision,
        "ticker": args.ticker,
        "date": args.date,
        "output_dir": args.output_dir,
        "duration_s": round(duration, 1),
    }
    print(json.dumps(payload), flush=True)

    if tg is not None:
        try:
            notify_success(tg[0], tg[1], args.output_dir, decision)
        except TelegramSendError as te:
            print(f"telegram notify failed: {te}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
