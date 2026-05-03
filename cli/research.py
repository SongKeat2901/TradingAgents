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
from pathlib import Path

# Heavy imports stay at module level despite the daemonize-after-import order.
# This deviates from the strict reading of Phase 5's spec (which asked us to
# move heavy imports below the daemonize call) but is verified safe in practice:
# importing TradingAgentsGraph spawns 0 threads and opens 0 extra file
# descriptors. POSIX fork() of a single-threaded process with no extra FDs is
# clean. The original parent calls os._exit(0) which bypasses cleanup, so the
# loaded modules in the parent are abandoned without side effects. Module-level
# imports are kept here because tests monkeypatch e.g. `research.TradingAgentsGraph`
# directly, and moving the imports inside main() would break that pattern.
# If a future LangGraph or anthropic upgrade starts threads at import time,
# revisit this and move imports below the _daemonize() call in main().
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.research_progress import ProgressCallback
from cli.research_telegram import (
    TelegramSendError,
    notify_failure,
    notify_success,
)
from cli.research_writer import write_research_outputs


def _daemonize(log_path: str) -> None:
    """Detach from the controlling terminal so the binary runs fire-and-forget.

    Standard double-fork + setsid pattern:
      - first fork: original parent prints child PID and exits 0
      - first child: setsid() to detach from controlling terminal
      - second fork: second-fork parent exits, leaving a grandchild
        orphaned to init (PPID=1) so it cannot acquire a TTY later
      - grandchild: redirect stdin/stdout/stderr to log_path
    """
    pid1 = os.fork()
    if pid1 > 0:
        print(f"started pid={pid1}", flush=True)
        os._exit(0)

    os.setsid()

    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)  # stdout
    os.dup2(log_fd, 2)  # stderr
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)  # stdin
    os.close(log_fd)
    os.close(devnull)


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
    p.add_argument(
        "--pacing-seconds", type=float, default=3.0,
        help=(
            "Minimum seconds between LLM calls (shared across deep+quick "
            "clients). Spreads burst to stay under output-tokens-per-minute "
            "rate limits on subscription auth. 0 disables pacing."
        ),
    )
    p.add_argument(
        "--no-daemonize", action="store_true",
        help="Skip the self-daemonize step (for tests / direct foreground use).",
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
    config["pacing_seconds"] = args.pacing_seconds
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

    # Self-daemonize by default. We can't trust callers (especially the
    # OpenClaw trader agent's synth-bash) to wrap us in nohup/&/Popen
    # correctly — they tend to invent broken patterns (nohup fails on macOS
    # without a TTY; subprocess.Popen + terminate() kills us prematurely).
    # By forking ourselves on entry, the parent process exits within ~1s
    # regardless of how we were called, and the grandchild runs to completion.
    # Foreground use (tests, direct CLI) passes --no-daemonize.
    if not args.no_daemonize:
        log_path = (
            Path(args.output_dir).parent.parent / "logs"
            / f"tradingresearch-{args.date}-{args.ticker}.log"
        )
        _daemonize(str(log_path))

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
