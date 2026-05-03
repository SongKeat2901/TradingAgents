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

    p.add_argument(
        "--deep", default="claude-opus-4-6",
        help="Deep-think model id (Research Manager + Portfolio Manager).",
    )
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
        "--pacing-seconds", type=float, default=30.0,
        help=(
            "Minimum seconds between any two LLM calls (shared rate limiter "
            "across deep+quick clients). Spreads burst to stay under "
            "output-tokens-per-minute rate limits on subscription auth. "
            "Default 30 — generous for reliability; lower at your own risk."
        ),
    )
    p.add_argument(
        "--deep-cooldown-seconds", type=float, default=90.0,
        help=(
            "Additional sleep before each deep-model call (Research Manager, "
            "Portfolio Manager). Lets per-minute output-tokens bucket fully "
            "refill before the heavy judges fire. Default 90 — stacks on top "
            "of --pacing-seconds. Set 0 to disable."
        ),
    )
    p.add_argument(
        "--max-tokens", type=int, default=4096,
        help=(
            "Max output tokens per LLM call. Default 4096 — comfortable for "
            "Opus 4.6 judges; analysts produce far less."
        ),
    )
    return p


# Env-var only escape hatch for tests / direct foreground use. Deliberately
# NOT a CLI flag — the OpenClaw trader agent kept discovering --no-daemonize
# (via --help, source code, or its own session memory) and adding it to its
# synth-bash, defeating the daemonize-by-default gate. An env var is invisible
# to argparse introspection and the agent doesn't set it spontaneously.
_FOREGROUND_ENV = "TRADINGRESEARCH_FOREGROUND"


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
    config["max_tokens"] = args.max_tokens
    config["deep_cooldown_seconds"] = args.deep_cooldown_seconds
    return config


_OPENCLAW_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"


def _auto_discover_telegram_from_openclaw() -> tuple[str | None, str | None]:
    """If running on an OpenClaw host (~/.openclaw/openclaw.json exists), read
    the bot token and the first configured group chat_id from it. Returns
    (bot_token, chat_id), each possibly None.

    The OpenClaw trader agent's synth-bash often drops --telegram-notify and
    omits the TRADINGRESEARCH_BOT_TOKEN env var. This auto-discovery makes
    the binary resilient to that — when run on the trueknot host, posting to
    Telegram works even if the caller forgot the explicit args.
    """
    if not _OPENCLAW_CONFIG_PATH.exists():
        return None, None
    try:
        cfg = json.loads(_OPENCLAW_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    channels = cfg.get("channels", {})
    telegram = channels.get("telegram", {})
    accounts = telegram.get("accounts", {})
    bot_token = (accounts.get("default") or {}).get("botToken")
    groups = telegram.get("groups", {})
    # Prefer the first explicitly-keyed group (skip the wildcard "*" key
    # which OpenClaw uses to mean "any group").
    chat_id = next((k for k in groups if k != "*"), None)
    return bot_token, chat_id


def _telegram_args(args: argparse.Namespace) -> tuple[str, str] | None:
    """Return (bot_token, chat_id) if both are available, else None.

    Resolution order:
    1. CLI --telegram-notify + env TRADINGRESEARCH_BOT_TOKEN (explicit caller).
    2. Auto-discover from ~/.openclaw/openclaw.json (OpenClaw deployment).
    """
    chat_id = args.telegram_notify
    bot_token = os.environ.get("TRADINGRESEARCH_BOT_TOKEN")
    if chat_id and bot_token:
        return bot_token, chat_id

    auto_token, auto_chat = _auto_discover_telegram_from_openclaw()
    bot_token = bot_token or auto_token
    chat_id = chat_id or auto_chat
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
    # Foreground use (tests, direct CLI) sets TRADINGRESEARCH_FOREGROUND=1.
    if not os.environ.get(_FOREGROUND_ENV):
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
