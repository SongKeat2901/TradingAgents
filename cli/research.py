"""Headless CLI: run TradingAgents end-to-end and emit decision JSON + report files."""

from __future__ import annotations

import argparse
import json
import sys
import time

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from cli.research_progress import ProgressCallback
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _build_config(args)
    progress = ProgressCallback()

    progress.on_node_start("research")
    started = time.monotonic()
    graph = TradingAgentsGraph(debug=False, config=config)
    final_state, _decision = graph.propagate(args.ticker, args.date)
    write_research_outputs(final_state, args.output_dir)
    duration = time.monotonic() - started
    progress.on_node_done("research", duration_s=duration)

    payload = {
        "decision": final_state.get("final_trade_decision", "UNKNOWN"),
        "ticker": args.ticker,
        "date": args.date,
        "output_dir": args.output_dir,
        "duration_s": round(duration, 1),
    }
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
