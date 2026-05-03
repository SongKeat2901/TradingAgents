import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "output_dir": "results/",
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Claude Code OAuth: where to read the access token from.
    # "keychain" — macOS keychain (default; for local dev on the user's MacBook).
    # "openclaw_profile" — read from an OpenClaw auth-profiles.json file
    #   (used when running as a TrueKnot OpenClaw skill on Farm 1 mini).
    "claude_code_token_source": "keychain",
    "claude_code_openclaw_profile_path": None,
    "claude_code_openclaw_profile_name": "anthropic:default",
    # Phase 5: minimum seconds between LLM calls. Shared between deep + quick
    # clients via a langchain_core InMemoryRateLimiter. 0 disables pacing.
    # Subscription-OAuth users on Anthropic typically need 3-5s to stay under
    # output-tokens-per-minute burst limits.
    "pacing_seconds": 30.0,
    # Additional sleep before each deep-model call (only used when deep
    # is on the ChatAnthropic path — see deep_via_cli below).
    "deep_cooldown_seconds": 90.0,
    # Phase 5: route the deep judges (Research Manager, Portfolio Manager)
    # through `claude -p` subprocess instead of langchain_anthropic
    # ChatAnthropic. Empirically the only path that doesn't 429 on Sonnet/
    # Opus via subscription OAuth. Quick client (Haiku) stays on
    # ChatAnthropic so analysts retain bind_tools() for yfinance/alpha_vantage.
    "deep_via_cli": True,
    # Optional override for the claude CLI path. Defaults to PATH lookup.
    # On the trueknot host claude is at
    # /Users/trueknot/.nvm/versions/node/v24.14.1/bin/claude — set this
    # in the env there if it's not on PATH for the daemon subprocess.
    "claude_code_cli_path": None,

    # WARNING: do not set both deep_think_llm and quick_think_llm to the same
    # Sonnet variant on subscription auth — that doubles burst risk on the
    # 4 analyst turns plus the 2 deep judges. Keep haiku for quick.

    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
