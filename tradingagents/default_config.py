import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
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
    # Additional sleep before each deep-model call. Lets per-minute
    # output-tokens bucket fully refill before Research Manager / Portfolio
    # Manager fire. Stacks on top of pacing_seconds.
    "deep_cooldown_seconds": 90.0,

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
