from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

load_dotenv()

config = DEFAULT_CONFIG.copy()

# Use Claude Code OAuth credentials from local keychain instead of an API key.
# Requires: Claude Code CLI installed and logged in (`claude /login`).
config["llm_provider"]    = "claude_code"
config["deep_think_llm"]  = "claude-sonnet-4-6"
config["quick_think_llm"] = "claude-haiku-4-5"
config["anthropic_effort"] = "medium"

config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis":      "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data":     "yfinance",
    "news_data":            "yfinance",
}

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
