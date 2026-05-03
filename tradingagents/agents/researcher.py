"""Researcher — deterministic data fetcher (no LLM).

Replaces the bind_tools pattern in the original 4 analysts. Pulls all
data the multi-agent pipeline needs once, up front, and writes it to
`<output_dir>/raw/` as JSON / Markdown. Every downstream agent reads
from raw/ — no agent-side data fetching, no ReAct loops over tools.

Wraps the existing dataflows utilities (yfinance, alpha_vantage) as
plain Python functions called from this single deterministic step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _fetch_financials(ticker: str, date: str) -> dict[str, Any]:
    """Pull fundamentals + balance sheet + cashflow + income statement for one ticker."""
    from tradingagents.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_income_statement,
    )
    return {
        "ticker": ticker,
        "trade_date": date,
        "fundamentals": get_fundamentals.invoke({"ticker": ticker, "curr_date": date}),
        "balance_sheet": get_balance_sheet.invoke({"ticker": ticker, "curr_date": date}),
        "cashflow": get_cashflow.invoke({"ticker": ticker, "curr_date": date}),
        "income_statement": get_income_statement.invoke({"ticker": ticker, "curr_date": date}),
    }


def _fetch_news(ticker: str, date: str) -> dict[str, Any]:
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_global_news, get_news
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    return {
        "ticker_news": get_news.invoke({"ticker": ticker, "start_date": start, "end_date": date}),
        "global_news": get_global_news.invoke({"curr_date": date}),
    }


def _fetch_insider(ticker: str, date: str) -> dict[str, Any]:
    # get_insider_transactions accepts only `ticker`; date is unused but kept for signature symmetry.
    from tradingagents.agents.utils.agent_utils import get_insider_transactions
    return {"transactions": get_insider_transactions.invoke({"ticker": ticker})}


def _fetch_social(ticker: str, date: str) -> dict[str, Any]:
    # No dedicated social tool exists; reuse get_news as a sentiment-adjacent source.
    # T8/T9 prompts know to focus on social/sentiment-relevant items.
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_news
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=30)).strftime("%Y-%m-%d")
    return {"social_news": get_news.invoke({"ticker": ticker, "start_date": start, "end_date": date})}


def _fetch_prices(ticker: str, date: str) -> dict[str, Any]:
    """5y OHLCV history. Note: get_stock_data returns a string; T6 parses for spot/ATR/etc."""
    from datetime import datetime, timedelta
    from tradingagents.agents.utils.agent_utils import get_stock_data
    end = datetime.strptime(date, "%Y-%m-%d")
    start = (end - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return {
        "ohlcv": get_stock_data.invoke(
            {"symbol": ticker, "start_date": start, "end_date": date}
        ),
    }


def _fetch_indicators(ticker: str, date: str) -> dict[str, Any]:
    """Pull a fixed set of indicators per ticker. T5 (TA agent) parses the strings."""
    from tradingagents.agents.utils.agent_utils import get_indicators
    indicators = ("close_50_sma", "close_200_sma", "rsi", "macd", "boll_ub", "boll_lb", "atr")
    return {
        ind: get_indicators.invoke({"symbol": ticker, "indicator": ind, "curr_date": date})
        for ind in indicators
    }


def fetch_research_pack(state: dict) -> None:
    """Fetch all data needed by the multi-agent pipeline. Writes to raw/.

    Required state keys: `company_of_interest`, `trade_date`, `peers`, `raw_dir`.
    """
    ticker = state["company_of_interest"]
    date = state["trade_date"]
    peers = state.get("peers", [])
    raw = Path(state["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    # Per-ticker bundles
    financials = _fetch_financials(ticker, date)
    news = _fetch_news(ticker, date)
    insider = _fetch_insider(ticker, date)
    social = _fetch_social(ticker, date)
    prices = _fetch_prices(ticker, date)
    indicators = _fetch_indicators(ticker, date)

    # Peers (one financials bundle per peer)
    peers_data = {p: _fetch_financials(p, date) for p in peers}

    # Reference: single source of truth for prices.
    # prices["ohlcv"] is a raw string from get_stock_data; T6 parses numeric values.
    # indicators keys are indicator names; T6 parses the returned strings.
    reference = {
        "ticker": ticker,
        "trade_date": date,
        "reference_price": None,  # T6 (market analyst) extracts from prices["ohlcv"]
        "reference_price_source": f"yfinance close {date}",
        "spot_50dma": indicators.get("close_50_sma"),
        "spot_200dma": indicators.get("close_200_sma"),
        "ytd_high": None,  # T6 extracts from prices["ohlcv"]
        "ytd_low": None,   # T6 extracts from prices["ohlcv"]
        "atr_14": indicators.get("atr"),
    }

    # Write everything
    (raw / "financials.json").write_text(json.dumps(financials, indent=2, default=str), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers_data, indent=2, default=str), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps(insider, indent=2, default=str), encoding="utf-8")
    (raw / "social.json").write_text(json.dumps(social, indent=2, default=str), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps(prices, indent=2, default=str), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")
