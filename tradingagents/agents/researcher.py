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

from tradingagents.agents.utils.raw_data import raw_dir_for


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
        "fundamentals": get_fundamentals.invoke({"ticker": ticker, "date": date}),
        "balance_sheet": get_balance_sheet.invoke({"ticker": ticker, "date": date}),
        "cashflow": get_cashflow.invoke({"ticker": ticker, "date": date}),
        "income_statement": get_income_statement.invoke({"ticker": ticker, "date": date}),
    }


def _fetch_news(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_global_news, get_news
    return {
        "ticker_news": get_news.invoke({"ticker": ticker, "date": date}),
        "global_news": get_global_news.invoke({"date": date}),
    }


def _fetch_insider(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_insider_transactions
    return {"transactions": get_insider_transactions.invoke({"ticker": ticker, "date": date})}


def _fetch_social(ticker: str, date: str) -> dict[str, Any]:
    # Reuse the news tool for social by convention; downstream prompts know
    # to focus on social/sentiment indicators in this view.
    from tradingagents.agents.utils.agent_utils import get_news
    return {"social_news": get_news.invoke({"ticker": ticker, "date": date, "social": True})}


def _fetch_prices(ticker: str, date: str) -> dict[str, Any]:
    """5y OHLCV + close on date + YTD high/low + ATR."""
    from tradingagents.agents.utils.agent_utils import get_stock_data
    return get_stock_data.invoke({"ticker": ticker, "date": date, "lookback_years": 5})


def _fetch_indicators(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_indicators
    return get_indicators.invoke({"ticker": ticker, "date": date})


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

    # Reference: single source of truth for prices
    reference = {
        "ticker": ticker,
        "trade_date": date,
        "reference_price": prices.get("close_on_date"),
        "reference_price_source": f"yfinance close {date}",
        "spot_50dma": indicators.get("sma_50"),
        "spot_200dma": indicators.get("sma_200"),
        "ytd_high": prices.get("ytd_high"),
        "ytd_low": prices.get("ytd_low"),
        "atr_14": prices.get("atr_14"),
    }

    # Write everything
    (raw / "financials.json").write_text(json.dumps(financials, indent=2, default=str), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers_data, indent=2, default=str), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps(insider, indent=2, default=str), encoding="utf-8")
    (raw / "social.json").write_text(json.dumps(social, indent=2, default=str), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps(prices, indent=2, default=str), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")
