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
import re
from pathlib import Path
from typing import Any, Optional


_OHLCV_HEADER = "Date,Open,High,Low,Close,Volume"


def _parse_ohlcv_rows(ohlcv_str: str) -> list[tuple[str, float, float, float]]:
    """Parse the get_stock_data CSV string into (date, high, low, close) rows.

    Skips comment lines (#...) and the header row. Returns rows in the order
    they appear (chronological for yfinance). Malformed rows are skipped.
    """
    rows: list[tuple[str, float, float, float]] = []
    for line in ohlcv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            rows.append((parts[0], float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return rows


def _close_on_or_before(rows: list[tuple[str, float, float, float]], date: str) -> Optional[float]:
    """Return the Close on `date` or the most recent trading day before it."""
    candidate: Optional[float] = None
    for d, _h, _l, c in rows:
        if d <= date:
            candidate = c
        else:
            break
    return candidate


def _ytd_high_low(rows: list[tuple[str, float, float, float]], date: str) -> tuple[Optional[float], Optional[float]]:
    """Return (max High, min Low) across rows in the same calendar year as `date`, up to and including `date`."""
    year = date[:4]
    highs: list[float] = []
    lows: list[float] = []
    for d, h, l, _c in rows:
        if d.startswith(year) and d <= date:
            highs.append(h)
            lows.append(l)
    return (max(highs) if highs else None, min(lows) if lows else None)


def _latest_indicator_value(indicator_str: str) -> Optional[float]:
    """Pull the most recent numeric value from a get_indicators output string.

    Format is `## <ind> values from <start> to <end>:\\n\\n<DATE>: <val>\\n...`
    The first non-N/A `<DATE>: <number>` line is the most recent observation.
    """
    if not indicator_str or not isinstance(indicator_str, str):
        return None
    for m in re.finditer(r"^\d{4}-\d{2}-\d{2}:\s*([0-9]+(?:\.[0-9]+)?)\s*$", indicator_str, re.MULTILINE):
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None

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

    # Reference: single source of truth for prices. Parse the CSV / indicator
    # strings to numeric values here so downstream agents (especially the PM's
    # QC #7 self-audit) can verify price citations against a real number.
    rows = _parse_ohlcv_rows(prices.get("ohlcv", ""))
    close_on_date = _close_on_or_before(rows, date)
    ytd_high, ytd_low = _ytd_high_low(rows, date)
    reference = {
        "ticker": ticker,
        "trade_date": date,
        "reference_price": close_on_date,
        "reference_price_source": f"yfinance close on or before {date}",
        "spot_50dma": _latest_indicator_value(indicators.get("close_50_sma", "")),
        "spot_200dma": _latest_indicator_value(indicators.get("close_200_sma", "")),
        "ytd_high": ytd_high,
        "ytd_low": ytd_low,
        "atr_14": _latest_indicator_value(indicators.get("atr", "")),
    }

    # Write everything
    (raw / "financials.json").write_text(json.dumps(financials, indent=2, default=str), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers_data, indent=2, default=str), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps(insider, indent=2, default=str), encoding="utf-8")
    (raw / "social.json").write_text(json.dumps(social, indent=2, default=str), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps(prices, indent=2, default=str), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")

    # Phase-6 stochasticity mitigation: pure-Python deterministic classifier.
    # See tradingagents/agents/utils/classifier.py + the design spec at
    # docs/superpowers/specs/2026-05-04-deterministic-classifier-design.md
    from tradingagents.agents.utils.classifier import compute_classification
    classification = compute_classification(reference, prices.get("ohlcv", ""))
    (raw / "classification.json").write_text(
        json.dumps(classification, indent=2, default=str), encoding="utf-8"
    )

    # Phase-6.2 calendar.json is written by PM Pre-flight (which runs before
    # this node and has the peer list). Read-only here.

    # Phase-6.4 deterministic peer ratios: compute authoritative
    # capex/revenue + op margin + P/E from peers_data (already in memory)
    # and append a verbatim "## Peer ratios" block to pm_brief.md (which
    # PM Pre-flight already created). The peer-ratios block must land
    # AFTER the Phase 6.2 calendar table and Phase 6.3 SEC filing footer.
    # Lives in the Researcher (not PM Pre-flight) because peers.json is
    # only written here — PM Pre-flight runs before this node and would
    # find peers_path.exists() == False. See docs/superpowers/specs/
    # 2026-05-05-deterministic-peer-ratios-design.md.
    # Phase 6.4 invariant: the deterministic peer-ratios block must land in
    # pm_brief.md every run, or downstream LLM agents fill the void with
    # fabricated peer numbers (RCL 2026-05-06: peers.json was {}, the block
    # never appended, decision.md cited NCLH/CCL/VIK ratios that came from
    # nowhere). Three guarded paths replace the prior silent-skip:
    #
    #   1. pm_brief.md missing → PM Pre-flight failed; raise.
    #   2. peers_data empty   → upstream peer-discovery returned nothing;
    #                           raise rather than ship a peer-less brief.
    #   3. all peers unavailable → write peer_ratios.json with the
    #                              `_unavailable` list AND append an explicit
    #                              "do not fabricate" warning block so the
    #                              LLM sees the gap and refuses to invent
    #                              numbers.
    pm_brief_path = raw / "pm_brief.md"
    if not pm_brief_path.exists():
        raise RuntimeError(
            "Phase 6.4 invariant: pm_brief.md does not exist before the "
            "Researcher's peer-ratios block runs. PM Pre-flight likely "
            "failed silently; investigate before re-running."
        )
    if not peers_data:
        raise RuntimeError(
            "Phase 6.4 invariant: peers_data is empty (peers.json wrote `{}`). "
            "Upstream peer-discovery returned no peers; the LLM will fabricate "
            "peer ratios downstream if this run is allowed to proceed. Fix the "
            "peer-lookup path for this ticker rather than shipping without a "
            "peer-ratios block."
        )

    from tradingagents.agents.utils.peer_ratios import (
        compute_peer_ratios,
        format_peer_ratios_block,
    )
    ratios = compute_peer_ratios(peers_data, date)
    (raw / "peer_ratios.json").write_text(
        json.dumps(ratios, indent=2, default=str),
        encoding="utf-8",
    )

    # If every peer is unavailable, the standard format renders a table of
    # `(unavailable)` rows — technically correct but the same trailing
    # "Use these values verbatim" footer can read as "use these unavailable
    # cells", which the LLM may interpret as license to substitute memory.
    # Override with an explicit "do not fabricate" warning instead.
    unavailable = set(ratios.get("_unavailable", []))
    peer_keys = [k for k in ratios.keys() if k not in ("trade_date", "_unavailable")]
    if peer_keys and unavailable == set(peer_keys):
        peer_block = (
            f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {date})\n\n"
            "**All peers unavailable** — yfinance returned degenerate or missing "
            "data (revenue/operating-income/capex rows) for every peer in "
            "raw/peers.json. **Do not cite peer ratios in this report.** If a "
            "peer comparison is essential to the thesis, flag it as `(peer data "
            "unavailable)` and do not invent figures from memory.\n"
        )
    else:
        peer_block = format_peer_ratios_block(ratios)
        if not peer_block:
            # Defensive: peers_data was non-empty but format returned ""
            # (all entries were non-dict, etc.). Surface the gap loudly.
            peer_block = (
                f"\n\n## Peer ratios (computed from raw/peers.json, trade_date {date})\n\n"
                "**Peer-ratios table could not be rendered** from raw/peers.json. "
                "**Do not cite peer ratios in this report.**\n"
            )

    with open(pm_brief_path, "a", encoding="utf-8") as f:
        f.write(peer_block)
