"""PM Pre-flight node — opener role for the PM persona.

Sets the research mandate before analysts run:
- Validates ticker (trading day, sector, market cap class)
- Classifies the actual business model (overrides yfinance sector if wrong;
  motivated by spec Flaw 1 — MARA was tagged Financial Services but is a
  Bitcoin miner)
- Identifies 2-4 peers for comparison
- Reads memory log for past lessons on this ticker or pattern
- Specifies questions this run must answer

Output: `<raw_dir>/pm_brief.md` (Markdown). Side-effect on state:
populates `pm_brief` (full text) and `peers` (parsed list).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

_logger = logging.getLogger(__name__)


_SYSTEM = """\
You are the Portfolio Manager performing pre-flight due diligence on \
$TICKER for trade date $DATE.

Produce a Markdown brief with EXACTLY these sections (use the headers below verbatim):

# PM Pre-flight Brief: $TICKER $DATE

## Ticker validation
- Trading day: <day-of-week + date>
- Sector (yfinance): <yfinance sector>
- Market cap: <Mega-cap / Large-cap / Mid-cap / Small-cap>

## Business model classification
- yfinance sector: <yfinance sector>
- Actual business model: **<plain-English description>**

Interpretation rules for analysts:
- <bullet rule 1>
- <bullet rule 2>
- <bullet rule 3>

The "Actual business model" overrides yfinance when yfinance's sector is \
structurally misleading. Examples of mismatch: a Bitcoin miner tagged as \
Financial Services, a SPAC tagged as Shell Companies, a biotech tagged as \
Pharmaceuticals when it has no revenue. Call out the mismatch explicitly.

## Peer set
- <PEER_TICKER>: <one-line rationale>
- <PEER_TICKER>: <one-line rationale>
- <PEER_TICKER>: <one-line rationale>

Pick 2-4 peers based on actual business model, not yfinance sector. For \
broad-market ETFs (SPY, QQQ, etc.), write "(none — index ETF; ...)" \
instead of a peer list.

## Past-lesson summary
- <Any prior decision on this ticker, or similar pattern, from the memory log>

## What this run must answer
1. <specific question>
2. <specific question>
3. <specific question>

Be concrete and falsifiable. No vague questions like "what's the outlook?"."""


_PEER_LINE = re.compile(r"^- ([A-Z]{1,5}): ", re.MULTILINE)


def _extract_peers(brief: str) -> list[str]:
    """Pull peer tickers from the Peer set section."""
    # Find the "## Peer set" section and the next "## " section
    match = re.search(r"## Peer set\s*\n(.*?)(?=^## |\Z)", brief, re.DOTALL | re.MULTILINE)
    if not match:
        return []
    section = match.group(1)
    return _PEER_LINE.findall(section)


def create_pm_preflight_node(llm):
    """Factory: returns the PM Pre-flight LangGraph node function."""

    def pm_preflight_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = Path(state["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        messages = [
            SystemMessage(content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)),
            HumanMessage(content=f"Produce the PM Pre-flight brief for {ticker} on {date}."),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        brief = raw_content if raw_content else str(result)

        (raw_dir / "pm_brief.md").write_text(brief, encoding="utf-8")
        peers = _extract_peers(brief)
        if not peers and "## Peer set" in brief:
            _logger.warning(
                "PM Pre-flight: Peer set section present but no peers extracted "
                "for %s — check LLM format drift (expected '- TICKER: rationale')",
                ticker,
            )

        return {
            "messages": [result] if raw_content is not None else [],
            "pm_brief": brief,
            "peers": peers,
        }

    return pm_preflight_node
