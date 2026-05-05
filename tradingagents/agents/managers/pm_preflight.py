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

## Fiscal calendar context
- Reporting fiscal year: <e.g., July–June for MSFT, January–December for most US firms>
- Most recently reported quarter: <e.g., FY26 Q2 = calendar Q4 2025, reported Jan 2026>
- Next earnings print expected: <approximate month + the EXACT fiscal quarter label>

This section is non-negotiable. Downstream agents (analysts, debaters, PM Final) \
must cite the next earnings catalyst with the correct fiscal-quarter label. \
Common error to avoid: calling Microsoft's late-July print "Q3 FY26" — Microsoft's \
FY runs July–June, so the late-July 2026 print is **Q4 FY26** (covering Apr–Jun \
2026), not Q3. Verify the fiscal calendar against the actual ticker's reporting cadence.

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

Be concrete and falsifiable. No vague questions like "what's the outlook?".

# Temporal anchor

Treat the trade date as "today". Events dated before it have already
occurred — never write them as "data to follow", "upcoming", or
"data to be reported". A "Reporting status" table will be programmatically
appended to your output listing the most-recent and next-expected earnings
dates for each ticker; those dates are authoritative and you do not need
to enumerate them yourself in the brief."""


_PEER_LINE = re.compile(
    r"^-\s+(?:\*{1,2})?([A-Z]{1,5})(?:\*{1,2})?\s*:\s",
    re.MULTILINE,
)


def _extract_peers(brief: str) -> list[str]:
    """Pull peer tickers from the Peer set section."""
    # Find the "## Peer set" section and the next "## " section
    match = re.search(r"## Peer set\s*\n(.*?)(?=^## |\Z)", brief, re.DOTALL | re.MULTILINE)
    if not match:
        return []
    section = match.group(1)
    return _PEER_LINE.findall(section)


def _format_calendar_block(raw_dir: str) -> str:
    """Format raw/calendar.json as a 'Reporting status' Markdown block for
    appending to pm_brief.md after the LLM call.

    Returns "" if calendar.json is missing or all tickers are unavailable —
    in which case downstream agents fall back to LLM judgment for temporal
    reasoning (same INDETERMINATE pattern as the classifier).
    """
    import json as _json
    cal_path = Path(raw_dir) / "calendar.json"
    if not cal_path.exists():
        return ""
    try:
        cal = _json.loads(cal_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError:
        return ""

    trade_date = cal.get("trade_date", "?")
    unavailable_set = set(cal.get("_unavailable", []))

    rows = []
    for key, val in cal.items():
        if key in ("trade_date", "_unavailable"):
            continue
        if key in unavailable_set or val.get("unavailable"):
            rows.append(
                f"| {key} | (yfinance unavailable) | unknown | (yfinance unavailable) |"
            )
            continue
        last = val.get("last_reported", "?")
        period = val.get("fiscal_period", "?")
        nxt = val.get("next_expected") or "(unknown)"
        rows.append(
            f"| {key} | {period} reported {last} | already happened | {nxt} |"
        )

    if not rows:
        return ""

    table = "\n".join(rows)
    return (
        f"\n\n## Reporting status (relative to trade_date {trade_date})\n\n"
        "| Ticker | Most recent earnings | Status | Next expected |\n"
        "|---|---|---|---|\n"
        f"{table}\n\n"
        "*Use these dates verbatim. Do not write \"data to follow\" or "
        "\"upcoming\" for rows marked \"already happened\" — they happened "
        "before the trade date. Treat them as rear-view information that "
        "should inform fundamental and sentiment reasoning. The \"next "
        "expected\" dates are the forward catalyst windows.*\n"
    )


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

        # Phase-6.2 temporal-anchor: compute the deterministic earnings
        # calendar with the peer list we just extracted, write calendar.json,
        # then append the Reporting status block AFTER the LLM-written content
        # so dates can never be paraphrased. PM Pre-flight runs before the
        # Researcher, so this is the earliest point in the graph where peers
        # are known. See docs/superpowers/specs/2026-05-05-deterministic-earnings-calendar-design.md.
        from tradingagents.agents.utils.calendar import compute_calendar
        import json as _json
        calendar = compute_calendar(date, [ticker] + peers)
        (raw_dir / "calendar.json").write_text(
            _json.dumps(calendar, indent=2, default=str), encoding="utf-8"
        )
        calendar_block = _format_calendar_block(state["raw_dir"])
        if calendar_block:
            with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
                f.write(calendar_block)
            brief = brief + calendar_block

        # Phase-6.3 filing-anchor: fetch the most recent 10-Q/10-K filed on
        # or before the trade date from SEC EDGAR. This catches the case
        # where the LLM hallucinates "filing pending" for a document that's
        # already public. Only fetched for the primary ticker (peers add
        # noise + rate-limit pressure on EDGAR).
        from tradingagents.agents.utils.sec_edgar import fetch_latest_filing, format_for_prompt as format_sec_filing
        try:
            filing = fetch_latest_filing(ticker, date)
        except Exception:
            filing = {"unavailable": True, "reason": "fetcher raised", "ticker": ticker}
        sec_filing_md = format_sec_filing(filing)
        if sec_filing_md:
            (raw_dir / "sec_filing.md").write_text(sec_filing_md, encoding="utf-8")
            # Append a one-line filing-status note to pm_brief so the
            # PM/analysts/debaters see it inline.
            footer = (
                f"\n## Recent SEC filing (relative to trade_date {date})\n\n"
                f"- **{filing['ticker']} {filing['form']} filed {filing['filing_date']}** — "
                f"contents already public on the trade date; full text in "
                f"raw/sec_filing.md. Treat as **known data**, never as "
                f"\"pending adjudication\" or \"awaiting filing\".\n"
            )
            with open(raw_dir / "pm_brief.md", "a", encoding="utf-8") as f:
                f.write(footer)
            brief = brief + footer

        # NOTE: Phase 6.4 deterministic peer-ratios injection lives in
        # researcher.py (which writes peers.json). PM Pre-flight runs
        # BEFORE the Researcher, so peers.json doesn't exist here yet.
        # See tradingagents/agents/researcher.py for the actual wiring
        # and docs/superpowers/specs/2026-05-05-deterministic-peer-ratios-design.md.

        return {
            "messages": [result] if raw_content is not None else [],
            "pm_brief": brief,
            "peers": peers,
        }

    return pm_preflight_node
