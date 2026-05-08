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

from tradingagents.agents.utils.structured import invoke_with_empty_retry

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
    # `- TICKER: ...` / `- **TICKER**: ...` / `- *TICKER*: ...` and now also
    # `- **TICKER** (Company Name): ...` (the format the LLM emitted for
    # TSCO 2026-05-06, which the prior regex's strict `**:` requirement
    # silently dropped — peers.json wrote `{}` and the run crashed at the
    # Phase 6.4 invariant gate). The optional `(...)` group consumes a
    # parenthesized company-name expansion before the colon.
    r"^-\s+(?:\*{1,2})?([A-Z]{1,5})(?:\*{1,2})?(?:\s+\([^)\n]+\))?\s*:\s",
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
            # Phase 6.8: separate structural-N/A (passive ETF / index — no
            # earnings concept) from transient-unavailable (yfinance returned
            # nothing for what should be a reporting equity). The former
            # gets a clear "passive instrument" label; the latter keeps the
            # original "yfinance unavailable" so the LLM doesn't fabricate.
            if val.get("structural"):
                instrument = (val.get("instrument_type") or "passive").lower()
                rows.append(
                    f"| {key} | (N/A — {instrument}; no earnings reporting) | "
                    f"n/a | (N/A — {instrument}) |"
                )
            else:
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


def _fetch_canonical_identity(ticker: str) -> str:
    """Phase 7.6: fetch yfinance authoritative identity for the ticker
    and format as a system-prompt prefix.

    Closes the same-symbol-different-company failure mode (e.g., `ASX`
    resolves in yfinance to ASE Technology Holding Co — Taiwan
    semiconductor packaging — but the LLM's prior knowledge maps `ASX`
    to the Australian Securities Exchange operator, generating a brief
    about the wrong company entirely). By injecting yfinance's
    canonical longName / country / sector / industry as IMMUTABLE
    ground truth, we override the LLM's training-data pattern match.

    Returns "" on any yfinance error — graceful degradation; the LLM
    falls back to its existing identification logic.
    """
    try:
        import yfinance as yf
        info = getattr(yf.Ticker(ticker), "info", None) or {}
    except Exception:  # noqa: BLE001 — yfinance is best-effort
        return ""

    long_name = info.get("longName") or info.get("shortName")
    if not long_name:
        return ""

    country = info.get("country") or "?"
    sector = info.get("sector") or "?"
    industry = info.get("industry") or "?"
    quote_type = (info.get("quoteType") or "?").upper()
    market_cap = info.get("marketCap")
    cap_str = ""
    if isinstance(market_cap, (int, float)) and market_cap > 0:
        if market_cap >= 200e9:
            cap_str = "Mega-cap"
        elif market_cap >= 10e9:
            cap_str = "Large-cap"
        elif market_cap >= 2e9:
            cap_str = "Mid-cap"
        else:
            cap_str = "Small-cap"

    return (
        f"# AUTHORITATIVE TICKER IDENTITY (yfinance, fetched at run time)\n\n"
        f"For ticker `{ticker}`, yfinance returns the following canonical "
        f"identity. Your brief MUST describe THIS company, even if your "
        f"prior knowledge of `{ticker}` matches a different same-symbol "
        f"entity in another market.\n\n"
        f"- **Long name:** {long_name}\n"
        f"- **Country:** {country}\n"
        f"- **Sector:** {sector}\n"
        f"- **Industry:** {industry}\n"
        f"- **Instrument type:** {quote_type}\n"
        f"- **Market-cap classification:** {cap_str or '(unavailable)'}\n\n"
        f"Concrete example of the failure mode this section prevents: the "
        f"ticker `ASX` ambiguates between ASE Technology Holding Co (Taiwan, "
        f"semiconductor packaging) and the Australian Securities Exchange "
        f"operator. yfinance maps the symbol to the former; a "
        f"prior LLM run wrote a brief describing the latter, then suggested "
        f"foreign-listed exchange peers (S68.SI / 388.HK / DB1.DE) that "
        f"yfinance can't fetch under US-style ticker calls — peers.json "
        f"wrote `{{}}` and the run aborted at the Phase 6.4 fail-loud gate. "
        f"Use the yfinance identity above as ground truth; pick US-tradable "
        f"peers compatible with yfinance's peer-discovery (e.g., AMKR for "
        f"ASE Technology, not foreign exchange operators).\n\n"
        f"---\n\n"
    )


def create_pm_preflight_node(llm):
    """Factory: returns the PM Pre-flight LangGraph node function."""

    def pm_preflight_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = Path(state["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Phase 7.6: prefix the system prompt with yfinance authoritative
        # identity. Empty on yfinance error (graceful degradation).
        identity_prefix = _fetch_canonical_identity(ticker)

        system_prompt = _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
        if identity_prefix:
            system_prompt = identity_prefix + system_prompt

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Produce the PM Pre-flight brief for {ticker} on {date}."),
        ]
        result, brief = invoke_with_empty_retry(llm, messages, "PM Pre-flight")

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
            "messages": [result],
            "pm_brief": brief,
            "peers": peers,
        }

    return pm_preflight_node
