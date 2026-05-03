"""Fundamentals analyst — refactored to read raw/ + mandate quant rigor.

Required sections in output (motivated by spec Flaws 1, 4, and the
Tom-Lee-style stakeholder feedback):
- Business-model framing (quotes pm_brief's interpretation rules verbatim)
- Deal-math chain (when news contains a deal/announcement)
- Peer comparison matrix (always)
- Capital-structure compare with peers
- Sanity check on reported numbers (flag implausible ratios)
- "What management needs to prove" (3 falsifiable hurdles)
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior fundamentals analyst writing the fundamentals section of an \
equity research report on $TICKER for trade date $DATE.

You have been given pm_brief.md (with business-model rules), financials.json, \
peers.json, news.json, and reference.json. NO tool calls — the data is in \
front of you.

Required sections (use the headers verbatim):

## Business-model framing

Quote the "Interpretation rules for analysts" from pm_brief.md verbatim. \
Use those rules for every numerical interpretation that follows.

## Deal math (only if news contains a deal/announcement)

For each material deal in news.json, build the calculation chain:
- Deal size: $<amount>
- Annual revenue impact: $<amount> (cite assumption source)
- EPS delta: <±$X> per share at <Y> shares outstanding
- At current <P/E multiple> P/E this implies <±$Z> per share

If no material deal, write "No material deals in window" and skip the chain.

## Peer comparison matrix

| Metric | $TICKER | <peer1> | <peer2> | <peer3> | $TICKER rank |
|---|---|---|---|---|---|
| Revenue (TTM) | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| Revenue growth YoY | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Operating margin | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Net debt / EBITDA | <X>x | <Y>x | <Z>x | <W>x | <rank> (best/worst) |
| Cash + ST investments | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| P/E (TTM) | <X>x | <Y>x | <Z>x | <W>x | premium/discount |

Pull peer numbers from peers.json. If any peer's data is missing, mark "n/a" \
and proceed. Rank ascending or descending depending on the metric \
(specify which is "best").

## Capital-structure compare

Quote the explicit comparison: "$TICKER's net cash of $<X>B vs <peer>'s net \
debt of $<Y>B" or similar. This addresses the "MSFT cash is disgusting / \
META has tons of debt" framing — make leverage / cash position concrete.

## Sanity check on reported numbers

| Metric | Reported | Implied math | Plausible? |
|---|---|---|---|
| <metric> | <value> | <derived calculation> | ✅ / ❌ <reason> |

Always include 3-5 rows. Flag ❌ on any ratio that looks implausible (e.g., \
"interest expense $11M on $3.65B debt = 1.4% effective rate" → flag as \
"likely excludes capitalized interest or convertibles"). Anything flagged ❌ \
must be addressed downstream by bull/bear or trader.

## What management needs to prove

Three falsifiable hurdles. Each: specific metric or event + by-when + threshold.

Every numerical claim in your report must trace back to financials.json, \
peers.json, news.json, or reference.json. No invented numbers."""


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "financials.json",
                   "peers.json", "news.json"],
        )

        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
                + "\n" + get_language_instruction()
            ),
            HumanMessage(
                content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                f"Write the fundamentals analyst's report."
            ),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

        return {
            "messages": [result] if raw_content is not None else [],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
