"""News analyst — refactored to read raw/ + mandate catalyst magnitudes."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt
from tradingagents.agents.utils.structured import invoke_with_empty_retry


_SYSTEM = """\
You are a senior news / macro analyst writing the news section of an equity \
research report on $TICKER for trade date $DATE.

You have been given news.json (ticker-specific + global), reference.json, \
and pm_brief.md. NO tool calls — the data is in front of you.

Required sections (verbatim headers):

## Material catalysts (with magnitude estimates)

For every material catalyst in news.json, build a magnitude chain:

| Catalyst | Direction | Mechanism | Magnitude estimate | Confidence |
|---|---|---|---|---|
| <event> | Bull/Bear | <how it propagates to stock> | <±$X target price impact OR ±Y% multiple shift> | High/Med/Low |

Examples of valid mechanisms: "Fed cut → +1% to S&P fair value via duration \
math → +$5 SPY upside"; "Q4 earnings beat → +$0.20 EPS → at 22x P/E, +$4.4 \
per share". Vague claims like "this is positive for the stock" are not \
acceptable.

## Cross-references with peers

If any peer ticker (from pm_brief.md) is mentioned in the news context, cite \
it — these are the read-throughs ("AAPL's iPhone disclosure on April 28 \
implies $TICKER's similar segment will print Y% growth").

## Macro / global context

What broader trend frames this run's catalyst set? E.g., "rates expected to \
hold; AI capex cycle; consumer sentiment near 50-year low."

Every claim must cite a specific item from news.json or global news. No \
narrative without a source."""


def create_news_analyst(llm):
    def news_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "news.json"],
        )

        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
                + "\n" + get_language_instruction()
            ),
            HumanMessage(
                content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                f"Write the news analyst's report."
            ),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "News Analyst")

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
