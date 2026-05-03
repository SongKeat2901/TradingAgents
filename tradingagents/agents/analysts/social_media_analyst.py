"""Social/sentiment analyst — refactored to read raw/ + mandate numerical sentiment."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior social/sentiment analyst writing the sentiment section of \
an equity research report on $TICKER for trade date $DATE.

Required sections (verbatim headers):

## Sentiment indicators (with numbers)

| Source | Metric | Value | Trend (7d) | Interpretation |
|---|---|---|---|---|
| <source> | <metric> | <number or %> | <±X> | <crowd-psychology read> |

No vague claims like "sentiment is bullish." Every row must have a specific \
number or percentage, even if estimated from a sample size you state.

## Conviction asymmetry

Are bulls or bears more convinced? Cite specific signals — "X% of mentions \
include price targets above current spot vs Y% targeting downside" or \
similar. Quantify.

## Crowd-vs-data divergence

Where does sentiment disagree with the fundamentals analyst's data? \
Disagreement is a signal. Cite specific conflicts.

## Risk to my read

What sentiment signal would invalidate this analysis?"""


def create_social_media_analyst(llm):
    def social_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "social.json"],
        )

        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
                + "\n" + get_language_instruction()
            ),
            HumanMessage(
                content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                f"Write the sentiment analyst's report."
            ),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

        return {
            "messages": [result] if raw_content is not None else [],
            "sentiment_report": report,
        }

    return social_analyst_node
