"""Market analyst — refactored to read raw/ instead of bind_tools().

Reads the TA Agent's technicals.md as canonical level analysis, then writes
its own commentary that may agree, extend, or challenge specific levels.
The disagreement surfaces in the bull/bear debate downstream.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt
from tradingagents.agents.utils.structured import extract_llm_content


_SYSTEM = """\
You are a senior market analyst writing the technical commentary section of \
an equity research report on $TICKER for trade date $DATE.

The TA Agent has already produced raw/technicals.md with the canonical level \
analysis. Your job is NOT to redo it — your job is to:

1. Quote the 2-3 levels you consider MOST important and explain WHY.
2. Either agree with the TA Agent's setup classification OR challenge it \
   with specific evidence ("TA Agent classified consolidation; I disagree \
   because volume on $X day was Y% above average and price broke...").
3. Map the technical setup onto a TRADING PLAYBOOK: \
   "if SPY breaks $500 on volume, expected next stop $487 (200-DMA); \
    if it holds, base for $510-520 retest."

Required sections in your output:

## Most important levels (2-3)
For each: price, type, your reason it matters more than the others.

## Setup assessment
Either "I agree with the TA Agent's <classification>" + supporting evidence,
or "I disagree with the TA Agent's <classification>; correct read is <X>" \
+ specific contradicting evidence.

## Trading playbook
- If <condition>: <expected next move> with <triggers/levels>
- If <opposite condition>: <expected next move> with <triggers/levels>

## Risk to my own read
What would have to be true for me to be wrong? Be specific."""


def create_market_analyst(llm):
    def market_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "technicals.md", "prices.json"],
        )

        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
                + "\n" + get_language_instruction()
            ),
            HumanMessage(
                content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                f"Write the market analyst's commentary."
            ),
        ]
        result = llm.invoke(messages)
        report = extract_llm_content(result, "Market Analyst")

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
