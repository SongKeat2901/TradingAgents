"""TA Agent — owns historical level identification with crowd psychology.

Reads `raw/prices.json` (5y OHLCV) + `raw/pm_brief.md` for context. Produces
`raw/technicals.md` with the mandated section structure. Other agents (Market
analyst, Bull/Bear) read this file and may agree, disagree, or extend.

Motivated by stakeholder feedback for "needle-see-blood" technical analysis:
identifying major historical levels (prior cycle highs, swing points, Fib
zones) AND explaining why crowds will trade them (stop clusters, retests,
round numbers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior technical analyst. Your job is to identify the price levels \
that matter for $TICKER and explain why crowds will trade at each one.

You have been given the 5-year price history, current technical indicators, \
and the PM pre-flight brief. Produce a Markdown report with EXACTLY these \
sections (use the headers verbatim):

## Major historical levels

| Level | Price | Type | Why crowds trade here |
|---|---|---|---|
| <name> | $<price> | Resistance/Support | <crowd-psychology rationale> |

Identify 3-7 levels. Examples of valid levels: prior cycle highs, all-time \
highs, swing pivots, 200-day SMA, 50-day SMA, Fibonacci 0.382/0.618 \
retracements, round numbers ($100, $500), volume-profile peaks.

## Volume profile zones

- Heaviest accumulation: $<low>-$<high> (<X>% of volume)
- Volume gap: $<low>-$<high> (<Y>% of volume — slip-through zone)

## Current technical state

Narrative on RSI, MACD, moving-average stack, divergences. Cite specific \
numbers from the indicators data.

## Setup classification

One of: breakout / breakdown / consolidation / distribution / accumulation. \
Justify in 1-2 sentences.

## Asymmetry

- Upside to next major resistance: $<price> (<+X>%)
- Downside to next major support: $<price> (<-Y>%)
- Reward/risk: <ratio>:1

Quantify both the magnitude AND the implied reward/risk ratio.

Cite specific levels from the price data — no vague "support somewhere below" \
language. Every claim must trace to a specific number from prices.json or \
indicators.json."""


def create_ta_agent_node(llm):
    """Factory: returns the TA Agent LangGraph node function."""

    def ta_agent_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        raw_dir = state["raw_dir"]

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "prices.json"],
        )

        messages = [
            SystemMessage(content=_SYSTEM.replace("$TICKER", ticker)),
            HumanMessage(content=f"Produce the technicals report for {ticker}.\n\n{context}"),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

        (Path(raw_dir) / "technicals.md").write_text(report, encoding="utf-8")

        return {
            "messages": [result] if raw_content is not None else [],
            "technicals_report": report,
        }

    return ta_agent_node
