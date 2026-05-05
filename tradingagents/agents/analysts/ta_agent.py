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


def _load_classification_block(raw_dir: str) -> str:
    """Format raw/classification.json as a 'DETERMINISTIC CLASSIFICATION' block
    for injection into the TA agent SystemMessages.

    Returns an empty string if the file is missing or contains INDETERMINATE,
    in which case the agent falls back to legacy LLM-judged classification.
    """
    import json as _json
    cls_path = Path(raw_dir) / "classification.json"
    if not cls_path.exists():
        return ""
    try:
        cls = _json.loads(cls_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError:
        return ""
    if cls.get("setup_class") in (None, "INDETERMINATE"):
        return ""
    return (
        "\n\n# DETERMINISTIC CLASSIFICATION (use this verbatim — do NOT override)\n\n"
        f"Setup classification: {cls['setup_class']}\n"
        f"Asymmetry:\n"
        f"  - Upside to ${cls.get('upside_target')} ({cls.get('upside_pct'):+.2f}%)\n"
        f"  - Downside to ${cls.get('downside_target')} ({cls.get('downside_pct'):+.2f}%)\n"
        f"  - Reward/risk ratio: {cls.get('reward_risk_ratio')}:1\n"
        f"Rationale (deterministic, audit trail): {cls.get('rationale', '')}\n\n"
        "You MUST use exactly this Setup classification in your "
        '"## Setup classification" section and these exact upside/downside '
        'numbers in your "## Asymmetry" section. You may add prose, '
        "qualifying language, and additional context — but the classification "
        "name and the asymmetry numbers are fixed.\n\n"
        "If — and ONLY if — you disagree with the classification (e.g., you "
        "see a chart pattern the rules missed), document the disagreement "
        'under a "## Notes for next pass" subsection limited to ≤3 bullets, '
        "≤30 words each. Otherwise, omit that subsection entirely. Do not "
        "use it as a place to monologue about the rule engine's design or "
        "to write 'Action for downstream agents' — the rules are load-bearing "
        "for cross-run consistency; your prose is for nuance, not "
        "pipeline meta-commentary.\n\n"
        "Do not append any 'Report prepared', 'Data sources', or 'Framework' "
        "metadata footer at the bottom of your output — that information "
        "lives in git history, not in trader-facing reports.\n"
    )


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


_SYSTEM_V2 = """\
You are the TA Agent doing a second-pass review for $TICKER on $DATE.

Your v1 setup is in raw/technicals.md. The four analysts have now produced \
their reports (Market, Fundamentals, News, Sentiment). Your job is to read \
their reports plus your own v1 read, then emit a refined technical setup that \
addresses any analyst pushback that materially affects the technical view.

Produce a Markdown report with EXACTLY these sections (use the headers verbatim):

## Revisions from v1

For each material revision, name the analyst whose pushback caused the change \
(verbatim quote ≤30 words) and state what changed and why.

If no revision is warranted, this section reads exactly:
"No revisions — analyst reports did not surface evidence to revise v1's classification."

## Major historical levels

[Same table format as v1: Level | Price | Type | Why crowds trade here]

## Volume profile zones

- Heaviest accumulation: $<low>-$<high>
- Volume gap: $<low>-$<high>

## Current technical state

Narrative on RSI, MACD, moving-average stack, divergences.

## Setup classification

One of: breakout / breakdown / consolidation / distribution / accumulation.

## Asymmetry

- Upside to next major resistance: $<price> (<+X>%)
- Downside to next major support: $<price> (<-Y>%)
- Reward/risk: <ratio>:1

The v2 view is what every downstream agent (bull/bear/RM/trader/risk team/PM) \
will reason over. Cite specific numbers. Address fundamental concerns when they \
materially shift the technical read."""


def create_ta_agent_v2_node(llm):
    """Factory: returns the TA Agent v2 LangGraph node function.

    Reads v1 + four analyst reports + raw/reference.json + raw/prices.json.
    Writes raw/technicals_v2.md and overwrites state.technicals_report.
    """

    def ta_agent_v2_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]

        v1_context = format_for_prompt(
            raw_dir,
            files=["technicals.md", "reference.json", "prices.json", "sec_filing.md"],
        )
        analyst_block = (
            f"\n## Market Analyst Report\n{state.get('market_report', '(missing)')}\n\n"
            f"## Fundamentals Analyst Report\n{state.get('fundamentals_report', '(missing)')}\n\n"
            f"## News Analyst Report\n{state.get('news_report', '(missing)')}\n\n"
            f"## Sentiment Analyst Report\n{state.get('sentiment_report', '(missing)')}\n"
        )

        classification_block = _load_classification_block(raw_dir)
        messages = [
            SystemMessage(
                content=_SYSTEM_V2.replace("$TICKER", ticker).replace("$DATE", date)
                + classification_block
            ),
            HumanMessage(content=(
                f"Produce the v2 technicals report for {ticker} on {date}. "
                f"Below are the v1 setup, the four analyst reports, and the "
                f"reference snapshot. Refine and emit v2.\n\n"
                f"{v1_context}\n{analyst_block}"
            )),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

        (Path(raw_dir) / "technicals_v2.md").write_text(report, encoding="utf-8")

        return {
            "messages": [result] if raw_content is not None else [],
            "technicals_report": report,
        }

    return ta_agent_v2_node


def create_ta_agent_node(llm):
    """Factory: returns the TA Agent LangGraph node function."""

    def ta_agent_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        raw_dir = state["raw_dir"]

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "prices.json"],
        )

        classification_block = _load_classification_block(raw_dir)
        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker) + classification_block
            ),
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
