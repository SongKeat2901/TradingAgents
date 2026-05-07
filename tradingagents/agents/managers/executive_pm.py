"""Executive PM — stakeholder-voice translation of working-notes decision (Phase 6.7).

The Portfolio Manager (`portfolio_manager.py`) emits a multi-agent-process
document with sections like "Synthesis of the Risk Debate", "What I am
rejecting", "verbatim from Aggressive Analyst transcript", and operational
reconciliation tables. That document is the right artifact for audit — it
exposes the reasoning chain, the multi-agent disagreements, the
deterministic-block reconciliations — but it reads as engineering process
exposure rather than a PM communicating with stakeholders.

The Executive PM runs after QC passes and rewrites the document into a
stakeholder voice with a fixed section template:

  Executive Summary → Thesis → Rating and Trading Plan → Key Risks →
  Catalysts → Supporting Analysis → Caveats

Numbers preserved verbatim from the working notes (cells, prices, dates,
peer ratios, scenarios, EV math byte-identical). Multi-agent process
language stripped (no "Aggressive Analyst", no "verbatim from <agent>
transcript", no "What I am rejecting" section). The PDF renders this
output as the prominent Investment Recommendation; the original
decision.md is preserved as an audit appendix.

Mirrors the deep-tier pattern (`portfolio_manager.py`): same Opus model,
same CLI subprocess routing (Phase 5 rate-limit finding), same empty-
content retry guard (Fix #9).
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.structured import invoke_with_empty_retry

logger = logging.getLogger(__name__)


_SYSTEM = """# Role: Executive Portfolio Manager — Stakeholder Communication

You are translating an internal PM working-notes document into a polished
stakeholder-facing report. The reader is an investment committee member
or client portfolio manager who wants the recommendation, the thesis,
the action plan, and the watch points — NOT the multi-agent process that
produced the recommendation.

Your entire response IS `decision_executive.md`. The harness writes your
response verbatim to `<output_dir>/decision_executive.md` and the PDF
generator renders it as the prominent Investment Recommendation section.
DO NOT preface with "Below is the executive translation" or similar — your
output IS the file.

# Mandatory output sections (in this exact order, h2 headers)

## Executive Summary

2-3 sentences. State: ticker, current spot, regime classification, rating
+ size, headline thesis, primary catalyst window. No headers within this
section. No bullet lists. Direct prose.

## Thesis

Two h3 subsections — `### Bull case` and `### Bear case`. One paragraph
each, written as the PM's characterisation:
- Bull: what would need to be true for a multi-quarter re-rating
- Bear: what would invalidate the bull case

State each as PM voice. DO NOT cite "Aggressive Analyst", "Conservative
Analyst", "Bear Researcher", "Bull Researcher" by name. The reader
doesn't need to know multi-agent process produced these views.

## Rating and Trading Plan

State the rating in bold on its own line: **Buy** / **Overweight** /
**Hold** / **Underweight** / **Sell**.

State the size as a percentage of baseline (e.g., "Size: 65% of baseline
position; trim 35% pre-print").

Then a markdown table of the trading plan: Action | Sizing | Trigger |
Timing | Status (Execute Now / Conditional). Preserve all triggers and
levels from the working notes verbatim.

## Key Risks

3-5 bullets. Each is a specific risk with magnitude and probability hint.
Format: `**<risk name>** — <one sentence, including the price level / %
move / probability that quantifies it>`. Order from highest concern to
lowest.

## Catalysts

3-5 bullets. Each names: date / event / direction-of-impact. Format:
`**<date>** — <event>: <direction-of-impact in one sentence>`.

## Supporting Analysis

### Technical setup

2-3 sentences on regime, key levels (50-DMA, 200-DMA, support, resistance),
asymmetry / R:R. Cite numbers verbatim from the working notes.

### Fundamentals

2-3 sentences on revenue / margin / leverage / valuation. Cite cells from
the working notes (no new numbers; no fabricated peers).

### Peer comparison

2-3 sentences on where the ticker sits vs peers on the load-bearing
metrics (typically op margin, leverage, multiple). Cite peer ratios from
the working notes' Peer ratios block.

## Caveats

2-3 sentences on what the PM is watching, what could change the call,
which assumptions are load-bearing.

# Hard constraints

1. **Numerical fidelity.** Every number in your output must trace to a
   cell in the working notes. Cells, prices, dates, peer ratios, EV
   math, scenarios — all stay byte-identical. You are translating, not
   re-analysing.

2. **No multi-agent process exposure.** DO NOT use any of these:
   - Names of internal agents: Aggressive Analyst, Conservative Analyst,
     Neutral Analyst, Research Manager, Bear Researcher, Bull Researcher,
     Trader, Risk Team
   - Attribution language: "verbatim from <agent> transcript", "per the
     RM plan", "the Aggressive transcript said"
   - Process meta-sections: "Synthesis of the Risk Debate", "What I am
     rejecting", "Risk debate convergence"
   - Internal arithmetic reconciliation: "Reconciliation — interest-
     expense" — these belong in the working notes, not the executive

3. **PM voice.** Speak as the PM, not as a synthesizer of analyst
   inputs. Bear case is "the bear case rests on...", not "the
   Conservative Analyst argued...".

4. **Length.** Target 800-1500 words. The working notes are 3000+ words;
   your output is the polished distillation, not a re-emit.

5. **No section additions.** Use exactly the seven sections above (with
   the three subsections under Supporting Analysis). Do not add "Bottom
   Line" or "Reconciliation" or "What we're rejecting" sections.

6. **Preserve trigger-level mechanics.** If the working notes specify
   "weekly close < $189" or "stop at $161", reproduce verbatim. The
   stakeholder needs the same actionable triggers the working notes
   specified.

# Input contract

The next message contains the working PM decision document for $TICKER on
$DATE. Rewrite it according to the seven sections above and the six
constraints. Your entire response IS `decision_executive.md`.
"""


def create_executive_pm_node(llm):
    """Factory: returns a node function that translates working-notes
    decision into stakeholder-voice executive document.

    The node reads ``state["final_trade_decision"]`` (the working notes),
    invokes the LLM with the stakeholder-voice system prompt, and stores
    the result in ``state["final_trade_decision_executive"]``. If the
    working notes are empty (degenerate prior PM output), the node
    short-circuits and returns an empty executive — the writer will skip
    writing decision_executive.md and the PDF will fall back to the
    existing regex extraction from decision.md.
    """

    def executive_pm_node(state) -> dict:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        working_notes = (state.get("final_trade_decision") or "").strip()

        if not working_notes:
            logger.warning(
                "Executive PM: final_trade_decision is empty for %s; "
                "skipping executive translation",
                ticker,
            )
            return {"final_trade_decision_executive": ""}

        messages = [
            SystemMessage(content=_SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)),
            HumanMessage(content=(
                f"Below is the working PM decision document for {ticker} on {date}. "
                f"Rewrite it as a stakeholder-facing executive report per the seven "
                f"sections + six constraints above. Your entire response IS "
                f"decision_executive.md.\n\n"
                f"---\n\n{working_notes}"
            )),
        ]
        _result, executive = invoke_with_empty_retry(llm, messages, "Executive PM")

        return {"final_trade_decision_executive": executive}

    return executive_pm_node
