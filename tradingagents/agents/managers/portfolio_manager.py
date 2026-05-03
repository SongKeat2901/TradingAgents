"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

_MANDATED_SECTIONS = """

# Mandatory sections in your decision.md (Quant Research Rebuild — 2026-05-03)

Your decision.md MUST start with these two sections in this order:

## Inputs to this decision

- **Reference price:** $<reference_price> (<reference_price_source>)
- **Peers compared:** <ticker list with one-line rationale per peer>
- **Past decisions referenced:** <ticker> <date> (<rating>; outcome <±X>%, alpha <±Y>%) \
  — invoked to argue <…>
- **Memory-log lessons applied:** <bullets with source line refs>
- **Catalysts in window:** <upcoming events with dates>
- **Data freshness:** <financials period>, <news through date>

## 12-Month Scenario Analysis

| Scenario | Probability | 12-Mo Price Target | Return | Key drivers |
|---|---:|---:|---:|---|
| Bull | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Base | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Bear | <pct>% | $<price> | <±pct>% | <named events / metrics> |

**Expected Value:** <calculation> = $<EV> (<±pct>% from spot $<spot>)
**Rating implication:** <BUY/HOLD/SELL> (<one-line bridge from EV to rating>)

After these two sections, continue with the existing memo format (synthesis, \
trading plan, what you're rejecting, etc.).

# Hard rules for the scenario table

- Probabilities must sum to exactly 100%.
- All three price targets must be specific dollar values, not ranges.
- Each scenario lists at least one named, falsifiable catalyst (not narrative).
- Rating must logically derive from EV.

# Hard rules for the Inputs section

- Reference price must equal the value in raw/reference.json.
- Past-decision citations must include the actual rating and outcome.
- Stakeholder reading the decision.md cold (without analyst reports) must \
be able to understand the framing from the Inputs section alone.
"""


_QC_CHECKLIST = """

# Self-correction QC checklist (Quant Research Rebuild — 2026-05-03)

Before emitting your final decision.md, apply this 13-item checklist to your \
draft. If ANY item fails, revise the draft in place, then re-apply the \
checklist. Do not emit until every item passes.

1. Probabilities sum to exactly 100%.
2. All three price targets are specific dollar values (not ranges).
3. Each scenario lists at least one named, falsifiable catalyst.
4. Rating logically derives from EV — e.g., EV materially below spot → \
SELL or UNDERWEIGHT, not HOLD.
5. Execution triggers are falsifiable (named price / level / date).
6. (Flaw 8) Re-entry / upgrade triggers must be reachable in at least one \
scenario in the table. Example: if Bull peaks at $14 but you state \
"re-enter below $18," that's inconsistent — either revise the trigger or \
revise the scenario.
7. (Flaw 2) Every bare "<ticker> at <trade_date>" price citation matches \
reference_price ± $0.01. Other prices (article quotes, intraday) must \
carry an explicit time/source qualifier.
8. (Flaw 3) Every cited analyst position has a verbatim ≤ 30-word quote \
attributed by section. Statements like "Neutral's math, applied honestly, \
supports Sell" require a direct quote. If the cited claim is not in the \
source, the synthesis is invalid — re-synthesize.
9. (Flaw 5) Cross-section numerical consistency. Compare the same numerical \
claim (cash runway, target price, percentage move, etc.) across analyst \
reports + debate transcripts. Any claim that appears with different values \
in different sections must be reconciled in decision.md under a \
"Reconciliation" subsection.
10. (Flaw 4) Sanity-check flags from the fundamentals analyst are addressed \
in either the bull/bear debate or the trader's plan. Flagged items cannot \
be silently ignored.
11. Inputs section is present and complete in decision.md.
12. Peer comparisons cite specific numbers, not vague comparisons.
13. All claimed numbers trace back to raw/*.json data.

If revision after one self-correction loop still fails, see the push-back \
retry rules in the next section."""


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}""" + _MANDATED_SECTIONS + _QC_CHECKLIST

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
