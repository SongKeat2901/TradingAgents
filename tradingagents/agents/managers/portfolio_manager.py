"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import json as _json
import re as _re
from pathlib import Path

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)

_OUTPUT_CONTRACT = """

# Output contract (READ THIS FIRST — Quant Research Rebuild — 2026-05-03)

Your entire response IS decision.md. The harness writes your response \
verbatim to `<output_dir>/decision.md`, attaches it to a PDF report, and \
forwards it to downstream consumers (trader, PM memory log, Telegram).

Therefore:
- DO NOT write "Decision written to <path>" or "see attached file" — your \
output IS the file.
- DO NOT emit a summary of what you "would have" produced — emit the full \
mandated format below in its entirety.
- DO NOT omit any mandated section ("Inputs to this decision", "12-Month \
Scenario Analysis", "Reconciliation" if conflicts found, full synthesis, \
trading plan, caveats).
- The QC checklist below is applied to the document you emit. Pass-2 \
self-correction means revising sections IN PLACE inside your response, not \
emitting a meta-summary about which checks passed.

A response under ~3000 words is almost certainly missing mandated content; \
expand to full required depth before emitting."""


_MANDATED_SECTIONS = """

# Mandatory sections in decision.md (Quant Research Rebuild — 2026-05-03)

Your response MUST start with these two sections in this order:

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

Before emitting your response, apply this 13-item checklist to your draft. \
If ANY item fails, revise the draft IN PLACE (rewrite the affected sections \
inside your response), then re-apply the checklist. Do not emit until every \
item passes. Do not narrate which checks passed — the audit is internal; \
the reader sees only the final corrected document.

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


_RETRY_DIRECTIVE = """

# Push-back retry (Pass 3 — substantive disagreement)

If after self-correction your draft still has a substantive disagreement \
with upstream synthesis (Research Manager's investment plan or Risk team's \
debate), emit a structured retry signal as the LAST line of your output, \
in this exact format on its own line:

PM_RETRY_SIGNAL: {"target": "research_manager", "feedback": "<≤200-word specific instruction>"}

Or:

PM_RETRY_SIGNAL: {"target": "risk_team", "feedback": "<≤200-word specific instruction>"}

The feedback should be specific, e.g., "Bull case scenarios assume Azure \
≥32% but no analyst quantified what AI capex ROI would have to be for that \
to clear; have RM re-do scenarios with explicit ROI hurdle."

Hard cap: max 1 retry per run. If state.pm_retries == 1, you cannot push \
back further; you must accept the best-available output and note remaining \
concerns in decision.md under "Caveats from PM."

Do NOT emit PM_RETRY_SIGNAL on the first pass unless you genuinely cannot \
ship the report. The signal triggers re-running upstream nodes (~3-4 extra \
LLM calls); use sparingly."""


def _parse_retry_signal(text: str) -> dict | None:
    """Look for `PM_RETRY_SIGNAL: {...}` on the last lines of the report."""
    m = _re.search(
        r"^PM_RETRY_SIGNAL:\s*(\{.+?\})\s*$",
        text,
        flags=_re.MULTILINE,
    )
    if not m:
        return None
    try:
        sig = _json.loads(m.group(1))
        if sig.get("target") in ("research_manager", "risk_team"):
            return sig
    except _json.JSONDecodeError:
        return None
    return None


def _load_reference_block(state: dict) -> str:
    """Load raw/reference.json and format the canonical price snapshot the PM
    must use verbatim. Falls back to an empty string when raw/ is unavailable.
    """
    raw_dir = state.get("raw_dir")
    if not raw_dir:
        return ""
    try:
        ref_path = Path(raw_dir) / "reference.json"
        if not ref_path.exists():
            return ""
        ref = _json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return ""

    def fmt(v):
        return f"${v}" if v is not None else "(unavailable)"

    return (
        "\n**Canonical reference snapshot (use these exact values in the Inputs section):**\n"
        f"- reference_price: {fmt(ref.get('reference_price'))}\n"
        f"- reference_price_source: {ref.get('reference_price_source', '(unavailable)')}\n"
        f"- trade_date: {ref.get('trade_date', '(unavailable)')}\n"
        f"- spot_50dma: {fmt(ref.get('spot_50dma'))}\n"
        f"- spot_200dma: {fmt(ref.get('spot_200dma'))}\n"
        f"- ytd_high: {fmt(ref.get('ytd_high'))}\n"
        f"- ytd_low: {fmt(ref.get('ytd_low'))}\n"
        f"- atr_14: {fmt(ref.get('atr_14'))}\n"
        "Cite reference_price and trade_date verbatim from this snapshot — \
do not infer either from analyst transcripts or earlier debate prose. \
Other prices (article quotes, intraday) must carry an explicit time/source qualifier.\n"
    )


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

        reference_block = _load_reference_block(state)

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}{reference_block}

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

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}""" + _OUTPUT_CONTRACT + _MANDATED_SECTIONS + _QC_CHECKLIST + _RETRY_DIRECTIVE

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

        retry_signal = _parse_retry_signal(final_trade_decision)
        out = {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }
        if retry_signal and state.get("pm_retries", 0) < 1:
            out["pm_feedback"] = retry_signal["feedback"]
            out["pm_retries"] = state.get("pm_retries", 0) + 1
            out["pm_retry_target"] = retry_signal["target"]
        return out

    return portfolio_manager_node
