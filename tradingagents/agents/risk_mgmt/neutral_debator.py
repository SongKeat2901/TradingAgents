_PM_FEEDBACK_HANDLER = """

# Handling PM feedback (when re-invoked on retry)

If state.pm_feedback is set and non-empty, this is your second pass after \
the PM disagreed with your first investment plan. You must:

1. Quote the PM's feedback verbatim at the top of your revised plan, in a \
"## Addressing PM feedback" section.
2. Specifically address each concern raised. If the PM said "scenarios \
need explicit ROI hurdle," your scenarios must now include an explicit \
ROI hurdle.
3. Acknowledge what changed in your revised plan vs the first draft.

Do not silently ignore the feedback. Do not produce an identical plan."""


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        technicals_report = state.get("technicals_report", "")

        trader_decision = state["trader_investment_plan"]

        pm_feedback = state.get("pm_feedback", "")
        feedback_block = (
            f"\n\nPM_FEEDBACK_FROM_PRIOR_PASS:\n{pm_feedback}\n"
            if pm_feedback else ""
        )

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan. You prioritize a well-rounded approach, evaluating the upsides and downsides while factoring in broader market trends, potential economic shifts, and diversification strategies.Here is the trader's decision:

{trader_decision}

Your task is to challenge both the Aggressive and Conservative Analysts, pointing out where each perspective may be overly optimistic or overly cautious. Use insights from the following data sources to support a moderate, sustainable strategy to adjust the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the conservative analyst: {current_conservative_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage actively by analyzing both sides critically, addressing weaknesses in the aggressive and conservative arguments to advocate for a more balanced approach. Challenge each of their points to illustrate why a moderate risk strategy might offer the best of both worlds, providing growth potential while safeguarding against extreme volatility. Focus on debating rather than simply presenting data, aiming to show that a balanced view can lead to the most reliable outcomes.

**Length and form constraints:** ≤400 words total. Dense paragraphs only — do NOT invent rhetorical subsection headers like "The Real Problem Both Analyses Miss" or "Where Both Analyses Break Down"; those produce 8-page essays a portfolio manager will not read. State your strongest 2-3 points, each ≤120 words, anchored to a specific number from the analyst reports or refined technicals. The PM will adjudicate; you do not need to dismantle every claim the other analysts make.{feedback_block}""" + _PM_FEEDBACK_HANDLER

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
