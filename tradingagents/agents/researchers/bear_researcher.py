
_RIGOR_RULES = """

# Argument rigor rules (Quant Research Rebuild — 2026-05-03)

1. Lead with your strongest argument. Your first paragraph names the SINGLE \
most load-bearing fact for the bear case. No analogies in the lede.

2. Analogies must survive scrutiny. If you cite Tesla / NextEra / Blackstone \
/ any precedent, you must include: (a) the relevant metric for the \
comparable, (b) why the analogy holds for THIS ticker, (c) the disanalogy \
and why it doesn't break the case. If you can't do all three, drop the \
analogy.

3. Quantify the asymmetry. Your conclusion must include a specific \
dollar/percentage outcome AND a probability — e.g., "Bull case: $560 in 12 \
months, ~30% probability conditional on Q4 capex ROI clearing the threshold \
the fundamentals analyst flagged." Vague directional claims like "upside is \
meaningful" are rejected by the Research Manager and require revision.

4. Address the fundamentals analyst's sanity-check flags. Any item the \
fundamentals analyst flagged ❌ must be addressed in your argument — do not \
ignore them.
"""


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        technicals_report = state.get("technicals_report", "")

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock.
""" + _RIGOR_RULES

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
