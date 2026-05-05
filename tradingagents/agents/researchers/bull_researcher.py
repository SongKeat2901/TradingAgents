
_RIGOR_RULES = """

# Argument rigor rules (Quant Research Rebuild — 2026-05-03)

1. Lead with your strongest argument. Your first paragraph names the SINGLE \
most load-bearing fact for the bull case. No analogies in the lede.

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

5. Length and form discipline. ≤500 words total. Dense paragraphs only — \
do NOT invent rhetorical subsection headers like "Contrarian Signal #1:", \
"The Bull's Most Load-Bearing Defense", "ARGUMENT 1:", or "Why X Is \
Backwards"; those produce multi-page essays the Research Manager and PM \
will not read. State your 3-4 strongest points, each ≤120 words, anchored \
to specific numbers from the analyst reports. The Research Manager will \
adjudicate; you do not need to dismantle every claim the bear makes.
"""


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        technicals_report = state.get("technicals_report", "")

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Dense, evidence-anchored response. Address the bear's strongest counterpoint with one specific data point, then move on. No rhetorical headers, no monologue.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
""" + _RIGOR_RULES

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
