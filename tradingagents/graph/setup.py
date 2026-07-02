# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node
from tradingagents.agents.managers.executive_pm import create_executive_pm_node
from tradingagents.agents.managers.qc_agent import create_qc_agent_node
from tradingagents.agents.analysts.ta_agent import create_ta_agent_node, create_ta_agent_v2_node
from tradingagents.agents.analysts.fundamentals_roles import (
    create_financial_statement_analyst,
    create_risk_redflags_analyst,
    create_catalysts_ownership_analyst,
    create_competitive_quality_analyst,
    create_fundamentals_aggregator,
)
from tradingagents.agents.researcher import fetch_research_pack

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        config: Dict[str, Any] | None = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.config = config

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes (quant-research rebuild: no tool loops — analysts read raw/)
        analyst_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm
            )

        # Fundamentals is split into 4 role-specific analyst nodes + a
        # deterministic aggregator (FA-101 Phase 3), not a single node like
        # market/social/news. Built separately from `analyst_nodes` below
        # because it needs 5 distinct display names wired in a fixed
        # sub-chain rather than the generic single-node-per-type pattern.
        fundamentals_role_nodes = {}
        if "fundamentals" in selected_analysts:
            fundamentals_role_nodes = {
                "Financial-Statement Analyst": create_financial_statement_analyst(
                    self.quick_thinking_llm
                ),
                "Risk & Red-Flags Analyst": create_risk_redflags_analyst(
                    self.quick_thinking_llm
                ),
                "Catalysts & Ownership Analyst": create_catalysts_ownership_analyst(
                    self.quick_thinking_llm
                ),
                "Competitive-Quality Analyst": create_competitive_quality_analyst(
                    self.quick_thinking_llm
                ),
                "Fundamentals Aggregator": create_fundamentals_aggregator(),
            }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Quant-research rebuild nodes (2026-05-03)
        pm_preflight_node = create_pm_preflight_node(self.deep_thinking_llm)
        ta_agent_node = create_ta_agent_node(self.quick_thinking_llm)
        ta_agent_v2_node = create_ta_agent_v2_node(self.quick_thinking_llm)
        # QC agent runs on a Sonnet-tier (quick) model to keep cost low; the
        # checklist work doesn't require Opus reasoning.
        qc_agent_node = create_qc_agent_node(self.quick_thinking_llm, self.config)
        # Phase 6.7 Executive PM runs on the deep tier (Opus) — stakeholder
        # voice + numerical fidelity require careful prose, and it's the
        # last LLM call in the pipeline so the marginal cost is small.
        executive_pm_node = create_executive_pm_node(self.deep_thinking_llm)

        def researcher_node(state):
            """Wraps the Python data fetcher as a LangGraph node."""
            fetch_research_pack(state)
            return {"raw_dir": state["raw_dir"]}

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add new prefix nodes
        workflow.add_node("PM Preflight", pm_preflight_node)
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("TA Agent", ta_agent_node)
        workflow.add_node("TA Agent v2", ta_agent_v2_node)
        workflow.add_node("QC Agent", qc_agent_node)
        workflow.add_node("Executive PM", executive_pm_node)

        # Add analyst nodes to the graph (no tool-loop nodes in rebuild)
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)

        # Add the 5 fundamentals role/aggregator nodes under their exact
        # display names (not the generic "Fundamentals Analyst" pattern).
        for name, node in fundamentals_role_nodes.items():
            workflow.add_node(name, node)

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # New prefix: START → PM Preflight → Researcher → TA Agent → first analyst
        workflow.add_edge(START, "PM Preflight")
        workflow.add_edge("PM Preflight", "Researcher")
        workflow.add_edge("Researcher", "TA Agent")

        # The "fundamentals" slot expands to a fixed 5-node sub-chain
        # (Financial-Statement → Risk & Red-Flags → Catalysts & Ownership →
        # Competitive-Quality → Fundamentals Aggregator); every other
        # analyst type is still a single node. `entry_node`/`exit_node`
        # let the generic chain-building loop below treat both uniformly.
        def entry_node(analyst_type):
            if analyst_type == "fundamentals":
                return "Financial-Statement Analyst"
            return f"{analyst_type.capitalize()} Analyst"

        def exit_node(analyst_type):
            if analyst_type == "fundamentals":
                return "Fundamentals Aggregator"
            return f"{analyst_type.capitalize()} Analyst"

        first_analyst = selected_analysts[0]
        workflow.add_edge("TA Agent", entry_node(first_analyst))

        # Connect analysts in sequence (no tool-loop conditional edges)
        for i, analyst_type in enumerate(selected_analysts):
            if analyst_type == "fundamentals":
                workflow.add_edge(
                    "Financial-Statement Analyst", "Risk & Red-Flags Analyst"
                )
                workflow.add_edge(
                    "Risk & Red-Flags Analyst", "Catalysts & Ownership Analyst"
                )
                workflow.add_edge(
                    "Catalysts & Ownership Analyst", "Competitive-Quality Analyst"
                )
                workflow.add_edge(
                    "Competitive-Quality Analyst", "Fundamentals Aggregator"
                )

            current_exit = exit_node(analyst_type)
            if i < len(selected_analysts) - 1:
                next_analyst = selected_analysts[i + 1]
                workflow.add_edge(current_exit, entry_node(next_analyst))
            else:
                workflow.add_edge(current_exit, "TA Agent v2")
        workflow.add_edge("TA Agent v2", "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        # PM retry conditional edges:
        # - PM_RETRY_SIGNAL → push back to RM or Risk team (Pass-3 mechanism)
        # - otherwise → QC Agent for an independent audit
        def pm_router(state):
            if state.get("pm_retries", 0) >= 1:
                return "QC Agent"
            target = state.get("pm_retry_target")
            if target == "research_manager":
                return "Research Manager"
            if target == "risk_team":
                return "Aggressive Analyst"
            return "QC Agent"

        workflow.add_conditional_edges(
            "Portfolio Manager",
            pm_router,
            {
                "Research Manager": "Research Manager",
                "Aggressive Analyst": "Aggressive Analyst",
                "QC Agent": "QC Agent",
            },
        )

        # QC routing: PASS → Executive PM (Phase 6.7 stakeholder-voice
        # translation) → END; FAIL with retries left → re-run PM with feedback.
        def qc_router(state):
            if state.get("qc_passed", False):
                return "Executive PM"
            return "Portfolio Manager"

        workflow.add_conditional_edges(
            "QC Agent",
            qc_router,
            {
                "Portfolio Manager": "Portfolio Manager",
                "Executive PM": "Executive PM",
            },
        )
        workflow.add_edge("Executive PM", END)

        return workflow
