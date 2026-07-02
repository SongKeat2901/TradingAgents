"""FA-101 Phase 3 Task 4 — graph wiring for the 4 fundamentals role nodes
+ aggregator (replacing the single monolithic "Fundamentals Analyst" node).

Graph-inspection surface: ``GraphSetup.setup_graph()`` returns the
*uncompiled* ``langgraph.graph.StateGraph`` builder (see
``tradingagents/graph/trading_graph.py`` where ``self.workflow =
graph_setup.setup_graph(...)`` is compiled separately). That builder
exposes ``.nodes`` (dict keyed by display name) and ``.edges`` (a set of
``(source, target)`` tuples) BEFORE compilation — the same surface
``TradingAgentsGraph`` production code drives. Constructing a bare
``GraphSetup`` with stub LLMs (as ``setup_graph`` never invokes the LLMs,
only stores them) avoids the heavier CLI-model monkeypatching that
``test_smoke_pipeline.py`` needs for the *compiled* graph.
"""
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic


_ROLE_NODE_NAMES = [
    "Financial-Statement Analyst",
    "Risk & Red-Flags Analyst",
    "Catalysts & Ownership Analyst",
    "Competitive-Quality Analyst",
    "Fundamentals Aggregator",
]


def _build_workflow(selected_analysts):
    gs = GraphSetup(
        quick_thinking_llm=MagicMock(),
        deep_thinking_llm=MagicMock(),
        tool_nodes={},
        conditional_logic=ConditionalLogic(),
        config={},
    )
    return gs.setup_graph(selected_analysts)


def test_role_nodes_registered_and_fundamentals_analyst_absent():
    wf = _build_workflow(["market", "social", "news", "fundamentals"])
    node_names = set(wf.nodes.keys())

    for name in _ROLE_NODE_NAMES:
        assert name in node_names, f"missing role node: {name}"

    assert "Fundamentals Analyst" not in node_names


def test_role_chain_wired_sequentially_and_aggregator_feeds_ta_v2():
    """Binding assertion: the edge into "TA Agent v2" originates from
    "Fundamentals Aggregator" ONLY — no individual role node has a direct
    edge to TA Agent v2."""
    wf = _build_workflow(["market", "social", "news", "fundamentals"])
    edges = wf.edges

    assert ("News Analyst", "Financial-Statement Analyst") in edges
    assert ("Financial-Statement Analyst", "Risk & Red-Flags Analyst") in edges
    assert ("Risk & Red-Flags Analyst", "Catalysts & Ownership Analyst") in edges
    assert ("Catalysts & Ownership Analyst", "Competitive-Quality Analyst") in edges
    assert ("Competitive-Quality Analyst", "Fundamentals Aggregator") in edges
    assert ("Fundamentals Aggregator", "TA Agent v2") in edges

    edges_into_ta_v2 = {src for (src, dst) in edges if dst == "TA Agent v2"}
    assert edges_into_ta_v2 == {"Fundamentals Aggregator"}


def test_fundamentals_not_selected_no_role_nodes_added():
    """Gating preserved: without "fundamentals" in selected_analysts, none
    of the 5 nodes exist and the chain closes to TA Agent v2 exactly as
    before (from the prior generic analyst)."""
    wf = _build_workflow(["market", "social", "news"])
    node_names = set(wf.nodes.keys())

    for name in _ROLE_NODE_NAMES + ["Fundamentals Analyst"]:
        assert name not in node_names

    edges_into_ta_v2 = {src for (src, dst) in wf.edges if dst == "TA Agent v2"}
    assert edges_into_ta_v2 == {"News Analyst"}


def test_role_chain_handles_fundamentals_not_last():
    """The 5-node block must slot in correctly regardless of where
    "fundamentals" sits in selected_analysts (entry from whatever precedes
    it, exit from the aggregator to whatever follows it)."""
    wf = _build_workflow(["fundamentals", "market"])
    node_names = set(wf.nodes.keys())

    for name in _ROLE_NODE_NAMES:
        assert name in node_names

    edges = wf.edges
    assert ("TA Agent", "Financial-Statement Analyst") in edges
    assert ("Fundamentals Aggregator", "Market Analyst") in edges

    edges_into_ta_v2 = {src for (src, dst) in edges if dst == "TA Agent v2"}
    assert edges_into_ta_v2 == {"Market Analyst"}
