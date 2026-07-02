"""FA-101 Phase 4 Task 4 — each fundamentals role node has a conditional
self-loop edge (retry the failed role) that advances to the next node once its
deterministic self-check passes or the retry cap is hit.

Inspection surface is the uncompiled StateGraph builder's ``.branches`` dict
(see test_graph_role_split.py): ``wf.branches[src]["router"].ends`` maps the
router return values ("retry"/"advance") to target nodes.
"""
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from tradingagents.graph.setup import GraphSetup, make_role_router
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.agents.analysts.fundamentals_roles import ROLE_RETRY_CAP


_ROLE_ADVANCE = {
    "Financial-Statement Analyst": "Risk & Red-Flags Analyst",
    "Risk & Red-Flags Analyst": "Catalysts & Ownership Analyst",
    "Catalysts & Ownership Analyst": "Competitive-Quality Analyst",
    "Competitive-Quality Analyst": "Fundamentals Aggregator",
}


def _build_workflow(selected_analysts):
    gs = GraphSetup(
        quick_thinking_llm=MagicMock(),
        deep_thinking_llm=MagicMock(),
        tool_nodes={},
        conditional_logic=ConditionalLogic(),
        config={},
    )
    return gs.setup_graph(selected_analysts)


def test_each_role_node_self_loops_and_advances():
    wf = _build_workflow(["market", "social", "news", "fundamentals"])
    for src, dst in _ROLE_ADVANCE.items():
        ends = wf.branches[src]["router"].ends
        assert ends["retry"] == src, f"{src} retry must self-loop"
        assert ends["advance"] == dst, f"{src} advance must go to {dst}"


def test_aggregator_has_no_self_loop():
    """The aggregator is deterministic — it must NOT be a conditional/retry
    node; it advances unconditionally to TA Agent v2."""
    wf = _build_workflow(["market", "social", "news", "fundamentals"])
    assert "Fundamentals Aggregator" not in wf.branches
    assert ("Fundamentals Aggregator", "TA Agent v2") in wf.edges


def test_role_router_advances_on_pass():
    router = make_role_router("p_passed", "p_retries", ROLE_RETRY_CAP)
    assert router({"p_passed": True, "p_retries": 0}) == "advance"


def test_role_router_retries_when_failed_under_cap():
    router = make_role_router("p_passed", "p_retries", ROLE_RETRY_CAP)
    assert router({"p_passed": False, "p_retries": 0}) == "retry"
    assert router({"p_passed": False, "p_retries": 1}) == "retry"


def test_role_router_advances_when_cap_hit_even_if_failed():
    """Fail-open: at the cap the node advances regardless of passed."""
    router = make_role_router("p_passed", "p_retries", ROLE_RETRY_CAP)
    assert router({"p_passed": False, "p_retries": ROLE_RETRY_CAP}) == "advance"
