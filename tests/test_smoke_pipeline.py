"""
Fast pipeline smoke gate — runs in <5 s, no real LLM, no network.

WHAT THIS COVERS
----------------
1. Import smoke — key top-level modules (cli.research, cli.cadence_followup,
   tradingagents.graph.trading_graph) import cleanly without missing deps or
   circular imports.  Catches typos, deleted files, bad __init__.py wires.

2. Graph wiring — TradingAgentsGraph constructs end-to-end with the production
   claude_code / via_cli=True config and a monkeypatched ClaudeCliChatModel so
   no subprocess is spawned.  Asserts every expected node name is present in
   the compiled graph.  Catches GraphSetup regressions (missing add_node /
   add_edge, renamed nodes).

3. Deterministic blocks on inline fixtures — exercises
   compute_classification, compute_peer_ratios, and compute_intrinsic_value
   with the smallest valid inputs to confirm the module is importable, the
   public API hasn't changed shape, and the result is well-formed (right
   top-level keys / types).  Does NOT re-test the logic that the dedicated
   test files already cover exhaustively.

PRE-CADENCE GATE: run this test before kicking a 21-ticker cadence:
    .venv/bin/python -m pytest tests/test_smoke_pipeline.py -q
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_llm():
    """A MagicMock that looks like a LangChain BaseChatModel."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    m = MagicMock()
    m.invoke.return_value = AIMessage(content="stub response")
    return m


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------

def test_import_cli_research():
    """cli.research imports without error (catches missing deps / bad wiring)."""
    import cli.research  # noqa: F401


def test_import_cli_cadence_followup():
    """cli.cadence_followup imports without error."""
    import cli.cadence_followup  # noqa: F401


def test_import_trading_graph():
    """tradingagents.graph.trading_graph imports without error."""
    import tradingagents.graph.trading_graph  # noqa: F401


def test_import_graph_setup():
    """tradingagents.graph.setup imports without error."""
    import tradingagents.graph.setup  # noqa: F401


def test_import_deterministic_utils():
    """All three deterministic-block modules import cleanly."""
    import tradingagents.agents.utils.classifier  # noqa: F401
    import tradingagents.agents.utils.peer_ratios  # noqa: F401
    import tradingagents.agents.utils.intrinsic_value  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Graph wiring
# ---------------------------------------------------------------------------

# Expected node names from setup.py (as of Phase 6.7).
_EXPECTED_NODES = {
    "PM Preflight",
    "Researcher",
    "TA Agent",
    "TA Agent v2",
    "Market Analyst",
    "Social Analyst",
    "News Analyst",
    "Financial-Statement Analyst",
    "Risk & Red-Flags Analyst",
    "Catalysts & Ownership Analyst",
    "Competitive-Quality Analyst",
    "Fundamentals Aggregator",
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Neutral Analyst",
    "Conservative Analyst",
    "Portfolio Manager",
    "QC Agent",
    "Executive PM",
}


def test_graph_wiring_nodes_present(tmp_path, monkeypatch):
    """TradingAgentsGraph builds offline with via_cli config.

    ClaudeCliChatModel is monkeypatched so no subprocess is spawned.
    The compiled graph node set is asserted against the expected list.
    """
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    # Patch ClaudeCliChatModel.__init__ to a no-op so constructing it offline
    # never tries to discover or spawn the claude CLI.
    import tradingagents.llm_clients.claude_cli_chat_model as _cli_mod

    fake_invoke_return = AIMessage(content="stub")

    class _StubCliModel(_cli_mod.ClaudeCliChatModel):
        """Offline stub: construction and invoke succeed without a subprocess."""

        def __init__(self, **kwargs):
            # Bypass BaseChatModel's pydantic __init__ so we don't need a
            # real model alias.  Just store attrs directly.
            object.__setattr__(self, "model", kwargs.get("model", "opus"))
            object.__setattr__(self, "cli_path", "claude")
            object.__setattr__(self, "timeout_seconds", 10.0)
            object.__setattr__(self, "max_retries", 0)
            object.__setattr__(self, "retry_base_delay", 0.0)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult
            return ChatResult(generations=[ChatGeneration(message=fake_invoke_return)])

        @property
        def _llm_type(self):
            return "stub-cli"

    monkeypatch.setattr(_cli_mod, "ClaudeCliChatModel", _StubCliModel)

    # Override dirs so the constructor's os.makedirs doesn't write to ~/.tradingagents
    config = {
        "llm_provider": "claude_code",
        "deep_think_llm": "claude-opus-4-8",
        "quick_think_llm": "claude-sonnet-4-6",
        "deep_via_cli": True,
        "quick_via_cli": True,
        "backend_url": None,
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "memory_log_path": str(tmp_path / "memory" / "trading_memory.md"),
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "pacing_seconds": 0,
        "deep_cooldown_seconds": 0,
        "checkpoint_enabled": False,
        "output_language": "English",
        "output_dir": str(tmp_path / "output"),
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
        "tool_vendors": {},
    }

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    g = TradingAgentsGraph(
        selected_analysts=["market", "social", "news", "fundamentals"],
        config=config,
    )

    # Compiled LangGraph exposes its nodes via .graph.nodes (a dict keyed by
    # node name).  "__start__" and "__end__" are framework internals we skip.
    compiled_nodes = {
        name
        for name in g.graph.nodes
        if not name.startswith("__")
    }

    missing = _EXPECTED_NODES - compiled_nodes
    assert not missing, (
        f"Graph is missing expected node(s): {sorted(missing)}\n"
        f"Actual nodes: {sorted(compiled_nodes)}"
    )


# ---------------------------------------------------------------------------
# 3. Deterministic blocks — smoke (well-formed output, no raise)
# ---------------------------------------------------------------------------

# Minimal OHLCV fixture — 100 flat days, reused from test_classifier.py shape.
def _flat_ohlcv(spot=400.0, days=100):
    from datetime import date, timedelta

    end = date(2026, 5, 1)
    rows = [
        (
            (end - timedelta(days=days - 1 - i)).isoformat(),
            spot, spot + 0.5, spot - 0.5, spot, 20_000_000,
        )
        for i in range(days)
    ]
    header = "# Stock data\nDate,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    body = "\n".join(
        f"{d},{o},{h},{l},{c},{v},0.0,0.0" for (d, o, h, l, c, v) in rows
    )
    return header + body + "\n"


def test_smoke_compute_classification_returns_well_formed_dict():
    """compute_classification returns a dict with the required keys and types."""
    from tradingagents.agents.utils.classifier import compute_classification

    ref = {
        "ticker": "MSFT",
        "trade_date": "2026-05-01",
        "reference_price": 400.0,
        "spot_50dma": 390.0,
        "spot_200dma": 440.0,
        "ytd_high": 480.0,
        "ytd_low": 380.0,
        "atr_14": 8.0,
    }
    out = compute_classification(ref, _flat_ohlcv())

    assert isinstance(out, dict), "compute_classification must return a dict"
    for key in ("setup_class", "upside_target", "downside_target", "rationale",
                "broken_level", "broken_level_type", "volume_confirmed"):
        assert key in out, f"missing key: {key}"
    assert isinstance(out["setup_class"], str)
    assert isinstance(out["rationale"], str)


def test_smoke_compute_peer_ratios_returns_well_formed_dict():
    """compute_peer_ratios returns a dict with trade_date and _unavailable."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "GOOGL": {
            "ticker": "GOOGL",
            "trade_date": "2026-05-01",
            "fundamentals": (
                "# Company Fundamentals for GOOGL\n"
                "PE Ratio (TTM): 29.0\nForward PE: 26.0\n"
            ),
            "income_statement": (
                "# header\n"
                "Total Revenue,109900000000,100000000000\n"
                "Operating Income,39700000000,35000000000\n"
            ),
            "cashflow": (
                "# header\n"
                "Capital Expenditure,-35700000000,-30000000000\n"
            ),
            "balance_sheet": "",
        }
    }

    out = compute_peer_ratios(peers_data, "2026-05-01")

    assert isinstance(out, dict), "compute_peer_ratios must return a dict"
    assert out["trade_date"] == "2026-05-01"
    assert isinstance(out["_unavailable"], list)
    assert "GOOGL" in out
    g = out["GOOGL"]
    for key in ("latest_quarter_capex_to_revenue", "latest_quarter_op_margin",
                "ttm_pe", "forward_pe"):
        assert key in g, f"GOOGL result missing key: {key}"


def test_smoke_compute_intrinsic_value_returns_well_formed_dict():
    """compute_intrinsic_value returns a dict with profile + fair_value keys."""
    from tradingagents.agents.utils.intrinsic_value import compute_intrinsic_value

    fin = {
        "ticker": "ACME",
        "financial_currency": "USD",
        "fundamentals": (
            "Name: Acme\nSector: Technology\nMarket Cap: 1000000000\n"
            "PE Ratio (TTM): 20\nForward PE: 16\nEPS (TTM): 5.0\n"
            "Forward EPS: 6.0\nBeta: 1.2\nEBITDA: 200000000\n"
            "Net Income: 100000000\nFree Cash Flow: 90000000\n"
            "Revenue (TTM): 800000000\n"
        ),
        "income_statement": (
            "\nTax Rate For Calcs,0.15,0.15\n"
            "Diluted Average Shares,50000000,50000000\n"
            "EBIT,150000000,140000000\n"
        ),
        "cashflow": "",
        "balance_sheet": "",
    }

    out = compute_intrinsic_value(
        fin,
        {"net_debt": -50_000_000},
        {"reference_price": 100.0},
        {"PEER": {"ttm_pe": 18}},
        risk_free=0.04,
        ticker="ACME",
    )

    assert isinstance(out, dict), "compute_intrinsic_value must return a dict"
    for key in ("profile", "fair_value", "methods", "skipped_methods"):
        assert key in out, f"missing top-level key: {key}"
    fv = out["fair_value"]
    for band in ("bear", "base", "bull"):
        assert band in fv, f"fair_value missing band: {band}"
    assert isinstance(out["skipped_methods"], list)
