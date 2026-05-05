"""Tests for the deterministic peer-ratios module (Phase-6.4)."""
import pytest

pytestmark = pytest.mark.unit


def _stub_peer(income_rows: list[tuple[str, list[float]]],
               cashflow_rows: list[tuple[str, list[float]]],
               fundamentals_text: str = "") -> dict:
    """Build a minimal peers.json sub-tree for one peer."""
    def _csv(rows):
        out = "# header line\n"
        for name, vals in rows:
            out += name + "," + ",".join(str(v) for v in vals) + "\n"
        return out
    return {
        "ticker": "TEST",
        "trade_date": "2026-05-01",
        "fundamentals": fundamentals_text,
        "balance_sheet": "",
        "cashflow": _csv(cashflow_rows),
        "income_statement": _csv(income_rows),
    }


def test_compute_peer_ratios_happy_path():
    """Per-peer Q1 capex/revenue + op margin compute correctly; PE parses
    from the fundamentals text block."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "GOOGL": _stub_peer(
            income_rows=[
                ("Total Revenue", [109_900_000_000, 100_000_000_000]),
                ("Operating Income", [39_700_000_000, 35_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-35_700_000_000, -30_000_000_000]),
            ],
            fundamentals_text=(
                "# Company Fundamentals for GOOGL\n"
                "Name: Alphabet Inc.\n"
                "PE Ratio (TTM): 29.23341\n"
                "Forward PE: 26.67876\n"
            ),
        ),
    }

    out = compute_peer_ratios(peers_data, "2026-05-01")

    assert out["trade_date"] == "2026-05-01"
    assert out["_unavailable"] == []
    g = out["GOOGL"]
    # 35.7B / 109.9B = 32.48%
    assert abs(g["latest_quarter_capex_to_revenue"] - 32.48) < 0.05
    # 39.7B / 109.9B = 36.12%
    assert abs(g["latest_quarter_op_margin"] - 36.12) < 0.05
    assert abs(g["ttm_pe"] - 29.23) < 0.01
    assert abs(g["forward_pe"] - 26.68) < 0.01
    assert "peers.json" in g["source"].lower()


def test_compute_peer_ratios_handles_missing_capex_column():
    """If a peer has no Capital Expenditure row, the peer enters _unavailable;
    other peers still populate."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "GOOD": _stub_peer(
            income_rows=[
                ("Total Revenue", [100_000_000_000]),
                ("Operating Income", [25_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-10_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 20.0\nForward PE: 18.0\n",
        ),
        "BAD": _stub_peer(
            income_rows=[
                ("Total Revenue", [50_000_000_000]),
            ],
            cashflow_rows=[
                # NO Capital Expenditure row
                ("Operating Cash Flow", [10_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 15.0\nForward PE: 14.0\n",
        ),
    }

    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert "GOOD" in out
    assert out["GOOD"]["latest_quarter_capex_to_revenue"] == 10.0
    assert "BAD" in out
    assert out["BAD"].get("unavailable") is True
    assert "BAD" in out["_unavailable"]
    assert "GOOD" not in out["_unavailable"]


def test_compute_peer_ratios_handles_zero_revenue():
    """Zero revenue must not raise ZeroDivisionError; peer enters _unavailable."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "ZEROREV": _stub_peer(
            income_rows=[
                ("Total Revenue", [0.0]),
                ("Operating Income", [0.0]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-1_000_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 0\nForward PE: 0\n",
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["ZEROREV"].get("unavailable") is True
    assert "ZEROREV" in out["_unavailable"]


def test_compute_peer_ratios_parses_pe_from_fundamentals_text():
    """The PE parser extracts TTM and Forward PE from the multi-line text block
    yfinance dumps as the `fundamentals` field."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "MSFT": _stub_peer(
            income_rows=[
                ("Total Revenue", [82_886_000_000]),
                ("Operating Income", [38_398_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-30_876_000_000]),
            ],
            fundamentals_text=(
                "# Company Fundamentals for MSFT\n"
                "Name: Microsoft Corporation\n"
                "Sector: Technology\n"
                "Market Cap: 3000000000000\n"
                "PE Ratio (TTM): 31.5\n"
                "Forward PE: 28.4\n"
                "PEG Ratio: 2.1\n"
            ),
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["MSFT"]["ttm_pe"] == 31.5
    assert out["MSFT"]["forward_pe"] == 28.4


def test_compute_peer_ratios_handles_unparseable_pe():
    """If the fundamentals text doesn't contain PE Ratio lines, the peer's
    ttm_pe / forward_pe are None — but capex/revenue + op margin still
    populate. Only PE-related fields go None; the peer is NOT marked
    _unavailable just for missing PE data."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "NOPEDATA": _stub_peer(
            income_rows=[
                ("Total Revenue", [50_000_000_000]),
                ("Operating Income", [10_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-5_000_000_000]),
            ],
            fundamentals_text="(empty fundamentals)",
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-01")
    assert out["NOPEDATA"]["ttm_pe"] is None
    assert out["NOPEDATA"]["forward_pe"] is None
    assert out["NOPEDATA"]["latest_quarter_capex_to_revenue"] == 10.0
    assert "NOPEDATA" not in out["_unavailable"]


def test_compute_peer_ratios_top_level_metadata():
    """Output dict has top-level trade_date + _unavailable; trade_date matches input."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    out = compute_peer_ratios({}, "2026-05-01")
    assert out["trade_date"] == "2026-05-01"
    assert out["_unavailable"] == []
    # Empty input — no peers in output beyond bookkeeping
    assert set(out.keys()) == {"trade_date", "_unavailable"}
