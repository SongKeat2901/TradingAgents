"""Tests for the deterministic peer-ratios module (Phase-6.4)."""
import pytest

pytestmark = pytest.mark.unit


def _stub_peer(income_rows: list[tuple[str, list[float]]],
               cashflow_rows: list[tuple[str, list[float]]],
               fundamentals_text: str = "",
               balance_sheet_rows: list[tuple[str, list[float]]] | None = None) -> dict:
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
        "balance_sheet": _csv(balance_sheet_rows) if balance_sheet_rows else "",
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


# ---------------------------------------------------------------------------
# Phase-6.4 leverage extension (2026-05-07): Net Debt + TTM EBITDA + ND/EBITDA
# ---------------------------------------------------------------------------

def test_compute_peer_ratios_extracts_net_debt_and_ebitda():
    """When balance_sheet has Net Debt and fundamentals has EBITDA, the
    output dict must include net_debt + ttm_ebitda + nd_ebitda. Regression
    for the 2026-05-06 MARA decision that fabricated `RIOT EV/EBITDA ~12×,
    CIFR ND/EBITDA ~1.5×, CLSK op margin ~5%` (actual CLSK op margin
    -37.83%, sign-flipped) — the Phase-6.4 deterministic block had no
    leverage cells to anchor the LLM."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "RIOT": _stub_peer(
            income_rows=[
                ("Total Revenue", [167_219_000]),
                ("Operating Income", [-121_348_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-131_649_000]),
            ],
            balance_sheet_rows=[
                ("Net Debt", [636_497_000]),
                ("Total Debt", [877_185_000]),
                ("Cash And Cash Equivalents", [205_666_000]),
            ],
            fundamentals_text=(
                "PE Ratio (TTM): 0\nForward PE: 0\nEBITDA: -326712000\n"
            ),
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-06")
    r = out["RIOT"]
    assert r["net_debt"] == 636_497_000
    assert r["ttm_ebitda"] == -326_712_000
    # EBITDA negative → ND/EBITDA uninterpretable → None (formatter renders n/m)
    assert r["nd_ebitda"] is None


def test_compute_peer_ratios_nd_ebitda_when_positive_ebitda():
    """ND/EBITDA computes when TTM EBITDA > 0 (typical mature peer case)."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "DVN": _stub_peer(
            income_rows=[
                ("Total Revenue", [3_840_000_000]),
                ("Operating Income", [785_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-870_000_000]),
            ],
            balance_sheet_rows=[
                ("Net Debt", [6_960_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 12.33\nForward PE: 9.49\nEBITDA: 5_000_000_000\n".replace("_", ""),
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-06")
    d = out["DVN"]
    assert d["net_debt"] == 6_960_000_000
    assert d["ttm_ebitda"] == 5_000_000_000
    assert d["nd_ebitda"] is not None
    assert abs(d["nd_ebitda"] - 1.39) < 0.05  # 6.96 / 5.00


def test_compute_peer_ratios_falls_back_to_total_debt_minus_cash():
    """If balance_sheet lacks the Net Debt row, compute net_debt from
    Total Debt - (Cash + STI)."""
    from tradingagents.agents.utils.peer_ratios import compute_peer_ratios

    peers_data = {
        "NONETDEBTROW": _stub_peer(
            income_rows=[
                ("Total Revenue", [10_000_000_000]),
                ("Operating Income", [2_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-1_000_000_000]),
            ],
            balance_sheet_rows=[
                # Net Debt row deliberately absent
                ("Total Debt", [3_000_000_000]),
                ("Cash And Cash Equivalents", [800_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 15.0\nForward PE: 13.0\nEBITDA: 1_500_000_000\n".replace("_", ""),
        ),
    }
    out = compute_peer_ratios(peers_data, "2026-05-06")
    p = out["NONETDEBTROW"]
    assert p["net_debt"] == 2_200_000_000  # 3B - 0.8B
    assert p["ttm_ebitda"] == 1_500_000_000
    # 2.2 / 1.5 = 1.467
    assert abs(p["nd_ebitda"] - 1.47) < 0.05


def test_format_peer_ratios_block_renders_seven_columns():
    """The rendered table must include Net Debt + TTM EBITDA + ND/EBITDA
    columns AFTER the four original columns; (n/m) renders for negative
    EBITDA peers."""
    from tradingagents.agents.utils.peer_ratios import (
        compute_peer_ratios,
        format_peer_ratios_block,
    )

    peers_data = {
        "DVN": _stub_peer(
            income_rows=[
                ("Total Revenue", [3_840_000_000]),
                ("Operating Income", [785_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-870_000_000]),
            ],
            balance_sheet_rows=[
                ("Net Debt", [6_960_000_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 12.33\nForward PE: 9.49\nEBITDA: 5000000000\n",
        ),
        "RIOT": _stub_peer(
            income_rows=[
                ("Total Revenue", [167_219_000]),
                ("Operating Income", [-121_348_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-131_649_000]),
            ],
            balance_sheet_rows=[
                ("Net Debt", [636_497_000]),
            ],
            fundamentals_text="PE Ratio (TTM): 0\nForward PE: 0\nEBITDA: -326712000\n",
        ),
    }
    ratios = compute_peer_ratios(peers_data, "2026-05-06")
    block = format_peer_ratios_block(ratios)

    # Header has 7 data columns
    assert "Net Debt" in block
    assert "TTM EBITDA" in block
    assert "ND/EBITDA" in block
    # DVN: positive EBITDA → ND/EBITDA renders (~1.39x)
    assert "$6.96B" in block
    assert "1.39x" in block
    # RIOT: negative EBITDA → ND/EBITDA renders (n/m), not a fabricated number
    assert "(n/m)" in block
    # Footer warns about negative-EBITDA peers
    assert "(n/m)" in block.lower() or "uninterpretable" in block.lower()


def test_format_peer_ratios_block_unavailable_row_has_seven_columns():
    """When a peer is unavailable, the (unavailable) row must still match the
    new 7-column width to keep the markdown table structurally valid."""
    from tradingagents.agents.utils.peer_ratios import (
        compute_peer_ratios,
        format_peer_ratios_block,
    )

    peers_data = {
        "GOOD": _stub_peer(
            income_rows=[
                ("Total Revenue", [100_000_000_000]),
                ("Operating Income", [25_000_000_000]),
            ],
            cashflow_rows=[
                ("Capital Expenditure", [-10_000_000_000]),
            ],
            balance_sheet_rows=[("Net Debt", [10_000_000_000])],
            fundamentals_text="PE Ratio (TTM): 20.0\nForward PE: 18.0\nEBITDA: 30000000000\n",
        ),
        "BAD": _stub_peer(
            income_rows=[("Total Revenue", [50_000_000_000])],  # no Op Income
            cashflow_rows=[("Operating Cash Flow", [10_000_000_000])],  # no Capex
            fundamentals_text="(empty)",
        ),
    }
    ratios = compute_peer_ratios(peers_data, "2026-05-06")
    block = format_peer_ratios_block(ratios)

    # Find the BAD row and count cells
    for line in block.split("\n"):
        if line.startswith("| BAD |"):
            # 1 ticker + 7 data + trailing empty = 9 pipe-separated cells
            cell_count = line.count("|")
            assert cell_count == 9, f"BAD row has {cell_count} pipes; expected 9"
            assert line.count("(unavailable)") == 7
            break
    else:
        raise AssertionError("BAD row not found in rendered block")
