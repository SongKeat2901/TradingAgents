"""Tests for the deterministic relative-multiples block (Phase-1 Task 4)."""
import pytest

from tradingagents.agents.utils.relative_multiples import (
    compute_relative_multiples,
    format_relative_multiples_block,
)

pytestmark = pytest.mark.unit

_FIN = {
    "revenue_ttm": 40000000000.0, "ebit": 10000000000.0, "ebitda": 15000000000.0,
    "fcf": 9000000000.0, "net_income": 8000000000.0, "total_equity": 40000000000.0,
    "eps": 5.0, "forward_eps": 6.0, "diluted_shares": 2000000000.0,
}
_PEERS = {
    "A": {"market_cap": 50e9, "net_debt": 5e9, "ttm_ebitda": 6e9, "ttm_pe": 18.0,
          "forward_pe": 15.0, "latest_quarter_op_margin": 20.0, "nd_ebitda": 0.83},
    "B": {"market_cap": 70e9, "net_debt": 10e9, "ttm_ebitda": 8e9, "ttm_pe": 22.0,
          "forward_pe": 19.0, "latest_quarter_op_margin": 18.0, "nd_ebitda": 1.25},
}


def test_subject_multiples_and_ev_tie_out():
    m = compute_relative_multiples(_FIN, market_cap=100e9, net_debt=8e9, peers=_PEERS, forward_eps=6.0)
    assert m["subject"]["ev"] == 108e9                 # market_cap + net_debt
    assert m["subject"]["ev_ebitda"] == round(108e9 / 15e9, 2)
    assert m["subject"]["ev_sales"] == round(108e9 / 40e9, 2)
    assert m["subject"]["p_b"] == round(100e9 / 40e9, 2)
    # forward P/E basis: market_cap / (forward_eps * diluted_shares)
    # = 100e9 / (6.0 * 2e9) = 100e9 / 12e9 = 8.333... -> 8.33
    assert m["subject"]["p_e_fwd"] == round(100e9 / (6.0 * 2000000000.0), 2)
    assert m["subject"]["p_e_fwd"] == 8.33
    # trailing subject P/E = market_cap / net_income = 100e9 / 8e9 = 12.5
    assert m["subject"]["p_e_ttm"] == 12.5
    # peer median EV/EBITDA = median([55/6, 80/8]) = median([9.17, 10.0])
    assert m["peer_median"]["ev_ebitda"] == round((round(55e9/6e9,2) + round(80e9/8e9,2)) / 2, 2)


def test_missing_inputs_na():
    m = compute_relative_multiples({}, market_cap=None, net_debt=None, peers={})
    assert m["subject"]["ev"] is None
    assert m["subject"]["p_e_fwd"] is None
    block = format_relative_multiples_block(m, "2026-05-01")
    assert "## Relative valuation multiples" in block
    assert "n/a (data unavailable)" in block


def test_block_header_and_mandate():
    m = compute_relative_multiples(_FIN, 100e9, 8e9, _PEERS, 6.0)
    block = format_relative_multiples_block(m, "2026-05-01")
    assert "EV/EBITDA" in block and "P/B" in block
    assert "EV = market cap + net debt" in block
    assert "verbatim" in block


def test_unavailable_peer_excluded_from_median_no_crash():
    """A Task-3 `{"unavailable": True, "reason": ...}` peer (or one merely
    missing market_cap/net_debt/ttm_ebitda) must be skipped from the
    peer-median EV/EBITDA computation, not crash it."""
    peers = dict(_PEERS)
    peers["C"] = {"unavailable": True, "reason": "missing rows: Total Revenue"}
    peers["D"] = {"market_cap": 90e9, "ttm_ebitda": 9e9}  # missing net_debt
    m = compute_relative_multiples(_FIN, market_cap=100e9, net_debt=8e9, peers=peers, forward_eps=6.0)
    # median unchanged from the two fully-populated peers (A, B)
    assert m["peer_median"]["ev_ebitda"] == round((round(55e9/6e9,2) + round(80e9/8e9,2)) / 2, 2)


def test_forward_pe_none_when_diluted_shares_missing():
    fin = dict(_FIN)
    del fin["diluted_shares"]
    m = compute_relative_multiples(fin, market_cap=100e9, net_debt=8e9, peers=_PEERS, forward_eps=6.0)
    assert m["subject"]["p_e_fwd"] is None
