"""Tests for the deterministic net-debt block (Phase-6.5)."""
import pytest

pytestmark = pytest.mark.unit


# Real balance_sheet rows from the 2026-05-06 cadence. The header row is the
# yfinance format (leading empty cell + comma-separated date columns; col 0 =
# most-recent quarter).
_MSTR_BS = (
    "# Balance Sheet data for MSTR (quarterly)\n"
    "# Data retrieved on: 2026-05-06 01:06:54\n"
    "\n"
    ",2025-12-31,2025-09-30,2025-06-30,2025-03-31,2024-12-31,2024-09-30\n"
    "Net Debt,5888685000.0,8119618000.0,8112535000.0,8080383000.0,7153558000.0,\n"
    "Total Debt,8236290000.0,8222065000.0,8213848000.0,8194372000.0,7248078000.0,\n"
    "Long Term Debt,8158842000.0,8173587000.0,8158839000.0,8170528000.0,7193987000.0,\n"
    "Current Debt,31313000.0,316000.0,3791000.0,1234000.0,200000.0,\n"
    "Capital Lease Obligations,46135000.0,48162000.0,51218000.0,53691000.0,56403000.0,\n"
    "Cash And Cash Equivalents,2301470000.0,54285000.0,84221000.0,108156000.0,38116000.0,\n"
    "Cash Cash Equivalents And Short Term Investments,2301470000.0,54285000.0,84221000.0,108156000.0,38116000.0,\n"
)

_APA_BS = (
    "# Balance Sheet data for APA (quarterly)\n"
    "\n"
    ",2025-12-31,2025-09-30,2025-06-30,2025-03-31,2024-12-31\n"
    "Net Debt,3977000000.0,4013000000.0,4100000000.0,4150000000.0,4200000000.0\n"
    "Total Debt,4590000000.0,4591000000.0,4595000000.0,4600000000.0,4650000000.0\n"
    "Long Term Debt,4280000000.0,4275000000.0,4280000000.0,4285000000.0,4290000000.0\n"
    "Current Debt,213000000.0,213000000.0,213000000.0,213000000.0,360000000.0\n"
    "Capital Lease Obligations,97000000.0,103000000.0,108000000.0,113000000.0,120000000.0\n"
    "Cash And Cash Equivalents,516000000.0,475000000.0,420000000.0,400000000.0,450000000.0\n"
    "Cash Cash Equivalents And Short Term Investments,516000000.0,475000000.0,420000000.0,400000000.0,450000000.0\n"
)


def test_compute_net_debt_extracts_yfinance_row_when_present():
    from tradingagents.agents.utils.net_debt import compute_net_debt
    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _MSTR_BS})

    assert out["unavailable"] is False
    assert out["as_of_quarter"] == "2025-12-31"
    assert out["net_debt"] == 5_888_685_000.0
    assert out["net_debt_source"] == "yfinance"
    assert out["total_debt"] == 8_236_290_000.0
    assert out["long_term_debt"] == 8_158_842_000.0
    assert out["current_debt"] == 31_313_000.0
    assert out["capital_lease_obligations"] == 46_135_000.0
    assert out["cash_and_equivalents"] == 2_301_470_000.0
    assert out["cash_plus_short_term_investments"] == 2_301_470_000.0


def test_compute_net_debt_falls_back_when_yfinance_row_missing():
    """If the Net Debt row is absent, compute as Total Debt − (Cash + STI)
    and surface `net_debt_source = "computed"` so the LLM can disclose."""
    from tradingagents.agents.utils.net_debt import compute_net_debt

    bs = (
        ",2025-12-31\n"
        "Total Debt,8236290000.0\n"
        "Cash Cash Equivalents And Short Term Investments,2301470000.0\n"
    )
    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": bs})

    assert out["unavailable"] is False
    assert out["net_debt"] == 8_236_290_000.0 - 2_301_470_000.0
    assert out["net_debt_source"] == "computed"


def test_compute_net_debt_unavailable_when_total_debt_missing():
    from tradingagents.agents.utils.net_debt import compute_net_debt

    bs = (
        ",2025-12-31\n"
        "Cash And Cash Equivalents,2301470000.0\n"
    )
    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": bs})

    assert out["unavailable"] is True
    assert "Total Debt" in (out["unavailable_reason"] or "")


def test_compute_net_debt_handles_empty_balance_sheet():
    from tradingagents.agents.utils.net_debt import compute_net_debt
    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": ""})

    assert out["unavailable"] is True
    assert out["net_debt"] is None


def test_compute_net_debt_handles_non_dict_input():
    from tradingagents.agents.utils.net_debt import compute_net_debt
    out = compute_net_debt(None)  # type: ignore[arg-type]

    assert out["unavailable"] is True


def test_format_net_debt_block_renders_authoritative_cells_for_mstr():
    """Smoke test: the rendered block must contain the exact MSTR cells the
    LLM should quote — not paraphrases. Regression for the APA $6.0B
    fabrication where the LLM invented a Total Debt cell that wasn't in raw."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _MSTR_BS})
    block = format_net_debt_block(nd)

    assert "## Net debt" in block
    assert "trade_date 2026-05-06" in block
    assert "quarter ending 2025-12-31" in block
    # Authoritative Net Debt line — the single number to cite
    assert "Authoritative Net Debt: $5.89B" in block
    assert "source: yfinance" in block
    # Cells that anchored APA's fabricated math — Total Debt $4.59B and
    # similar must render verbatim.
    assert "$8.24B" in block  # MSTR Total Debt
    assert "Long Term Debt" in block
    assert "Cash + Short Term Investments" in block
    # Anti-fabrication footer
    assert "verbatim" in block.lower()
    assert "do not introduce cells not in this table" in block.lower()


def test_format_net_debt_block_renders_apa_cells_correctly():
    """APA-specific regression: raw Total Debt is $4.59B, not the $6.0B
    that decision.md fabricated. The rendered block must show $4.59B."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _APA_BS})
    block = format_net_debt_block(nd)

    assert "Authoritative Net Debt: $3.98B" in block
    assert "$4.59B" in block  # Total Debt — the cell APA's report fabricated
    assert "$516M" in block  # Cash And Cash Equivalents (sub-1B → M scale)
    # Capital Lease Obligations — APA has $97M, must render as $97M
    assert "$97M" in block


def test_format_net_debt_block_returns_empty_when_unavailable():
    """Caller is responsible for the "unavailable" warning block; the
    formatter just returns "" so the caller can detect and override."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": ""})
    assert format_net_debt_block(nd) == ""


def test_format_b_handles_billions_millions_thousands():
    from tradingagents.agents.utils.net_debt import _fmt_b

    assert _fmt_b(5_888_685_000.0) == "$5.89B"
    assert _fmt_b(516_000_000.0) == "$516M"  # Sub-1B drops to millions
    assert _fmt_b(97_000_000.0) == "$97M"
    assert _fmt_b(31_313_000.0) == "$31M"
    assert _fmt_b(None) == "(n/a)"
    # Negative values (e.g., net cash position) handled correctly
    assert _fmt_b(-2_500_000_000.0) == "$-2.50B"
