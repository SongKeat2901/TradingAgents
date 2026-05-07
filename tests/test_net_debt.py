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
    assert "do not introduce cells not present in either source" in block.lower()


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


# ---------------------------------------------------------------------------
# Phase 6.5 v2 (2026-05-07): Other STI fallback + 10-Q period crosscheck note
# ---------------------------------------------------------------------------

# SOFI 2026-05-06 balance sheet — has Cash + Other STI but NO composite row.
# Real cells from raw/financials.json on macmini-trueknot.
_SOFI_BS = (
    "# Balance Sheet for SOFI (quarterly)\n"
    "\n"
    ",2025-12-31,2025-09-30,2025-06-30,2025-03-31,2024-12-31\n"
    "Total Debt,1934625000.0,2050000000.0,2100000000.0,2200000000.0,2300000000.0\n"
    "Long Term Debt,1329682000.0,1400000000.0,1450000000.0,1500000000.0,1600000000.0\n"
    "Current Debt,486000000.0,500000000.0,500000000.0,550000000.0,560000000.0\n"
    "Capital Lease Obligations,118943000.0,120000000.0,120000000.0,120000000.0,125000000.0\n"
    "Other Short Term Investments,2430980000.0,2393242000.0,2266588000.0,2153456000.0,1804043000.0\n"
    "Cash And Cash Equivalents,4929452000.0,3246351000.0,2122502000.0,2085697000.0,2538293000.0\n"
)


def test_compute_net_debt_includes_other_sti_when_composite_row_missing():
    """SOFI 2026-05-06 audit: the prior block surfaced cash_plus_sti = $4.93B
    (Cash only) when actual liquid assets were $4.93B + $2.43B Other STI =
    $7.36B. Composite row absent; Other STI must be added to the aggregate."""
    from tradingagents.agents.utils.net_debt import compute_net_debt

    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _SOFI_BS})

    assert out["unavailable"] is False
    assert out["cash_and_equivalents"] == 4_929_452_000.0
    assert out["other_short_term_investments"] == 2_430_980_000.0
    assert out["short_term_investments"] is None
    # The aggregate is now Cash + Other STI = $4.93B + $2.43B = $7.36B
    assert abs(out["cash_plus_short_term_investments"] - 7_360_432_000.0) < 1.0
    # Net Debt = Total Debt − (Cash + STI) = 1.935B − 7.36B = −$5.43B
    # (yfinance Net Debt row absent in this fixture; falls back to computed)
    assert out["net_debt_source"] == "computed"
    assert abs(out["net_debt"] - (1_934_625_000.0 - 7_360_432_000.0)) < 1.0


def test_compute_net_debt_uses_short_term_investments_when_other_absent():
    """When `Short Term Investments` row exists (some tickers use this name
    instead of the `Other` variant), it must also feed the composite."""
    from tradingagents.agents.utils.net_debt import compute_net_debt

    bs = (
        ",2025-12-31\n"
        "Total Debt,1000000000.0\n"
        "Cash And Cash Equivalents,500000000.0\n"
        "Short Term Investments,300000000.0\n"
    )
    out = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": bs})
    assert out["short_term_investments"] == 300_000_000.0
    assert out["cash_plus_short_term_investments"] == 800_000_000.0


def test_format_net_debt_block_renders_sti_breakdown_line_when_present():
    """When STI is present as a separate row from cash, the rendered block
    must include an explicit `Short Term Investments` line so the LLM sees
    the breakdown rather than just the composite."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _SOFI_BS})
    block = format_net_debt_block(nd)

    assert "Short Term Investments" in block
    assert "$2.43B" in block  # Other STI cell — surfaced explicitly
    assert "$4.93B" in block  # Cash cell — distinct from STI
    assert "$7.36B" in block  # Composite cell (Cash + STI)


def test_format_net_debt_block_omits_sti_line_when_only_composite_present():
    """When only the composite row is present (typical case for non-fintech),
    the breakdown line is omitted to keep the table clean."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _MSTR_BS})
    block = format_net_debt_block(nd)

    # MSTR doesn't have separate STI row, so no "Short Term Investments" line
    # except in the existing "Cash + Short Term Investments" composite line
    sti_lines = [
        line for line in block.split("\n")
        if line.startswith("| Short Term Investments")
    ]
    assert sti_lines == []


def test_format_net_debt_block_relabels_negative_as_net_cash():
    """Phase 6.8 stakeholder-polish: negative net debt → relabel as
    "Net Cash" with positive magnitude, instead of `Net Debt: $-4.08B`
    which forces the reader to flip the sign mentally. The COIN
    2026-05-06 run surfaced this — −$4.08B is net-cash-positive, not
    "negative net debt"."""
    from tradingagents.agents.utils.net_debt import format_net_debt_block

    nd = {
        "trade_date": "2026-05-06",
        "as_of_quarter": "2025-12-31",
        "net_debt": -4_082_607_000.0,  # COIN net-cash-positive
        "net_debt_source": "computed",
        "total_debt": 7_830_000_000.0,
        "long_term_debt": 5_940_000_000.0,
        "current_debt": 1_720_000_000.0,
        "capital_lease_obligations": 173_000_000.0,
        "cash_and_equivalents": 11_290_000_000.0,
        "cash_plus_short_term_investments": 11_910_000_000.0,
        "unavailable": False,
    }
    block = format_net_debt_block(nd)

    # No awkward `$-X.XXB` artifacts in the rendered block (it should
    # have been flipped to a positive Net Cash value).
    assert "Net Cash" in block
    assert "Authoritative Net Cash: $4.08B" in block
    # Section header also relabels for the net-cash case
    assert "## Net cash" in block
    # The old confusing rendering must NOT appear
    assert "$-4.08B" not in block
    assert "Authoritative Net Debt:" not in block


def test_format_net_debt_block_keeps_net_debt_label_when_positive():
    """Sanity: positive net debt (typical leveraged company) still
    renders as Net Debt; the relabel only fires for negative values."""
    from tradingagents.agents.utils.net_debt import format_net_debt_block

    nd = {
        "trade_date": "2026-05-06",
        "as_of_quarter": "2025-12-31",
        "net_debt": 5_888_685_000.0,  # MSTR net-debt-positive
        "net_debt_source": "yfinance",
        "total_debt": 8_236_290_000.0,
        "long_term_debt": 8_158_842_000.0,
        "current_debt": 31_313_000.0,
        "capital_lease_obligations": 46_135_000.0,
        "cash_and_equivalents": 2_301_470_000.0,
        "cash_plus_short_term_investments": 2_301_470_000.0,
        "unavailable": False,
    }
    block = format_net_debt_block(nd)

    assert "Authoritative Net Debt: $5.89B" in block
    assert "## Net debt" in block
    # No accidental "Net Cash" relabel
    assert "Authoritative Net Cash" not in block


def test_format_net_debt_block_includes_period_crosscheck_note():
    """The block must explicitly tell the LLM that 10-Q cells in raw/sec_filing.md
    may be more current than balance_sheet col 0 — the AMD 2026-05-06 finding
    where decision.md cited 10-Q (Mar 28) cells while pm_brief had Dec 31 data,
    period mismatch never reconciled."""
    from tradingagents.agents.utils.net_debt import compute_net_debt, format_net_debt_block

    nd = compute_net_debt({"trade_date": "2026-05-06", "balance_sheet": _MSTR_BS})
    block = format_net_debt_block(nd)

    # The note must mention sec_filing.md and disclose the AMD-style period gap
    assert "sec_filing.md" in block
    assert "10-Q" in block
    assert "disclose" in block.lower() or "crosscheck" in block.lower() or "period" in block.lower()
