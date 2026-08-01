"""Tests for the PM Pass-3 push-back retry mechanism."""
import pytest

pytestmark = pytest.mark.unit


def test_pm_system_prompt_documents_retry_signal():
    """PM must know how to emit a structured retry decision."""
    from tradingagents.agents.managers.portfolio_manager import _RETRY_DIRECTIVE
    assert "retry" in _RETRY_DIRECTIVE.lower()
    assert "research_manager" in _RETRY_DIRECTIVE
    assert "risk_team" in _RETRY_DIRECTIVE
    # Cap rule
    assert "max" in _RETRY_DIRECTIVE.lower() or "1" in _RETRY_DIRECTIVE


def test_research_manager_handles_pm_feedback():
    """Research Manager prompt must reference pm_feedback when set."""
    from tradingagents.agents.managers.research_manager import _PM_FEEDBACK_HANDLER
    assert "pm_feedback" in _PM_FEEDBACK_HANDLER
    assert "address" in _PM_FEEDBACK_HANDLER.lower()


def test_risk_debators_handle_pm_feedback():
    """Each risk debator must reference pm_feedback in its prompt."""
    from tradingagents.agents.risk_mgmt.aggressive_debator import _PM_FEEDBACK_HANDLER as agg
    from tradingagents.agents.risk_mgmt.conservative_debator import _PM_FEEDBACK_HANDLER as con
    from tradingagents.agents.risk_mgmt.neutral_debator import _PM_FEEDBACK_HANDLER as neu
    for handler in (agg, con, neu):
        assert "pm_feedback" in handler


def test_pm_brief_slice_includes_all_deterministic_valuation_sections():
    """wk31 cadence audits (TXN C / AMKR B / STM+MSFT provenance claims): the
    PM's prompt injection sliced only 4 section headers, so it never saw
    Accounting ratios / Relative valuation multiples / Intrinsic value — and
    for net-CASH tickers the section header reads "## Net cash", which the
    "## Net debt" matcher misses. The PM then honestly-but-wrongly disclaimed
    those blocks ("not supplied this run"), and under QC pressure escalated to
    a fabricated filesystem-search attestation (TXN)."""
    from tradingagents.agents.managers.portfolio_manager import _slice_pm_brief_sections

    brief = "\n".join([
        "# PM Pre-flight Brief",
        "prose intro",
        "## Peer ratios (computed from raw/peers.json, trade_date 2026-07-31)",
        "| peer | pe |",
        "## Net cash (computed from raw/financials.json balance_sheet, trade_date 2026-07-31, col 0 = quarter ending 2026-06-30)",
        "**Authoritative Net Cash: $2.01B**",
        "## Intrinsic value (computed from fundamentals & balance-sheet data, trade_date 2026-07-31)",
        "| base | $355.50 |",
        "## Accounting ratios (computed from raw/financials.json, trade_date 2026-07-31, latest quarter 2026-06-30)",
        "| ROE | 11.94% |",
        "## Relative valuation multiples (computed from raw/financials.json + raw/peer_ratios.json, trade_date 2026-07-31)",
        "| P/FCF | 50.02x |",
        "## 12-month scenario probabilities (block-bootstrap MC on 36-mo history)",
        "| bull | 0.54 |",
        "## Liquidity / Volume profile (computed from raw/prices.json)",
        "| VAH | 456.77 |",
        "## Institutional & insider ownership (yfinance, 13F-derived)",
        "| holder | pct |",
    ])
    sliced = _slice_pm_brief_sections(brief)
    assert "## Peer ratios" in sliced
    assert "Authoritative Net Cash: $2.01B" in sliced
    assert "## Accounting ratios" in sliced and "11.94%" in sliced
    assert "## Relative valuation multiples" in sliced and "50.02x" in sliced
    assert "## Intrinsic value" in sliced and "$355.50" in sliced
    assert "## 12-month scenario probabilities" in sliced
    assert "## Liquidity / Volume profile" in sliced
    assert "Institutional & insider ownership" not in sliced  # not a PM section


def test_pm_brief_slice_still_matches_net_debt_header():
    from tradingagents.agents.managers.portfolio_manager import _slice_pm_brief_sections

    brief = (
        "# Brief\n"
        "## Net debt (computed from raw/financials.json balance_sheet, trade_date 2026-07-31, col 0 = quarter ending 2026-06-30)\n"
        "**Authoritative Net Debt: $7.05B**\n"
        "## Filing surface\nx\n"
    )
    sliced = _slice_pm_brief_sections(brief)
    assert "Authoritative Net Debt: $7.05B" in sliced
    assert "Filing surface" not in sliced
