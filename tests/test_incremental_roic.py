import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.accounting_ratios import (
    compute_accounting_ratios, format_accounting_ratios_block, _incremental_roic,
)


def _fin(ebit, debt, eq, pretax=100, taxp=0):
    return {
        "annual_series": {"ebit": ebit, "total_debt": debt, "total_equity": eq,
                          "revenue": [], "diluted_eps": [], "fcf": []},
        "pretax_income_annual": pretax, "tax_provision_annual": taxp,
    }


def test_clean_incremental_roic_20pct():
    # EBIT 100->120, IC 500->600 (debt+equity), tax rate 0 -> ΔNOPAT 20 / ΔIC 100 = 20%
    pct, label, span = _incremental_roic(_fin([120, 100], [100, 100], [500, 400]))
    assert pct == 20.0 and span == "1y" and label == "eff 0%"


def test_tax_rate_applied():
    # rate = 25/100 = 0.25 -> ΔNOPAT = 20*0.75 = 15 / ΔIC 100 = 15%
    pct, label, _ = _incremental_roic(_fin([120, 100], [100, 100], [500, 400], pretax=100, taxp=25))
    assert pct == 15.0 and label == "eff 25%"


def test_tax_default_when_no_pretax():
    pct, label, _ = _incremental_roic(_fin([120, 100], [100, 100], [500, 400], pretax=None, taxp=None))
    assert label == "default 21%"
    assert pct == round((20 * 0.79) / 100 * 100, 1)  # 15.8


def test_non_positive_delta_ic_is_none():
    # IC shrinks (600 -> 500) -> ΔIC negative -> None
    pct, _, _ = _incremental_roic(_fin([120, 100], [100, 100], [400, 500]))
    assert pct is None


def test_fewer_than_two_years_is_none():
    pct, _, _ = _incremental_roic(_fin([120], [100], [500]))
    assert pct is None


def test_out_of_band_suppressed():
    # huge EBIT jump on tiny ΔIC -> >200% -> suppressed
    pct, _, _ = _incremental_roic(_fin([1000, 100], [100, 100], [301, 300]))
    assert pct is None


def test_block_renders_incremental_roic_row():
    block = format_accounting_ratios_block(
        compute_accounting_ratios(_fin([120, 100], [100, 100], [500, 400]), wacc=0.09, net_debt=None),
        trade_date="2026-07-01", as_of="2026-07-01")
    assert "Incremental ROIC" in block
