import json
import pytest
from tradingagents.agents.utils.financials_parser import parse_financials
from tradingagents.agents.utils.accounting_ratios import compute_accounting_ratios, format_accounting_ratios_block
from tradingagents.agents.utils.relative_multiples import compute_relative_multiples, format_relative_multiples_block

pytestmark = pytest.mark.unit


def test_blocks_compose_end_to_end(tmp_path):
    # This mirrors exactly what researcher.py will do inline.
    from tests.test_financials_parser import _BUNDLE
    fin = parse_financials(_BUNDLE)
    ar = compute_accounting_ratios(fin, wacc=0.09, net_debt={"net_debt": 8e9})
    rm = compute_relative_multiples(fin, market_cap=fin["market_cap"], net_debt=8e9, peers={})
    ar_block = format_accounting_ratios_block(ar, fin["trade_date"], fin["as_of_quarter"])
    rm_block = format_relative_multiples_block(rm, fin["trade_date"])
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(ar_block)
        f.write(rm_block)
    text = pm.read_text(encoding="utf-8")
    assert "## Accounting ratios" in text
    assert "## Relative valuation multiples" in text


def test_blocks_compose_when_intrinsic_value_unavailable(tmp_path):
    # Regression for F1: an intrinsic-value failure leaves `iv = None`
    # (researcher.py initializes `iv = None` before the IV try-block, so a
    # raise inside compute_intrinsic_value never leaves `iv` unbound). Both
    # accounting-ratios and relative-multiples are independent of `iv` — they
    # only use it opportunistically for wacc/market_cap — so they must still
    # render even in this degraded state, using wacc=None and market_cap
    # falling back to fin_parsed["market_cap"].
    from tests.test_financials_parser import _BUNDLE
    iv = None  # simulates compute_intrinsic_value raising before assignment
    fin = parse_financials(_BUNDLE)
    wacc = (iv.get("inputs", {}) or {}).get("wacc") if isinstance(iv, dict) else None
    assert wacc is None
    ar = compute_accounting_ratios(fin, wacc=wacc, net_debt={"net_debt": 8e9})

    mc = (iv.get("inputs", {}) or {}).get("market_cap") if isinstance(iv, dict) else None
    if mc is None:
        mc = fin.get("market_cap")
    assert mc == fin["market_cap"]
    rm = compute_relative_multiples(fin, market_cap=mc, net_debt=8e9, peers={},
                                     forward_eps=fin.get("forward_eps"))

    ar_block = format_accounting_ratios_block(ar, fin["trade_date"], fin["as_of_quarter"])
    rm_block = format_relative_multiples_block(rm, fin["trade_date"])
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(ar_block)
        f.write(rm_block)
    text = pm.read_text(encoding="utf-8")
    assert "## Accounting ratios" in text
    assert "## Relative valuation multiples" in text
