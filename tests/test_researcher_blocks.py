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
