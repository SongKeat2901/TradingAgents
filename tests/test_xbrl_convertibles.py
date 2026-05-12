"""Tests for the Phase 7.11 XBRL convertibles extractor."""
import pytest

pytestmark = pytest.mark.unit


# Minimal inline-XBRL fixture modeled on MARA's actual 10-Q structure.
# Two tranches with face amounts, conversion prices, coupons, ratios.
_FIXTURE_HTML = """
<html xmlns:ix="..." xmlns:xbrli="..." xmlns:xbrldi="..." xmlns:us-gaap="..." xmlns:mara="...">
<xbrli:context id="c_dec2026_period">
  <xbrldi:explicitMember dimension="us-gaap:LongtermDebtTypeAxis">
    mara:ConvertibleSeniorNotesDueDecember2026Member
  </xbrldi:explicitMember>
</xbrli:context>
<xbrli:context id="c_aug2032_period">
  <xbrldi:explicitMember dimension="us-gaap:LongtermDebtTypeAxis">
    mara:ConvertibleSeniorNotesDueAugust2032Member
  </xbrldi:explicitMember>
</xbrli:context>
<xbrli:context id="c_unrelated">
  <xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">
    us-gaap:CommonStockMember
  </xbrldi:explicitMember>
</xbrli:context>

<ix:nonFraction name="us-gaap:DebtInstrumentFaceAmount" contextRef="c_dec2026_period" scale="6" decimals="-6" unitRef="usd">747.50</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentConvertibleConversionPrice1" contextRef="c_dec2026_period" decimals="2" unitRef="usdPerShare">76.17</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentConvertibleConversionRatio1" contextRef="c_dec2026_period" decimals="6">0.013128</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentInterestRateStatedPercentage" contextRef="c_dec2026_period" decimals="4">0.0100</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentInterestRateEffectivePercentage" contextRef="c_dec2026_period" decimals="4">0.0100</ix:nonFraction>

<ix:nonFraction name="us-gaap:DebtInstrumentFaceAmount" contextRef="c_aug2032_period" scale="6" decimals="-6" unitRef="usd">1025.00</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentConvertibleConversionPrice1" contextRef="c_aug2032_period" decimals="2" unitRef="usdPerShare">20.26</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentConvertibleConversionRatio1" contextRef="c_aug2032_period" decimals="6">0.049362</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentInterestRateStatedPercentage" contextRef="c_aug2032_period" decimals="4">0.0000</ix:nonFraction>
<ix:nonFraction name="us-gaap:DebtInstrumentInterestRateEffectivePercentage" contextRef="c_aug2032_period" decimals="4">0.0010</ix:nonFraction>

<!-- Unrelated fact in a non-tranche context: should NOT be extracted -->
<ix:nonFraction name="us-gaap:DebtInstrumentFaceAmount" contextRef="c_unrelated" scale="6">999.99</ix:nonFraction>
</html>
"""


def test_extract_returns_empty_for_html_without_xbrl():
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    assert extract_convertibles_from_html("") == []
    assert extract_convertibles_from_html("<html><p>Just prose</p></html>") == []


def test_extract_returns_one_dict_per_tranche():
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    assert len(result) == 2
    tranches = {t["tranche"] for t in result}
    assert tranches == {"December2026", "August2032"}


def test_extract_handles_scale_attribute():
    """scale=6 on a value 747.50 → $747,500,000 (= 747.5 × 10^6)."""
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    by_tranche = {t["tranche"]: t for t in result}
    assert by_tranche["December2026"]["face_amount"] == 747_500_000.0
    assert by_tranche["August2032"]["face_amount"] == 1_025_000_000.0


def test_extract_captures_conversion_price_and_ratio():
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    by_tranche = {t["tranche"]: t for t in result}
    assert by_tranche["December2026"]["conversion_price"] == 76.17
    assert abs(by_tranche["December2026"]["conversion_ratio"] - 0.013128) < 1e-7
    assert by_tranche["August2032"]["conversion_price"] == 20.26


def test_extract_captures_interest_rates():
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    by_tranche = {t["tranche"]: t for t in result}
    assert abs(by_tranche["December2026"]["interest_rate_stated"] - 0.0100) < 1e-7
    assert abs(by_tranche["August2032"]["interest_rate_stated"] - 0.0000) < 1e-7
    assert abs(by_tranche["August2032"]["interest_rate_effective"] - 0.0010) < 1e-7


def test_extract_skips_facts_in_non_tranche_context():
    """The $999.99 fact in c_unrelated context (CommonStockMember) must
    NOT appear in any tranche bucket."""
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    # No tranche should have $999.99M face
    for t in result:
        if t["face_amount"]:
            assert t["face_amount"] != 999_990_000.0


def test_extract_sorts_by_face_amount_descending():
    """Largest tranche first — operators scanning the markdown block see
    the biggest exposure at the top."""
    from tradingagents.agents.utils.xbrl_convertibles import extract_convertibles_from_html
    result = extract_convertibles_from_html(_FIXTURE_HTML)
    # August2032 ($1025M) should come before December2026 ($747M)
    assert result[0]["tranche"] == "August2032"
    assert result[1]["tranche"] == "December2026"


def test_format_block_returns_empty_for_no_tranches():
    from tradingagents.agents.utils.xbrl_convertibles import format_convertibles_block
    assert format_convertibles_block([]) == ""


def test_format_block_renders_table_with_required_columns():
    from tradingagents.agents.utils.xbrl_convertibles import (
        extract_convertibles_from_html,
        format_convertibles_block,
    )
    tranches = extract_convertibles_from_html(_FIXTURE_HTML)
    block = format_convertibles_block(tranches, spot=12.94, ticker="MARA")
    # Block must include the per-tranche table
    assert "August2032" in block
    assert "December2026" in block
    assert "$1.02B" in block  # face amount rendered ($1025M → $1.02B at 2dp)
    assert "$76.17" in block  # conversion price
    assert "$20.26" in block
    # Rally-to-ITM column when spot provided
    assert "Rally to ITM" in block
    # August2032 rally: ($20.26 - $12.94) / $12.94 = +56.6%
    assert "+56." in block
    # Total face amount summary
    assert "Total face amount" in block


def test_format_block_omits_rally_column_when_spot_missing():
    """When spot price isn't provided, the rally-to-ITM column shouldn't
    appear (the block is still useful, just without the comparison)."""
    from tradingagents.agents.utils.xbrl_convertibles import (
        extract_convertibles_from_html,
        format_convertibles_block,
    )
    tranches = extract_convertibles_from_html(_FIXTURE_HTML)
    block = format_convertibles_block(tranches, ticker="MARA")
    assert "Rally to ITM" not in block
