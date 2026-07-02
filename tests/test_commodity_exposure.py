import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.commodity_exposure import (
    compute_commodity_exposure, format_commodity_block,
)


def test_energy_high():
    r = compute_commodity_exposure({"sector": "Energy", "industry": "Oil & Gas E&P"})
    assert r["exposure"] == "high"
    assert "crude" in r["primary_inputs"].lower()


def test_airline_high_jet_fuel():
    r = compute_commodity_exposure({"sector": "Industrials", "industry": "Airlines"})
    assert r["exposure"] == "high" and "jet fuel" in r["primary_inputs"]


def test_software_low():
    r = compute_commodity_exposure({"sector": "Technology", "industry": "Software—Infrastructure"})
    assert r["exposure"] == "low"
    assert r["classified"] is True


def test_unclassified_low_honest():
    r = compute_commodity_exposure({"sector": "", "industry": ""})
    assert r["exposure"] == "low" and r["classified"] is False


def test_industry_beats_sector_specificity():
    # Airline sits under the Industrials sector, but the airline industry keyword
    # (jet fuel, high) must win over the generic industrials (moderate) rule.
    r = compute_commodity_exposure({"sector": "Industrials", "industry": "Airlines"})
    assert r["exposure"] == "high"


def test_block_renders_verbatim_mandate():
    block = format_commodity_block(compute_commodity_exposure(
        {"sector": "Basic Materials", "industry": "Steel"}))
    assert "## Commodity input exposure" in block
    assert "high" in block and "verbatim" in block
