"""Pro-deck technique C: capex funding bridge + FCF-inflection thesis + dated
"what to watch" catalyst (deck pp64-65).

These are earnings-call / guidance-derived items with NO free structured
source, so the honest mechanism is grounded role directives: quote the filing
(targeted excerpts) / news, else "not disclosed" — never fabricate a funding
split or a guidance number.
"""
import pytest

from tradingagents.agents.analysts.fundamentals_roles import (
    _REQUIRED_CATALYSTS,
    _SYSTEM_CATALYSTS,
    _SYSTEM_FINANCIAL,
)

pytestmark = pytest.mark.unit


def test_financial_role_has_capex_funding_bridge_directive():
    assert "Capex funding bridge" in _SYSTEM_FINANCIAL
    # grounded in the targeted excerpts, with the honest fallback
    assert "customer prepayment" in _SYSTEM_FINANCIAL
    assert "not disclosed" in _SYSTEM_FINANCIAL
    # the deck's framing: decompose the scary number into funding sources
    assert "funding" in _SYSTEM_FINANCIAL


def test_financial_role_has_fcf_inflection_directive():
    assert "FCF trajectory" in _SYSTEM_FINANCIAL
    # trajectory must come from actual quarterly columns, not memory
    assert "quarterly" in _SYSTEM_FINANCIAL
    assert "inflection" in _SYSTEM_FINANCIAL.lower()


def test_catalysts_role_has_dated_inflection_section():
    assert "## Dated inflection to watch" in _SYSTEM_CATALYSTS
    # grounded in the deterministic calendar block
    assert "Reporting status" in _SYSTEM_CATALYSTS
    assert "Next expected" in _SYSTEM_CATALYSTS
    # falsifiable framing: name the metric that confirms/refutes at the event
    assert "confirm" in _SYSTEM_CATALYSTS.lower()


def test_dated_inflection_is_a_required_header():
    assert "## Dated inflection to watch" in _REQUIRED_CATALYSTS


def test_capex_bridge_paragraph_forbids_fabricated_guidance():
    # the capex-bridge directive ITSELF must forbid inventing guidance numbers
    paras = [p for p in _SYSTEM_FINANCIAL.split("\n\n") if "Capex funding bridge" in p]
    assert paras, "capex-bridge directive paragraph missing"
    low = paras[0].lower()
    assert "do not" in low and ("invent" in low or "fabricat" in low)
