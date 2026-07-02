"""Pro-deck technique E: bear-concern -> reframe-with-numbers voice.

The deck's whole method (pp63-70): open each theme with the market's top bear
concern, then test it against the numbers. Our version must stay two-sided —
when the data does NOT defuse the concern, the role says the concern stands
(research honesty, not promotion).
"""
import pytest

from tradingagents.agents.analysts.fundamentals_roles import (
    _REQUIRED_CATALYSTS,
    _REQUIRED_FINANCIAL,
    _REQUIRED_QUALITY,
    _REQUIRED_RISK,
    _SYSTEM_CATALYSTS,
    _SYSTEM_FINANCIAL,
    _SYSTEM_QUALITY,
    _SYSTEM_RISK,
)

pytestmark = pytest.mark.unit

_ALL_SYSTEMS = {
    "financial": _SYSTEM_FINANCIAL,
    "risk": _SYSTEM_RISK,
    "catalysts": _SYSTEM_CATALYSTS,
    "quality": _SYSTEM_QUALITY,
}


def test_every_role_has_the_bear_reframe_section():
    for name, system in _ALL_SYSTEMS.items():
        assert "## Top bear concern, tested" in system, f"{name} missing bear-reframe section"


def test_directive_demands_numbers_not_narrative():
    for name, system in _ALL_SYSTEMS.items():
        para = system.split("## Top bear concern, tested", 1)[1][:1200]
        assert "number" in para.lower(), f"{name} reframe not numbers-grounded"


def test_directive_keeps_two_sided_honesty():
    # must allow the bear case to WIN: "concern stands" fallback, no forced bull spin
    for name, system in _ALL_SYSTEMS.items():
        para = system.split("## Top bear concern, tested", 1)[1][:1200]
        assert "stands" in para.lower(), f"{name} reframe lacks the concern-stands fallback"


def test_bear_reframe_is_required_in_all_roles():
    for req in (_REQUIRED_FINANCIAL, _REQUIRED_RISK, _REQUIRED_CATALYSTS, _REQUIRED_QUALITY):
        assert "## Top bear concern, tested" in req
