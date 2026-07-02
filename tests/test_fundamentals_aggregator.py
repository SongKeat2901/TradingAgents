import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.analysts.fundamentals_roles import create_fundamentals_aggregator


def _state(**kw):
    base = {"fundamentals_financial_report": "FIN", "fundamentals_riskflags_report": "RISK",
            "fundamentals_catalysts_report": "CAT", "fundamentals_quality_report": "QUAL"}
    base.update(kw)
    return base


def test_aggregates_all_four_in_order():
    out = create_fundamentals_aggregator()(_state())
    r = out["fundamentals_report"]
    assert r.index("FIN") < r.index("RISK") < r.index("CAT") < r.index("QUAL")
    assert "# Fundamentals" in r
    for h in ("Financial-Statement", "Risk & Red-Flags", "Catalysts & Ownership", "Competitive-Quality"):
        assert h in r


def test_missing_role_gets_placeholder_not_dropped():
    out = create_fundamentals_aggregator()(_state(fundamentals_catalysts_report=""))
    r = out["fundamentals_report"]
    assert "unavailable" in r.lower()
    assert "FIN" in r and "QUAL" in r  # others intact


def test_missing_key_does_not_raise():
    out = create_fundamentals_aggregator()({})  # no role keys at all
    assert "fundamentals_report" in out  # never raises
