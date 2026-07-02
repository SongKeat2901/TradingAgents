"""Forward-EPS × exit-multiple price-target grid (pro-deck technique A).

Deck reference: Tiger 30-Jun FA Outlook p69 — project EPS on the consensus
growth path, price it at 20x/25x/30x, and show implied-P/E compression at a
flat price. Pure compute (no network) so the grid is unit-testable.
"""
import pytest

from tradingagents.agents.utils.eps_scenario import (
    compute_eps_scenario,
    format_eps_scenario_block,
)

pytestmark = pytest.mark.unit


_EST = {"eps_0y": 8.0, "eps_1y": 10.0, "growth_1y": 0.25,
        "n_analysts_0y": 40, "n_analysts_1y": 38, "source": "yfinance earnings_estimate"}


def test_happy_path_grid():
    g = compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=6.0)
    assert g["available"] is True
    years = g["years"]
    assert len(years) == 4
    # +1y EPS is the consensus +1y number verbatim, not extrapolated
    assert years[0]["eps"] == pytest.approx(10.0)
    # +2..+4y extrapolate at growth_1y
    assert years[1]["eps"] == pytest.approx(12.5)
    assert years[2]["eps"] == pytest.approx(15.625, abs=0.01)
    assert years[3]["eps"] == pytest.approx(19.53, abs=0.01)
    # price at multiples
    assert years[0]["price_at_20x"] == pytest.approx(200.0)
    assert years[0]["price_at_25x"] == pytest.approx(250.0)
    assert years[0]["price_at_30x"] == pytest.approx(300.0)
    # implied P/E at flat price compresses monotonically
    pes = [y["implied_pe_flat_price"] for y in years]
    assert pes[0] == pytest.approx(20.0)
    assert pes == sorted(pes, reverse=True)


def test_current_pe_anchor_row_present_when_trailing_eps_given():
    g = compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=6.0)
    # trailing P/E 33.33x anchors the "multiple holds" scenario
    assert g["inputs"]["current_pe_ttm"] == pytest.approx(33.33, abs=0.01)
    assert g["years"][0]["price_at_current_pe"] == pytest.approx(333.33, abs=0.1)


def test_current_pe_omitted_when_trailing_eps_absent_or_nonpositive():
    g = compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=None)
    assert g["inputs"]["current_pe_ttm"] is None
    assert "price_at_current_pe" not in g["years"][0]
    g2 = compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=-1.0)
    assert g2["inputs"]["current_pe_ttm"] is None


def test_growth_fallback_derived_from_0y_to_1y_when_growth_missing():
    est = dict(_EST, growth_1y=None)
    g = compute_eps_scenario(price=200.0, estimates=est)
    assert g["available"] is True
    # implied growth = 10/8 - 1 = 25%
    assert g["inputs"]["growth"] == pytest.approx(0.25)
    assert "derived" in g["inputs"]["growth_source"]


def test_eps_1y_fallback_to_forward_eps():
    est = {"eps_0y": None, "eps_1y": None, "growth_1y": 0.20}
    g = compute_eps_scenario(price=100.0, estimates=est, forward_eps=5.0)
    assert g["available"] is True
    assert g["years"][0]["eps"] == pytest.approx(5.0)


def test_unavailable_when_no_eps():
    g = compute_eps_scenario(price=100.0, estimates={"eps_0y": None, "eps_1y": None, "growth_1y": 0.2})
    assert g["available"] is False
    assert "EPS" in g["reason"]


def test_unavailable_when_eps_nonpositive():
    est = dict(_EST, eps_0y=-2.0, eps_1y=-1.0)
    g = compute_eps_scenario(price=100.0, estimates=est)
    assert g["available"] is False


def test_unavailable_when_no_growth_at_all():
    est = {"eps_0y": None, "eps_1y": 5.0, "growth_1y": None}
    g = compute_eps_scenario(price=100.0, estimates=est)
    assert g["available"] is False
    assert "growth" in g["reason"].lower()


def test_unavailable_on_implausible_growth():
    est = dict(_EST, growth_1y=0.9)  # >60%/yr — 4y extrapolation not meaningful
    g = compute_eps_scenario(price=100.0, estimates=est)
    assert g["available"] is False
    assert "growth" in g["reason"].lower()
    est2 = dict(_EST, growth_1y=-0.6)
    assert compute_eps_scenario(price=100.0, estimates=est2)["available"] is False


def test_unavailable_when_price_missing():
    g = compute_eps_scenario(price=None, estimates=_EST)
    assert g["available"] is False


def test_format_block_happy_path():
    g = compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=6.0)
    block = format_eps_scenario_block(g, "2026-07-01")
    assert "## Forward-EPS price-target grid" in block
    assert "2026-07-01" in block
    assert "+1y" in block and "+4y" in block
    assert "20x" in block and "25x" in block and "30x" in block
    assert "Implied P/E" in block
    # honest labeling: out-years are an extrapolation, not guidance
    assert "extrapolat" in block.lower()
    assert "verbatim" in block  # cite-verbatim mandate
    # values from the grid appear
    assert "10.00" in block and "12.50" in block


def test_format_block_unavailable():
    g = compute_eps_scenario(price=100.0, estimates={"eps_0y": None, "eps_1y": None, "growth_1y": None})
    block = format_eps_scenario_block(g, "2026-07-01")
    assert "## Forward-EPS price-target grid" in block
    assert "unavailable" in block
    assert "Do not cite" in block


def test_block_composes_into_pm_brief(tmp_path):
    block = format_eps_scenario_block(
        compute_eps_scenario(price=200.0, estimates=_EST, trailing_eps=6.0), "2026-07-01")
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    assert "## Forward-EPS price-target grid" in pm.read_text(encoding="utf-8")


def test_researcher_wires_eps_scenario():
    import inspect
    from tradingagents.agents import researcher
    src = inspect.getsource(researcher)
    assert "eps_scenario" in src
    assert "fetch_eps_estimates" in src
    assert "format_eps_scenario_block" in src


def test_financial_role_prompt_cites_grid():
    from tradingagents.agents.analysts.fundamentals_roles import _SYSTEM_FINANCIAL
    assert "Forward-EPS price-target grid" in _SYSTEM_FINANCIAL
