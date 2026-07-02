import pytest

pytestmark = pytest.mark.unit

from tradingagents.agents.utils.distress_screens import (
    compute_goodwill_flag, format_goodwill_block,
)


def test_elevated_by_equity():
    r = compute_goodwill_flag({"goodwill": 60, "total_assets": 200, "total_equity": 100})
    assert r["pct_equity"] == 60.0 and r["pct_assets"] == 30.0
    assert r["flag"] == "elevated"


def test_normal_when_small():
    r = compute_goodwill_flag({"goodwill": 5, "total_assets": 500, "total_equity": 100})
    assert r["flag"] == "normal"
    assert r["pct_equity"] == 5.0


def test_no_goodwill_reported():
    r = compute_goodwill_flag({"goodwill": None, "total_assets": 500, "total_equity": 100})
    assert r["reported"] is False
    assert "no goodwill reported" in format_goodwill_block(r).lower()


def test_missing_equity_degrades_to_na():
    r = compute_goodwill_flag({"goodwill": 40, "total_assets": None, "total_equity": 0})
    assert r["pct_equity"] is None and r["pct_assets"] is None
    assert r["flag"] == "normal"  # no ratio crossed a threshold (both undefined)
    block = format_goodwill_block(r)
    assert "n/a (data unavailable)" in block


def test_block_renders_flag_and_verbatim_mandate():
    block = format_goodwill_block(compute_goodwill_flag(
        {"goodwill": 60, "total_assets": 200, "total_equity": 100}))
    assert "## Goodwill / impairment screen" in block
    assert "elevated" in block and "verbatim" in block
