# tests/test_recent_closes_wiring.py
import pytest
from tradingagents.agents.utils.recent_closes import compute_recent_closes, format_recent_closes_block

pytestmark = pytest.mark.unit


def test_block_composes_into_pm_brief(tmp_path):
    from tests.test_recent_closes import _PRICES
    rc = compute_recent_closes(_PRICES, "2026-06-30")
    block = format_recent_closes_block(rc)
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    text = pm.read_text(encoding="utf-8")
    assert "## Recent closes" in text
    assert "| 2026-06-29 | $368.57 |" in text
