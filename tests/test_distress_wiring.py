import pytest
from tradingagents.agents.utils.distress_screens import compute_altman_z, format_distress_block

pytestmark = pytest.mark.unit


def test_block_composes_into_pm_brief(tmp_path):
    from tests.test_distress_screens import _HEALTHY
    block = format_distress_block(compute_altman_z(_HEALTHY))
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    assert "## Distress screen (Altman Z″)" in pm.read_text(encoding="utf-8")
