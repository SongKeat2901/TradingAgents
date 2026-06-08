import pytest
from tradingagents.cadence.batch import find_latest_batch

pytestmark = pytest.mark.unit


def _mk(base, name, with_decision=True):
    d = base / name
    d.mkdir(parents=True)
    if with_decision:
        (d / "decision.md").write_text("- **Reference price:** $1.00\n")
    return d


def test_picks_newest_date_completed_only_excludes_dotted(tmp_path):
    pre = tmp_path / "preaudit"
    _mk(pre, "2026-06-04-AAA")
    _mk(pre, "2026-06-05-BBB")
    _mk(pre, "2026-06-05-CCC", with_decision=False)
    _mk(pre, "2026-06-05-DDD.pre-cadence")
    date, runs = find_latest_batch(pre)
    assert date == "2026-06-05"
    assert [r.name for r in runs] == ["2026-06-05-BBB"]


def test_empty_base_returns_none(tmp_path):
    pre = tmp_path / "preaudit"
    pre.mkdir()
    assert find_latest_batch(pre) == (None, [])
