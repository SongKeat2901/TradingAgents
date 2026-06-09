import pytest
from tradingagents.cadence import publish

pytestmark = pytest.mark.unit


class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_refresh_invokes_updater_with_python(tmp_path):
    calls = []
    def runner(args, **kw):
        calls.append(args)
        return FakeProc(out="updated 9 rows")
    ok = publish.refresh_summary_sheet(python="/venv/bin/python",
                                       script="/u/update_register.py",
                                       account="acct", run=runner)
    assert ok is True
    assert calls[0][0] == "/venv/bin/python"
    assert "/u/update_register.py" in calls[0]


def test_refresh_returns_false_on_nonzero(tmp_path):
    runner = lambda args, **kw: FakeProc(rc=2, err="boom")
    assert publish.refresh_summary_sheet(python="p", script="s",
                                         account="a", run=runner) is False
