import pytest
from tradingagents.cadence.models import (
    FlagDisposition, FlagVerdict, RunData, RunVerdict,
)

pytestmark = pytest.mark.unit


def test_flag_disposition_values():
    assert FlagDisposition.DISMISS.value == "dismiss"
    assert FlagDisposition.CORRECT_BY_DESIGN.value == "correct"
    assert FlagDisposition.NEEDS_ADJUDICATION.value == "adjudicate"


def test_flag_verdict_defaults():
    fv = FlagVerdict(phase="phase_7_5_net_debt",
                     disposition=FlagDisposition.DISMISS, reason="x")
    assert fv.detail == {}


def test_run_verdict_holds_lists():
    rv = RunVerdict(ticker="AAPL", grade="A", flag_verdicts=[],
                    auto_dismissed=[], needs_adjudication=[])
    assert rv.grade == "A"
    assert rv.needs_adjudication == []
