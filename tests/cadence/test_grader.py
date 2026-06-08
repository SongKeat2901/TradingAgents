import pytest
from tradingagents.cadence.models import RunData
from tradingagents.cadence.grader import grade_run

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None):
    return RunData(ticker="T", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value=intrinsic or {},
                   peer_ratios={"ASX": {"nd_ebitda": 1.27}, "TSM": {"nd_ebitda": 0.33}},
                   financials={}, reference_price=10.0)


def test_all_dismissed_grades_A():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_dollars": 163000000000.0,
        "match_text": "$163B authorized buyback"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "A"
    assert len(rv.auto_dismissed) == 1
    assert rv.needs_adjudication == []


def test_null_fair_value_still_A():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MINOR", "type": "skipped_non_usd_reporter"}]}}
    rv = grade_run(_run(v, intrinsic={"fair_value": {"base": None}}))
    assert rv.grade == "A"


def test_unknown_blocking_flag_holds():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_dollars": 50000000000.0, "match_text": "Net debt is $50B"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "HOLD"
    assert len(rv.needs_adjudication) == 1


def test_minor_unknown_does_not_block():
    v = {"phase_x": {"violations": [{"severity": "MINOR", "type": "whatever"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "A"
