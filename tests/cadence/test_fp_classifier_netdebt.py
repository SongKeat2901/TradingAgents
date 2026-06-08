import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None):
    return RunData(ticker="AAPL", trade_date="2026-06-05", run_dir="/x",
                   validation=validation,
                   intrinsic_value=intrinsic or {"inputs": {"net_debt": 39139000000.0}},
                   peer_ratios={}, financials={}, reference_price=177.0)


def test_buyback_dollar_grab_is_dismissed():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 163000000000.0,
        "match_text": "$39.14B Net Debt against a >$163B authorized buyback and "
                      "$82.6B H1 operating cash flow indicates no balance-sheet stress."}]}}
    verdicts = classify_run_flags(_run(v))
    assert len(verdicts) == 1
    assert verdicts[0].disposition is FlagDisposition.DISMISS
    assert "buyback" in verdicts[0].reason.lower()


def test_operating_cashflow_fcf_grab_is_dismissed():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net cash", "claimed_dollars": 2540000000.0,
        "match_text": "FCF Q1 2026 = -$2,540M: Net cash from operating activities "
                      "$1,096M - capex $3,636M = -$2,540M (cashflow col 0)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_non_usd_skip_is_correct_by_design():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MINOR", "type": "skipped_non_usd_reporter"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.CORRECT_BY_DESIGN


def test_unknown_definitional_drift_escalates():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 50000000000.0,
        "match_text": "Net debt is $50B per the balance sheet."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
