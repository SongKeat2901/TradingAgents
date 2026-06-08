import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation):
    peers = {"ASX": {"nd_ebitda": 1.27}, "TSM": {"nd_ebitda": 0.33}}
    return RunData(ticker="AMKR", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value={}, peer_ratios=peers,
                   financials={}, reference_price=64.95)


def test_respectively_mismap_is_dismissed():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "1.27x",
        "actual_value": "0.33",
        "match_text": "Net Debt $293M vs. ASX $159.79B, TSM $936.16B; "
                      "ND/EBITDA 1.27x and 0.33x respectively)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_peer_metric_no_respectively_escalates():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "1.27x",
        "actual_value": "0.33",
        "match_text": "TSM ND/EBITDA is 1.27x."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


def test_peer_metric_value_matches_no_other_peer_escalates():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "9.99x",
        "actual_value": "0.33",
        "match_text": "ND/EBITDA 9.99x and 0.33x respectively)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
