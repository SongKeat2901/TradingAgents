import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation):
    return RunData(ticker="ASX", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value={}, peer_ratios={},
                   financials={}, reference_price=34.03)


def test_from_to_miswire_is_dismissed():
    v = {"phase_7_1_price_date": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_close",
        "claimed_date": "2026-05-28", "claimed_price": 34.03, "actual_close": 40.6,
        "match_text": "ine from the $40.60 May 28 close to $34.03 compressed RSI from"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS
    assert "from" in verdicts[0].reason.lower()


def test_genuine_wrong_close_escalates():
    v = {"phase_7_1_price_date": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_close",
        "claimed_date": "2026-05-28", "claimed_price": 34.03, "actual_close": 40.6,
        "match_text": "the May 28 close was $34.03 per my notes"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
