import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None, peers=None):
    return RunData(ticker="SUBJ", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value=intrinsic or {},
                   peer_ratios=peers or {}, financials={}, reference_price=10.0)


def test_ebitda_ttm_peer_guard_now_fires_escalates():
    # metric 'EBITDA (TTM)', claimed matches subject AND the named peer's ttm_ebitda
    # cell -> ambiguous -> must ESCALATE (previously dismissed: peer_key was None).
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "COMP", "metric": "EBITDA (TTM)", "claimed_value": "-$50M",
        "match_text": "COMP EBITDA (TTM) -$50M"}]}}
    run = _run(v, intrinsic={"inputs": {"ebitda": -50000000.0}},
               peers={"COMP": {"ttm_ebitda": -50000000.0}})
    verdicts = classify_run_flags(run)
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


def test_would_imply_no_longer_dismisses_genuine_drift():
    # 'would imply' (DCF prose) + boilerplate 'is authoritative' must NOT dismiss.
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 120000000000.0,
        "match_text": ("At net debt of $120B the EV/EBITDA would imply a distressed "
                       "valuation. Our $120B net debt figure is authoritative for this report.")}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


def test_aapl_broad_netcash_still_dismissed():
    # regression: the real AAPL disclosure uses 'broadest definition' + authoritative -> still DISMISS
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net cash", "claimed_dollars": 61900000000.0,
        "match_text": ("Including non-current marketable securities would put AAPL in net cash "
                       "territory: ~$61.9B net cash on the broadest definition. The $39.14B "
                       "yfinance figure is authoritative for this report.")}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS
