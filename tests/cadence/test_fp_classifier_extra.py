import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None, peers=None):
    return RunData(ticker="SUBJ", trade_date="2026-06-05", run_dir="/x",
                   validation=validation,
                   intrinsic_value=intrinsic or {},
                   peer_ratios=peers or {}, financials={}, reference_price=10.0)


# ---- Pattern A: subject's own metric mis-attributed to a peer ----
def test_subject_metric_misattributed_to_peer_is_dismissed():
    # −$27M is the SUBJECT's own EBITDA; validator blamed peer LITE.
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "LITE", "metric": "TTM EBITDA", "claimed_value": "−$27M",
        "match_text": "| Net Debt / EBITDA | (n/m) TTM EBITDA −$27M; Net Cash $159M | 1.22x | ... |"}]}}
    run = _run(v, intrinsic={"inputs": {"ebitda": -27000000.0}},
               peers={"LITE": {"ttm_ebitda": 509000000.0}})
    verdicts = classify_run_flags(run)
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_peer_metric_matching_peer_not_subject_escalates():
    # claimed matches the PEER's cell, not the subject -> a real peer claim -> escalate
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "LITE", "metric": "TTM EBITDA", "claimed_value": "$509M",
        "match_text": "LITE TTM EBITDA $509M is healthy."}]}}
    run = _run(v, intrinsic={"inputs": {"ebitda": -27000000.0}},
               peers={"LITE": {"ttm_ebitda": 509000000.0}})
    verdicts = classify_run_flags(run)
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


def test_peer_metric_no_subject_data_escalates():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "LITE", "metric": "TTM EBITDA", "claimed_value": "−$27M",
        "match_text": "TTM EBITDA −$27M somewhere."}]}}
    run = _run(v, intrinsic={}, peers={})   # no subject inputs -> can't confirm -> escalate
    verdicts = classify_run_flags(run)
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


# ---- Pattern B: broad-definition net-cash disclosure ----
def test_broad_netcash_disclosure_is_dismissed():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net cash", "claimed_dollars": 61900000000.0,
        "match_text": ("Including non-current marketable securities ($78.09B) would imply AAPL is "
                       "in net cash territory: approximately $61.9B net cash on the broadest "
                       "definition. The $39.14B yfinance figure is authoritative for this report.")}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_plain_netdebt_drift_without_disclosure_markers_escalates():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 50000000000.0,
        "match_text": "Net debt is $50B per the balance sheet."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
