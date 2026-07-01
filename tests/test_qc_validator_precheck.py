# tests/test_qc_validator_precheck.py
import pytest
from tradingagents.agents.managers.qc_agent import (
    _decision_blocking_violations,
    format_validator_feedback,
    VALIDATOR_RETRY_CAP,
)

pytestmark = pytest.mark.unit

# Shape mirrors run_phase_7_validators() output: per-phase dict with a "violations"
# list; each violation is a dict carrying "severity" and "file".
_RESULTS = {
    "phase_7_1_price_date": {"violations": [
        {"severity": "MATERIAL", "file": "decision.md", "type": "wrong_close",
         "claimed_date": "2026-06-29", "claimed_price": 359.90, "actual_close": 368.57,
         "match_text": "below Jun 29 close $359.90"},
        {"severity": "MATERIAL", "file": "analyst_fundamentals.md", "type": "wrong_close",
         "claimed_price": 100.0, "actual_close": 110.0},          # upstream -> excluded (v1)
        {"severity": "MINOR", "file": "decision.md", "type": "no_prices_data"},  # minor -> excluded
    ]},
    "phase_7_5_net_debt": {"violations": [
        {"severity": "MATERIAL", "file": "decision.md", "type": "definitional_drift",
         "claimed_dollars": 29000000000.0, "closest_canonical": 31423000000.0},
    ]},
    "phase_8_scenario_probability": {"violations": [
        {"severity": "MATERIAL", "type": "prob_sum", "detail": "probabilities sum to 95%, must be 100%"},
        # note: phase-8 violations may lack a "file" key; must still be kept
    ]},
    "total_violations": 5,
    "blocking_violations": 4,
}


def test_cap_is_two():
    assert VALIDATOR_RETRY_CAP == 2


def test_filter_keeps_only_decision_blocking_plus_scenario():
    keep = _decision_blocking_violations(_RESULTS)
    types = sorted(v["type"] for v in keep)
    # decision.md price + decision.md net-debt + scenario (fileless) => 3
    assert types == ["definitional_drift", "prob_sum", "wrong_close"]
    # the analyst_fundamentals.md and MINOR ones are excluded
    assert all(v.get("file", "decision.md") == "decision.md" or v.get("_phase", "").startswith("phase_8")
               for v in keep)


def test_empty_when_no_blocking():
    clean = {"phase_7_1_price_date": {"violations": [
        {"severity": "MINOR", "file": "decision.md", "type": "x"}]},
        "total_violations": 1, "blocking_violations": 0}
    assert _decision_blocking_violations(clean) == []


def test_formatter_actionable_lines():
    fb = format_validator_feedback(_decision_blocking_violations(_RESULTS))
    assert "368.57" in fb          # authoritative price surfaced
    assert "359.90" in fb          # claimed price surfaced
    assert "full document" in fb.lower() or "re-emit" in fb.lower()


def test_initial_state_has_qc_validator_retries():
    from tradingagents.graph.propagation import Propagator
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x")
    assert st["qc_validator_retries"] == 0
