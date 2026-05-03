"""Tests for the PM Pass-2 self-correction QC checklist."""
import pytest

pytestmark = pytest.mark.unit


def test_qc_checklist_in_pm_system_prompt():
    """The PM system prompt must include the 13 QC items."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST

    must_contain = [
        "sum to exactly 100%",          # item 1
        "specific dollar values",       # item 2
        "named, falsifiable catalyst",  # item 3
        "Rating logically derives",     # item 4
        "Execution triggers are falsifiable",  # item 5
        "reachable in at least one scenario",  # item 6 (Flaw 8)
        "reference_price",              # item 7 (Flaw 2)
        "verbatim",                     # item 8 (Flaw 3)
        "Cross-section numerical consistency",  # item 9 (Flaw 5)
        "Sanity-check flags",           # item 10 (Flaw 4)
        "Inputs section",               # item 11
        "Peer comparisons cite specific",  # item 12
        "trace back to",                # item 13
    ]
    for keyword in must_contain:
        assert keyword in _QC_CHECKLIST, f"QC checklist missing: {keyword}"


def test_qc_checklist_has_self_correction_directive():
    """The system prompt must instruct the PM to self-correct on failure."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST
    # The instruction to apply the checklist before final output
    assert "self-correct" in _QC_CHECKLIST.lower() or "revise" in _QC_CHECKLIST.lower()
