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


def test_output_contract_forbids_summary_emission():
    """The PM must be told its response IS decision.md, not a pointer to one."""
    from tradingagents.agents.managers.portfolio_manager import _OUTPUT_CONTRACT
    assert "Your entire response IS decision.md" in _OUTPUT_CONTRACT
    assert "DO NOT write" in _OUTPUT_CONTRACT
    assert "DO NOT emit a summary" in _OUTPUT_CONTRACT


def test_load_reference_block_pulls_canonical_values(tmp_path):
    """The reference block must surface raw/reference.json values verbatim
    so the PM cites reference_price + trade_date from a single source.
    """
    import json
    from tradingagents.agents.managers.portfolio_manager import _load_reference_block

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(json.dumps({
        "ticker": "MSFT",
        "trade_date": "2026-05-01",
        "reference_price": 410.0,
        "reference_price_source": "yfinance close on or before 2026-05-01",
        "spot_50dma": 405.0,
        "spot_200dma": 380.0,
        "ytd_high": 460.0,
        "ytd_low": 379.0,
        "atr_14": 4.2,
    }), encoding="utf-8")

    block = _load_reference_block({"raw_dir": str(raw)})
    assert "$410.0" in block
    assert "2026-05-01" in block
    assert "yfinance close on or before 2026-05-01" in block
    assert "verbatim" in block.lower()


def test_load_reference_block_returns_empty_when_raw_dir_missing():
    from tradingagents.agents.managers.portfolio_manager import _load_reference_block
    assert _load_reference_block({}) == ""
    assert _load_reference_block({"raw_dir": "/nonexistent"}) == ""


def test_pm_preflight_prompt_documents_fiscal_calendar():
    """PM Pre-flight must require a Fiscal calendar context section so the PM
    Final doesn't mislabel earnings quarters (e.g., calling MSFT's late-July
    print 'Q3 FY26' when it's Q4 FY26).
    """
    from tradingagents.agents.managers.pm_preflight import _SYSTEM
    assert "## Fiscal calendar context" in _SYSTEM
    assert "fiscal-quarter label" in _SYSTEM
    # The example in the prompt names the specific past mistake
    assert "Q4 FY26" in _SYSTEM
