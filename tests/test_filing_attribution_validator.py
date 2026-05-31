"""Phase 9 P3 — filing-attribution validator tests."""

import pytest

pytestmark = pytest.mark.unit

_STUB = (
    "# SEC Filing — AAPL 10-Q\n\n"
    "> ⚠️ **XBRL ENCODING WARNING**: prose footnotes (e.g. \"Note 5\") are NOT "
    "available as readable text.\n\n**Do NOT cite specific Note numbers**.\n"
)
_READABLE = (
    "# SEC Filing — XYZ 10-Q\n\nNote 2 — Revenue. The company recognized ...\n"
    "Note 7 — Debt. Total borrowings were $5.0B ...\n"
)


def test_stub_filing_flags_note_citations():
    from tradingagents.validators.filing_attribution_validator import (
        validate_filing_attribution,
    )
    text = (
        "Per Note 2 the Services gross margin was 76.7%.\n"
        "Note 7 discloses the $163.8B buyback authorization.\n"
    )
    v = validate_filing_attribution(text, "decision.md", _STUB)
    assert len(v) == 2
    assert all(x.type == "fabricated_note_citation" and x.severity == "MATERIAL" for x in v)
    assert {x.line_no for x in v} == {1, 2}


def test_readable_filing_does_not_flag_notes():
    """When the filing has readable prose footnotes, citing Note N is fine."""
    from tradingagents.validators.filing_attribution_validator import (
        validate_filing_attribution,
    )
    text = "Per Note 2 revenue rose; Note 7 covers debt."
    assert validate_filing_attribution(text, "decision.md", _READABLE) == []


def test_no_filing_text_is_safe():
    from tradingagents.validators.filing_attribution_validator import (
        validate_filing_attribution,
    )
    assert validate_filing_attribution("Per Note 2 ...", "decision.md", None) == []
    assert validate_filing_attribution("Per Note 2 ...", "decision.md", "") == []


def test_stub_detection_helper():
    from tradingagents.validators.filing_attribution_validator import filing_is_xbrl_stub
    assert filing_is_xbrl_stub(_STUB) is True
    assert filing_is_xbrl_stub(_READABLE) is False
    assert filing_is_xbrl_stub(None) is False


def test_runner_gates_on_fabricated_note_citation(tmp_path):
    """End-to-end: a stub filing + a Note citation in decision.md must produce
    a blocking violation in run_phase_7_validators."""
    import json
    from cli.research_validation import run_phase_7_validators

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sec_filing.md").write_text(_STUB, encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({"company_of_interest": "AAPL"}), encoding="utf-8")
    (tmp_path / "decision.md").write_text(
        "Rating: Overweight.\n\nPer Note 5 the intangibles were $25.8B.\n", encoding="utf-8"
    )

    results = run_phase_7_validators(tmp_path)
    fa = results.get("phase_9_filing_attribution", {})
    assert fa.get("violations"), "filing-attribution violation not reported"
    assert results["blocking_violations"] >= 1
