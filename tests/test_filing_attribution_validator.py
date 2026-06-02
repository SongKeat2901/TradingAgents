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


_STUB2 = "⚠️ **XBRL ENCODING WARNING**: prose footnotes are NOT readable.\n"


def test_strip_fabricated_note_citations_keeps_numbers():
    from tradingagents.validators.filing_attribution_validator import (
        strip_fabricated_note_citations, validate_filing_attribution,
    )
    text = (
        "Buyback $163.8B. Source: raw/sec_filing.md Note 7 (\"remaining $63.8B\").\n"
        "Gross intangibles $37,767M vs $24,950M per Note 5 — flag.\n"
        "Services GM 76.7% (Note 2).\n"
    )
    out, n = strip_fabricated_note_citations(text, _STUB2)
    assert n >= 3
    # numbers preserved
    assert "$163.8B" in out and "$37,767M" in out and "76.7%" in out
    # no fabricated Note citations remain
    assert "Note 5" not in out and "Note 7" not in out and "Note 2" not in out
    # and the validator now passes on the stripped text
    assert validate_filing_attribution(out, "decision.md", _STUB2) == []


def test_strip_is_noop_for_readable_filing():
    from tradingagents.validators.filing_attribution_validator import (
        strip_fabricated_note_citations,
    )
    text = "Per Note 2 revenue rose."
    out, n = strip_fabricated_note_citations(text, _READABLE)
    assert out == text and n == 0


def test_substantiated_note_quote_not_flagged_under_xbrl_warning():
    """RKLB 2026-05-28: the fetched 10-Q carried the XBRL-warning header yet
    its body DID contain the cited note prose. A Note citation whose verbatim
    quote is present in the filing is substantiated, not fabricated, and must
    not be flagged — even though filing_is_xbrl_stub() is True."""
    from tradingagents.validators.filing_attribution_validator import (
        validate_filing_attribution,
    )
    # Stub warning header AND real note prose (the filing renders "$ 2,219,756").
    sec = (
        "> ⚠️ **XBRL ENCODING WARNING**: prose footnotes are NOT readable.\n\n"
        "3. REVENUES\nRemaining backlog totaled $ 2,219,756 as of March 31, 2026, "
        "of which approximately 36% is expected to be recognized within 12 months.\n"
    )
    text = (
        'Backlog $2,219,756K verified verbatim from 10-Q Note 3 '
        '("Remaining backlog totaled $2,219,756K").\n'
    )
    assert validate_filing_attribution(text, "decision.md", sec) == []


def test_fabricated_note_quote_still_flagged():
    """A Note citation whose quoted text is ABSENT from the filing remains a
    fabricated citation (AAPL-class protection preserved)."""
    from tradingagents.validators.filing_attribution_validator import (
        validate_filing_attribution,
    )
    text = 'Per Note 5 ("a long fabricated quote not in the filing at all here").\n'
    v = validate_filing_attribution(text, "decision.md", _STUB)
    assert len(v) == 1 and v[0].type == "fabricated_note_citation"
