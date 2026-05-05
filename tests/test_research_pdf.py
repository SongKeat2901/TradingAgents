"""Tests for the cover-page model label resolution in cli.research_pdf.

The PDF cover page used to hardcode "Opus 4.6 judges · Haiku 4.5 analysts"
regardless of the actual run config. The c5c41e4 audit (2026-05-05) caught
this: real runs may use Sonnet, Opus 4.7, etc., and the cover page misled.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_humanize_model_id_renders_canonical_names():
    from cli.research_pdf import _humanize_model_id

    assert _humanize_model_id("claude-opus-4-6") == "Opus 4.6"
    assert _humanize_model_id("claude-sonnet-4-6") == "Sonnet 4.6"
    assert _humanize_model_id("claude-haiku-4-5") == "Haiku 4.5"
    assert _humanize_model_id("claude-opus-4-7") == "Opus 4.7"


def test_humanize_model_id_falls_back_to_raw_id_for_unknown_pattern():
    from cli.research_pdf import _humanize_model_id

    assert _humanize_model_id("gpt-5.4-mini") == "gpt-5.4-mini"
    assert _humanize_model_id("custom-thing") == "custom-thing"


def test_humanize_model_id_handles_none_and_empty():
    from cli.research_pdf import _humanize_model_id

    assert _humanize_model_id(None) == "(unknown)"
    assert _humanize_model_id("") == "(unknown)"


def test_resolve_model_label_reads_state_json_meta(tmp_path):
    """When state.json has _meta with deep/quick model ids, the cover label
    renders 'Opus 4.7 judges · Sonnet 4.6 analysts' (or whatever was used)."""
    from cli.research_pdf import _resolve_model_label

    (tmp_path / "state.json").write_text(json.dumps({
        "company_of_interest": "MSFT",
        "_meta": {
            "deep_think_llm": "claude-opus-4-7",
            "quick_think_llm": "claude-sonnet-4-6",
            "llm_provider": "claude_code",
        },
    }), encoding="utf-8")

    label = _resolve_model_label(tmp_path)
    assert label == "Opus 4.7 judges · Sonnet 4.6 analysts"


def test_resolve_model_label_falls_back_when_meta_missing(tmp_path):
    """Old runs that pre-date the _meta block must not break PDF generation."""
    from cli.research_pdf import _resolve_model_label

    (tmp_path / "state.json").write_text(json.dumps({
        "company_of_interest": "MSFT",
    }), encoding="utf-8")

    assert _resolve_model_label(tmp_path) == "(model not recorded)"


def test_resolve_model_label_falls_back_when_state_json_missing(tmp_path):
    """No state.json at all → fall back gracefully."""
    from cli.research_pdf import _resolve_model_label

    assert _resolve_model_label(tmp_path) == "(model not recorded)"


def test_resolve_model_label_falls_back_on_malformed_json(tmp_path):
    """Truncated / corrupt state.json must not crash PDF generation."""
    from cli.research_pdf import _resolve_model_label

    (tmp_path / "state.json").write_text("{ not valid json", encoding="utf-8")
    assert _resolve_model_label(tmp_path) == "(model not recorded)"


# ============================================================================
# Phase 6.5 — Executive-format PDF: vocabulary cleanup + summary extractor
# ============================================================================


def test_clean_agentic_vocabulary_strips_agent_role_names():
    """Multi-agent role names ('Trader's', 'Aggressive Risk Analyst', etc.)
    must be replaced with executive-friendly equivalents in front-of-document
    text. The user wants the operational language confined to the appendix."""
    from cli.research_pdf import _clean_agentic_vocabulary

    src = (
        "The Trader's HOLD proposal at $413.62 is correctly aligned. "
        "The Research Manager's verdict adopts the same trigger architecture. "
        "All three risk analysts converge on trim direction; "
        "Aggressive's strongest punch is the 55/45 ratio inversion."
    )
    out = _clean_agentic_vocabulary(src)
    # Internal agent vocabulary must NOT appear unchanged in the output.
    assert "The Trader's" not in out
    assert "Research Manager's verdict" not in out
    assert "Aggressive's strongest punch" not in out
    # Replacements should still preserve the underlying meaning.
    assert "trader proposal (HOLD)" in out
    assert "research synthesis" in out
    assert "All three risk perspectives" in out  # capitalised in source
    assert "aggressive case's strongest argument" in out


def test_clean_agentic_vocabulary_strips_file_paths():
    """Internal file paths leak too often into the front-of-document narrative.
    Replace them with descriptive prose."""
    from cli.research_pdf import _clean_agentic_vocabulary

    src = (
        "Recomputed from raw/peers.json: GOOGL 32.46%. "
        "The 10-Q text in raw/sec_filing.md confirms this. "
        "See raw/financials.json for the underlying quarterly data."
    )
    out = _clean_agentic_vocabulary(src)
    assert "raw/peers.json" not in out
    assert "raw/sec_filing.md" not in out
    assert "raw/financials.json" not in out
    assert "the peer dataset" in out
    assert "the 10-Q text" in out
    assert "the financials dataset" in out


def test_clean_agentic_vocabulary_strips_internal_pass_labels():
    """v1/v2 technical-pass labels are internal; an executive doesn't care
    that there were two passes."""
    from cli.research_pdf import _clean_agentic_vocabulary

    src = "MSFT Technical Analysis — v2 Report. The TA v2 view classifies this as DOWNTREND."
    out = _clean_agentic_vocabulary(src)
    assert "v2 Report" not in out
    assert "TA v2" not in out
    assert "Technical Analysis" in out


def test_clean_agentic_vocabulary_replaces_qc_framework_refs():
    """QC checklist item numbers leak occasionally; rephrase to plain English."""
    from cli.research_pdf import _clean_agentic_vocabulary

    src = "Item 16a failure: The PM cited GOOGL 4.9% with a caveat. The QC verdict was FAIL."
    out = _clean_agentic_vocabulary(src)
    assert "Item 16a" not in out
    assert "QC verdict" not in out
    assert "the numerical-trace check" in out
    assert "quality verdict" in out


def test_clean_agentic_vocabulary_idempotent_on_clean_text():
    """Polished text should pass through unchanged — replacements must not
    create runaway substitution chains."""
    from cli.research_pdf import _clean_agentic_vocabulary

    src = "MSFT trades at $413.62 with a Hold rating. EV is +2.09% above spot."
    assert _clean_agentic_vocabulary(src) == src


def test_extract_section_pulls_named_section():
    """Helper extracts a markdown section header + body until the next
    same-or-higher header."""
    from cli.research_pdf import _extract_section

    md = (
        "# Title\n"
        "\n"
        "## First section\n"
        "first body\n"
        "\n"
        "## Bottom Line\n"
        "**Rating: Hold.**\n"
        "Body of the bottom line.\n"
        "\n"
        "## Next section\n"
        "next body\n"
    )
    out = _extract_section(md, r"^## Bottom Line\s*$")
    assert out is not None
    assert out.startswith("## Bottom Line")
    assert "Rating: Hold" in out
    assert "Next section" not in out
    assert "first body" not in out


def test_extract_section_returns_none_if_missing():
    from cli.research_pdf import _extract_section
    assert _extract_section("# Title\n\n## Body\n", r"^## Nonexistent\s*$") is None


def test_build_executive_summary_md_includes_verdict_and_scenarios(tmp_path):
    """The executive summary must surface both the rating verdict (renamed
    from 'Bottom Line') and the 12-month scenario table."""
    from cli.research_pdf import _build_executive_summary_md

    decision_md = (
        "# MSFT — 2026-05-05\n\n"
        "## Inputs to this decision\n"
        "Reference price: $413.62.\n\n"
        "## 12-Month Scenario Analysis\n\n"
        "| Scenario | Probability | Target | Return | Drivers |\n"
        "|---|---|---|---|---|\n"
        "| Bull | 25% | $480.00 | +16.05% | confirmation |\n"
        "| Base | 45% | $425.00 | +2.75% | neutral |\n"
        "| Bear | 30% | $370.00 | -10.55% | breakdown |\n\n"
        "**Expected Value:** $422.25.\n\n"
        "## Synthesis of the Risk Debate\n"
        "(detail belongs in the recommendation, not the summary)\n\n"
        "## Bottom Line\n\n"
        "**Rating: Hold.** Maintain existing position. "
        "Hard stop at $396.44.\n"
    )

    out = _build_executive_summary_md(decision_md)
    # Verdict (renamed from Bottom Line)
    assert "## Verdict" in out
    assert "Rating: Hold" in out
    assert "Hard stop at $396.44" in out
    # Scenario table
    assert "12-Month Scenario Analysis" in out
    assert "Bull | 25%" in out or "Bull" in out
    assert "$480.00" in out
    # Synthesis section must NOT be in the summary (operational detail)
    assert "Synthesis of the Risk Debate" not in out
    # Inputs section should also be excluded (operational detail)
    assert "Inputs to this decision" not in out


def test_build_executive_summary_md_handles_empty_decision():
    from cli.research_pdf import _build_executive_summary_md
    assert "no decision" in _build_executive_summary_md("").lower()


def test_build_executive_summary_md_handles_decision_without_named_sections():
    """Defensive: if decision.md doesn't have a recognised structure, return
    a placeholder rather than a misleading blank page."""
    from cli.research_pdf import _build_executive_summary_md

    out = _build_executive_summary_md("# MSFT\n\nSome free-form prose without expected headers.\n")
    assert "executive summary unavailable" in out.lower()
