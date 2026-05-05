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
