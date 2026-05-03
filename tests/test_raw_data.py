"""Tests for raw/ data helpers used by all agents."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_raw_dir_for_returns_subdir(tmp_path):
    from tradingagents.agents.utils.raw_data import raw_dir_for
    out = tmp_path / "out"
    assert raw_dir_for(str(out)) == str(out / "raw")


def test_load_json_returns_parsed_dict(tmp_path):
    from tradingagents.agents.utils.raw_data import load_json
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "financials.json").write_text(json.dumps({"ticker": "MSFT", "revenue": 245}), encoding="utf-8")
    data = load_json(str(raw), "financials.json")
    assert data == {"ticker": "MSFT", "revenue": 245}


def test_load_json_returns_none_when_missing(tmp_path):
    from tradingagents.agents.utils.raw_data import load_json
    raw = tmp_path / "raw"
    raw.mkdir()
    assert load_json(str(raw), "nope.json") is None


def test_load_text_reads_md(tmp_path):
    from tradingagents.agents.utils.raw_data import load_text
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "technicals.md").write_text("# Levels\n", encoding="utf-8")
    assert load_text(str(raw), "technicals.md") == "# Levels\n"


def test_format_for_prompt_concatenates_with_section_headers(tmp_path):
    from tradingagents.agents.utils.raw_data import format_for_prompt
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "pm_brief.md").write_text("# PM Brief\n\nContent.", encoding="utf-8")
    (raw / "financials.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")
    out = format_for_prompt(str(raw), files=["pm_brief.md", "financials.json"])
    assert "## raw/pm_brief.md" in out
    assert "Content." in out
    assert "## raw/financials.json" in out
    assert '"k": "v"' in out
