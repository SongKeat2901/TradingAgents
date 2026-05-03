"""Tests for the progress callback used by tradingresearch CLI."""

from io import StringIO

import pytest

pytestmark = pytest.mark.unit


def test_callback_emits_done_lines():
    from cli.research_progress import ProgressCallback

    out = StringIO()
    cb = ProgressCallback(stream=out)
    cb.on_node_start("Market Analyst")
    cb.on_node_done("Market Analyst", duration_s=2.34)

    lines = out.getvalue().strip().splitlines()
    assert any("[Market Analyst] start" in line for line in lines)
    assert any("[Market Analyst] done" in line for line in lines)
    assert any("2.3" in line for line in lines)  # duration formatted to 1 dp


def test_callback_handles_unicode_node_names():
    from cli.research_progress import ProgressCallback

    out = StringIO()
    cb = ProgressCallback(stream=out)
    cb.on_node_done("Risk-Manager", duration_s=0.05)
    assert "[Risk-Manager] done" in out.getvalue()
