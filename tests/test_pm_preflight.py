"""Tests for PM Pre-flight node."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


_VALID_BRIEF = """\
# PM Pre-flight Brief: MSFT 2026-05-01

## Ticker validation
- Trading day: Friday 2026-05-01
- Sector (yfinance): Technology / Software
- Market cap: Mega-cap

## Business model classification
- yfinance sector: Technology / Software
- Actual business model: **Cloud + productivity software + AI infrastructure**.

Interpretation rules for analysts:
- Revenue is enterprise software + cloud (Azure) + Office.

## Peer set
- GOOG: nearest cloud + AI infra peer
- META: nearest scale + AI infra peer
- AAPL: nearest mega-cap tech peer

## Past-lesson summary
- No prior decision on this ticker in memory log.

## What this run must answer
1. Is Azure growth durable?
2. Is AI capex generating ROI?
3. Is the multiple still defensible?
"""


def test_pm_preflight_writes_brief_to_raw_dir(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)

    brief_path = tmp_path / "raw" / "pm_brief.md"
    assert brief_path.exists()
    content = brief_path.read_text(encoding="utf-8")
    assert "Business model classification" in content
    assert out["pm_brief"] == content


def test_pm_preflight_extracts_peers_from_brief(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)
    assert sorted(out["peers"]) == ["AAPL", "GOOG", "META"]


def test_pm_preflight_extracts_peers_with_markdown_bold(tmp_path):
    """LLMs sometimes emit '- **GOOGL**: ...' instead of '- GOOGL: ...'.
    The regex must tolerate optional markdown bold/italic around the ticker."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    bold_brief = """\
# PM Pre-flight Brief: MSFT 2026-05-01

## Peer set
- **GOOGL**: Hyperscaler peer (GCP)
- *AMZN*: AWS comp (italic variant)
- ORCL: plain form
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=bold_brief)
    node = create_pm_preflight_node(fake_llm)
    out = node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    })
    assert sorted(out["peers"]) == ["AMZN", "GOOGL", "ORCL"]


def test_pm_preflight_handles_no_peers_etf(tmp_path):
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    spy_brief = """\
# PM Pre-flight Brief: SPY 2026-05-01

## Business model classification
- yfinance sector: ETF / Index
- Actual business model: **S&P 500 index ETF** — peer comparison not applicable.

## Peer set
(none — index ETF; compare to other broad-market ETFs only on request)
"""
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=spy_brief)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "SPY",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)
    assert out["peers"] == []


def test_pm_preflight_appends_calendar_block_to_brief(tmp_path):
    """If raw/calendar.json exists, pm_brief.md must end with a deterministic
    'Reporting status' block appended after the LLM-written content."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "calendar.json").write_text(_json.dumps({
        "trade_date": "2026-05-01",
        "_unavailable": [],
        "MSFT": {
            "last_reported": "2026-04-29",
            "fiscal_period": "FY26 Q3",
            "next_expected": "2026-07-25",
            "source": "yfinance",
        },
        "GOOGL": {
            "last_reported": "2026-04-22",
            "fiscal_period": "Q1 2026",
            "next_expected": "2026-07-23",
            "source": "yfinance",
        },
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# PM Brief: MSFT 2026-05-01\n\n## Peer set\n- GOOGL: hyperscaler peer\n")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert brief.startswith("# PM Brief: MSFT 2026-05-01")
    assert "## Reporting status (relative to trade_date 2026-05-01)" in brief
    assert "MSFT" in brief
    assert "FY26 Q3 reported 2026-04-29" in brief
    assert "GOOGL" in brief
    assert "Q1 2026 reported 2026-04-22" in brief
    assert "already happened" in brief
    assert "2026-07-25" in brief
    assert "2026-07-23" in brief


def test_pm_preflight_skips_calendar_block_when_calendar_missing(tmp_path):
    """If raw/calendar.json doesn't exist, pm_brief.md should contain only
    the LLM content. No fallback fabrication."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    # No calendar.json

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief content\n\n## Peer set\n- AAPL: peer\n")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "Reporting status" not in brief
    assert brief.startswith("# Brief content")


def test_pm_preflight_calendar_block_renders_unavailable_tickers(tmp_path):
    """A ticker marked unavailable in calendar.json should render '(yfinance unavailable)'."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "calendar.json").write_text(_json.dumps({
        "trade_date": "2026-05-01",
        "_unavailable": ["TICKERX"],
        "MSFT": {
            "last_reported": "2026-04-29",
            "fiscal_period": "FY26 Q3",
            "next_expected": "2026-07-25",
            "source": "yfinance",
        },
        "TICKERX": {"unavailable": True, "reason": "yfinance returned no earnings dates"},
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="# Brief")

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    }
    node(state)

    brief = (raw / "pm_brief.md").read_text(encoding="utf-8")
    assert "TICKERX" in brief
    assert "yfinance unavailable" in brief.lower() or "(yfinance unavailable)" in brief


def test_pm_preflight_system_prompt_has_temporal_anchor():
    """Option A: PM Pre-flight _SYSTEM must include a Temporal anchor section
    instructing the LLM not to fabricate past-vs-future status."""
    from tradingagents.agents.managers.pm_preflight import _SYSTEM
    assert "Temporal anchor" in _SYSTEM or "trade date as \"today\"" in _SYSTEM
    assert "data to follow" in _SYSTEM or "already occurred" in _SYSTEM
