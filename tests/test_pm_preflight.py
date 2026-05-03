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
