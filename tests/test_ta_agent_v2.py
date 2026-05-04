"""Tests for the TA Agent v2 (post-analyst reconciliation pass)."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


_VALID_V2 = """\
## Revisions from v1

The Fundamentals analyst flagged 970 bps FCF margin compression that the v1 \
"accumulation" classification did not address. Setup classification revised \
from "accumulation" to "distribution range" based on this fundamental drag.

## Major historical levels

| Level | Price | Type | Why crowds trade here |
|---|---|---|---|
| 200-day SMA | $466 | Resistance | Long-term trend; institutional rebalancing |

## Volume profile zones

- Heavy accumulation: $410-$430

## Current technical state

RSI 54, MACD negative.

## Setup classification

Distribution range.

## Asymmetry

- Upside: $466 (+14.4%)
- Downside: $356 (-12.6%)
- Reward/risk: 1.1:1
"""


def _stub_state(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "technicals.md").write_text("# v1 placeholder", encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps({"reference_price": 407.78}), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps({"ohlcv": "..."}), encoding="utf-8")
    return {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "market_report": "Market analyst said: setup looks like accumulation.",
        "fundamentals_report": "Fundamentals analyst said: 970bps FCF margin compression.",
        "news_report": "News analyst said: no MSFT catalysts in window.",
        "sentiment_report": "Sentiment analyst said: zero social mentions.",
    }


def test_ta_agent_v2_writes_technicals_v2_md(tmp_path):
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_v2_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_V2)

    node = create_ta_agent_v2_node(fake_llm)
    out = node(_stub_state(tmp_path))

    v2_path = tmp_path / "raw" / "technicals_v2.md"
    assert v2_path.exists()
    content = v2_path.read_text(encoding="utf-8")
    assert "Revisions from v1" in content
    assert "Setup classification" in content
    assert out["technicals_report"] == content


def test_ta_agent_v2_overwrites_state_technicals_report(tmp_path):
    """Downstream agents read state.technicals_report; v2 must overwrite v1's value."""
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_v2_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_V2)

    state = _stub_state(tmp_path)
    state["technicals_report"] = "v1 contents that should be overwritten"

    node = create_ta_agent_v2_node(fake_llm)
    out = node(state)
    assert "v1 contents that should be overwritten" not in out["technicals_report"]
    assert "Distribution range" in out["technicals_report"]


def test_ta_agent_v2_system_prompt_lists_revision_section_and_5_mandated():
    from tradingagents.agents.analysts.ta_agent import _SYSTEM_V2
    assert "Revisions from v1" in _SYSTEM_V2
    for required in ("Major historical levels", "Volume profile zones",
                     "Current technical state", "Setup classification", "Asymmetry"):
        assert required in _SYSTEM_V2


def test_ta_agent_v2_includes_all_4_analyst_reports_in_user_message(tmp_path):
    """The v2 prompt must include every analyst's text so v2 can reconcile."""
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_v2_node

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_V2)
    node = create_ta_agent_v2_node(fake_llm)
    node(_stub_state(tmp_path))

    call_args = fake_llm.invoke.call_args
    messages = call_args.args[0]
    user = messages[1].content
    assert "Market analyst said: setup looks like accumulation" in user
    assert "Fundamentals analyst said: 970bps FCF margin compression" in user
    assert "News analyst said: no MSFT catalysts" in user
    assert "Sentiment analyst said: zero social mentions" in user
