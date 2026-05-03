"""Tests for the TA Agent."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


_VALID_TA = """\
## Major historical levels

| Level | Price | Type | Why crowds trade here |
|---|---|---|---|
| Prior cycle high | $560 | Resistance | 2024 swing high; large stop cluster above |
| 200-day SMA | $487 | Support | Long-term trend; institutional rebalancing |
| Fib 0.618 retrace | $446 | Support | Algo bounce zone |

## Volume profile zones

- Heavy accumulation: $480-$510
- Volume gap: $410-$430

## Current technical state

RSI 58, MACD positive, 50-SMA above 200-SMA.

## Setup classification

Consolidation within uptrend.

## Asymmetry

- Upside to next resistance: $560 (+8%)
- Downside to next support: $446 (-14%)
- Reward/risk: 0.6:1
"""


def test_ta_agent_writes_technicals_md(tmp_path):
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "prices.json").write_text(json.dumps({"history_5y": []}), encoding="utf-8")
    (raw / "pm_brief.md").write_text("# Brief", encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps({
        "ticker": "MSFT", "reference_price": 410.0, "spot_50dma": 405.0, "spot_200dma": 380.0
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_TA)

    node = create_ta_agent_node(fake_llm)
    out = node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    })

    technicals = (raw / "technicals.md").read_text(encoding="utf-8")
    assert "Major historical levels" in technicals
    assert out["technicals_report"] == technicals


def test_ta_agent_includes_required_sections_in_prompt(tmp_path):
    """The TA prompt must instruct the model to produce all 5 mandated sections."""
    from tradingagents.agents.analysts.ta_agent import _SYSTEM
    for required in ("Major historical levels", "Volume profile zones",
                     "Current technical state", "Setup classification", "Asymmetry"):
        assert required in _SYSTEM
