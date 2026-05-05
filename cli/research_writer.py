"""Write research outputs from a final AgentState dict to disk.

Output layout per run (under output_dir):
  decision.md
  analyst_market.md
  analyst_social.md
  analyst_news.md
  analyst_fundamentals.md
  debate_bull_bear.md
  debate_risk.md
  state.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _decision_md(state: dict[str, Any]) -> str:
    """Render the PM's full mandated decision under a ticker/date header.

    The PM emits the full document (Inputs / Scenarios / Reconciliation / etc.)
    in `final_trade_decision`. The same string also lands in
    `risk_debate_state.judge_decision`. We render it once under the header.
    """
    ticker = state.get("company_of_interest", "?")
    date = state.get("trade_date", "?")
    pm_body = (
        state.get("final_trade_decision")
        or state.get("risk_debate_state", {}).get("judge_decision", "")
    ).strip()
    return f"# {ticker} — {date}\n\n{pm_body}\n"


def _analyst_md(title: str, body: str) -> str:
    return f"# {title}\n\n{body or '_(no report)_'}\n"


def _bull_bear_md(state: dict[str, Any]) -> str:
    debate = state.get("investment_debate_state", {})
    return (
        "# Bull vs Bear Debate\n\n"
        f"## Bull\n\n{debate.get('bull_history', '_(empty)_')}\n\n"
        f"## Bear\n\n{debate.get('bear_history', '_(empty)_')}\n\n"
        f"## Research Manager Decision\n\n{debate.get('judge_decision', '_(empty)_')}\n"
    )


def _risk_md(state: dict[str, Any]) -> str:
    risk = state.get("risk_debate_state", {})
    # The PM's judge_decision is the canonical content of decision.md and is
    # rendered as its own section in the PDF. Including it here too produces
    # a duplicated 9-page block in the final PDF.
    return (
        "# Risk Team Debate\n\n"
        f"## Aggressive\n\n{risk.get('aggressive_history', '_(empty)_')}\n\n"
        f"## Neutral\n\n{risk.get('neutral_history', '_(empty)_')}\n\n"
        f"## Conservative\n\n{risk.get('conservative_history', '_(empty)_')}\n"
    )


def write_research_outputs(state: dict[str, Any], output_dir: str) -> list[Path]:
    """Write all report files. Returns the list of paths written."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, str]] = [
        ("decision.md", _decision_md(state)),
        ("analyst_market.md", _analyst_md("Market Analyst", state.get("market_report", ""))),
        ("analyst_social.md", _analyst_md("Social Sentiment Analyst", state.get("sentiment_report", ""))),
        ("analyst_news.md", _analyst_md("News Analyst", state.get("news_report", ""))),
        ("analyst_fundamentals.md", _analyst_md("Fundamentals Analyst", state.get("fundamentals_report", ""))),
        ("debate_bull_bear.md", _bull_bear_md(state)),
        ("debate_risk.md", _risk_md(state)),
    ]

    written: list[Path] = []
    for name, content in files:
        p = out / name
        p.write_text(content, encoding="utf-8")
        written.append(p)

    state_path = out / "state.json"
    state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    written.append(state_path)
    return written
