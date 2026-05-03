# Quant Research Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current 4-analyst-with-bind_tools graph with a data-grounded, quant-rigorous, self-correcting multi-agent pipeline that produces sell-side-quality equity research with explicit calculation chains, peer matrices, technical levels, and a mandatory 12-month Bull/Base/Bear scenario table with EV.

**Architecture:** Insert three new components in front of the existing 12-persona graph: PM Pre-flight (Sonnet, opener role) → Researcher (Python, deterministic data fetcher writing `raw/`) → TA Agent (Sonnet). The 4 analysts are refactored to read `raw/` instead of using `bind_tools()`. Bull/Bear get prompt-rigor rules. PM Final gains a 3-pass model with a 13-item QC checklist and a push-back retry loop to Research Manager or Risk team (max 1 retry).

**Tech Stack:** Python 3.13, langchain_core BaseChatModel, existing `cli/claude_cli_chat_model.py` for Sonnet/Opus paths, pytest with `unit`/`integration` markers, existing yfinance + alpha_vantage tools (called as plain Python functions, not via bind_tools).

**Spec:** `docs/superpowers/specs/2026-05-03-quant-research-rebuild-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tradingagents/agents/researcher.py` | Deterministic data fetcher; writes `raw/{financials,peers,news,prices,social,reference}.json` |
| `tradingagents/agents/utils/raw_data.py` | Helper functions for reading raw/ files in agent prompts |
| `tradingagents/agents/managers/pm_preflight.py` | PM Pre-flight node (Sonnet); writes `raw/pm_brief.md` |
| `tradingagents/agents/analysts/ta_agent.py` | TA Agent node (Sonnet); writes `raw/technicals.md` |
| `tests/test_researcher.py` | Researcher unit tests |
| `tests/test_raw_data.py` | raw/ helper tests |
| `tests/test_pm_preflight.py` | PM Pre-flight tests |
| `tests/test_ta_agent.py` | TA Agent tests |
| `tests/test_pm_qc_checklist.py` | PM Pass-2 QC checklist tests |
| `tests/test_pm_retry_loop.py` | PM Pass-3 retry loop tests |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/utils/agent_states.py` | Add `pm_brief`, `peers`, `raw_dir`, `technicals_report`, `pm_feedback`, `pm_retries` fields |
| `tradingagents/agents/analysts/market_analyst.py` | Refactor: drop bind_tools; read raw/; consume technicals.md |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | Refactor: drop bind_tools; mandate peer matrix, sanity-check, capital-structure |
| `tradingagents/agents/analysts/news_analyst.py` | Refactor: drop bind_tools; mandate catalyst magnitudes |
| `tradingagents/agents/analysts/social_media_analyst.py` | Refactor: drop bind_tools; mandate numerical sentiment |
| `tradingagents/agents/researchers/bull_researcher.py` | Prompt rigor: lead-with-strongest, analogies-must-survive, quantified asymmetry |
| `tradingagents/agents/researchers/bear_researcher.py` | Same rigor rules |
| `tradingagents/agents/managers/research_manager.py` | Read `pm_feedback` from state on retry; address it explicitly |
| `tradingagents/agents/managers/portfolio_manager.py` | 3-pass model: draft → QC self-correct → optional push-back retry; mandate Inputs + Scenarios sections |
| `tradingagents/agents/risk_mgmt/{aggressive,conservative,neutral}_debator.py` | Read `pm_feedback` from state on retry |
| `tradingagents/graph/setup.py` | Insert PM Pre-flight + Researcher (as node) + TA Agent in graph; add retry conditional edges |
| `tradingagents/graph/trading_graph.py` | Wire new config keys; route deep judges still via Opus CLI |
| `tradingagents/default_config.py` | Add `claude_code_research_llm` (Sonnet via CLI) for analyst tier |
| `cli/research_pdf.py` | Include `raw/pm_brief.md` and `raw/technicals.md` sections in PDF |

---

## Task 1: AgentState extensions

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Create: `tests/test_agent_state_extensions.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for the new AgentState fields added by the quant-research rebuild."""
import pytest

pytestmark = pytest.mark.unit


def test_agent_state_has_quant_research_fields():
    from tradingagents.agents.utils.agent_states import AgentState
    annotations = AgentState.__annotations__
    for field in ("pm_brief", "peers", "raw_dir", "technicals_report",
                  "pm_feedback", "pm_retries"):
        assert field in annotations, f"AgentState missing field: {field}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_agent_state_extensions.py -v
```

Expected: FAIL with AssertionError — fields missing.

- [ ] **Step 3: Add fields to AgentState**

Open `tradingagents/agents/utils/agent_states.py`. Locate the `AgentState` TypedDict. Add the new fields at the end of the dict body (preserving the existing order and trailing comma):

```python
    # Quant-research rebuild (2026-05-03 spec): pre-flight + raw data folder + retry loop
    pm_brief: str
    peers: List[str]
    raw_dir: str
    technicals_report: str
    pm_feedback: str
    pm_retries: int
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_agent_state_extensions.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tests/test_agent_state_extensions.py
git commit -m "feat(state): add quant-research-rebuild AgentState fields"
```

---

## Task 2: raw/ data helper

**Files:**
- Create: `tradingagents/agents/utils/raw_data.py`
- Create: `tests/test_raw_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_raw_data.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_raw_data.py -v
```

Expected: ImportError on `tradingagents.agents.utils.raw_data`.

- [ ] **Step 3: Implement helper**

Create `tradingagents/agents/utils/raw_data.py`:

```python
"""Helpers for reading the per-run `raw/` data folder.

The Researcher writes deterministic data files to `<output_dir>/raw/`.
All downstream agents read from this folder via these helpers — never
via bind_tools or ad-hoc fetches. Single source of truth, deterministic
data path, easy to inspect and test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional


def raw_dir_for(output_dir: str) -> str:
    """Return the `raw/` subdirectory path under the run's output_dir."""
    return str(Path(output_dir) / "raw")


def load_json(raw_dir: str, filename: str) -> Optional[Any]:
    """Read and parse a JSON file from raw/. Returns None if missing."""
    p = Path(raw_dir) / filename
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_text(raw_dir: str, filename: str) -> Optional[str]:
    """Read a text file from raw/. Returns None if missing."""
    p = Path(raw_dir) / filename
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def format_for_prompt(raw_dir: str, files: Iterable[str]) -> str:
    """Concatenate raw/ files for inclusion in an LLM prompt.

    Each file becomes a section with header `## raw/<filename>` followed
    by its contents. Missing files become `## raw/<filename>\n_(missing)_`.
    JSON files are pretty-printed for readability.
    """
    parts: list[str] = []
    for fname in files:
        p = Path(raw_dir) / fname
        parts.append(f"## raw/{fname}\n")
        if not p.exists():
            parts.append("_(missing)_\n")
            continue
        if fname.endswith(".json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                parts.append("```json\n")
                parts.append(json.dumps(data, indent=2))
                parts.append("\n```\n")
            except json.JSONDecodeError:
                parts.append(p.read_text(encoding="utf-8"))
        else:
            parts.append(p.read_text(encoding="utf-8"))
        parts.append("\n")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_raw_data.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/raw_data.py tests/test_raw_data.py
git commit -m "feat(agents): raw/ data helper for reading shared per-run files"
```

---

## Task 3: Researcher (deterministic data fetcher)

**Files:**
- Create: `tradingagents/agents/researcher.py`
- Create: `tests/test_researcher.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_researcher.py`:

```python
"""Tests for the Researcher data fetcher."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _stub_state(tmp_path, peers=("GOOG", "META")):
    return {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "peers": list(peers),
        "raw_dir": str(tmp_path / "raw"),
    }


def test_researcher_writes_all_expected_files(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    # Stub the data tools so the test doesn't network out
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {"ticker": t, "revenue": 100})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {"items": []})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {"sentiment": 0.5})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {
        "ticker": t, "close_on_date": 410.0, "history_5y": []
    })
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "rsi_14": 58.0, "macd": 1.2, "sma_50": 405.0, "sma_200": 380.0
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    raw = Path(state["raw_dir"])
    for f in ("financials.json", "peers.json", "news.json", "insider.json",
              "social.json", "prices.json", "reference.json"):
        assert (raw / f).exists(), f"missing: {f}"


def test_researcher_writes_reference_with_required_keys(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {
        "close_on_date": 410.0,
        "history_5y": [],
        "ytd_high": 460.0,
        "ytd_low": 380.0,
        "atr_14": 4.2,
    })
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "sma_50": 405.0, "sma_200": 380.0
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    ref = json.loads((Path(state["raw_dir"]) / "reference.json").read_text())
    assert ref["ticker"] == "MSFT"
    assert ref["trade_date"] == "2026-05-01"
    assert ref["reference_price"] == 410.0
    assert ref["reference_price_source"].startswith("yfinance close")
    assert ref["spot_50dma"] == 405.0
    assert ref["spot_200dma"] == 380.0
    assert ref["ytd_high"] == 460.0
    assert ref["ytd_low"] == 380.0
    assert ref["atr_14"] == 4.2


def test_researcher_writes_peer_per_ticker(tmp_path, monkeypatch):
    from tradingagents.agents import researcher
    fetched = {}

    def fake_financials(t, d):
        fetched.setdefault(t, 0)
        fetched[t] += 1
        return {"ticker": t, "revenue": 100 if t == "MSFT" else 200}

    monkeypatch.setattr(researcher, "_fetch_financials", fake_financials)
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"close_on_date": 1.0})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {})

    state = _stub_state(tmp_path, peers=("GOOG", "META", "AAPL"))
    researcher.fetch_research_pack(state)

    peers_data = json.loads((Path(state["raw_dir"]) / "peers.json").read_text())
    assert set(peers_data.keys()) == {"GOOG", "META", "AAPL"}
    assert peers_data["GOOG"]["revenue"] == 200
    # Main ticker also fetched
    assert "MSFT" in fetched
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_researcher.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement Researcher**

Create `tradingagents/agents/researcher.py`:

```python
"""Researcher — deterministic data fetcher (no LLM).

Replaces the bind_tools pattern in the original 4 analysts. Pulls all
data the multi-agent pipeline needs once, up front, and writes it to
`<output_dir>/raw/` as JSON / Markdown. Every downstream agent reads
from raw/ — no agent-side data fetching, no ReAct loops over tools.

Wraps the existing dataflows utilities (yfinance, alpha_vantage) as
plain Python functions called from this single deterministic step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.raw_data import raw_dir_for


def _fetch_financials(ticker: str, date: str) -> dict[str, Any]:
    """Pull fundamentals + balance sheet + cashflow + income statement for one ticker."""
    from tradingagents.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_income_statement,
    )
    return {
        "ticker": ticker,
        "trade_date": date,
        "fundamentals": get_fundamentals.invoke({"ticker": ticker, "date": date}),
        "balance_sheet": get_balance_sheet.invoke({"ticker": ticker, "date": date}),
        "cashflow": get_cashflow.invoke({"ticker": ticker, "date": date}),
        "income_statement": get_income_statement.invoke({"ticker": ticker, "date": date}),
    }


def _fetch_news(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_global_news, get_news
    return {
        "ticker_news": get_news.invoke({"ticker": ticker, "date": date}),
        "global_news": get_global_news.invoke({"date": date}),
    }


def _fetch_insider(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_insider_transactions
    return {"transactions": get_insider_transactions.invoke({"ticker": ticker, "date": date})}


def _fetch_social(ticker: str, date: str) -> dict[str, Any]:
    # Reuse the news tool for social by convention; downstream prompts know
    # to focus on social/sentiment indicators in this view.
    from tradingagents.agents.utils.agent_utils import get_news
    return {"social_news": get_news.invoke({"ticker": ticker, "date": date, "social": True})}


def _fetch_prices(ticker: str, date: str) -> dict[str, Any]:
    """5y OHLCV + close on date + YTD high/low + ATR."""
    from tradingagents.agents.utils.agent_utils import get_stock_data
    return get_stock_data.invoke({"ticker": ticker, "date": date, "lookback_years": 5})


def _fetch_indicators(ticker: str, date: str) -> dict[str, Any]:
    from tradingagents.agents.utils.agent_utils import get_indicators
    return get_indicators.invoke({"ticker": ticker, "date": date})


def fetch_research_pack(state: dict) -> None:
    """Fetch all data needed by the multi-agent pipeline. Writes to raw/.

    Required state keys: `company_of_interest`, `trade_date`, `peers`, `raw_dir`.
    """
    ticker = state["company_of_interest"]
    date = state["trade_date"]
    peers = state.get("peers", [])
    raw = Path(state["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)

    # Per-ticker bundles
    financials = _fetch_financials(ticker, date)
    news = _fetch_news(ticker, date)
    insider = _fetch_insider(ticker, date)
    social = _fetch_social(ticker, date)
    prices = _fetch_prices(ticker, date)
    indicators = _fetch_indicators(ticker, date)

    # Peers (one financials bundle per peer)
    peers_data = {p: _fetch_financials(p, date) for p in peers}

    # Reference: single source of truth for prices
    reference = {
        "ticker": ticker,
        "trade_date": date,
        "reference_price": prices.get("close_on_date"),
        "reference_price_source": f"yfinance close {date}",
        "spot_50dma": indicators.get("sma_50"),
        "spot_200dma": indicators.get("sma_200"),
        "ytd_high": prices.get("ytd_high"),
        "ytd_low": prices.get("ytd_low"),
        "atr_14": prices.get("atr_14"),
    }

    # Write everything
    (raw / "financials.json").write_text(json.dumps(financials, indent=2, default=str), encoding="utf-8")
    (raw / "peers.json").write_text(json.dumps(peers_data, indent=2, default=str), encoding="utf-8")
    (raw / "news.json").write_text(json.dumps(news, indent=2, default=str), encoding="utf-8")
    (raw / "insider.json").write_text(json.dumps(insider, indent=2, default=str), encoding="utf-8")
    (raw / "social.json").write_text(json.dumps(social, indent=2, default=str), encoding="utf-8")
    (raw / "prices.json").write_text(json.dumps(prices, indent=2, default=str), encoding="utf-8")
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_researcher.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_researcher.py
git commit -m "feat(researcher): deterministic data fetcher writes raw/ folder"
```

---

## Task 4: PM Pre-flight node

**Files:**
- Create: `tradingagents/agents/managers/pm_preflight.py`
- Create: `tests/test_pm_preflight.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pm_preflight.py`:

```python
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
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement PM Pre-flight**

Create `tradingagents/agents/managers/pm_preflight.py`:

```python
"""PM Pre-flight node — opener role for the PM persona.

Sets the research mandate before analysts run:
- Validates ticker (trading day, sector, market cap class)
- Classifies the actual business model (overrides yfinance sector if wrong;
  motivated by spec Flaw 1 — MARA was tagged Financial Services but is a
  Bitcoin miner)
- Identifies 2-4 peers for comparison
- Reads memory log for past lessons on this ticker or pattern
- Specifies questions this run must answer

Output: `<raw_dir>/pm_brief.md` (Markdown). Side-effect on state:
populates `pm_brief` (full text) and `peers` (parsed list).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


_SYSTEM = """\
You are the Portfolio Manager performing pre-flight due diligence on \
$TICKER for trade date $DATE.

Produce a Markdown brief with EXACTLY these sections (use the headers below verbatim):

# PM Pre-flight Brief: $TICKER $DATE

## Ticker validation
- Trading day: <day-of-week + date>
- Sector (yfinance): <yfinance sector>
- Market cap: <Mega-cap / Large-cap / Mid-cap / Small-cap>

## Business model classification
- yfinance sector: <yfinance sector>
- Actual business model: **<plain-English description>**

Interpretation rules for analysts:
- <bullet rule 1>
- <bullet rule 2>
- <bullet rule 3>

The "Actual business model" overrides yfinance when yfinance's sector is \
structurally misleading. Examples of mismatch: a Bitcoin miner tagged as \
Financial Services, a SPAC tagged as Shell Companies, a biotech tagged as \
Pharmaceuticals when it has no revenue. Call out the mismatch explicitly.

## Peer set
- <PEER_TICKER>: <one-line rationale>
- <PEER_TICKER>: <one-line rationale>
- <PEER_TICKER>: <one-line rationale>

Pick 2-4 peers based on actual business model, not yfinance sector. For \
broad-market ETFs (SPY, QQQ, etc.), write "(none — index ETF; ...)" \
instead of a peer list.

## Past-lesson summary
- <Any prior decision on this ticker, or similar pattern, from the memory log>

## What this run must answer
1. <specific question>
2. <specific question>
3. <specific question>

Be concrete and falsifiable. No vague questions like "what's the outlook?"."""


_PEER_LINE = re.compile(r"^- ([A-Z]{1,5}): ", re.MULTILINE)


def _extract_peers(brief: str) -> list[str]:
    """Pull peer tickers from the Peer set section."""
    # Find the "## Peer set" section and the next "## " section
    match = re.search(r"## Peer set\s*\n(.*?)(?=^## |\Z)", brief, re.DOTALL | re.MULTILINE)
    if not match:
        return []
    section = match.group(1)
    return _PEER_LINE.findall(section)


def create_pm_preflight_node(llm):
    """Factory: returns the PM Pre-flight LangGraph node function."""

    def pm_preflight_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = Path(state["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)),
            ("user", f"Produce the PM Pre-flight brief for {ticker} on {date}."),
        ])
        result = (prompt | llm).invoke({})
        brief = result.content if hasattr(result, "content") else str(result)

        (raw_dir / "pm_brief.md").write_text(brief, encoding="utf-8")
        peers = _extract_peers(brief)

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "pm_brief": brief,
            "peers": peers,
        }

    return pm_preflight_node
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/pm_preflight.py tests/test_pm_preflight.py
git commit -m "feat(pm): pre-flight node with business-model classification + peer set"
```

---

## Task 5: TA Agent

**Files:**
- Create: `tradingagents/agents/analysts/ta_agent.py`
- Create: `tests/test_ta_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ta_agent.py`:

```python
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
```

- [ ] **Step 2: Run to fail**

```bash
.venv/bin/python -m pytest tests/test_ta_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement TA Agent**

Create `tradingagents/agents/analysts/ta_agent.py`:

```python
"""TA Agent — owns historical level identification with crowd psychology.

Reads `raw/prices.json` (5y OHLCV) + `raw/pm_brief.md` for context. Produces
`raw/technicals.md` with the mandated section structure. Other agents (Market
analyst, Bull/Bear) read this file and may agree, disagree, or extend.

Motivated by stakeholder feedback for "needle-see-blood" technical analysis:
identifying major historical levels (prior cycle highs, swing points, Fib
zones) AND explaining why crowds will trade them (stop clusters, retests,
round numbers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior technical analyst. Your job is to identify the price levels \
that matter for $TICKER and explain why crowds will trade at each one.

You have been given the 5-year price history, current technical indicators, \
and the PM pre-flight brief. Produce a Markdown report with EXACTLY these \
sections (use the headers verbatim):

## Major historical levels

| Level | Price | Type | Why crowds trade here |
|---|---|---|---|
| <name> | $<price> | Resistance/Support | <crowd-psychology rationale> |

Identify 3-7 levels. Examples of valid levels: prior cycle highs, all-time \
highs, swing pivots, 200-day SMA, 50-day SMA, Fibonacci 0.382/0.618 \
retracements, round numbers ($100, $500), volume-profile peaks.

## Volume profile zones

- Heaviest accumulation: $<low>-$<high> (<X>% of volume)
- Volume gap: $<low>-$<high> (<Y>% of volume — slip-through zone)

## Current technical state

Narrative on RSI, MACD, moving-average stack, divergences. Cite specific \
numbers from the indicators data.

## Setup classification

One of: breakout / breakdown / consolidation / distribution / accumulation. \
Justify in 1-2 sentences.

## Asymmetry

- Upside to next major resistance: $<price> (<+X>%)
- Downside to next major support: $<price> (<-Y>%)
- Reward/risk: <ratio>:1

Quantify both the magnitude AND the implied reward/risk ratio.

Cite specific levels from the price data — no vague "support somewhere below" \
language. Every claim must trace to a specific number from prices.json or \
indicators.json."""


def create_ta_agent_node(llm):
    """Factory: returns the TA Agent LangGraph node function."""

    def ta_agent_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        raw_dir = state["raw_dir"]

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "prices.json"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM.replace("$TICKER", ticker)),
            ("user", f"Produce the technicals report for {ticker}.\n\n{context}"),
        ])
        result = (prompt | llm).invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        (Path(raw_dir) / "technicals.md").write_text(report, encoding="utf-8")

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "technicals_report": report,
        }

    return ta_agent_node
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ta_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/ta_agent.py tests/test_ta_agent.py
git commit -m "feat(ta-agent): owns level identification with crowd-psychology rationale"
```

---

## Task 6: Refactor Market analyst (drop bind_tools, consume technicals.md)

**Files:**
- Modify: `tradingagents/agents/analysts/market_analyst.py`
- Modify: `tests/test_research_cli.py` (if Market analyst is in any existing test path)

- [ ] **Step 1: Read the existing market analyst**

```bash
cat tradingagents/agents/analysts/market_analyst.py
```

The current implementation uses `bind_tools()` with `[get_stock_data, get_indicators]`. The refactor: drop tools, read raw/, banter on top of TA Agent's output.

- [ ] **Step 2: Replace the file**

Replace the entire contents of `tradingagents/agents/analysts/market_analyst.py` with:

```python
"""Market analyst — refactored to read raw/ instead of bind_tools().

Reads the TA Agent's technicals.md as canonical level analysis, then writes
its own commentary that may agree, extend, or challenge specific levels.
The disagreement surfaces in the bull/bear debate downstream.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior market analyst writing the technical commentary section of \
an equity research report on $TICKER for trade date $DATE.

The TA Agent has already produced raw/technicals.md with the canonical level \
analysis. Your job is NOT to redo it — your job is to:

1. Quote the 2-3 levels you consider MOST important and explain WHY.
2. Either agree with the TA Agent's setup classification OR challenge it \
   with specific evidence ("TA Agent classified consolidation; I disagree \
   because volume on $X day was Y% above average and price broke...").
3. Map the technical setup onto a TRADING PLAYBOOK: \
   "if SPY breaks $500 on volume, expected next stop $487 (200-DMA); \
    if it holds, base for $510-520 retest."

Required sections in your output:

## Most important levels (2-3)
For each: price, type, your reason it matters more than the others.

## Setup assessment
Either "I agree with the TA Agent's <classification>" + supporting evidence,
or "I disagree with the TA Agent's <classification>; correct read is <X>" \
+ specific contradicting evidence.

## Trading playbook
- If <condition>: <expected next move> with <triggers/levels>
- If <opposite condition>: <expected next move> with <triggers/levels>

## Risk to my own read
What would have to be true for me to be wrong? Be specific."""


def create_market_analyst(llm):
    def market_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "technicals.md", "prices.json"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
             + "\n" + get_language_instruction()),
            ("user",
             f"For your reference: {instrument_context}\n\n{context}\n\n"
             f"Write the market analyst's commentary."),
        ])
        result = (prompt | llm).invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "market_report": report,
        }

    return market_analyst_node
```

- [ ] **Step 3: Run all tests to catch regressions**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all pass. The market analyst test (if any) should still pass because the function signature `create_market_analyst(llm)` is unchanged. The refactor is internal.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/analysts/market_analyst.py
git commit -m "refactor(market-analyst): read raw/, consume TA agent, write playbook"
```

---

## Task 7: Refactor Fundamentals analyst (peer matrix, sanity check, capital structure)

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`

- [ ] **Step 1: Replace the file**

Replace the entire contents of `tradingagents/agents/analysts/fundamentals_analyst.py` with:

```python
"""Fundamentals analyst — refactored to read raw/ + mandate quant rigor.

Required sections in output (motivated by spec Flaws 1, 4, and the
Tom-Lee-style stakeholder feedback):
- Business-model framing (quotes pm_brief's interpretation rules verbatim)
- Deal-math chain (when news contains a deal/announcement)
- Peer comparison matrix (always)
- Capital-structure compare with peers
- Sanity check on reported numbers (flag implausible ratios)
- "What management needs to prove" (3 falsifiable hurdles)
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior fundamentals analyst writing the fundamentals section of an \
equity research report on $TICKER for trade date $DATE.

You have been given pm_brief.md (with business-model rules), financials.json, \
peers.json, news.json, and reference.json. NO tool calls — the data is in \
front of you.

Required sections (use the headers verbatim):

## Business-model framing

Quote the "Interpretation rules for analysts" from pm_brief.md verbatim. \
Use those rules for every numerical interpretation that follows.

## Deal math (only if news contains a deal/announcement)

For each material deal in news.json, build the calculation chain:
- Deal size: $<amount>
- Annual revenue impact: $<amount> (cite assumption source)
- EPS delta: <±$X> per share at <Y> shares outstanding
- At current <P/E multiple> P/E this implies <±$Z> per share

If no material deal, write "No material deals in window" and skip the chain.

## Peer comparison matrix

| Metric | $TICKER | <peer1> | <peer2> | <peer3> | $TICKER rank |
|---|---|---|---|---|---|
| Revenue (TTM) | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| Revenue growth YoY | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Operating margin | <X>% | <Y>% | <Z>% | <W>% | <rank> |
| Net debt / EBITDA | <X>x | <Y>x | <Z>x | <W>x | <rank> (best/worst) |
| Cash + ST investments | $<X>B | $<Y>B | $<Z>B | $<W>B | <rank> |
| P/E (TTM) | <X>x | <Y>x | <Z>x | <W>x | premium/discount |

Pull peer numbers from peers.json. If any peer's data is missing, mark "n/a" \
and proceed. Rank ascending or descending depending on the metric \
(specify which is "best").

## Capital-structure compare

Quote the explicit comparison: "$TICKER's net cash of $<X>B vs <peer>'s net \
debt of $<Y>B" or similar. This addresses the "MSFT cash is disgusting / \
META has tons of debt" framing — make leverage / cash position concrete.

## Sanity check on reported numbers

| Metric | Reported | Implied math | Plausible? |
|---|---|---|---|
| <metric> | <value> | <derived calculation> | ✅ / ❌ <reason> |

Always include 3-5 rows. Flag ❌ on any ratio that looks implausible (e.g., \
"interest expense $11M on $3.65B debt = 1.4% effective rate" → flag as \
"likely excludes capitalized interest or convertibles"). Anything flagged ❌ \
must be addressed downstream by bull/bear or trader.

## What management needs to prove

Three falsifiable hurdles. Each: specific metric or event + by-when + threshold.

Every numerical claim in your report must trace back to financials.json, \
peers.json, news.json, or reference.json. No invented numbers."""


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "financials.json",
                   "peers.json", "news.json"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
             + "\n" + get_language_instruction()),
            ("user",
             f"For your reference: {instrument_context}\n\n{context}\n\n"
             f"Write the fundamentals analyst's report."),
        ])
        result = (prompt | llm).invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py
git commit -m "refactor(fundamentals): read raw/, mandate peer matrix + sanity check + capital structure"
```

---

## Task 8: Refactor News analyst (catalyst magnitudes)

**Files:**
- Modify: `tradingagents/agents/analysts/news_analyst.py`

- [ ] **Step 1: Replace the file**

Replace the entire contents of `tradingagents/agents/analysts/news_analyst.py` with:

```python
"""News analyst — refactored to read raw/ + mandate catalyst magnitudes."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior news / macro analyst writing the news section of an equity \
research report on $TICKER for trade date $DATE.

You have been given news.json (ticker-specific + global), reference.json, \
and pm_brief.md. NO tool calls — the data is in front of you.

Required sections (verbatim headers):

## Material catalysts (with magnitude estimates)

For every material catalyst in news.json, build a magnitude chain:

| Catalyst | Direction | Mechanism | Magnitude estimate | Confidence |
|---|---|---|---|---|
| <event> | Bull/Bear | <how it propagates to stock> | <±$X target price impact OR ±Y% multiple shift> | High/Med/Low |

Examples of valid mechanisms: "Fed cut → +1% to S&P fair value via duration \
math → +$5 SPY upside"; "Q4 earnings beat → +$0.20 EPS → at 22x P/E, +$4.4 \
per share". Vague claims like "this is positive for the stock" are not \
acceptable.

## Cross-references with peers

If any peer ticker (from pm_brief.md) is mentioned in the news context, cite \
it — these are the read-throughs ("AAPL's iPhone disclosure on April 28 \
implies $TICKER's similar segment will print Y% growth").

## Macro / global context

What broader trend frames this run's catalyst set? E.g., "rates expected to \
hold; AI capex cycle; consumer sentiment near 50-year low."

Every claim must cite a specific item from news.json or global news. No \
narrative without a source."""


def create_news_analyst(llm):
    def news_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "news.json"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
             + "\n" + get_language_instruction()),
            ("user",
             f"For your reference: {instrument_context}\n\n{context}\n\n"
             f"Write the news analyst's report."),
        ])
        result = (prompt | llm).invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "news_report": report,
        }

    return news_analyst_node
```

- [ ] **Step 2: Run tests**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/agents/analysts/news_analyst.py
git commit -m "refactor(news): read raw/, mandate catalyst magnitude estimates"
```

---

## Task 9: Refactor Social analyst (numerical sentiment)

**Files:**
- Modify: `tradingagents/agents/analysts/social_media_analyst.py`

- [ ] **Step 1: Replace the file**

Replace the entire contents of `tradingagents/agents/analysts/social_media_analyst.py` with:

```python
"""Social/sentiment analyst — refactored to read raw/ + mandate numerical sentiment."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.raw_data import format_for_prompt


_SYSTEM = """\
You are a senior social/sentiment analyst writing the sentiment section of \
an equity research report on $TICKER for trade date $DATE.

Required sections (verbatim headers):

## Sentiment indicators (with numbers)

| Source | Metric | Value | Trend (7d) | Interpretation |
|---|---|---|---|---|
| <source> | <metric> | <number or %> | <±X> | <crowd-psychology read> |

No vague claims like "sentiment is bullish." Every row must have a specific \
number or percentage, even if estimated from a sample size you state.

## Conviction asymmetry

Are bulls or bears more convinced? Cite specific signals — "X% of mentions \
include price targets above current spot vs Y% targeting downside" or \
similar. Quantify.

## Crowd-vs-data divergence

Where does sentiment disagree with the fundamentals analyst's data? \
Disagreement is a signal. Cite specific conflicts.

## Risk to my read

What sentiment signal would invalidate this analysis?"""


def create_social_media_analyst(llm):
    def social_analyst_node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)

        context = format_for_prompt(
            raw_dir,
            files=["pm_brief.md", "reference.json", "social.json"],
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             _SYSTEM.replace("$TICKER", ticker).replace("$DATE", date)
             + "\n" + get_language_instruction()),
            ("user",
             f"For your reference: {instrument_context}\n\n{context}\n\n"
             f"Write the sentiment analyst's report."),
        ])
        result = (prompt | llm).invoke({})
        report = result.content if hasattr(result, "content") else str(result)

        return {
            "messages": [result] if hasattr(result, "content") else [],
            "sentiment_report": report,
        }

    return social_analyst_node
```

- [ ] **Step 2: Run tests + commit**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
git add tradingagents/agents/analysts/social_media_analyst.py
git commit -m "refactor(social): read raw/, mandate numerical sentiment indicators"
```

---

## Task 10: Bull/Bear prompt rigor

**Files:**
- Modify: `tradingagents/agents/researchers/bull_researcher.py`
- Modify: `tradingagents/agents/researchers/bear_researcher.py`

- [ ] **Step 1: Read existing**

```bash
cat tradingagents/agents/researchers/bull_researcher.py tradingagents/agents/researchers/bear_researcher.py
```

These nodes don't use bind_tools; they take the existing analyst reports from state and argue. The change here is purely prompt rigor.

- [ ] **Step 2: Update Bull researcher prompt**

Open `tradingagents/agents/researchers/bull_researcher.py`. Locate the system prompt construction. Add the following text to the existing system prompt (find the section that instructs the bull on its argument structure and append):

```python
_RIGOR_RULES = """

# Argument rigor rules (Quant Research Rebuild — 2026-05-03)

1. Lead with your strongest argument. Your first paragraph names the SINGLE \
most load-bearing fact for the bull case. No analogies in the lede.

2. Analogies must survive scrutiny. If you cite Tesla / NextEra / Blackstone \
/ any precedent, you must include: (a) the relevant metric for the \
comparable, (b) why the analogy holds for THIS ticker, (c) the disanalogy \
and why it doesn't break the case. If you can't do all three, drop the \
analogy.

3. Quantify the asymmetry. Your conclusion must include a specific \
dollar/percentage outcome AND a probability — e.g., "Bull case: $560 in 12 \
months, ~30% probability conditional on Q4 capex ROI clearing the threshold \
the fundamentals analyst flagged." Vague directional claims like "upside is \
meaningful" are rejected by the Research Manager and require revision.

4. Address the fundamentals analyst's sanity-check flags. Any item the \
fundamentals analyst flagged ❌ must be addressed in your argument — do not \
ignore them.
"""
```

Then concatenate `_RIGOR_RULES` to the existing system prompt string.

- [ ] **Step 3: Same change for Bear researcher**

Open `tradingagents/agents/researchers/bear_researcher.py`. Add the same `_RIGOR_RULES` block (with "bull" replaced by "bear" in rule 1's lede instruction; everything else is identical) and concatenate it to its system prompt.

- [ ] **Step 4: Run all tests + commit**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
git add tradingagents/agents/researchers/bull_researcher.py tradingagents/agents/researchers/bear_researcher.py
git commit -m "feat(debaters): rigor rules — lead with strongest, analogies survive scrutiny, quantified asymmetry"
```

---

## Task 11: PM Final mandatory output structure (Inputs + Scenarios + EV)

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py`

- [ ] **Step 1: Read existing**

```bash
cat tradingagents/agents/managers/portfolio_manager.py
```

- [ ] **Step 2: Add mandated sections to system prompt**

Open `tradingagents/agents/managers/portfolio_manager.py`. Locate the system prompt where the PM is told what to produce. Add this block to the system prompt (concatenate at the end of the existing system message before any closing instructions):

```python
_MANDATED_SECTIONS = """

# Mandatory sections in your decision.md (Quant Research Rebuild — 2026-05-03)

Your decision.md MUST start with these two sections in this order:

## Inputs to this decision

- **Reference price:** $<reference_price> (<reference_price_source>)
- **Peers compared:** <ticker list with one-line rationale per peer>
- **Past decisions referenced:** <ticker> <date> (<rating>; outcome <±X>%, alpha <±Y>%) \
  — invoked to argue <…>
- **Memory-log lessons applied:** <bullets with source line refs>
- **Catalysts in window:** <upcoming events with dates>
- **Data freshness:** <financials period>, <news through date>

## 12-Month Scenario Analysis

| Scenario | Probability | 12-Mo Price Target | Return | Key drivers |
|---|---:|---:|---:|---|
| Bull | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Base | <pct>% | $<price> | <±pct>% | <named events / metrics> |
| Bear | <pct>% | $<price> | <±pct>% | <named events / metrics> |

**Expected Value:** <calculation> = $<EV> (<±pct>% from spot $<spot>)
**Rating implication:** <BUY/HOLD/SELL> (<one-line bridge from EV to rating>)

After these two sections, continue with the existing memo format (synthesis, \
trading plan, what you're rejecting, etc.).

# Hard rules for the scenario table

- Probabilities must sum to exactly 100%.
- All three price targets must be specific dollar values, not ranges.
- Each scenario lists at least one named, falsifiable catalyst (not narrative).
- Rating must logically derive from EV.

# Hard rules for the Inputs section

- Reference price must equal the value in raw/reference.json.
- Past-decision citations must include the actual rating and outcome.
- Stakeholder reading the decision.md cold (without analyst reports) must \
be able to understand the framing from the Inputs section alone.
"""
```

Concatenate `_MANDATED_SECTIONS` to the existing system message.

- [ ] **Step 3: Run all tests + commit**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
git add tradingagents/agents/managers/portfolio_manager.py
git commit -m "feat(pm): mandate Inputs + 12-month scenarios + EV in decision.md"
```

---

## Task 12: PM Pass-2 self-correction QC checklist

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Create: `tests/test_pm_qc_checklist.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pm_qc_checklist.py`:

```python
"""Tests for the PM Pass-2 self-correction QC checklist."""
import pytest

pytestmark = pytest.mark.unit


def test_qc_checklist_in_pm_system_prompt():
    """The PM system prompt must include the 13 QC items."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST

    must_contain = [
        "sum to exactly 100%",          # item 1
        "specific dollar values",       # item 2
        "named, falsifiable catalyst",  # item 3
        "Rating logically derives",     # item 4
        "Execution triggers are falsifiable",  # item 5
        "reachable in at least one scenario",  # item 6 (Flaw 8)
        "reference_price",              # item 7 (Flaw 2)
        "verbatim",                     # item 8 (Flaw 3)
        "Cross-section numerical consistency",  # item 9 (Flaw 5)
        "Sanity-check flags",           # item 10 (Flaw 4)
        "Inputs section",               # item 11
        "Peer comparisons cite specific",  # item 12
        "trace back to",                # item 13
    ]
    for keyword in must_contain:
        assert keyword in _QC_CHECKLIST, f"QC checklist missing: {keyword}"


def test_qc_checklist_has_self_correction_directive():
    """The system prompt must instruct the PM to self-correct on failure."""
    from tradingagents.agents.managers.portfolio_manager import _QC_CHECKLIST
    # The instruction to apply the checklist before final output
    assert "self-correct" in _QC_CHECKLIST.lower() or "revise" in _QC_CHECKLIST.lower()
```

- [ ] **Step 2: Run to fail**

```bash
.venv/bin/python -m pytest tests/test_pm_qc_checklist.py -v
```

Expected: ImportError on `_QC_CHECKLIST`.

- [ ] **Step 3: Add `_QC_CHECKLIST` constant + concat to system prompt**

In `tradingagents/agents/managers/portfolio_manager.py`, add a module-level constant:

```python
_QC_CHECKLIST = """

# Self-correction QC checklist (Quant Research Rebuild — 2026-05-03)

Before emitting your final decision.md, apply this 13-item checklist to your \
draft. If ANY item fails, revise the draft in place, then re-apply the \
checklist. Do not emit until every item passes.

1. Probabilities sum to exactly 100%.
2. All three price targets are specific dollar values (not ranges).
3. Each scenario lists at least one named, falsifiable catalyst.
4. Rating logically derives from EV — e.g., EV materially below spot → \
SELL or UNDERWEIGHT, not HOLD.
5. Execution triggers are falsifiable (named price / level / date).
6. (Flaw 8) Re-entry / upgrade triggers must be reachable in at least one \
scenario in the table. Example: if Bull peaks at $14 but you state \
"re-enter below $18," that's inconsistent — either revise the trigger or \
revise the scenario.
7. (Flaw 2) Every bare "<ticker> at <trade_date>" price citation matches \
reference_price ± $0.01. Other prices (article quotes, intraday) must \
carry an explicit time/source qualifier.
8. (Flaw 3) Every cited analyst position has a verbatim ≤ 30-word quote \
attributed by section. Statements like "Neutral's math, applied honestly, \
supports Sell" require a direct quote. If the cited claim is not in the \
source, the synthesis is invalid — re-synthesize.
9. (Flaw 5) Cross-section numerical consistency. Compare the same numerical \
claim (cash runway, target price, percentage move, etc.) across analyst \
reports + debate transcripts. Any claim that appears with different values \
in different sections must be reconciled in decision.md under a \
"Reconciliation" subsection.
10. (Flaw 4) Sanity-check flags from the fundamentals analyst are addressed \
in either the bull/bear debate or the trader's plan. Flagged items cannot \
be silently ignored.
11. Inputs section is present and complete in decision.md.
12. Peer comparisons cite specific numbers, not vague comparisons.
13. All claimed numbers trace back to raw/*.json data.

If revision after one self-correction loop still fails, see the push-back \
retry rules in the next section."""
```

Then concatenate `_QC_CHECKLIST` to the system message string (the same string `_MANDATED_SECTIONS` was appended to).

- [ ] **Step 4: Run tests + commit**

```bash
.venv/bin/python -m pytest tests/test_pm_qc_checklist.py tests/ -q --tb=short
git add tradingagents/agents/managers/portfolio_manager.py tests/test_pm_qc_checklist.py
git commit -m "feat(pm): 13-item self-correction QC checklist"
```

---

## Task 13: PM Pass-3 push-back retry mechanism

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/risk_mgmt/{aggressive,conservative,neutral}_debator.py`
- Create: `tests/test_pm_retry_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pm_retry_loop.py`:

```python
"""Tests for the PM Pass-3 push-back retry mechanism."""
import pytest

pytestmark = pytest.mark.unit


def test_pm_system_prompt_documents_retry_signal():
    """PM must know how to emit a structured retry decision."""
    from tradingagents.agents.managers.portfolio_manager import _RETRY_DIRECTIVE
    assert "retry" in _RETRY_DIRECTIVE.lower()
    assert "research_manager" in _RETRY_DIRECTIVE
    assert "risk_team" in _RETRY_DIRECTIVE
    # Cap rule
    assert "max" in _RETRY_DIRECTIVE.lower() or "1" in _RETRY_DIRECTIVE


def test_research_manager_handles_pm_feedback():
    """Research Manager prompt must reference pm_feedback when set."""
    from tradingagents.agents.managers.research_manager import _PM_FEEDBACK_HANDLER
    assert "pm_feedback" in _PM_FEEDBACK_HANDLER
    assert "address" in _PM_FEEDBACK_HANDLER.lower()


def test_risk_debators_handle_pm_feedback():
    """Each risk debator must reference pm_feedback in its prompt."""
    from tradingagents.agents.risk_mgmt.aggressive_debator import _PM_FEEDBACK_HANDLER as agg
    from tradingagents.agents.risk_mgmt.conservative_debator import _PM_FEEDBACK_HANDLER as con
    from tradingagents.agents.risk_mgmt.neutral_debator import _PM_FEEDBACK_HANDLER as neu
    for handler in (agg, con, neu):
        assert "pm_feedback" in handler
```

- [ ] **Step 2: Run to fail**

```bash
.venv/bin/python -m pytest tests/test_pm_retry_loop.py -v
```

Expected: ImportError on `_RETRY_DIRECTIVE` / `_PM_FEEDBACK_HANDLER`.

- [ ] **Step 3: Add `_RETRY_DIRECTIVE` to PM**

In `tradingagents/agents/managers/portfolio_manager.py`, add a module-level constant:

```python
_RETRY_DIRECTIVE = """

# Push-back retry (Pass 3 — substantive disagreement)

If after self-correction your draft still has a substantive disagreement \
with upstream synthesis (Research Manager's investment plan or Risk team's \
debate), emit a structured retry signal as the LAST line of your output, \
in this exact format on its own line:

PM_RETRY_SIGNAL: {"target": "research_manager", "feedback": "<≤200-word specific instruction>"}

Or:

PM_RETRY_SIGNAL: {"target": "risk_team", "feedback": "<≤200-word specific instruction>"}

The feedback should be specific, e.g., "Bull case scenarios assume Azure \
≥32% but no analyst quantified what AI capex ROI would have to be for that \
to clear; have RM re-do scenarios with explicit ROI hurdle."

Hard cap: max 1 retry per run. If state.pm_retries == 1, you cannot push \
back further; you must accept the best-available output and note remaining \
concerns in decision.md under "Caveats from PM."

Do NOT emit PM_RETRY_SIGNAL on the first pass unless you genuinely cannot \
ship the report. The signal triggers re-running upstream nodes (~3-4 extra \
LLM calls); use sparingly."""
```

Concatenate `_RETRY_DIRECTIVE` to the system message. Then add a parser at the end of `portfolio_manager_node`:

```python
import json as _json
import re as _re


def _parse_retry_signal(text: str) -> dict | None:
    """Look for `PM_RETRY_SIGNAL: {…}` on the last lines of the report."""
    m = _re.search(
        r"^PM_RETRY_SIGNAL:\s*(\{.+?\})\s*$",
        text,
        flags=_re.MULTILINE,
    )
    if not m:
        return None
    try:
        sig = _json.loads(m.group(1))
        if sig.get("target") in ("research_manager", "risk_team"):
            return sig
    except _json.JSONDecodeError:
        return None
    return None
```

In the node function, after computing `report`, add:

```python
        retry_signal = _parse_retry_signal(report)
        out = {
            "messages": [result] if hasattr(result, "content") else [],
            "final_trade_decision": report,
        }
        if retry_signal and state.get("pm_retries", 0) < 1:
            out["pm_feedback"] = retry_signal["feedback"]
            out["pm_retries"] = state.get("pm_retries", 0) + 1
            out["pm_retry_target"] = retry_signal["target"]
        return out
```

- [ ] **Step 4: Add `_PM_FEEDBACK_HANDLER` to Research Manager**

In `tradingagents/agents/managers/research_manager.py`, add a module-level constant:

```python
_PM_FEEDBACK_HANDLER = """

# Handling PM feedback (when re-invoked on retry)

If state.pm_feedback is set and non-empty, this is your second pass after \
the PM disagreed with your first investment plan. You must:

1. Quote the PM's feedback verbatim at the top of your revised plan, in a \
"## Addressing PM feedback" section.
2. Specifically address each concern raised. If the PM said "scenarios \
need explicit ROI hurdle," your scenarios must now include an explicit \
ROI hurdle.
3. Acknowledge what changed in your revised plan vs the first draft.

Do not silently ignore the feedback. Do not produce an identical plan."""
```

Concatenate to the system prompt. In the node function, locate where the prompt is built and add at the top of the user message:

```python
        pm_feedback = state.get("pm_feedback", "")
        feedback_block = (
            f"\n\nPM_FEEDBACK_FROM_PRIOR_PASS:\n{pm_feedback}\n"
            if pm_feedback else ""
        )
```

Then prepend `feedback_block` into the user-facing prompt content where it will reach the model.

- [ ] **Step 5: Same for the 3 risk debators**

Add the analogous `_PM_FEEDBACK_HANDLER` to each of:
- `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- `tradingagents/agents/risk_mgmt/conservative_debator.py`
- `tradingagents/agents/risk_mgmt/neutral_debator.py`

Each gets the same constant content and the same prompt-injection pattern as Research Manager (Step 4).

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/test_pm_retry_loop.py tests/ -q --tb=short
```

Expected: 3 new + all existing pass.

- [ ] **Step 7: Commit**

```bash
git add tradingagents/agents/managers/portfolio_manager.py tradingagents/agents/managers/research_manager.py tradingagents/agents/risk_mgmt/ tests/test_pm_retry_loop.py
git commit -m "feat(pm): push-back retry signal + RM/risk-team feedback handlers"
```

---

## Task 14: Wire new nodes into trading_graph

**Files:**
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `tradingagents/default_config.py`

- [ ] **Step 1: Read existing setup.py**

```bash
cat tradingagents/graph/setup.py | head -200
```

- [ ] **Step 2: Add new nodes + edges in setup.py**

In `tradingagents/graph/setup.py`, locate `setup_graph()`. Before the existing analyst nodes are added, insert:

```python
        from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node
        from tradingagents.agents.analysts.ta_agent import create_ta_agent_node
        from tradingagents.agents.researcher import fetch_research_pack
```

After the imports section but before nodes are added to the workflow, create the new node instances:

```python
        # Quant-research rebuild nodes (2026-05-03)
        pm_preflight_node = create_pm_preflight_node(self.deep_thinking_llm)
        ta_agent_node = create_ta_agent_node(self.quick_thinking_llm)

        def researcher_node(state):
            """Wraps the Python data fetcher as a LangGraph node."""
            fetch_research_pack(state)
            return {"raw_dir": state["raw_dir"]}
```

Add nodes to workflow:

```python
        workflow.add_node("PM Preflight", pm_preflight_node)
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("TA Agent", ta_agent_node)
```

Update the edge layout:
- Find where `START → first_analyst` is wired today. Replace it with: `START → PM Preflight → Researcher → TA Agent → first_analyst`.

```python
        # Replace the existing START → first_analyst with the rebuild prefix
        workflow.add_edge(START, "PM Preflight")
        workflow.add_edge("PM Preflight", "Researcher")
        workflow.add_edge("Researcher", "TA Agent")
        first_analyst = selected_analysts[0]
        workflow.add_edge("TA Agent", f"{first_analyst.capitalize()} Analyst")
```

(Remove the prior `workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")` line.)

For the retry edge from Portfolio Manager:

```python
        def pm_router(state):
            if state.get("pm_retries", 0) >= 1:
                return END
            target = state.get("pm_retry_target")
            if target == "research_manager":
                return "Research Manager"
            if target == "risk_team":
                return "Aggressive Analyst"  # entry point of risk debate
            return END

        workflow.add_conditional_edges(
            "Portfolio Manager",
            pm_router,
            {
                "Research Manager": "Research Manager",
                "Aggressive Analyst": "Aggressive Analyst",
                END: END,
            },
        )
```

(Remove the prior `workflow.add_edge("Portfolio Manager", END)` line.)

- [ ] **Step 3: Wire raw_dir in trading_graph.py**

In `tradingagents/graph/trading_graph.py`, locate where the initial state is constructed (in `propagate()` or `Propagator.create_initial_state`). Add `raw_dir` derivation:

```python
        # raw_dir under the per-run output_dir, used by all rebuild nodes.
        raw_dir = str(Path(self.config.get("output_dir", "/tmp")) / "raw")
        init_state["raw_dir"] = raw_dir
        init_state["peers"] = []  # populated by PM Preflight
        init_state["pm_brief"] = ""
        init_state["technicals_report"] = ""
        init_state["pm_feedback"] = ""
        init_state["pm_retries"] = 0
```

(Path import at top of file if not present: `from pathlib import Path`.)

- [ ] **Step 4: Drop bind_tools from analyst node tool_nodes (no longer needed)**

In `tradingagents/graph/trading_graph.py`, the existing `_create_tool_nodes()` builds ToolNode instances for each analyst. Since the refactored analysts no longer use bind_tools, the ToolNodes are unused; in `setup.py`, remove the `add_node(f"tools_{analyst_type}", ...)` lines and the `add_conditional_edges` wiring that loops back to tools. The analyst flow becomes simply `Analyst → next Analyst → Bull/Bear`.

Concretely in `setup.py`:

```python
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            # No more conditional edges to tools — analysts read raw/.
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_analyst, next_analyst)
            else:
                workflow.add_edge(current_analyst, "Bull Researcher")
```

(Remove the `tools_{analyst_type}` and `Msg Clear {…}` references; those nodes can stay defined for now and remain unused, or be removed in a follow-up cleanup.)

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all pass. Existing graph tests may need stub adjustments to provide `raw_dir` and stub the new nodes; if so, fix as needed.

- [ ] **Step 6: Smoke test on MacBook**

```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_test_oauth.py
```

Expected: PONG (regression check).

- [ ] **Step 7: Commit**

```bash
git add tradingagents/graph/setup.py tradingagents/graph/trading_graph.py
git commit -m "feat(graph): wire PM Preflight + Researcher + TA Agent + retry edges"
```

---

## Task 15: PDF includes new sections + final regression

**Files:**
- Modify: `cli/research_pdf.py`

- [ ] **Step 1: Add new sections to PDF template**

In `cli/research_pdf.py`, locate `_HTML_TEMPLATE`. Add two new sections at the start of the body (before the existing "Portfolio Manager — Final Decision" section):

```html
<h1>PM Pre-flight Brief</h1>
<div class="section-pretitle">Run mandate, business-model classification, peer set, framing.</div>
{pm_brief_html}

<h1>Technical Setup</h1>
<div class="section-pretitle">Major historical levels, volume zones, trading playbook.</div>
{technicals_html}
```

Then in the `build_research_pdf` function, render these from the raw/ folder:

```python
    pm_brief_html = render_md_from_path(out / "raw" / "pm_brief.md")
    technicals_html = render_md_from_path(out / "raw" / "technicals.md")
```

(Add a small helper `render_md_from_path(path)` that reads the file or returns "(missing)" — analogous to the existing `render_md(filename)` but for arbitrary paths.)

Pass the new variables into `_HTML_TEMPLATE.format(...)`.

- [ ] **Step 2: Run all tests + commit**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
git add cli/research_pdf.py
git commit -m "feat(pdf): include pm_brief.md and technicals.md sections"
```

- [ ] **Step 3: End-to-end regression on a known ticker**

```bash
ssh macmini-trueknot 'pkill -9 -f tradingresearch 2>&1; sleep 1
cd ~/tradingagents && git pull origin main 2>&1 | tail -3
.venv/bin/pip install -e . --quiet 2>&1 | tail -3
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043'
```

Wait ~12-18 min for completion. Verify:
- `decision.md` contains "## Inputs to this decision" and "## 12-Month Scenario Analysis" sections
- `raw/pm_brief.md` contains "## Business model classification"
- `raw/technicals.md` exists with all 5 mandated sections
- PDF in Telegram includes the new sections in the cover-page-then-content order

- [ ] **Step 4: Final commit + push + tag**

```bash
git push origin main
git tag phase6-quant-research-rebuild
```

---

## Self-review notes

**Spec coverage check:**
- ✅ PM Pre-flight (Task 4) — including Flaw 1 business-model classification
- ✅ Researcher + reference.json (Task 3) — Flaw 2 single-source-of-truth
- ✅ TA Agent (Task 5)
- ✅ 4 analysts refactored (Tasks 6-9) including peer matrix + sanity check + capital structure (Task 7) — Flaw 4 sanity-check
- ✅ Bull/Bear rigor (Task 10) — Flaw 6 strongest-arg-first, analogies-survive-scrutiny
- ✅ PM Inputs section + scenarios (Task 11) — Flaw 7 self-contained decision
- ✅ PM 13-item QC (Task 12) — Flaws 2, 3, 5, 8 enforcement
- ✅ Push-back retry + RM/Risk team feedback handlers (Task 13)
- ✅ Graph wiring (Task 14)
- ✅ PDF sections (Task 15)

**Type/method consistency:**
- `create_pm_preflight_node(llm)` — Task 4, used in Task 14
- `create_ta_agent_node(llm)` — Task 5, used in Task 14
- `fetch_research_pack(state)` — Task 3, used in Task 14
- `raw_dir_for(output_dir)` — Task 2, used by Researcher (could be inlined, kept as utility)
- State fields: `pm_brief`, `peers`, `raw_dir`, `technicals_report`, `pm_feedback`, `pm_retries` — defined Task 1, used Tasks 4, 5, 6-9, 11, 13, 14
- `_PM_FEEDBACK_HANDLER` constant in RM + 3 risk debators — same name, same shape, Task 13

**No placeholders:** every task has complete code; no "implement details" stubs.
