# Filing-anchor + Numerical Discipline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase-6.3 — fetch the most recent 10-Q/10-K from SEC EDGAR before each PM run, propagate it through Fundamentals/TA v2/PM via `raw/sec_filing.md`, and add QC items 15+16 (filing-anchor temporal correctness + multi-decimal numerical trace) — so the analyst tier stops calling already-public filings "pending" and stops fabricating ratios.

**Architecture:** PM Pre-flight (post-LLM) calls `sec_edgar.fetch_latest_filing(ticker, trade_date)`, writes the result to `raw/sec_filing.md`, and appends a one-line "filing already public" footer to `pm_brief.md`. Fundamentals + TA v2 + PM read `sec_filing.md` from raw via existing `format_for_prompt`. QC's 14-item checklist grows to 16: item 15 fails any "pending"/"awaiting" framing for filings whose text is in `raw/sec_filing.md`; item 16 fails any multi-decimal claim that doesn't trace verbatim to a `raw/` source cell. The implementation is already pre-written in the working tree — this plan structures the diff into per-task commits with TDD-style tests written or extended against the existing code (tests should pass on first run; if they don't, that's a real bug to fix).

**Tech Stack:** Python 3.13, `urllib.request` (no new dep), `html.parser.HTMLParser` (stdlib), pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-05-filing-anchor-numerical-discipline-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tradingagents/agents/utils/sec_edgar.py` | `fetch_latest_filing(ticker, trade_date)` + `format_for_prompt(filing)` — pulls latest 10-Q/10-K from EDGAR, strips HTML, returns dict or unavailable. Renders the Markdown block with the temporal-anchor instruction. |
| `tests/test_sec_edgar.py` | Unit tests: happy path, look-ahead-bias guard, unknown ticker, network failure, invalid date, content truncation, prompt-block content. |
| `tests/test_fundamentals_analyst.py` | New file. Asserts the Fundamentals system prompt has the YoY pre-write step + reads `sec_filing.md`. |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/managers/pm_preflight.py` | Fetch filing, write `raw/sec_filing.md`, append filing footer to `pm_brief.md`. |
| `tradingagents/agents/analysts/fundamentals_analyst.py` | Add YoY-from-financials.json mandatory pre-write step + filing-text mandatory read + add `sec_filing.md` to file list. |
| `tradingagents/agents/analysts/ta_agent.py` | Add `sec_filing.md` to TA v2 (only) file list. |
| `tradingagents/agents/managers/portfolio_manager.py` | Surface `sec_block` next to existing reference/technicals blocks in PM prompt. |
| `tradingagents/agents/managers/qc_agent.py` | Checklist 14 → 16 (item 15 = filing-anchor temporal correctness; item 16 = multi-decimal numerical trace). Update header counts. |
| `tests/test_pm_preflight.py` | +3 tests: writes `sec_filing.md` when filing available; omits when unavailable; handles fetcher exception gracefully. |
| `tests/test_ta_agent_v2.py` | +1 test: TA v2 reads `sec_filing.md` (TA v1 must not). |
| `tests/test_pm_qc_checklist.py` | +2 tests: PM prompt includes `sec_block` when `sec_filing.md` present; omits otherwise. |
| `tests/test_qc_agent.py` | +2 tests: checklist has 16 items + items 15/16 phrasing; QC fails draft that calls a filed 10-Q "pending". |

---

## Task 1: SEC EDGAR module + unit tests (foundation)

**Files:**
- Create: `tradingagents/agents/utils/sec_edgar.py` (already in working tree, untracked)
- Create: `tests/test_sec_edgar.py` (already in working tree, untracked)

The implementation and tests are already written. This task verifies they pass and commits them as the foundation for all downstream tasks.

- [ ] **Step 1: Confirm files are present and untracked**

```bash
git status --short tradingagents/agents/utils/sec_edgar.py tests/test_sec_edgar.py
```

Expected:
```
?? tests/test_sec_edgar.py
?? tradingagents/agents/utils/sec_edgar.py
```

- [ ] **Step 2: Run the SEC EDGAR unit tests**

```bash
.venv/bin/python -m pytest tests/test_sec_edgar.py -v
```

Expected: 8 passed (all 8 tests in `tests/test_sec_edgar.py` — happy path, look-ahead-bias skip, unknown ticker, network failure, invalid date, truncation, prompt-block content + temporal anchor, empty-on-unavailable).

If a test fails: read the failure carefully. The most likely cause is a stub mismatch (e.g., URL format change in `_FILING_URL`). Fix the code in `sec_edgar.py`, not the test.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: full suite passes. Record the test count for the next tasks.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/utils/sec_edgar.py tests/test_sec_edgar.py
git commit -m "$(cat <<'EOF'
feat(sec-edgar): pure-Python SEC EDGAR filing fetcher (Phase-6.3 foundation)

fetch_latest_filing pulls the most recent 10-Q/10-K filed on or before
trade_date from SEC EDGAR. Resolves CIK from a small in-module cache
plus fallback to SEC's directory; HTML-strips the primary document text
and truncates to a configurable budget. Includes 8 unit tests covering
happy path, look-ahead-bias guard, network failure, unknown ticker,
invalid date, and content truncation.

Foundation for the filing-anchor + numerical-discipline mitigation
spec'd at docs/superpowers/specs/2026-05-05-filing-anchor-numerical-discipline-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PM Pre-flight wiring + tests

**Files:**
- Modify: `tradingagents/agents/managers/pm_preflight.py` (already modified in working tree)
- Modify: `tests/test_pm_preflight.py` (extension — 3 new tests)

PM Pre-flight already has the EDGAR fetch wired in the working tree. This task adds the missing tests, runs them against the existing wiring, and commits both together.

- [ ] **Step 1: Append the 3 new tests to `tests/test_pm_preflight.py`**

Add at the bottom of the existing file (the `_stub_compute_calendar` autouse fixture at the top of the file already prevents network calls to yfinance — your stubs for `fetch_latest_filing` will compose correctly with it):

```python
def test_pm_preflight_writes_sec_filing_md_when_filing_available(tmp_path, monkeypatch):
    """If fetch_latest_filing returns a happy-path dict, PM Pre-flight writes
    raw/sec_filing.md AND appends a 'Recent SEC filing' footer to pm_brief.md."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    fake_filing = {
        "ticker": "MSFT",
        "form": "10-Q",
        "filing_date": "2026-04-29",
        "accession_number": "0001193125-26-191507",
        "primary_document": "msft-20260331.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm",
        "content": "Azure and other cloud services revenue increased 40%.",
        "content_truncated": False,
        "source": "sec.gov",
    }
    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: fake_filing,
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    node(state)

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert sec_path.exists()
    sec_content = sec_path.read_text(encoding="utf-8")
    assert "MSFT 10-Q filed 2026-04-29" in sec_content
    assert "Azure and other cloud services revenue increased 40%" in sec_content

    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "## Recent SEC filing (relative to trade_date 2026-05-01)" in brief
    assert "MSFT 10-Q filed 2026-04-29" in brief
    assert "treat as **known data**" in brief.lower() or "Treat as **known data**" in brief


def test_pm_preflight_omits_sec_filing_when_unavailable(tmp_path, monkeypatch):
    """If fetch_latest_filing returns unavailable, PM Pre-flight must NOT write
    raw/sec_filing.md and must NOT add the filing footer to pm_brief.md."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing",
        lambda t, d: {"unavailable": True, "reason": "EDGAR unreachable", "ticker": t},
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    node(state)

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert not sec_path.exists()
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "Recent SEC filing" not in brief


def test_pm_preflight_handles_fetcher_exception(tmp_path, monkeypatch):
    """If fetch_latest_filing raises, PM Pre-flight must degrade gracefully:
    no sec_filing.md written, no footer, pipeline still returns normally."""
    from tradingagents.agents.managers.pm_preflight import create_pm_preflight_node

    def _raises(t, d):
        raise RuntimeError("simulated EDGAR client crash")

    monkeypatch.setattr(
        "tradingagents.agents.utils.sec_edgar.fetch_latest_filing", _raises
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=_VALID_BRIEF)

    node = create_pm_preflight_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path / "raw"),
    }
    out = node(state)  # MUST NOT raise

    sec_path = tmp_path / "raw" / "sec_filing.md"
    assert not sec_path.exists()
    brief = (tmp_path / "raw" / "pm_brief.md").read_text(encoding="utf-8")
    assert "Recent SEC filing" not in brief
    assert out["pm_brief"] == brief
```

- [ ] **Step 2: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_pm_preflight.py -v
```

Expected: all PM Pre-flight tests pass (existing + 3 new).

If a test fails: the wiring in `pm_preflight.py` (the modifications already in the working tree) has a bug. Common failure modes:
- The `try/except` around `fetch_latest_filing` doesn't catch a non-`Exception` subclass (unlikely, but possible).
- The `monkeypatch.setattr` import path mismatches the actual import inside `pm_preflight.py`. The wiring imports `from tradingagents.agents.utils.sec_edgar import fetch_latest_filing` inside the function body — patching `tradingagents.agents.utils.sec_edgar.fetch_latest_filing` at the source-module level is the correct target.
- Footer phrasing differs from what the test asserts. Update the test assertion to match the actual footer (use whatever is in `pm_preflight.py` today as the canonical phrasing, but make sure the test still pins the temporal-anchor intent).

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/managers/pm_preflight.py tests/test_pm_preflight.py
git commit -m "$(cat <<'EOF'
feat(pm-preflight): fetch + propagate SEC filing (Phase-6.3 filing-anchor)

After the existing calendar-block append, PM Pre-flight now calls
sec_edgar.fetch_latest_filing(ticker, trade_date), writes raw/sec_filing.md
when a filing is available, and appends a 'Recent SEC filing' footer to
pm_brief.md so downstream agents (TA v2, Fundamentals, PM) see the filing
date and the temporal-anchor instruction inline.

Graceful degradation: if EDGAR is unreachable or the fetcher raises, no
file is written and no footer is added. Three new unit tests cover the
happy path, the unavailable path, and the fetcher-raises path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Fundamentals analyst — YoY pre-write + sec_filing read

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` (already modified in working tree)
- Create: `tests/test_fundamentals_analyst.py` (new file)

- [ ] **Step 1: Create `tests/test_fundamentals_analyst.py`**

Write the file:

```python
"""Tests for the Fundamentals analyst (Phase-6.3 numerical-discipline)."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.unit


def test_fundamentals_prompt_includes_yoy_preamble():
    """The system prompt must require a YoY computation pre-write step from
    financials.json so the analyst doesn't paraphrase ratios. Run #2 of the
    Phase-6.2 validation caught a fabricated 'capex/revenue 5.4%' (actual 37.3%)."""
    from tradingagents.agents.analysts.fundamentals_analyst import _SYSTEM
    assert "YoY computation from financials.json" in _SYSTEM
    assert "Revenue YoY" in _SYSTEM
    assert "Capex / revenue ratio" in _SYSTEM
    assert "DO NOT invent ratios" in _SYSTEM


def test_fundamentals_prompt_includes_sec_filing_read_step():
    """The system prompt must require the analyst to read raw/sec_filing.md
    when present and quote specific filing numbers (RPO, segment OpInc, etc.)."""
    from tradingagents.agents.analysts.fundamentals_analyst import _SYSTEM
    assert "raw/sec_filing.md" in _SYSTEM
    assert "Remaining Performance Obligations" in _SYSTEM
    assert "awaiting filing" in _SYSTEM or "pending adjudication" in _SYSTEM


def test_fundamentals_reads_sec_filing_md_when_present(monkeypatch, tmp_path):
    """Verify sec_filing.md is in the file list passed to format_for_prompt."""
    from tradingagents.agents.analysts import fundamentals_analyst

    captured = {}

    def fake_format(raw_dir, files):
        captured["files"] = list(files)
        return "stubbed context"

    monkeypatch.setattr(fundamentals_analyst, "format_for_prompt", fake_format)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="report body")

    node = fundamentals_analyst.create_fundamentals_analyst(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path),
    }
    node(state)

    assert "sec_filing.md" in captured["files"]
    # Sanity: existing files still present so we don't regress the prompt context.
    assert "pm_brief.md" in captured["files"]
    assert "financials.json" in captured["files"]
```

- [ ] **Step 2: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_fundamentals_analyst.py -v
```

Expected: 3 passed.

If `test_fundamentals_prompt_includes_yoy_preamble` fails: the working-tree edit to `fundamentals_analyst.py` doesn't have the exact phrasing the test expects. Open `fundamentals_analyst.py` and confirm the prompt addition is present; if the wording differs, update the test assertions to match the file's canonical phrasing as long as the core ideas (YoY, ratio, "DO NOT invent") are preserved.

If `test_fundamentals_reads_sec_filing_md_when_present` fails: the `format_for_prompt(...)` call site in `fundamentals_analyst.py` doesn't include `sec_filing.md` in its `files=` list. Open the file and confirm the list is `["pm_brief.md", "reference.json", "financials.json", "peers.json", "news.json", "sec_filing.md"]`.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py tests/test_fundamentals_analyst.py
git commit -m "$(cat <<'EOF'
feat(fundamentals): mandatory YoY pre-write + sec_filing.md read (Phase-6.3)

Adds two mandatory pre-write sections to the Fundamentals analyst system
prompt: (1) compute Revenue / OpInc / Capex YoY and Capex/Revenue ratio
from financials.json's quarterly columns before writing the report;
(2) read raw/sec_filing.md when present and quote specific filing numbers
(RPO, segment OpInc, Azure/cloud growth) verbatim. The format_for_prompt
file list now includes sec_filing.md.

Catches the Run-#2 failure mode where the analyst fabricated
"capex/revenue 5.4%" (actual 37.3%) by paraphrasing instead of computing
from raw data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: TA v2 reads sec_filing.md

**Files:**
- Modify: `tradingagents/agents/analysts/ta_agent.py` (already modified in working tree, single line change)
- Modify: `tests/test_ta_agent_v2.py` (extension — 1 new test)

- [ ] **Step 1: Append the new test to `tests/test_ta_agent_v2.py`**

```python
def test_ta_v2_reads_sec_filing_md_when_present(monkeypatch, tmp_path):
    """TA v2 must include sec_filing.md in its prompt context so the second-pass
    technical reviewer has fundamental context to anchor pattern interpretation.
    TA v1 (the first-pass narrow chart-read) intentionally does NOT load it."""
    from tradingagents.agents.analysts import ta_agent

    captured_v2 = {}

    def fake_format(raw_dir, files):
        captured_v2["files"] = list(files)
        return "stubbed context"

    monkeypatch.setattr(ta_agent, "format_for_prompt", fake_format)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="ta v2 report")

    node = ta_agent.create_ta_agent_v2_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(tmp_path),
        "market_report": "stub v1 report",
    }
    node(state)

    assert "sec_filing.md" in captured_v2["files"]
    assert "technicals.md" in captured_v2["files"]  # existing context preserved
```

You may need to add `from unittest.mock import MagicMock` and `from langchain_core.messages import AIMessage` at the top of the file if not already imported.

- [ ] **Step 2: Run the new test**

```bash
.venv/bin/python -m pytest tests/test_ta_agent_v2.py -v
```

Expected: all TA v2 tests pass (existing + 1 new).

If the new test fails: open `tradingagents/agents/analysts/ta_agent.py` and confirm `create_ta_agent_v2_node`'s `format_for_prompt(...)` call has `files=["technicals.md", "reference.json", "prices.json", "sec_filing.md"]`. The TA v1 node above must still NOT include `sec_filing.md`.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/analysts/ta_agent.py tests/test_ta_agent_v2.py
git commit -m "$(cat <<'EOF'
feat(ta-v2): read sec_filing.md for fundamental anchor (Phase-6.3)

The second-pass TA reviewer now reads raw/sec_filing.md when present so
chart-pattern interpretation can be cross-checked against the most recent
10-Q/10-K. Prevents the TA v2 'binary catalyst pending adjudication'
framing for filings already public on the trade date.

TA v1 (the first-pass narrow chart-read) intentionally still skips the
filing — its context budget should stay narrow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Portfolio Manager surfaces sec_block

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py` (already modified in working tree)
- Modify: `tests/test_pm_qc_checklist.py` (extension — 2 new tests)

- [ ] **Step 1: Append the 2 new tests to `tests/test_pm_qc_checklist.py`**

```python
def test_portfolio_manager_includes_sec_block_when_sec_filing_md_present(tmp_path, monkeypatch):
    """When raw/sec_filing.md exists, the PM prompt must include a 'Most recent
    SEC filing' block with the file's verbatim content + the temporal-anchor
    instruction. Catches the Run-#2 failure mode where the PM framed the
    already-public 10-Q as 'pending adjudication'."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    (raw / "sec_filing.md").write_text(
        "# SEC Filing — MSFT 10-Q filed 2026-04-29\n\n"
        "Azure and other cloud services revenue increased 40%.\n",
        encoding="utf-8",
    )

    captured = {}
    fake_llm = MagicMock()

    def _capture_invoke(messages):
        captured["prompt"] = "\n".join(
            m.content if hasattr(m, "content") else str(m) for m in messages
        )
        return AIMessage(content="## Inputs\n... full PM doc ...")

    fake_llm.invoke.side_effect = _capture_invoke

    node = create_portfolio_manager(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        # Minimum state the PM needs to render its prompt:
        "investment_debate_state": {"history": ""},
        "risk_debate_state": {"history": ""},
        "trader_investment_plan": "",
        "market_report": "stub",
        "sentiment_report": "stub",
        "news_report": "stub",
        "fundamentals_report": "stub",
        "technicals_report": "stub",
        "qc_feedback": "",
    }
    node(state)

    prompt = captured["prompt"]
    assert "Most recent SEC filing" in prompt
    assert "Azure and other cloud services revenue increased 40%" in prompt
    assert "treat as known data" in prompt.lower() or "treat as **known data**" in prompt.lower()


def test_portfolio_manager_omits_sec_block_when_sec_filing_md_missing(tmp_path):
    """When raw/sec_filing.md is absent, the PM prompt must NOT render a
    sec_block (no fabrication, no template residue)."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    # NO sec_filing.md

    captured = {}
    fake_llm = MagicMock()

    def _capture_invoke(messages):
        captured["prompt"] = "\n".join(
            m.content if hasattr(m, "content") else str(m) for m in messages
        )
        return AIMessage(content="## Inputs\n... full PM doc ...")

    fake_llm.invoke.side_effect = _capture_invoke

    node = create_portfolio_manager(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "investment_debate_state": {"history": ""},
        "risk_debate_state": {"history": ""},
        "trader_investment_plan": "",
        "market_report": "stub",
        "sentiment_report": "stub",
        "news_report": "stub",
        "fundamentals_report": "stub",
        "technicals_report": "stub",
        "qc_feedback": "",
    }
    node(state)

    prompt = captured["prompt"]
    assert "Most recent SEC filing" not in prompt
```

- [ ] **Step 2: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_pm_qc_checklist.py -v
```

Expected: all `test_pm_qc_checklist.py` tests pass (existing + 2 new).

If a test fails because the PM `node(state)` rejects the minimal state stub: open `portfolio_manager.py` and check what additional state keys the function accesses. Add the missing keys to the test stub. Do NOT change the production code to accept a smaller state — the test should mirror the real graph contract.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/managers/portfolio_manager.py tests/test_pm_qc_checklist.py
git commit -m "$(cat <<'EOF'
feat(pm): surface sec_filing.md content in PM prompt (Phase-6.3)

The PM prompt now includes a 'Most recent SEC filing' block when
raw/sec_filing.md exists, alongside the existing reference and technicals
blocks. The block carries the temporal-anchor instruction so the PM
cannot frame the already-public 10-Q as 'pending adjudication'.

Two new unit tests pin the present/absent behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: QC items 15+16 (filing-anchor + numerical-trace)

**Files:**
- Modify: `tradingagents/agents/managers/qc_agent.py` (already modified in working tree)
- Modify: `tests/test_qc_agent.py` (extension — 2 new tests)

- [ ] **Step 1: Append the 2 new tests to `tests/test_qc_agent.py`**

```python
def test_qc_checklist_has_16_items_and_filing_anchor_text():
    """The QC system prompt must (a) declare a 16-item checklist (was 14 pre-Phase-6.3),
    (b) include item 15 with key filing-anchor phrasing, (c) include item 16 with
    key numerical-trace phrasing."""
    from tradingagents.agents.managers.qc_agent import _SYSTEM

    assert "16-item checklist" in _SYSTEM
    # Item 15: filing-anchor temporal correctness
    assert "Filing-anchor temporal correctness" in _SYSTEM
    assert "raw/sec_filing.md" in _SYSTEM
    assert "pending adjudication" in _SYSTEM or "awaiting filing" in _SYSTEM
    # Item 16: numerical claims trace to source
    assert "Multi-decimal numerical claims" in _SYSTEM
    assert "trace" in _SYSTEM.lower()
    assert "raw/financials.json" in _SYSTEM


def test_qc_fails_pm_draft_calling_filed_10q_pending(tmp_path):
    """When raw/sec_filing.md exists and the PM draft frames its content as
    'pending adjudication', the QC verdict must be FAIL with feedback that
    references item 15. Catches the exact Run-#2 failure mode."""
    from tradingagents.agents.managers.qc_agent import create_qc_agent_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reference.json").write_text(
        '{"ticker": "MSFT", "trade_date": "2026-05-01", "reference_price": 410.0}',
        encoding="utf-8",
    )
    (raw / "sec_filing.md").write_text(
        "# SEC Filing — MSFT 10-Q filed 2026-04-29\n\n"
        "Azure revenue increased 40%.\n",
        encoding="utf-8",
    )

    pm_draft = (
        "## Inputs to this decision\n"
        "Reference price: $410.00 ...\n\n"
        "## Catalyst path\n"
        "The mid-May 10-Q is the binary catalyst pending adjudication ...\n"
    )

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content=(
        "Item 1: PASS\n"
        "...\n"
        "Item 15: FAIL — PM frames the already-filed 2026-04-29 10-Q as "
        "'pending adjudication' but raw/sec_filing.md contains its full text.\n"
        "Item 16: PASS\n"
        "QC_VERDICT: {\"status\": \"FAIL\", \"feedback\": "
        "\"Item 15 violation: the 10-Q referenced as 'pending' is already "
        "in raw/sec_filing.md (filed 2026-04-29). Rewrite the catalyst "
        "narrative around the NEXT 10-Q (~2026-07).\"}"
    ))

    node = create_qc_agent_node(fake_llm)
    state = {
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "final_trade_decision": pm_draft,
        "qc_retries": 0,
    }
    out = node(state)

    assert out.get("qc_passed") is False
    assert "Item 15" in out.get("qc_feedback", "") or "item 15" in out.get("qc_feedback", "").lower()
    assert "raw/sec_filing.md" in out.get("qc_feedback", "")
```

- [ ] **Step 2: Run the new tests**

```bash
.venv/bin/python -m pytest tests/test_qc_agent.py -v
```

Expected: all QC agent tests pass (existing + 2 new).

If `test_qc_checklist_has_16_items_and_filing_anchor_text` fails: open `qc_agent.py` and confirm the `_SYSTEM` constant declares "16-item checklist" (not "14-item") and that items 15 and 16 are present with the asserted key phrases. Update assertions if the file's canonical phrasing differs as long as the core ideas are preserved.

If `test_qc_fails_pm_draft_calling_filed_10q_pending` fails: the test stubs the LLM verdict, so a failure here implies the QC node's plumbing (state shape, return value contract) doesn't match the test stub. Compare with the existing passing test `test_qc_agent_fails_with_feedback_when_verdict_is_fail` for the canonical state shape.

- [ ] **Step 3: Run full unit suite to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/managers/qc_agent.py tests/test_qc_agent.py
git commit -m "$(cat <<'EOF'
feat(qc): items 15+16 — filing-anchor + numerical-trace (Phase-6.3)

Checklist grows 14 → 16:

Item 15 — Filing-anchor temporal correctness: FAIL any analyst quote or
PM claim that frames a filing in raw/sec_filing.md as "pending",
"awaiting filing", "not yet disclosed", or "the binary catalyst that
will reprice the trade". Filings already in raw/ are KNOWN DATA.

Item 16 — Multi-decimal numerical claims trace to source: FAIL any
"X% capex-to-revenue" / "Y bps compression" / "Zx multiple" that doesn't
trace verbatim to a cell in raw/financials.json, raw/sec_filing.md,
raw/peers.json, or raw/reference.json. Catches the Run-#2 fabricated
"5.4% capex-to-revenue" (actual 37.3%).

Two new unit tests: checklist phrasing pin + end-to-end FAIL on a
PM draft calling a filed 10-Q "pending adjudication".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: E2E validation on macmini

**Files:** none (operator step; outputs captured for the user-facing report)

- [ ] **Step 1: Push the branch + redeploy on macmini**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main && .venv/bin/pip install -e . --quiet'
ssh macmini-trueknot 'cd ~/tradingagents && git rev-parse --short HEAD'
```

Expected: HEAD on macmini matches the just-pushed SHA from Task 6.

- [ ] **Step 2: Refresh the OAuth token (8h TTL)**

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
```

Expected: a brief greeting response. The credentials.json mtime should advance.

- [ ] **Step 3: Run MSFT 2026-05-01**

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14–18 min for completion. Verify done:

```bash
ssh macmini-trueknot 'pgrep -fl tradingresearch | head -3'  # empty
ssh macmini-trueknot 'grep -E "^\[research\]" ~/.openclaw/data/logs/tradingresearch-2026-05-01-MSFT.log | tail -2'  # done line
```

- [ ] **Step 4: Inspect raw/sec_filing.md**

```bash
ssh macmini-trueknot 'head -30 ~/.openclaw/data/research/2026-05-01-MSFT/raw/sec_filing.md'
```

Expected:
- Title line `# SEC Filing — MSFT 10-Q filed 2026-04-29`
- Accession + URL lines
- The temporal-anchor instruction with "treat them as known data, NEVER as 'pending adjudication'"
- Then verbatim filing text (Azure / RPO / segment numbers should appear within the first ~5KB).

If `sec_filing.md` is missing entirely: check the log for an EDGAR fetch failure. Common causes: User-Agent rejected (look for HTTP 403), CIK lookup miss, primary-doc URL pattern change.

- [ ] **Step 5: Inspect pm_brief.md tail**

```bash
ssh macmini-trueknot 'tail -25 ~/.openclaw/data/research/2026-05-01-MSFT/raw/pm_brief.md'
```

Expected: ends with the Phase-6.2 "## Reporting status" calendar table THEN the Phase-6.3 "## Recent SEC filing (relative to trade_date 2026-05-01)" footer naming the same 2026-04-29 10-Q.

- [ ] **Step 6: Spot-check the Fundamentals report for YoY computation**

```bash
ssh macmini-trueknot '
grep -A 6 "Sanity check on reported numbers" ~/.openclaw/data/research/2026-05-01-MSFT/raw/analyst_fundamentals.md
'
```

Expected: the section contains the four computed ratios — Revenue YoY %, Operating Income YoY %, Capex YoY %, Capex/Revenue %. The Capex/Revenue figure must be in the 30–40% range (NOT the previously fabricated ~5%). If the value looks low, the analyst is still paraphrasing; the prompt change didn't fully bite. Escalate via the next item.

- [ ] **Step 7: Spot-check TA v2 for filing-pending phrasing**

```bash
ssh macmini-trueknot '
grep -in "pending adjudication\|pending\|awaiting filing\|binary catalyst" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals_v2.md || echo "NO PENDING-FILING PHRASING ✓"
'
```

Expected: `NO PENDING-FILING PHRASING ✓`. If matches appear, verify they refer to the NEXT 10-Q (~July 2026), not the just-filed 2026-04-29 one. Only the latter is a regression.

- [ ] **Step 8: Inspect QC verdict**

```bash
ssh macmini-trueknot '
tail -30 ~/.openclaw/data/research/2026-05-01-MSFT/raw/qc_verdict.md 2>/dev/null || \
grep -A 30 "QC_VERDICT" ~/.openclaw/data/research/2026-05-01-MSFT/raw/decision.md
'
```

Expected: `QC_VERDICT: {"status": "PASS"}` OR a FAIL with feedback citing real item-15 or item-16 violations (legitimate misses, not false positives on item 16).

- [ ] **Step 9: Report findings**

Pass criteria:
- ✅ `sec_filing.md` written with MSFT 10-Q content (including Azure/RPO numbers).
- ✅ `pm_brief.md` ends with the calendar block + the SEC-filing footer.
- ✅ Fundamentals "Sanity check" section has four computed YoY ratios; Capex/Revenue ≈ 35–40%.
- ✅ TA v2 + PM final do not call the 2026-04-29 10-Q "pending"/"awaiting".
- ✅ QC PASSes (or fails for legitimate reasons unrelated to items 15/16).

If all pass: report success — Phase 6.3 mitigations are working. The Run-#2 failure mode is closed.

If Fundamentals still fabricates ratios: tighten the prompt (e.g., add "Show your formula and the source columns used") or escalate to a deterministic Python pre-compute that writes `raw/fundamentals_yoy.json`, mirroring the calendar.json pattern.

If TA v2 still calls the filing "pending": the file is loaded but the LLM is ignoring the temporal anchor in the filing block header. Tighten by injecting a one-line system-prompt anchor in TA v2 similar to PM Pre-flight's calendar anchor.

If item 16 misfires (false positives on legitimate computed ratios): narrow item 16 to "ratios that don't appear in raw/ AND aren't computed from a raw cell whose formula is shown".

- [ ] **Step 10: Cleanup (optional)**

```bash
ssh macmini-trueknot 'ls ~/.openclaw/data/research/2026-05-01-MSFT*'
```

If you want to keep the run for archival, leave it. If not:

```bash
ssh macmini-trueknot 'rm -rf ~/.openclaw/data/research/2026-05-01-MSFT'
```

---

## Self-review notes

**Spec coverage:**
- ✅ `sec_edgar.py` module + 8 unit tests (Task 1)
- ✅ PM Pre-flight wiring + 3 unit tests (Task 2)
- ✅ Fundamentals YoY pre-write + sec_filing read + 3 unit tests (Task 3)
- ✅ TA v2 sec_filing read + 1 unit test (Task 4)
- ✅ Portfolio Manager sec_block + 2 unit tests (Task 5)
- ✅ QC items 15+16 + 2 unit tests (Task 6)
- ✅ E2E validation on macmini (Task 7)
- ✅ Out-of-scope items (8-K, historical 10-Qs, filing cache, XBRL) NOT touched
- ✅ Failure-mode reporting in Step 9 explicitly maps each pass criterion to a tightening recommendation

**Type / signature consistency:**
- `fetch_latest_filing(ticker: str, trade_date: str) -> dict` — referenced consistently across Task 1 (definition + tests), Task 2 (call site stub paths), and Task 7 (e2e expected output).
- `format_for_prompt(filing: dict) -> str` — same module path used in Task 1 stubs and Task 2 import.
- `sec_filing.md` filename — same in every task.
- Module path `tradingagents.agents.utils.sec_edgar` — same in every monkeypatch target.
- `raw/sec_filing.md` write site is in `pm_preflight.py` only (Task 2); Tasks 3–6 only READ this file.

**Placeholder scan:** No "TBD" / "TODO" / "implement later" / "similar to Task N" patterns. Each step shows exact code blocks, exact commands, exact expected output. Tasks 3 and 4 carry their own `from unittest.mock import MagicMock` reminder so an engineer reading those tasks out of order doesn't miss imports.

**Out-of-scope confirmation:** This plan does not add EDGAR caching, 8-K ingestion, XBRL parsing, historical 10-Q lookback, or peer-ticker filing fetches. Each is called out in the spec's "Out of scope" section and remains untouched.

**Rollback path:** Each task commits independently. Reverting Task 6 returns QC to 14 items but leaves `sec_filing.md` written and unused for downstream agents. Reverting Task 5 removes the PM block. Reverting Task 4 removes TA v2's read. Reverting Task 3 removes the Fundamentals prompt change. Reverting Task 2 stops `sec_filing.md` from being written. Reverting Task 1 removes the module entirely.

**Test count delta:** +19 new unit tests across the seven tasks (8 sec_edgar + 3 pm_preflight + 3 fundamentals + 1 ta_v2 + 2 pm_qc + 2 qc_agent — there are 3 fundamentals tests in Task 3, not 2 as initially scoped, because the Fundamentals system-prompt change has two distinct sub-additions worth pinning separately).
