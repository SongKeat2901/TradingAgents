# TA Judge Transparency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TA Agent v2 node that reads the four analyst reports + emits a refined technical view, force the PM to transcribe its agree/disagree relationship to v2, and extend the QC Agent to verify the transcription.

**Architecture:** Insert one new Sonnet node between the four analysts and the bull/bear debate. The node consumes `state.market_report` / `fundamentals_report` / `news_report` / `sentiment_report` plus `raw/technicals.md` (v1) plus `raw/reference.json` and `raw/prices.json`, and writes `raw/technicals_v2.md` (overwriting `state.technicals_report` with v2 contents). All downstream debate agents gain `state.technicals_report` in their prompts. PM Final's `_MANDATED_SECTIONS` gains a required "Technical setup adopted" subsection. QC Agent's `_SYSTEM` gains a 14th checklist item that verifies the subsection. PDF renders v2.

**Tech Stack:** Python 3.13, langchain_core BaseChatModel, existing langchain_core.messages SystemMessage/HumanMessage, pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-04-ta-judge-transparency-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tests/test_ta_agent_v2.py` | Unit tests for the v2 factory + node behavior |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/analysts/ta_agent.py` | Add `_SYSTEM_V2` constant + `create_ta_agent_v2_node(llm)` factory |
| `tradingagents/graph/setup.py` | Import + instantiate v2 node; wire between last analyst and Bull Researcher |
| `tradingagents/agents/researchers/bull_researcher.py` | Add Technicals Report block to prompt |
| `tradingagents/agents/researchers/bear_researcher.py` | Same |
| `tradingagents/agents/managers/research_manager.py` | Same |
| `tradingagents/agents/trader/trader.py` | Same |
| `tradingagents/agents/risk_mgmt/aggressive_debator.py` | Same |
| `tradingagents/agents/risk_mgmt/conservative_debator.py` | Same |
| `tradingagents/agents/risk_mgmt/neutral_debator.py` | Same |
| `tradingagents/agents/managers/portfolio_manager.py` | Add Technicals Report block + new "Technical setup adopted" subsection in `_MANDATED_SECTIONS` |
| `tradingagents/agents/managers/qc_agent.py` | Add Item 14 to `_SYSTEM` checklist |
| `tests/test_pm_qc_checklist.py` | Verify the new mandated subsection text in `_MANDATED_SECTIONS` |
| `tests/test_qc_agent.py` | Verify Item 14 enumeration in QC `_SYSTEM` |
| `cli/research_pdf.py` | Render `raw/technicals_v2.md` instead of `raw/technicals.md` for the Technical Setup section |

---

## Task 1: TA Agent v2 factory

**Files:**
- Modify: `tradingagents/agents/analysts/ta_agent.py`
- Create: `tests/test_ta_agent_v2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ta_agent_v2.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_ta_agent_v2.py -v
```

Expected: ImportError on `create_ta_agent_v2_node` and `_SYSTEM_V2`.

- [ ] **Step 3: Add `_SYSTEM_V2` and factory to `ta_agent.py`**

Open `tradingagents/agents/analysts/ta_agent.py`. After the existing `_SYSTEM` constant (and before `create_ta_agent_node`), add:

```python
_SYSTEM_V2 = """\
You are the TA Agent doing a second-pass review for $TICKER on $DATE.

Your v1 setup is in raw/technicals.md. The four analysts have now produced \
their reports (Market, Fundamentals, News, Sentiment). Your job is to read \
their reports plus your own v1 read, then emit a refined technical setup that \
addresses any analyst pushback that materially affects the technical view.

Produce a Markdown report with EXACTLY these sections (use the headers verbatim):

## Revisions from v1

For each material revision, name the analyst whose pushback caused the change \
(verbatim quote ≤30 words) and state what changed and why.

If no revision is warranted, this section reads exactly:
"No revisions — analyst reports did not surface evidence to revise v1's classification."

## Major historical levels

[Same table format as v1: Level | Price | Type | Why crowds trade here]

## Volume profile zones

- Heaviest accumulation: $<low>-$<high>
- Volume gap: $<low>-$<high>

## Current technical state

Narrative on RSI, MACD, moving-average stack, divergences.

## Setup classification

One of: breakout / breakdown / consolidation / distribution / accumulation.

## Asymmetry

- Upside to next major resistance: $<price> (<+X>%)
- Downside to next major support: $<price> (<-Y>%)
- Reward/risk: <ratio>:1

The v2 view is what every downstream agent (bull/bear/RM/trader/risk team/PM) \
will reason over. Cite specific numbers. Address fundamental concerns when they \
materially shift the technical read."""


def create_ta_agent_v2_node(llm):
    """Factory: returns the TA Agent v2 LangGraph node function.

    Reads v1 + four analyst reports + raw/reference.json + raw/prices.json.
    Writes raw/technicals_v2.md and overwrites state.technicals_report.
    """

    def ta_agent_v2_node(state: dict) -> dict[str, Any]:
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]

        v1_context = format_for_prompt(
            raw_dir,
            files=["technicals.md", "reference.json", "prices.json"],
        )
        analyst_block = (
            f"\n## Market Analyst Report\n{state.get('market_report', '(missing)')}\n\n"
            f"## Fundamentals Analyst Report\n{state.get('fundamentals_report', '(missing)')}\n\n"
            f"## News Analyst Report\n{state.get('news_report', '(missing)')}\n\n"
            f"## Sentiment Analyst Report\n{state.get('sentiment_report', '(missing)')}\n"
        )

        messages = [
            SystemMessage(content=_SYSTEM_V2.replace("$TICKER", ticker).replace("$DATE", date)),
            HumanMessage(content=(
                f"Produce the v2 technicals report for {ticker} on {date}. "
                f"Below are the v1 setup, the four analyst reports, and the "
                f"reference snapshot. Refine and emit v2.\n\n"
                f"{v1_context}\n{analyst_block}"
            )),
        ]
        result = llm.invoke(messages)
        raw_content = result.content if hasattr(result, "content") else None
        report = raw_content if raw_content else str(result)

        (Path(raw_dir) / "technicals_v2.md").write_text(report, encoding="utf-8")

        return {
            "messages": [result] if raw_content is not None else [],
            "technicals_report": report,
        }

    return ta_agent_v2_node
```

(`format_for_prompt`, `Path`, `SystemMessage`, `HumanMessage`, and `Any` are already imported by the existing `create_ta_agent_node`.)

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_ta_agent_v2.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/ta_agent.py tests/test_ta_agent_v2.py
git commit -m "feat(ta-agent): v2 factory — refines technicals after analyst pushback"
```

---

## Task 2: Wire TA v2 into the graph

**Files:**
- Modify: `tradingagents/graph/setup.py`

- [ ] **Step 1: Add the import**

Open `tradingagents/graph/setup.py`. Find the existing line:

```python
from tradingagents.agents.analysts.ta_agent import create_ta_agent_node
```

Replace with:

```python
from tradingagents.agents.analysts.ta_agent import create_ta_agent_node, create_ta_agent_v2_node
```

- [ ] **Step 2: Instantiate the v2 node**

In the same file, find the existing line:

```python
        ta_agent_node = create_ta_agent_node(self.quick_thinking_llm)
```

Add immediately after:

```python
        ta_agent_v2_node = create_ta_agent_v2_node(self.quick_thinking_llm)
```

- [ ] **Step 3: Add the workflow node**

Find:

```python
        workflow.add_node("TA Agent", ta_agent_node)
```

Add immediately after:

```python
        workflow.add_node("TA Agent v2", ta_agent_v2_node)
```

- [ ] **Step 4: Rewire the analyst → bull edge through TA v2**

Find the existing analyst-chain wiring:

```python
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"

            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_analyst, next_analyst)
            else:
                workflow.add_edge(current_analyst, "Bull Researcher")
```

Replace with:

```python
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"

            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_analyst, next_analyst)
            else:
                workflow.add_edge(current_analyst, "TA Agent v2")
        workflow.add_edge("TA Agent v2", "Bull Researcher")
```

- [ ] **Step 5: Run unit tests to confirm no regressions**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all 137 (or more) pass. The graph builds without error.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/graph/setup.py
git commit -m "feat(graph): wire TA Agent v2 between last analyst and Bull Researcher"
```

---

## Task 3: Add Technicals Report block to Bull researcher

**Files:**
- Modify: `tradingagents/agents/researchers/bull_researcher.py`

- [ ] **Step 1: Read existing bull_researcher.py prompt** to understand the current shape

```bash
grep -n "market_research_report\|fundamentals_report\|sentiment_report\|news_report" tradingagents/agents/researchers/bull_researcher.py
```

You'll see the four state fields are pulled into local variables and interpolated into the prompt template. Add a fifth.

- [ ] **Step 2: Add `technicals_report` to the state read + prompt**

Find:

```python
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
```

Replace with:

```python
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        technicals_report = state.get("technicals_report", "")
```

Find the prompt template's report block (lines that interpolate the four reports, near `Social media sentiment report:`). Add a fifth line for technicals. The existing block looks like:

```python
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
```

Add:

```python
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
```

- [ ] **Step 3: Run unit tests**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/researchers/bull_researcher.py
git commit -m "feat(bull): consume refined technicals_report from TA v2"
```

---

## Task 4: Add Technicals Report block to Bear researcher

**Files:**
- Modify: `tradingagents/agents/researchers/bear_researcher.py`

- [ ] **Step 1: Apply the same change as Task 3**

Open `tradingagents/agents/researchers/bear_researcher.py`. Find the four `state[...]` reads and add:

```python
        technicals_report = state.get("technicals_report", "")
```

Find the four-report interpolation block (`Social media sentiment report: {sentiment_report}` etc.) and add a fifth line:

```python
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
```

- [ ] **Step 2: Run unit tests**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/agents/researchers/bear_researcher.py
git commit -m "feat(bear): consume refined technicals_report from TA v2"
```

---

## Task 5: Add Technicals Report block to Research Manager + Trader + 3 risk debaters

**Files:**
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/trader/trader.py`
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py`

The pattern is identical to Tasks 3-4: read `technicals_report` from state, interpolate it in the user prompt with the line `Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}`. Apply to each of the five files.

- [ ] **Step 1: Update `research_manager.py`**

Find the existing four-report read block (look for `state["market_report"]` near the start of the node function). Add:

```python
        technicals_report = state.get("technicals_report", "")
```

Find where the four reports are interpolated into the prompt. Add the fifth line:

```python
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
```

- [ ] **Step 2: Update `trader.py`**

`trader.py` reads `state["investment_plan"]` rather than the four analyst reports — it inherits the technical view through the RM's plan. Add `technicals_report` so the trader sees the v2 view directly:

Find the `trader_node` function. After the `investment_plan = state["investment_plan"]` line, add:

```python
        technicals_report = state.get("technicals_report", "")
```

In the user-message content, append:

```
\n\nRefined technicals report (TA Agent v2): {technicals_report}
```

- [ ] **Step 3: Update `aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`**

Each debater has the same shape. For each file:

Find:

```python
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
```

Add:

```python
        technicals_report = state.get("technicals_report", "")
```

Find the four-report interpolation block in the prompt and add:

```python
Refined technicals report (TA Agent v2, post-analyst reconciliation): {technicals_report}
```

- [ ] **Step 4: Run unit tests**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/research_manager.py tradingagents/agents/trader/trader.py tradingagents/agents/risk_mgmt/
git commit -m "feat(downstream): RM + Trader + 3 risk debaters consume refined technicals_report"
```

---

## Task 6: PM Final — add Technicals block + mandated subsection

**Files:**
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tests/test_pm_qc_checklist.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_pm_qc_checklist.py`. Add the following test function at the bottom of the file:

```python
def test_mandated_sections_includes_technical_setup_adopted():
    """The PM's _MANDATED_SECTIONS must require the new 'Technical setup adopted'
    subsection so the disagreement with TA v2 is transcribed in decision.md."""
    from tradingagents.agents.managers.portfolio_manager import _MANDATED_SECTIONS
    assert "Technical setup adopted" in _MANDATED_SECTIONS
    assert "TA Agent v2 classification" in _MANDATED_SECTIONS
    # The three required choices
    assert "adopt" in _MANDATED_SECTIONS
    assert "partially adopt" in _MANDATED_SECTIONS
    assert "reject" in _MANDATED_SECTIONS
    # Reasoning length floor
    assert "≤80 words" in _MANDATED_SECTIONS or "<= 80 words" in _MANDATED_SECTIONS
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_pm_qc_checklist.py::test_mandated_sections_includes_technical_setup_adopted -v
```

Expected: FAIL — "Technical setup adopted" not in `_MANDATED_SECTIONS`.

- [ ] **Step 3: Extend `_MANDATED_SECTIONS` in `portfolio_manager.py`**

Open `tradingagents/agents/managers/portfolio_manager.py`. Find the `_MANDATED_SECTIONS` constant. Inside the section that begins `## Inputs to this decision`, add this block immediately AFTER the existing bullet list (after `**Data freshness:**` line) and BEFORE the `## 12-Month Scenario Analysis` heading:

```
**Technical setup adopted:**

- TA Agent v2 classification: <verbatim quote from raw/technicals_v2.md "Setup classification" section>
- My read: [adopt / partially adopt / reject]
- Reasoning: <≤80 words of evidence-based explanation citing specific analyst transcripts>
- Working setup for this decision: <one-line summary>

This subsection is non-negotiable. The TA Agent v2 produced a refined \
technical view after reading all four analyst reports; you must transcribe \
how your final decision relates to that view rather than routing around it. \
"adopt" means you accept v2's classification verbatim; "partially adopt" \
means you accept the classification but qualify it with named conditions; \
"reject" means you are overriding it and must cite specific analyst \
evidence that justifies the override.
```

- [ ] **Step 4: Add Technicals Report to PM user prompt**

In the same file, find the `portfolio_manager_node` function. Locate the line:

```python
        reference_block = _load_reference_block(state)
```

Add immediately after:

```python
        technicals_block = (
            f"\n\n**Refined technicals (TA Agent v2):**\n"
            f"{state.get('technicals_report', '(missing)')}\n"
        )
```

Find the `prompt = f"""..."""` assignment. Find the line that ends with `{instrument_context}{reference_block}`. Replace with:

```python
{instrument_context}{reference_block}{technicals_block}
```

Verify the resulting `f"""..."""` block has `instrument_context`, `reference_block`, and `technicals_block` interpolated. The qc_block (added in a previous commit) appears later in the same string.

- [ ] **Step 5: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_pm_qc_checklist.py -v
```

Expected: all PM QC tests (including the new one) pass.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/managers/portfolio_manager.py tests/test_pm_qc_checklist.py
git commit -m "feat(pm): mandate 'Technical setup adopted' subsection + consume v2 technicals"
```

---

## Task 7: QC Agent Item 14

**Files:**
- Modify: `tradingagents/agents/managers/qc_agent.py`
- Modify: `tests/test_qc_agent.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_qc_agent.py`. Find `test_qc_system_prompt_lists_all_13_items`. Replace its body with the 14-item version:

```python
def test_qc_system_prompt_lists_all_14_items():
    """The QC agent's system prompt must enumerate the 14 checklist items
    (13 original + 1 for the Technical setup adopted subsection)."""
    from tradingagents.agents.managers.qc_agent import _SYSTEM
    for n in range(1, 15):
        assert f"{n}." in _SYSTEM, f"QC system prompt missing item {n}"
    assert "QC_VERDICT:" in _SYSTEM
    # Item 14 specifics
    assert "Technical setup adopted" in _SYSTEM
    assert "verbatim" in _SYSTEM
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_qc_agent.py::test_qc_system_prompt_lists_all_14_items -v
```

Expected: FAIL — `14.` not in `_SYSTEM`.

- [ ] **Step 3: Extend QC `_SYSTEM` checklist**

Open `tradingagents/agents/managers/qc_agent.py`. Find the existing checklist line `13.` near the end of the checklist enumeration:

```
13. Numerical claims in the document trace back to raw/*.json or the analyst \
reports. No invented numbers, no "approximately" stand-ins for unsourced figures.
```

Add immediately after:

```
14. The "Technical setup adopted" subsection exists inside the Inputs section, \
names the TA Agent v2 classification verbatim (from raw/technicals_v2.md), \
picks one of {adopt, partially adopt, reject}, and provides ≥30-word reasoning \
that cites at least one specific analyst transcript. Skipping this subsection \
or filling it with vague phrasing like "I adopt the technical setup" without \
evidence → FAIL.
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_qc_agent.py -v
```

Expected: all QC tests pass (including the renamed `test_qc_system_prompt_lists_all_14_items`).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/qc_agent.py tests/test_qc_agent.py
git commit -m "feat(qc): item 14 — verify Technical setup adopted subsection"
```

---

## Task 8: PDF renders v2 technicals

**Files:**
- Modify: `cli/research_pdf.py`

- [ ] **Step 1: Switch the Technical Setup PDF section to read v2**

Open `cli/research_pdf.py`. Find the line in `build_research_pdf`:

```python
    technicals_html = render_md_from_path(out / "raw" / "technicals.md")
```

Replace with:

```python
    # Use the refined v2 view (post-analyst reconciliation) when available;
    # fall back to v1 if v2 is missing (e.g., older runs predating TA v2).
    technicals_v2 = out / "raw" / "technicals_v2.md"
    technicals_v1 = out / "raw" / "technicals.md"
    technicals_html = render_md_from_path(
        technicals_v2 if technicals_v2.exists() else technicals_v1
    )
```

- [ ] **Step 2: Run unit tests**

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: all pass.

- [ ] **Step 3: Smoke-build a PDF locally to confirm v2 renders**

If you have a recent `state.json` and `raw/technicals_v2.md` available locally (e.g., from a prior macmini run), regenerate the PDF and verify the Technical Setup section shows the v2 content (with the "Revisions from v1" header at the top of the section). Skip if no local state available — Task 9's e2e covers this.

- [ ] **Step 4: Commit**

```bash
git add cli/research_pdf.py
git commit -m "feat(pdf): render technicals_v2.md (v2 view) when available, fall back to v1"
```

---

## Task 9: End-to-end smoke test on macmini-trueknot

**Files:** none (operator step)

- [ ] **Step 1: Push the branch + redeploy on macmini**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main && .venv/bin/pip install -e . --quiet'
```

- [ ] **Step 2: Run a fresh MSFT 2026-05-01**

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~15-20 min for completion (one extra Sonnet call vs prior runs).

- [ ] **Step 3: Verify v2 file exists with required structure**

```bash
ssh macmini-trueknot '
cat ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals_v2.md | head -40
grep -E "^## (Revisions from v1|Major historical levels|Volume profile zones|Current technical state|Setup classification|Asymmetry)" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals_v2.md
'
```

Expected: 6 section headers (Revisions + 5 mandated). v1 + v2 both exist in `raw/`.

- [ ] **Step 4: Verify decision.md contains the new subsection**

```bash
ssh macmini-trueknot '
grep -A 5 "Technical setup adopted" ~/.openclaw/data/research/2026-05-01-MSFT/decision.md
'
```

Expected: subsection present with `TA Agent v2 classification:`, `My read:`, `Reasoning:`, `Working setup`.

- [ ] **Step 5: Verify QC retry behavior**

```bash
ssh macmini-trueknot '
python3 -c "
import json
s = json.load(open(\"/Users/trueknot/.openclaw/data/research/2026-05-01-MSFT/state.json\"))
print(\"qc_passed:\", s.get(\"qc_passed\"))
print(\"qc_retries:\", s.get(\"qc_retries\"))
print(\"qc_feedback len:\", len(s.get(\"qc_feedback\", \"\")))
"
'
```

Expected: `qc_passed: True`. If `qc_retries: 1`, that confirms QC caught and pushed back at least once. If `qc_retries: 0`, the PM nailed Item 14 on first pass.

- [ ] **Step 6: Verify the Telegram PDF renders the v2 technicals**

Open the PDF delivered to the chat and visually confirm the "Technical Setup" section starts with a "Revisions from v1" subsection.

- [ ] **Step 7: If everything passes, no commit needed**

If something fails, file a follow-up bug rather than reverting — the structural pieces are independent and individually rollback-able per the spec's "Rollback path".

---

## Self-review notes

**Spec coverage check:**
- ✅ TA Agent v2 module + factory (Task 1)
- ✅ TA v2 system prompt with "Revisions from v1" + 5 mandated sections (Task 1)
- ✅ TA v2 reads all four analyst reports + raw v1/reference/prices (Task 1)
- ✅ TA v2 always emits (no conditional skip) (Task 1)
- ✅ TA v2 overwrites state.technicals_report (Task 1)
- ✅ Graph wiring inserts v2 between last analyst and Bull Researcher (Task 2)
- ✅ Bull/Bear consume technicals_report (Tasks 3-4)
- ✅ RM/Trader/3 risk debaters consume technicals_report (Task 5)
- ✅ PM Final consumes technicals_report + has new mandated subsection (Task 6)
- ✅ QC Agent Item 14 verifies the subsection (Task 7)
- ✅ PDF renders v2 (Task 8)
- ✅ Market Analyst intentionally NOT modified (consumes v1 per T6 design) — verified by absence from file list
- ✅ End-to-end smoke verifies all of the above (Task 9)

**Type / signature consistency:**
- `create_ta_agent_v2_node(llm)` returns a closure node that takes `state` and returns a dict with `messages` and `technicals_report` keys — same shape as v1.
- `state.technicals_report` is a string (not a structured object) consistent with the existing AgentState field.
- All consumer additions read `state.get("technicals_report", "")` for backward compat with mid-pipeline test stubs.

**Placeholder scan:** No TBDs / TODOs. Each prompt-update task gives the exact line to add.

**Out of scope confirmation:** TA arithmetic errors, fundamentals net-cash inconsistency, and memory consistency are all explicitly noted in the spec as separate workstreams. Nothing in this plan touches those areas.
