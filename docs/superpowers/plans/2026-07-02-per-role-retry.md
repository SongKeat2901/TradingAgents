# Per-Role Retry (FA-101 Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each of the 4 fundamentals role nodes a deterministic self-check and a capped self-loop so a structurally-incomplete role is re-run on its own with targeted feedback — "retry the failed part, no full stack."

**Architecture:** Generalize the existing `qc_validator_retries` self-correct loop. Each role node, after invoking, runs `check_role_output` (required-section-headers present + length floor). On failure with retries left it returns `passed=False` + feedback + incremented retries; a per-role conditional edge self-loops the node (re-prompting with the feedback) or advances after the cap. Deterministic, fail-open, no downstream change.

**Tech Stack:** LangGraph `add_conditional_edges` (self-loop), `AgentState` plain-int control fields, `claude -p` CLI LLM (no bind_tools), pytest (`-m unit`).

## Global Constraints

- **Deterministic checks only** — no LLM in the loop. `check_role_output` reads required `## ` headers + a length floor; no file materialization.
- **`ROLE_RETRY_CAP = 2`**, fail-open: after the cap the node advances with whatever it produced. Never block the pipeline.
- **No downstream change, no new QC item.** The aggregator + 6 consumers are untouched; a role's finally-emitted report is what gets aggregated. Keep the partial report even when a node caps out (return the report, not `""`).
- **Avoid gratuitous retries (false positives).** Only ALWAYS-expected headers go in a role's required list; conditional sections (Catalysts' `## Deal math`, which the prompt lets collapse to "No material deals in window") are NOT required. Length floor is a low "obviously-truncated" value (600), NOT the in-call completeness target (`min_chars=1200`), so the node check doesn't fight `invoke_with_empty_retry`'s own soft retry.
- **No `bind_tools`.** Feedback is injected into the existing HumanMessage, mirroring `qc_feedback`.
- **Commit footer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Conventional-commit prefixes. Baseline unit count before this plan: **816**.

---

### Task 1: Per-role control state fields + init

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py` (after the 4 role-report keys)
- Modify: `tradingagents/graph/propagation.py` (the init dict that seeds `qc_retries`/`qc_validator_retries` etc.)
- Test: `tests/test_role_retry_state.py` (new)

**Interfaces:**
- Produces: for each `<role>` in {financial, riskflags, catalysts, quality}: `fundamentals_<role>_retries: int`, `fundamentals_<role>_feedback: str`, `fundamentals_<role>_passed: bool` — all plain (no reducer), seeded `0`/`""`/`False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_role_retry_state.py
import pytest
pytestmark = pytest.mark.unit

_ROLES = ["financial", "riskflags", "catalysts", "quality"]

def test_role_control_keys_present():
    from tradingagents.agents.utils.agent_states import AgentState
    ann = AgentState.__annotations__
    for r in _ROLES:
        for suf in ("retries", "feedback", "passed"):
            assert f"fundamentals_{r}_{suf}" in ann, f"missing fundamentals_{r}_{suf}"

def test_role_control_keys_initialized():
    from tradingagents.graph.propagation import Propagator
    init = Propagator().create_initial_state("MSFT", "2026-07-01")
    for r in _ROLES:
        assert init[f"fundamentals_{r}_retries"] == 0
        assert init[f"fundamentals_{r}_feedback"] == ""
        assert init[f"fundamentals_{r}_passed"] is False
```

Adapt `Propagator().create_initial_state(...)` to the REAL signature (READ `propagation.py` first — it may be `create_initial_state(company, date)` or similar).

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Add fields + init.** In `agent_states.py`, after `fundamentals_quality_report`:

```python
    fundamentals_financial_retries: int
    fundamentals_financial_feedback: str
    fundamentals_financial_passed: bool
    fundamentals_riskflags_retries: int
    fundamentals_riskflags_feedback: str
    fundamentals_riskflags_passed: bool
    fundamentals_catalysts_retries: int
    fundamentals_catalysts_feedback: str
    fundamentals_catalysts_passed: bool
    fundamentals_quality_retries: int
    fundamentals_quality_feedback: str
    fundamentals_quality_passed: bool
```

In `propagation.py`'s initial-state dict, beside `"qc_validator_retries": 0`, add all 12 seeds (`_retries: 0`, `_feedback: ""`, `_passed: False`).

- [ ] **Step 4: Run — pass + full suite (816 + tests).**

- [ ] **Step 5: Commit** `feat: add per-role retry control state (retries/feedback/passed x4)`

---

### Task 2: `check_role_output` + `format_role_feedback` + required-header lists

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_roles.py` (add helpers + `_REQUIRED_*` + `ROLE_RETRY_CAP`)
- Test: `tests/test_role_retry.py` (new)

**Interfaces:**
- Produces:
  - `ROLE_RETRY_CAP = 2`
  - `_REQUIRED_FINANCIAL = ["## Business-model framing", "## Peer comparison matrix", "## Capital-structure compare", "## Sanity check on reported numbers"]`
  - `_REQUIRED_RISK = ["## Risk & red flags"]`
  - `_REQUIRED_CATALYSTS = ["## Insider transactions", "## What management needs to prove", "## Sentiment & consensus"]`  (NOT `## Deal math` — conditional)
  - `_REQUIRED_QUALITY = ["## Competitive position", "## Capital-allocation track record", "## Ownership & governance"]`
  - `check_role_output(required_headers, report, min_chars=600) -> list[str]`
  - `format_role_feedback(issues) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_role_retry.py
import pytest
pytestmark = pytest.mark.unit
from tradingagents.agents.analysts import fundamentals_roles as fr

def test_missing_header_reported():
    report = "## Peer comparison matrix\n" + "x" * 800
    issues = fr.check_role_output(fr._REQUIRED_FINANCIAL, report)
    assert any("Business-model framing" in i for i in issues)
    assert any("Sanity check" in i for i in issues)
    assert not any("Peer comparison matrix" in i for i in issues)  # present -> not flagged

def test_complete_report_passes():
    report = "".join(h + "\n" for h in fr._REQUIRED_QUALITY) + "y" * 800
    assert fr.check_role_output(fr._REQUIRED_QUALITY, report) == []

def test_short_report_flagged():
    report = "".join(h + "\n" for h in fr._REQUIRED_RISK) + "short"
    issues = fr.check_role_output(fr._REQUIRED_RISK, report)
    assert any("too short" in i.lower() or "length" in i.lower() for i in issues)

def test_format_feedback_lists_issues():
    fb = fr.format_role_feedback(["missing section: ## Foo", "report too short"])
    assert "## Foo" in fb and "too short" in fb

def test_cap_constant():
    assert fr.ROLE_RETRY_CAP == 2
```

- [ ] **Step 2: Run — fail.**

- [ ] **Step 3: Implement** in `fundamentals_roles.py`:

```python
ROLE_RETRY_CAP = 2

_REQUIRED_FINANCIAL = ["## Business-model framing", "## Peer comparison matrix",
                       "## Capital-structure compare", "## Sanity check on reported numbers"]
_REQUIRED_RISK = ["## Risk & red flags"]
_REQUIRED_CATALYSTS = ["## Insider transactions", "## What management needs to prove",
                       "## Sentiment & consensus"]
_REQUIRED_QUALITY = ["## Competitive position", "## Capital-allocation track record",
                     "## Ownership & governance"]


def check_role_output(required_headers, report, min_chars=600):
    """Deterministic structural check: every required header present + a length
    floor. Returns human-readable issues; empty list == passed."""
    text = report or ""
    issues = [f"missing required section: {h}" for h in required_headers if h not in text]
    if len(text.strip()) < min_chars:
        issues.append(f"report too short ({len(text.strip())} chars < {min_chars})")
    return issues


def format_role_feedback(issues):
    lines = "\n".join(f"- {i}" for i in issues)
    return ("Your previous draft was incomplete. Fix these before rewriting the "
            "full section:\n" + lines)
```

- [ ] **Step 4: Run — pass + full suite.**

- [ ] **Step 5: Commit** `feat: deterministic role-output check + feedback formatter + required headers`

---

### Task 3: Node self-check + feedback injection (all 4 roles)

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_roles.py` (the 4 node bodies)
- Test: `tests/test_role_retry.py` (extend)

**Interfaces:**
- Consumes: `check_role_output`, `format_role_feedback`, `_REQUIRED_*`, `ROLE_RETRY_CAP`, the per-role state fields (Task 1).
- Produces (per role, e.g. financial): node returns `{"messages": [result], "fundamentals_financial_report": report, "fundamentals_financial_passed": passed, "fundamentals_financial_feedback": fb, "fundamentals_financial_retries": prior + (0 if passed else 1)}`. On re-entry the node reads the prior feedback and appends it to the HumanMessage.

**Node body change** (apply the analogous edit to all 4; financial shown):

```python
def create_financial_statement_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_FINANCIAL)
        prior_fb = state.get("fundamentals_financial_feedback", "")
        human = (f"For your reference: {instrument_context}\n\n{context}\n\n"
                 f"Write the financial-statement analysis.")
        if prior_fb:
            human += f"\n\n{prior_fb}"
        messages = [
            SystemMessage(content=_SYSTEM_FINANCIAL.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=human),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Financial-Statement Analyst", min_chars=1200)
        issues = check_role_output(_REQUIRED_FINANCIAL, report)
        passed = not issues
        prior = state.get("fundamentals_financial_retries", 0)
        return {
            "messages": [result],
            "fundamentals_financial_report": report,
            "fundamentals_financial_passed": passed,
            "fundamentals_financial_feedback": "" if passed else format_role_feedback(issues),
            "fundamentals_financial_retries": prior if passed else prior + 1,
        }
    return node
```

Use `_REQUIRED_RISK`/`_REQUIRED_CATALYSTS`/`_REQUIRED_QUALITY` and the matching state keys for the other 3.

- [ ] **Step 1: Write the failing tests** (extend `tests/test_role_retry.py`)

```python
class _StubSeq:
    """Returns queued responses in order; .content on each."""
    def __init__(self, *bodies): self._b = list(bodies); self.seen = []
    def invoke(self, msgs):
        self.seen.append(msgs[-1].content)
        class R: pass
        r = R(); r.content = self._b.pop(0) if self._b else self._b_last  # noqa
        return r

def _raw(tmp_path):
    import json, os
    for f in ("pm_brief.md","reference.json","financials.json","peers.json","sec_filing.md"):
        p = tmp_path / f
        p.write_text("{}" if f.endswith(".json") else "# stub", encoding="utf-8")
    return str(tmp_path)

def test_node_fails_check_sets_feedback(tmp_path):
    from tradingagents.agents.analysts.fundamentals_roles import create_financial_statement_analyst
    bad = "## Peer comparison matrix\n" + "x"*1300  # missing 3 required headers
    node = create_financial_statement_analyst(_StubSeq(bad))
    out = node({"company_of_interest":"MSFT","trade_date":"2026-07-01","raw_dir":_raw(tmp_path)})
    assert out["fundamentals_financial_passed"] is False
    assert out["fundamentals_financial_retries"] == 1
    assert "Business-model framing" in out["fundamentals_financial_feedback"]

def test_node_passes_clears_feedback(tmp_path):
    from tradingagents.agents.analysts.fundamentals_roles import create_financial_statement_analyst, _REQUIRED_FINANCIAL
    good = "".join(h+"\n" for h in _REQUIRED_FINANCIAL) + "x"*1300
    node = create_financial_statement_analyst(_StubSeq(good))
    out = node({"company_of_interest":"MSFT","trade_date":"2026-07-01","raw_dir":_raw(tmp_path)})
    assert out["fundamentals_financial_passed"] is True
    assert out["fundamentals_financial_feedback"] == ""
    assert out["fundamentals_financial_retries"] == 0

def test_node_injects_prior_feedback(tmp_path):
    from tradingagents.agents.analysts.fundamentals_roles import create_financial_statement_analyst, _REQUIRED_FINANCIAL
    good = "".join(h+"\n" for h in _REQUIRED_FINANCIAL) + "x"*1300
    stub = _StubSeq(good)
    node = create_financial_statement_analyst(stub)
    node({"company_of_interest":"MSFT","trade_date":"2026-07-01","raw_dir":_raw(tmp_path),
          "fundamentals_financial_feedback":"- missing required section: ## Peer comparison matrix"})
    assert "Peer comparison matrix" in stub.seen[-1]  # prior feedback reached the prompt
```

If `invoke_with_empty_retry` calls `llm.invoke` more than once on short output, make the stub return the SAME body each call (set `_b_last`); adapt the stub to the real `invoke_with_empty_retry` contract (READ `structured.py`).

- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement the 4 node bodies** as above.
- [ ] **Step 4: Run** focused + import smoke + full suite. All pass.
- [ ] **Step 5: Commit** `feat: role nodes self-check output + inject retry feedback into re-prompt`

---

### Task 4: Conditional self-loop edges in the graph

**Files:**
- Modify: `tradingagents/graph/setup.py` (the 4 role→role `add_edge` calls become `add_conditional_edges`)
- Test: `tests/test_graph_role_split.py` (extend) or `tests/test_graph_role_retry.py` (new)

**Interfaces:**
- Consumes: the per-role `_passed`/`_retries` state keys.
- Produces: each `"<Role> Analyst"` node has a conditional edge `{"retry": itself, "advance": <next node>}`.

**Router factory** (add to `setup.py` near `qc_router`, or `conditional_logic.py`):

```python
def make_role_router(passed_key, retries_key, cap):
    def router(state):
        if state.get(passed_key) or state.get(retries_key, 0) >= cap:
            return "advance"
        return "retry"
    return router
```

**Wiring** — replace the 4 sequential `add_edge` calls among the role nodes. For each role node `SRC` with next node `DST`:

```python
workflow.add_conditional_edges(
    SRC, make_role_router("fundamentals_<role>_passed", "fundamentals_<role>_retries", ROLE_RETRY_CAP),
    {"retry": SRC, "advance": DST},
)
```

Mapping: Financial-Statement→Risk & Red-Flags; Risk & Red-Flags→Catalysts & Ownership; Catalysts & Ownership→Competitive-Quality; Competitive-Quality→Fundamentals Aggregator. The entry edge (previous analyst → Financial-Statement) and the aggregator→TA v2 edge are unchanged. Import `ROLE_RETRY_CAP` from `fundamentals_roles`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph_role_retry.py
import pytest
pytestmark = pytest.mark.unit

def test_each_role_node_has_self_loop_and_advance():
    # Build the uncompiled StateGraph as test_graph_role_split.py does.
    from tradingagents.graph.setup import GraphSetup  # adapt to real construction
    wf = _build_default_graph_builder()  # reuse the helper/pattern from test_graph_role_split.py
    roles = {
        "Financial-Statement Analyst": "Risk & Red-Flags Analyst",
        "Risk & Red-Flags Analyst": "Catalysts & Ownership Analyst",
        "Catalysts & Ownership Analyst": "Competitive-Quality Analyst",
        "Competitive-Quality Analyst": "Fundamentals Aggregator",
    }
    for src, dst in roles.items():
        branch_targets = _conditional_targets(wf, src)  # inspect wf.branches/edges for the node
        assert src in branch_targets      # self-loop ("retry")
        assert dst in branch_targets      # advance
```

READ `test_graph_role_split.py` first and reuse its exact graph-inspection surface (`setup_graph()` → builder `.nodes`/`.edges`, and for conditional edges inspect `wf.branches`). Adapt `_conditional_targets` to however LangGraph exposes conditional-edge targets in this version. Binding assertions: for each role node, its conditional targets include BOTH itself (retry) and the correct next node (advance).

- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement** the router factory + swap the 4 edges to conditional.
- [ ] **Step 4: Run** graph test + import smoke (`.venv/bin/python -c "import tradingagents.graph.setup"`) + FULL suite. Investigate any regression (a self-loop must not create an infinite cycle — the cap guarantees termination; confirm no test hangs).
- [ ] **Step 5: Commit** `feat: per-role self-loop conditional edges (retry failed role, cap 2)`

---

## Post-plan (controller, after final review + merge)

- Merge `feat/fa101-per-role-retry` → `main`, push, deploy to mini.
- **Live smoke (mini):** a real run; confirm in logs that a healthy role passes with zero extra loops (no gratuitous retries) and — ideally — force one failure (e.g. temporarily tighten a required header) to observe a single self-loop re-run producing a complete section, then revert. Verify total runtime is not materially inflated on the happy path.
- With Phase 4 shipped, goal component (c) per-role retry is DONE. Remaining program items: FA-101 coverage gaps — Phase 2b (SEC-fetch 13F/13D/8-K/DEF 14A) and Phase 5 (red-flag screens, incremental ROIC).
