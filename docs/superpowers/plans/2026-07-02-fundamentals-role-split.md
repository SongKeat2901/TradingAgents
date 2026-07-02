# Fundamentals Role-Split (FA-101 Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `Fundamentals Analyst` node with 4 clear-role LangGraph nodes (Financial-Statement, Risk & Red-Flags, Catalysts & Ownership, Competitive-Quality) plus a deterministic aggregator that preserves the `fundamentals_report` contract, so no downstream consumer changes.

**Architecture:** 4 sequential LLM nodes each own a disjoint slice of today's 10 sections and write their own state key; a no-LLM aggregator concatenates them into `fundamentals_report`. Downstream (TA v2, bull, bear, 3 risk debators) is untouched.

**Tech Stack:** LangGraph `StateGraph`, `AgentState` TypedDict, `claude -p` CLI LLM (no `bind_tools`), pytest (`-m unit`).

## Global Constraints

- **No `bind_tools()` on any node** — CLI invocation only (project gotcha: bind_tools → direct-API 429s). Use `invoke_with_empty_retry(llm, messages, name, min_chars=…)`.
- **Preserve every mandate verbatim.** Each section's anti-fabrication / citation-discipline / net-debt-restatement / distress-cite text moves with the section, word-for-word from the current `tradingagents/agents/analysts/fundamentals_analyst.py` `_SYSTEM` (lines cited per task). No loosening, no paraphrase.
- **`$TICKER` / `$DATE` placeholders** are substituted in the factory via `.replace("$TICKER", ticker).replace("$DATE", date)` exactly as the current node does — keep them literal in prompt text.
- **Shared closing footer** appended to ALL 4 role prompts, verbatim from current lines 177-178: `"Every numerical claim in your report must trace back to financials.json, peers.json, news.json, reference.json, or insider.json. No invented numbers."`
- **The existing `fundamentals_report` state key stays** and remains what all downstream nodes read; only the aggregator writes it now.
- **Commit footer:** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Conventional-commit prefixes. Baseline unit count before this plan: **804**.

---

### Task 1: Add the 4 role state keys

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py` (after line 58, the `fundamentals_report` field)
- Test: `tests/test_agent_states_role_keys.py` (new)

**Interfaces:**
- Produces: 4 new `AgentState` fields — `fundamentals_financial_report`, `fundamentals_riskflags_report`, `fundamentals_catalysts_report`, `fundamentals_quality_report` (all `Annotated[str, …]`). Later tasks' node factories return dicts keyed by these; the aggregator reads them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_states_role_keys.py
import pytest
pytestmark = pytest.mark.unit

def test_role_report_keys_present():
    from tradingagents.agents.utils.agent_states import AgentState
    ann = AgentState.__annotations__
    for k in ("fundamentals_financial_report", "fundamentals_riskflags_report",
              "fundamentals_catalysts_report", "fundamentals_quality_report",
              "fundamentals_report"):
        assert k in ann, f"missing state key: {k}"
```

- [ ] **Step 2: Run — verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent_states_role_keys.py -q`
Expected: FAIL (new keys absent).

- [ ] **Step 3: Add the fields**

In `tradingagents/agents/utils/agent_states.py`, immediately after the `fundamentals_report` field (line 58), add:

```python
    fundamentals_financial_report: Annotated[str, "Financial-Statement role report"]
    fundamentals_riskflags_report: Annotated[str, "Risk & Red-Flags role report"]
    fundamentals_catalysts_report: Annotated[str, "Catalysts & Ownership role report"]
    fundamentals_quality_report: Annotated[str, "Competitive-Quality role report"]
```

Match the surrounding indentation (4 spaces, class body).

- [ ] **Step 4: Run — verify pass + full suite**

Run: `.venv/bin/python -m pytest tests/test_agent_states_role_keys.py -q` → PASS
Run: `.venv/bin/python -m pytest -q -m unit --tb=line` → 804 + 1 new, no regressions.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tests/test_agent_states_role_keys.py
git commit -m "feat: add 4 fundamentals role-report state keys"
```

---

### Task 2: The 4 role prompt modules + node factories

**Files:**
- Create: `tradingagents/agents/analysts/fundamentals_roles.py`
- Test: `tests/test_fundamentals_roles.py` (new)
- Reference (read, do NOT edit yet): `tradingagents/agents/analysts/fundamentals_analyst.py` — the source `_SYSTEM` text to partition.

**Interfaces:**
- Consumes: `format_for_prompt`, `invoke_with_empty_retry`, `build_instrument_context`, `get_language_instruction` (same imports as the current `fundamentals_analyst.py` lines 15-22).
- Produces: 4 factories, each `create_<role>(llm) -> node_fn(state) -> dict`. Return keys:
  - `create_financial_statement_analyst` → `{"messages": [result], "fundamentals_financial_report": report}`
  - `create_risk_redflags_analyst` → `{"messages": [result], "fundamentals_riskflags_report": report}`
  - `create_catalysts_ownership_analyst` → `{"messages": [result], "fundamentals_catalysts_report": report}`
  - `create_competitive_quality_analyst` → `{"messages": [result], "fundamentals_quality_report": report}`

**Prompt partition — move these EXACT blocks from the current `_SYSTEM` (`fundamentals_analyst.py`) into each role's `_SYSTEM`, verbatim:**

- **Financial-Statement** (`_SYSTEM_FINANCIAL`): the YoY pre-compute mandate (current lines 33-49), the SEC-filing pre-read mandate (lines 51-60), `## Business-model framing` (64-67), `## Peer comparison matrix` (79-92), `## Capital-structure compare` INCLUDING the net-debt-discipline paragraph (94-109), `## Sanity check on reported numbers` (123-132).
- **Risk & Red-Flags** (`_SYSTEM_RISK`): the "Distress screen discipline" paragraph (111-115) and "Manipulation screen discipline" paragraph (117-121), presented under a `## Risk & red flags` header, plus a directive to summarize solvency/risk-factor red flags grounded in `raw/sec_filing.md` risk factors. Cite the Altman Z″ and Beneish M blocks verbatim per those paragraphs.
- **Catalysts & Ownership** (`_SYSTEM_CATALYSTS`): `## Deal math` (69-77), `## Insider transactions` (134-142), `## What management needs to prove` (144-146), plus a `## Sentiment & consensus` directive: "When pm_brief.md carries a `## Sentiment & consensus` block, cite its short-interest %/days-to-cover and analyst rating + target upside verbatim; else 'not reported'."
- **Competitive-Quality** (`_SYSTEM_QUALITY`): `## Competitive position` (148-154), `## Capital-allocation track record` (156-161), `## Ownership & governance` (163-168), and the "Qualitative-claim discipline" paragraph (170-175).

Each `_SYSTEM_*` starts with a role-specific one-line intro replacing the current lines 25-31 preamble, e.g.:

```
You are the {ROLE} analyst writing that part of an equity research report on \
$TICKER for trade date $DATE. You have been given pm_brief.md and the raw data \
files below. NO tool calls — the data is in front of you.
```

and ends with the shared footer (Global Constraints). Keep `$TICKER`/`$DATE` literal.

**Files slice per role** (passed to `format_for_prompt`):
- Financial-Statement: `["pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md"]`
- Risk & Red-Flags: `["pm_brief.md", "reference.json", "financials.json", "sec_filing.md"]`
- Catalysts & Ownership: `["pm_brief.md", "reference.json", "news.json", "insider.json"]`
- Competitive-Quality: `["pm_brief.md", "reference.json", "financials.json", "sec_filing.md", "news.json"]`

**Factory template** (identical shape to current node lines 181-217; instantiate 4×):

```python
def create_financial_statement_analyst(llm):
    def node(state):
        ticker = state["company_of_interest"]
        date = state["trade_date"]
        raw_dir = state["raw_dir"]
        instrument_context = build_instrument_context(ticker)
        context = format_for_prompt(raw_dir, files=_FILES_FINANCIAL)
        messages = [
            SystemMessage(content=_SYSTEM_FINANCIAL.replace("$TICKER", ticker).replace("$DATE", date)
                          + "\n" + get_language_instruction()),
            HumanMessage(content=f"For your reference: {instrument_context}\n\n{context}\n\n"
                         f"Write the financial-statement analysis."),
        ]
        result, report = invoke_with_empty_retry(llm, messages, "Financial-Statement Analyst", min_chars=1200)
        return {"messages": [result], "fundamentals_financial_report": report}
    return node
```

Use `min_chars=1200` for each role (each is ~¼ the old 2000-char single report; keeps the empty-retry guard without over-demanding a focused section). The `HumanMessage` write-instruction line names the role.

- [ ] **Step 1: Write the failing tests** (`tests/test_fundamentals_roles.py`)

```python
import pytest
pytestmark = pytest.mark.unit
from tradingagents.agents.analysts import fundamentals_roles as fr

def test_financial_prompt_and_files():
    s = fr._SYSTEM_FINANCIAL
    assert "## Business-model framing" in s and "## Peer comparison matrix" in s
    assert "## Sanity check on reported numbers" in s
    assert "Revenue YoY" in s  # YoY pre-compute mandate moved here
    assert "restating a figure already shown" in s  # net-debt discipline preserved
    assert set(fr._FILES_FINANCIAL) == {"pm_brief.md", "reference.json", "financials.json", "peers.json", "sec_filing.md"}

def test_risk_prompt_and_files():
    s = fr._SYSTEM_RISK
    assert "Altman Z" in s and "Beneish M-score" in s
    assert "sec_filing.md" in fr._FILES_RISK

def test_catalysts_prompt_and_files():
    s = fr._SYSTEM_CATALYSTS
    assert "## Deal math" in s and "## Insider transactions" in s
    assert "## What management needs to prove" in s
    assert "Sentiment & consensus" in s
    assert set(fr._FILES_CATALYSTS) == {"pm_brief.md", "reference.json", "news.json", "insider.json"}

def test_quality_prompt_and_files():
    s = fr._SYSTEM_QUALITY
    assert "## Competitive position" in s and "## Capital-allocation track record" in s
    assert "## Ownership & governance" in s
    assert "not determinable from" in s  # qualitative-claim discipline preserved

def test_shared_footer_on_all():
    for s in (fr._SYSTEM_FINANCIAL, fr._SYSTEM_RISK, fr._SYSTEM_CATALYSTS, fr._SYSTEM_QUALITY):
        assert "No invented numbers." in s

def test_factories_return_role_keys():
    # a stub llm whose invoke returns an object with .content
    class _Stub:
        def invoke(self, msgs):
            class R:  # min_chars=1200 -> pad
                content = "x" * 1500
            return R()
    state = {"company_of_interest": "MSFT", "trade_date": "2026-05-05", "raw_dir": "/nonexistent"}
    # format_for_prompt tolerates a missing dir (returns missing-markers); if it raises, use tmp
    import tradingagents.agents.analysts.fundamentals_roles as m
    for factory, key in [
        (m.create_financial_statement_analyst, "fundamentals_financial_report"),
        (m.create_risk_redflags_analyst, "fundamentals_riskflags_report"),
        (m.create_catalysts_ownership_analyst, "fundamentals_catalysts_report"),
        (m.create_competitive_quality_analyst, "fundamentals_quality_report"),
    ]:
        out = factory(_Stub())(state)
        assert key in out and len(out[key]) >= 1200
```

If `format_for_prompt` raises on a nonexistent `raw_dir`, the implementer must create the 4 raw files under a `tmp_path` in `test_factories_return_role_keys` (empty JSON `{}` + a stub `pm_brief.md`) and set `raw_dir` to it. Adjust the test to whatever `format_for_prompt` actually needs; do NOT weaken the key/length asserts.

- [ ] **Step 2: Run — verify prompt tests fail** (module missing).

Run: `.venv/bin/python -m pytest tests/test_fundamentals_roles.py -q` → FAIL (ImportError).

- [ ] **Step 3: Implement `fundamentals_roles.py`** — same import block as `fundamentals_analyst.py` lines 15-22; define `_FILES_*` lists, `_SYSTEM_*` prompts (partition above), 4 factories (template above). Transcribe section text verbatim from the current `_SYSTEM`.

- [ ] **Step 4: Run — prompt tests + factory test pass; full suite**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_roles.py -q` → PASS
Run: `.venv/bin/python -c "import tradingagents.agents.analysts.fundamentals_roles"` → no error
Run: `.venv/bin/python -m pytest -q -m unit --tb=line` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_roles.py tests/test_fundamentals_roles.py
git commit -m "feat: 4 clear-role fundamentals analyst nodes (financial/risk/catalysts/quality)"
```

---

### Task 3: Deterministic aggregator node

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_roles.py` (add `create_fundamentals_aggregator`)
- Test: `tests/test_fundamentals_aggregator.py` (new)

**Interfaces:**
- Consumes: the 4 role keys from state.
- Produces: `create_fundamentals_aggregator() -> node_fn(state) -> {"fundamentals_report": combined}`. No `llm` arg (deterministic), no `messages` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fundamentals_aggregator.py
import pytest
pytestmark = pytest.mark.unit
from tradingagents.agents.analysts.fundamentals_roles import create_fundamentals_aggregator

def _state(**kw):
    base = {"fundamentals_financial_report": "FIN", "fundamentals_riskflags_report": "RISK",
            "fundamentals_catalysts_report": "CAT", "fundamentals_quality_report": "QUAL"}
    base.update(kw); return base

def test_aggregates_all_four_in_order():
    out = create_fundamentals_aggregator()(_state())
    r = out["fundamentals_report"]
    assert r.index("FIN") < r.index("RISK") < r.index("CAT") < r.index("QUAL")
    assert "# Fundamentals" in r
    for h in ("Financial-Statement", "Risk & Red-Flags", "Catalysts & Ownership", "Competitive-Quality"):
        assert h in r

def test_missing_role_gets_placeholder_not_dropped():
    out = create_fundamentals_aggregator()(_state(fundamentals_catalysts_report=""))
    r = out["fundamentals_report"]
    assert "unavailable" in r.lower()
    assert "FIN" in r and "QUAL" in r  # others intact

def test_missing_key_does_not_raise():
    out = create_fundamentals_aggregator()({})  # no role keys at all
    assert "fundamentals_report" in out  # never raises
```

- [ ] **Step 2: Run — verify fail** (`create_fundamentals_aggregator` undefined).

- [ ] **Step 3: Implement** in `fundamentals_roles.py`:

```python
_ROLE_SECTIONS = [
    ("Financial-Statement", "fundamentals_financial_report"),
    ("Risk & Red-Flags", "fundamentals_riskflags_report"),
    ("Catalysts & Ownership", "fundamentals_catalysts_report"),
    ("Competitive-Quality", "fundamentals_quality_report"),
]

def create_fundamentals_aggregator():
    def node(state):
        parts = ["# Fundamentals\n"]
        for title, key in _ROLE_SECTIONS:
            body = (state.get(key) or "").strip()
            if not body:
                body = f"_({title} section unavailable)_"
            parts.append(f"## {title}\n\n{body}\n")
        return {"fundamentals_report": "\n".join(parts)}
    return node
```

- [ ] **Step 4: Run — aggregator tests + full suite pass.**

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_roles.py tests/test_fundamentals_aggregator.py
git commit -m "feat: deterministic fundamentals aggregator preserving fundamentals_report contract"
```

---

### Task 4: Wire the 4 nodes + aggregator into the graph; retire the old node

**Files:**
- Modify: `tradingagents/graph/setup.py` (node creation ~lines 69-72, registration ~line 116, analyst-chain edges ~lines 129-144)
- Delete: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Delete: `tests/test_fundamentals_analyst.py`, `tests/test_fundamentals_prompt.py` (their assertions are migrated into `tests/test_fundamentals_roles.py` in Task 2 — confirm each distinctive assertion has an equivalent there; if any is missing, add it to `test_fundamentals_roles.py` in this task before deleting).
- Test: `tests/test_graph_role_split.py` (new)

**Interfaces:**
- Consumes: the 5 factories from `fundamentals_roles.py` (Task 2 + 3).
- Produces: a graph where `"fundamentals"` expands to 5 nodes; downstream `fundamentals_report` unchanged.

- [ ] **Step 1: Write the failing test** (`tests/test_graph_role_split.py`)

READ `setup.py` first to match the actual `GraphSetup` construction API used by existing tests (see how `test_ta_agent_v2.py` or others build it). Then:

```python
import pytest
pytestmark = pytest.mark.unit

def test_role_nodes_registered_and_aggregator_feeds_ta_v2():
    # Build the graph via the same entrypoint the pipeline uses; assert the
    # compiled graph contains the 4 role node names + the aggregator, and that
    # the aggregator (not a role node) has the edge into "TA Agent v2".
    from tradingagents.graph.setup import GraphSetup  # adjust to real symbol
    # ... construct with stub LLMs exactly as existing graph tests do ...
    # Assert node names present:
    for name in ["Financial-Statement Analyst", "Risk & Red-Flags Analyst",
                 "Catalysts & Ownership Analyst", "Competitive-Quality Analyst",
                 "Fundamentals Aggregator"]:
        assert name in graph_node_names
    assert "Fundamentals Analyst" not in graph_node_names
```

The implementer adapts this to whatever the real graph-inspection surface is (compiled graph `.nodes`, or the `StateGraph` builder before compile). The binding assertions: the 5 names present, `"Fundamentals Analyst"` absent, and the edge into `"TA Agent v2"` starts from `"Fundamentals Aggregator"`.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement the wiring.** In `setup.py`:
  - Replace the `create_fundamentals_analyst` import with the 5 factory imports from `fundamentals_roles`.
  - Where `"fundamentals"` is turned into a node (lines 69-72) and registered (line 116): register the 5 nodes under the display names above instead of one `"Fundamentals Analyst"`.
  - In the analyst-chain edge wiring (lines 129-144): keep market/social/news generic. After the last generic analyst, add edges `last_generic → "Financial-Statement Analyst" → "Risk & Red-Flags Analyst" → "Catalysts & Ownership Analyst" → "Competitive-Quality Analyst" → "Fundamentals Aggregator"`, and make the edge into `"TA Agent v2"` (currently line 144) originate from `"Fundamentals Aggregator"`.
  - Preserve the `selected_analysts` mechanism: `"fundamentals" in selected_analysts` gates this whole 5-node block; when absent, none are added (and the chain closes to TA v2 from the prior analyst as today).

- [ ] **Step 4: Migrate/confirm old-test assertions, then delete.** Before deleting `test_fundamentals_analyst.py` + `test_fundamentals_prompt.py`, diff their distinctive assertions against `test_fundamentals_roles.py`; add any missing one (e.g. `sec_filing.md` read-step, insider citation mandate, net-debt restatement, Altman/Beneish mandates, qualitative sections) to the matching role test. Then delete the two old files and `fundamentals_analyst.py`.

- [ ] **Step 5: Run — graph smoke + FULL suite + import smoke.**

Run: `.venv/bin/python -m pytest tests/test_graph_role_split.py -q` → PASS
Run: `.venv/bin/python -c "import tradingagents.graph.setup"` → no error (old import gone)
Run: `.venv/bin/python -m pytest -q -m unit --tb=line` → green; count = 804 − (deleted old-fundamentals tests) + (new role/aggregator/state/graph tests), no unexpected failures. Investigate any downstream test that referenced `create_fundamentals_analyst` or `"Fundamentals Analyst"`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: wire 4 role nodes + aggregator into graph; retire single fundamentals node"
```

---

## Post-plan (controller, after final review + merge)

- Merge `feat/fa101-role-split` → `main`, push, deploy to mini (`git pull` + `pip install -e .`).
- **Live smoke (mini):** one real ticker end-to-end (e.g. MSFT); confirm `raw/` has coherent output, the 4 role reports populate, the aggregated `fundamentals_report` and the PDF fundamentals section read correctly with no regression vs the pre-split report. This is the pattern that caught the EBIT-TTM + YoY bugs.
- Phase 4 (per-role retry) is the next plan; it builds directly on these nodes.
