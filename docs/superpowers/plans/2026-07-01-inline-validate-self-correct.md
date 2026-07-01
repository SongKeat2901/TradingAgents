# Inline Validate + Self-Correct at QC — Implementation Plan (Rerun-Reduction Phase B v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a decision.md-scoped Phase-7 validation failure self-correct in-graph (one PM re-draft with the exact violation as feedback) instead of failing the run to a full external rerun.

**Architecture:** Add a deterministic pre-check at the top of `qc_agent_node`: reuse the existing `write_research_outputs(state, output_dir)` to materialize preview files, run the existing `run_phase_7_validators(output_dir)`, keep only blocking violations whose `file == decision.md` (plus phase-8 scenario, which only scans decision.md), and on a hit return `qc_passed=False` + formatted `qc_feedback` + a bumped `qc_validator_retries` counter — the existing `qc_router` routes back to the PM whose prompt already injects `qc_feedback`. Self-verifying (re-runs each visit), capped at 2. Fail-open on any pre-check exception.

**Tech Stack:** Python 3, pytest (`unit` marker), LangGraph node, no new deps, no new network/LLM calls in the pre-check.

## Global Constraints

- **Reuse verbatim:** `cli/research_writer.py:write_research_outputs`, `cli/research_validation.py:run_phase_7_validators`, `graph/setup.py:qc_router`, and the PM's existing `qc_feedback` prompt slot. Do NOT modify `graph/setup.py` routing or `portfolio_manager.py`.
- **Decision.md-scoped only (v1):** keep a violation only if `severity != "MINOR"` AND (`file == "decision.md"` OR it is a `phase_8_scenario_probability` violation). Analyst/debate-originated violations are ignored here (v2).
- **Deterministic result is authoritative:** on a blocking decision.md violation, return immediately — do NOT run the LLM 18-item audit that visit.
- **Self-verifying, bounded:** the pre-check runs on every QC visit while `qc_validator_retries < VALIDATOR_RETRY_CAP` (**= 2**); it uses its OWN counter, separate from the LLM-QC `qc_retries`.
- **Fail-open:** any exception in the pre-check → log + fall through to the existing LLM audit (never worse than today).
- **Blocking semantics:** `severity != "MINOR"` (matches `research_validation.py:_is_blocking` / `_blocking`).
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root `/Users/songkeat/Documents/Python/Trading Agent/TradingAgents` (baseline **758** — do not regress).

---

## File Structure

- Modify: `tradingagents/agents/managers/qc_agent.py` — add `VALIDATOR_RETRY_CAP`, `_decision_blocking_violations`, `format_validator_feedback`, `_run_deterministic_precheck`, and the pre-check block in `qc_agent_node`; the factory takes `config`.
- Modify: `tradingagents/graph/setup.py` — pass `self.config` to `create_qc_agent_node(...)`.
- Modify: `tradingagents/agents/utils/agent_states.py` — add `qc_validator_retries: int`.
- Modify: `tradingagents/graph/propagation.py` — `create_initial_state` sets `"qc_validator_retries": 0`.
- Create test: `tests/test_qc_validator_precheck.py`.
- Modify test: `tests/test_reuse_raw_plumbing.py` OR new — assert `create_initial_state` includes `qc_validator_retries: 0` (fold into Task 2).

---

### Task 1: violation filter + feedback formatter

**Files:**
- Modify: `tradingagents/agents/managers/qc_agent.py` (add two pure helpers + the cap constant)
- Test: `tests/test_qc_validator_precheck.py` (filter/formatter portion)

**Interfaces:**
- Produces: `VALIDATOR_RETRY_CAP = 2`; `_decision_blocking_violations(results: dict) -> list[dict]`; `format_validator_feedback(violations: list[dict]) -> str`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qc_validator_precheck.py
import pytest
from tradingagents.agents.managers.qc_agent import (
    _decision_blocking_violations,
    format_validator_feedback,
    VALIDATOR_RETRY_CAP,
)

pytestmark = pytest.mark.unit

# Shape mirrors run_phase_7_validators() output: per-phase dict with a "violations"
# list; each violation is a dict carrying "severity" and "file".
_RESULTS = {
    "phase_7_1_price_date": {"violations": [
        {"severity": "MATERIAL", "file": "decision.md", "type": "wrong_close",
         "claimed_date": "2026-06-29", "claimed_price": 359.90, "actual_close": 368.57,
         "match_text": "below Jun 29 close $359.90"},
        {"severity": "MATERIAL", "file": "analyst_fundamentals.md", "type": "wrong_close",
         "claimed_price": 100.0, "actual_close": 110.0},          # upstream -> excluded (v1)
        {"severity": "MINOR", "file": "decision.md", "type": "no_prices_data"},  # minor -> excluded
    ]},
    "phase_7_5_net_debt": {"violations": [
        {"severity": "MATERIAL", "file": "decision.md", "type": "definitional_drift",
         "claimed_dollars": 29000000000.0, "closest_canonical": 31423000000.0},
    ]},
    "phase_8_scenario_probability": {"violations": [
        {"severity": "MATERIAL", "type": "prob_sum", "detail": "probabilities sum to 95%, must be 100%"},
        # note: phase-8 violations may lack a "file" key; must still be kept
    ]},
    "total_violations": 5,
    "blocking_violations": 4,
}


def test_cap_is_two():
    assert VALIDATOR_RETRY_CAP == 2


def test_filter_keeps_only_decision_blocking_plus_scenario():
    keep = _decision_blocking_violations(_RESULTS)
    types = sorted(v["type"] for v in keep)
    # decision.md price + decision.md net-debt + scenario (fileless) => 3
    assert types == ["definitional_drift", "prob_sum", "wrong_close"]
    # the analyst_fundamentals.md and MINOR ones are excluded
    assert all(v.get("file", "decision.md") == "decision.md" or v.get("_phase", "").startswith("phase_8")
               for v in keep)


def test_empty_when_no_blocking():
    clean = {"phase_7_1_price_date": {"violations": [
        {"severity": "MINOR", "file": "decision.md", "type": "x"}]},
        "total_violations": 1, "blocking_violations": 0}
    assert _decision_blocking_violations(clean) == []


def test_formatter_actionable_lines():
    fb = format_validator_feedback(_decision_blocking_violations(_RESULTS))
    assert "368.57" in fb          # authoritative price surfaced
    assert "359.90" in fb          # claimed price surfaced
    assert "full document" in fb.lower() or "re-emit" in fb.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py -q`
Expected: FAIL — `ImportError: cannot import name '_decision_blocking_violations'` (helpers not defined yet).

- [ ] **Step 3: Add the helpers to `qc_agent.py`**

Add near the top of `tradingagents/agents/managers/qc_agent.py` (module scope):

```python
VALIDATOR_RETRY_CAP = 2  # max in-graph self-correction attempts (self-verifying)

# Phase-7/8/9 result keys produced by run_phase_7_validators(); each maps to a
# {"violations": [ {severity, file, ...}, ... ]} dict.
_VALIDATOR_PHASE_KEYS = (
    "phase_7_1_price_date", "phase_7_2_quote_attribution",
    "phase_7_3_peer_metric", "phase_7_5_net_debt",
    "phase_8_scenario_probability", "phase_9_filing_attribution",
)


def _decision_blocking_violations(results: dict) -> list[dict]:
    """Keep blocking (severity != MINOR) violations the PM can fix by rewriting
    decision.md: those whose file == 'decision.md', plus all phase-8 scenario
    violations (which only ever concern decision.md and may omit a file key)."""
    out: list[dict] = []
    for key in _VALIDATOR_PHASE_KEYS:
        for v in (results.get(key, {}) or {}).get("violations", []) or []:
            if v.get("severity") == "MINOR":
                continue
            if key == "phase_8_scenario_probability" or v.get("file") == "decision.md":
                out.append({**v, "_phase": key})
    return out


def format_validator_feedback(violations: list[dict]) -> str:
    """Turn decision.md violations into actionable correction instructions for the PM."""
    lines = [
        "DETERMINISTIC validation found errors in your decision document. Re-emit the "
        "FULL document, correcting ONLY these flagged numbers to the authoritative values:",
    ]
    for v in violations:
        phase = v.get("_phase", "").replace("phase_", "").replace("_", " ").strip()
        label = v.get("type") or phase or "violation"
        claimed = v.get("claimed_price", v.get("claimed_value", v.get("claimed_dollars")))
        actual = v.get("actual_close", v.get("closest_canonical"))
        ctx = f' in: "{v["match_text"][:120]}"' if v.get("match_text") else ""
        if claimed is not None and actual is not None:
            lines.append(f"- [{label}]{ctx} — you wrote {claimed}; the authoritative "
                         f"value is {actual}. Restate it as {actual}.")
        elif v.get("detail"):
            lines.append(f"- [{label}]{ctx} — {v['detail']}")
        else:
            lines.append(f"- [{label}]{ctx} — correct this value to match raw/*.json.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/qc_agent.py tests/test_qc_validator_precheck.py
git commit -m "feat(qc): decision.md blocking-violation filter + PM feedback formatter"
```

---

### Task 2: `qc_validator_retries` state field

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py` (`AgentState`)
- Modify: `tradingagents/graph/propagation.py` (`create_initial_state`)
- Test: `tests/test_qc_validator_precheck.py` (add a plumbing test) OR `tests/test_reuse_raw_plumbing.py`

**Interfaces:**
- Produces: `state["qc_validator_retries"]` (int, default 0), read in Task 3.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_qc_validator_precheck.py
def test_initial_state_has_qc_validator_retries():
    from tradingagents.graph.propagation import Propagator
    st = Propagator().create_initial_state("MSFT", "2026-06-30", output_dir="/tmp/x")
    assert st["qc_validator_retries"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py::test_initial_state_has_qc_validator_retries -q`
Expected: FAIL — `KeyError: 'qc_validator_retries'`.

- [ ] **Step 3: Implement**

In `tradingagents/agents/utils/agent_states.py`, add to the `AgentState` TypedDict (near `qc_retries`):

```python
    qc_validator_retries: int  # Phase B: in-graph deterministic self-correction counter
```

In `tradingagents/graph/propagation.py` `create_initial_state`, add to the returned dict (near `"qc_retries": 0`):

```python
            "qc_validator_retries": 0,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tradingagents/graph/propagation.py tests/test_qc_validator_precheck.py
git commit -m "feat: add qc_validator_retries to agent state (Phase B counter)"
```

---

### Task 3: deterministic pre-check in `qc_agent_node`

**Files:**
- Modify: `tradingagents/agents/managers/qc_agent.py` (`create_qc_agent_node` factory + `qc_agent_node`)
- Modify: `tradingagents/graph/setup.py` (pass `self.config` to `create_qc_agent_node`)
- Test: `tests/test_qc_validator_precheck.py` (node-behavior tests)

**Interfaces:**
- Consumes: `_decision_blocking_violations`, `format_validator_feedback`, `VALIDATOR_RETRY_CAP` (Task 1); `state["qc_validator_retries"]` (Task 2); `write_research_outputs`, `run_phase_7_validators` (existing).
- Produces: the self-correct behavior. `create_qc_agent_node(llm, config=None)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_qc_validator_precheck.py
from unittest.mock import MagicMock, patch
import tradingagents.agents.managers.qc_agent as QC


def _node(llm):
    return QC.create_qc_agent_node(llm, config={})


def test_precheck_blocking_returns_feedback_and_skips_llm(tmp_path):
    llm = MagicMock()  # must NOT be called on a deterministic fail
    node = _node(llm)
    state = {"raw_dir": str(tmp_path / "raw"), "final_trade_decision": "buy",
             "qc_retries": 0, "qc_validator_retries": 0}
    with patch.object(QC, "write_research_outputs", lambda *a, **k: None), \
         patch.object(QC, "run_phase_7_validators", lambda *a, **k: _RESULTS):
        out = node(state)
    assert out["qc_passed"] is False
    assert out["qc_validator_retries"] == 1
    assert "368.57" in out["qc_feedback"]
    llm.assert_not_called()


def test_precheck_clean_falls_through_to_llm(tmp_path):
    # LLM returns a parseable PASS verdict so the node completes via the LLM path.
    llm = MagicMock()
    node = _node(llm)
    state = {"raw_dir": str(tmp_path / "raw"), "final_trade_decision": "buy",
             "qc_retries": 0, "qc_validator_retries": 0}
    clean = {"phase_7_1_price_date": {"violations": []}, "total_violations": 0, "blocking_violations": 0}
    with patch.object(QC, "write_research_outputs", lambda *a, **k: None), \
         patch.object(QC, "run_phase_7_validators", lambda *a, **k: clean), \
         patch.object(QC, "invoke_with_empty_retry", lambda *a, **k: (None, 'QC_VERDICT: {"status": "PASS"}')):
        out = node(state)
    assert out["qc_passed"] is True  # reached the LLM audit and passed


def test_precheck_skipped_when_cap_reached(tmp_path):
    llm = MagicMock()
    node = _node(llm)
    state = {"raw_dir": str(tmp_path / "raw"), "final_trade_decision": "buy",
             "qc_retries": 0, "qc_validator_retries": 2}  # >= VALIDATOR_RETRY_CAP
    called = {"validators": 0}
    def _boom(*a, **k):
        called["validators"] += 1
        return _RESULTS
    with patch.object(QC, "write_research_outputs", lambda *a, **k: None), \
         patch.object(QC, "run_phase_7_validators", _boom), \
         patch.object(QC, "invoke_with_empty_retry", lambda *a, **k: (None, 'QC_VERDICT: {"status": "PASS"}')):
        out = node(state)
    assert called["validators"] == 0     # pre-check skipped at cap
    assert out["qc_passed"] is True


def test_precheck_fails_open_on_exception(tmp_path):
    llm = MagicMock()
    node = _node(llm)
    state = {"raw_dir": str(tmp_path / "raw"), "final_trade_decision": "buy",
             "qc_retries": 0, "qc_validator_retries": 0}
    def _raise(*a, **k):
        raise RuntimeError("boom")
    with patch.object(QC, "write_research_outputs", _raise), \
         patch.object(QC, "invoke_with_empty_retry", lambda *a, **k: (None, 'QC_VERDICT: {"status": "PASS"}')):
        out = node(state)
    assert out["qc_passed"] is True      # exception -> fall through to LLM (PASS)
```

Note: adjust the `invoke_with_empty_retry` patch target to the actual name `qc_agent.py` imports for the LLM call (Step 3 lists it). If the verdict parser needs a specific `QC_VERDICT` format, match the string to what `_parse_verdict` accepts (READ `_parse_verdict`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py -q`
Expected: FAIL — `create_qc_agent_node()` takes 1 positional arg (no `config`) / `QC.write_research_outputs` attribute doesn't exist yet.

- [ ] **Step 3: Implement**

READ `qc_agent.py` `create_qc_agent_node`/`qc_agent_node` (factory ~line 221, node ~228) and the module imports first. Add module-level imports so tests can patch them on the module:

```python
from cli.research_writer import write_research_outputs
from cli.research_validation import run_phase_7_validators
```

(These are import-cycle-safe — verified: `cli/research_validation.py` imports `tradingagents.validators.*` lazily. If a cycle appears at import time, move both imports inside `_run_deterministic_precheck` instead.)

Add the pre-check helper:

```python
def _run_deterministic_precheck(state: dict, config) -> list[dict]:
    """Materialize preview outputs, run the Phase-7 validators, return the
    decision.md-scoped blocking violations. Fail-open: any error -> []."""
    try:
        output_dir = str(Path(state["raw_dir"]).parent)
        write_research_outputs(state, output_dir, config=config)
        results = run_phase_7_validators(output_dir)
        return _decision_blocking_violations(results)
    except Exception:
        logger.warning("QC deterministic pre-check failed; falling through to LLM audit",
                       exc_info=True)
        return []
```

Change the factory signature and add the pre-check at the TOP of `qc_agent_node` (before the `retries >= 1` check):

```python
def create_qc_agent_node(llm, config=None):
    def qc_agent_node(state: dict) -> dict[str, Any]:
        # --- Phase B: deterministic inline validation (before the LLM audit) ---
        val_retries = state.get("qc_validator_retries", 0)
        if val_retries < VALIDATOR_RETRY_CAP:
            blocking = _run_deterministic_precheck(state, config)
            if blocking:
                return {
                    "qc_passed": False,
                    "qc_feedback": format_validator_feedback(blocking),
                    "qc_validator_retries": val_retries + 1,
                }
        # --- existing LLM 18-item audit (unchanged) ---
        retries = state.get("qc_retries", 0)
        if retries >= 1:
            return {"qc_passed": True}
        # ... rest unchanged ...
```

In `tradingagents/graph/setup.py`, update the QC node creation to pass config:

```python
        qc_agent_node = create_qc_agent_node(self.quick_thinking_llm, self.config)
```

Confirm `write_research_outputs`' `config` param tolerates the config value passed (it's used only for the `state.json` `_meta` block). If it requires specific keys, `self.config` (the full run config) satisfies them; passing `config={}` in tests is fine because the preview `_meta` is irrelevant to validators.

- [ ] **Step 4: Run tests + import smoke**

Run: `.venv/bin/python -m pytest tests/test_qc_validator_precheck.py tests/test_qc_agent.py -q` and `.venv/bin/python -c "import tradingagents.agents.managers.qc_agent, tradingagents.graph.setup"`
Then full: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all pass. Confirm the existing `tests/test_qc_agent.py` (LLM-audit tests) still pass — the pre-check must be transparent when validators are clean/absent. Note the graph node isn't run end-to-end (LLM deps); verification is the mocked-node tests + import smoke + full suite.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/managers/qc_agent.py tradingagents/graph/setup.py tests/test_qc_validator_precheck.py
git commit -m "feat(qc): inline Phase-7 pre-check self-corrects decision.md via PM loop"
```

---

### Task 4: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 758 + new pre-check tests). Investigate any regression, especially in `tests/test_qc_agent.py` and `tests/test_pm_qc_checklist.py`.

- [ ] **Step 2: (Optional) live check**

On the mini, a run whose PM writes a wrong decision.md close should now self-correct: the QC node flags it, the PM re-drafts with the pinned value, and the run finishes with `blocking_violations == 0` — no exit-3. Not required for merge (mocked-node tests cover the logic).

---

## Out of scope (v2 / not this plan)

- Routing analyst_*.md/debate_*.md-originated violations back to their producing node (new qc_router edges + per-node feedback slots).
- The LangGraph checkpointer (crash recovery).
- Any change to the LLM 18-item audit, the validators' logic/tolerances, or the LLM-QC `qc_retries` cap.

## Self-Review

- **Spec coverage:** pre-check at QC reusing write_research_outputs + run_phase_7_validators (Task 3); decision.md + phase-8 blocking filter (Task 1 `_decision_blocking_violations`); actionable feedback (Task 1 `format_validator_feedback`); new `qc_validator_retries` counter default 0 (Task 2); cap=2 self-verifying + skip-LLM-on-fail + fail-open (Task 3 + tests); no changes to setup routing/PM prompt (only the config pass-through in setup.py). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step has real code; commands have expected output. The "adjust patch target / READ `_parse_verdict`" notes are concrete instructions tied to named symbols, not placeholders.
- **Type consistency:** `_decision_blocking_violations(results: dict) -> list[dict]` consumed by `format_validator_feedback(list[dict])` and by `_run_deterministic_precheck` → `qc_agent_node`; `create_qc_agent_node(llm, config=None)` matches the setup.py call site; `state["qc_validator_retries"]` produced in Task 2, read in Task 3; `VALIDATOR_RETRY_CAP` used consistently.
