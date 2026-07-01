# Phase B v1 — Inline Validation + Self-Correct at QC (Rerun-Reduction, Phase 3 of 3)

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Make a validation failure self-correct IN-GRAPH (one PM re-draft with the exact violation as feedback) instead of failing the run and forcing a full ~30-min external rerun. Phase 3 (final) of the rerun-reduction effort (C done → A done → **B**). This spec is **v1**: decision.md-scoped only. Routing analyst/debate-originated violations back to their producing node is a noted **v2**, out of scope here.

---

## Background

The deterministic Phase-7 validators run **post-hoc** (`cli/research.py`: `graph.propagate()` → `write_research_outputs()` → `run_phase_7_validators()`), after the graph has fully completed. A blocking violation only sets exit code 3 → the operator/skill re-invokes the whole pipeline. Phase C reduced how often that happens (pinning); Phase A made the reruns that remain cheaper (raw reuse). **Phase B removes the need for a rerun at all** for the most common case: a wrong number the PM wrote into `decision.md`.

The exploration confirmed a clean seam:
- By the time the **QC node** runs, every deterministic anchor (`raw/*.json`, `raw/technicals*.md`) is already on disk; only the top-level report files are missing, and each maps 1:1 to a state field QC already holds (`final_trade_decision`, `market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, `investment_debate_state`, `risk_debate_state`).
- `write_research_outputs(state, output_dir)` (`cli/research_writer.py:82-131`) materializes those preview files from state; `run_phase_7_validators(output_dir)` (`cli/research_validation.py`) then scans them exactly as the post-graph path does. Both are **pure/cheap (no LLM, no network)** and importing them from `qc_agent.py` has **no circular-import risk** (verified: `cli/research_validation.py` imports `tradingagents.validators.*` lazily inside the function).
- The existing retry loop is directly reusable: `qc_router` (`graph/setup.py:214-217`) routes `qc_passed=False` → **Portfolio Manager**; the PM prompt already injects `qc_feedback` (`portfolio_manager.py:427-445`, with a "re-emit the FULL document" directive). **No changes to `graph/setup.py` or `portfolio_manager.py` are needed.**

**Scope boundary (why v1 is decision.md-only):** `qc_router` has no edge back to an analyst/debate node, so only violations the PM can rewrite — those whose `file == decision.md` — are self-correctable through this loop. Phase-8 `scenario_probability` scans only `decision.md` and is fully covered. Violations attributed to `analyst_*.md`/`debate_*.md` (e.g. the net-debt commentary in `analyst_fundamentals.md`) are NOT self-correctable here; they remain caught post-hoc (full rerun as today) and are already reduced by Phase C's prompt discipline. Routing those back is v2.

## Design principles

- **Reuse, don't rebuild.** Reuse `write_research_outputs`, `run_phase_7_validators`, `qc_router`, and the PM's `qc_feedback` slot verbatim. New code is confined to a pre-check block in `qc_agent_node`, a state counter, and a violation→feedback formatter.
- **Deterministic result is authoritative.** On a deterministic blocking violation, skip the LLM 18-item audit for that visit (the violation is a hard fact) — saves an Opus call.
- **Self-verifying, bounded.** The pre-check runs on every QC visit (cheap), so a PM correction is re-validated in-graph. A dedicated counter caps in-graph corrections at **2** (separate from the LLM-QC `qc_retries`).
- **No behavior change when clean.** If the deterministic pre-check finds nothing (the common case once Phase C is doing its job), the node proceeds to the existing LLM audit unchanged.

## Components

### 1. New state field
`tradingagents/agents/utils/agent_states.py` `AgentState`: add `qc_validator_retries: int` (default 0, set in `propagation.create_initial_state` alongside `qc_retries`).

### 2. Deterministic pre-check in `qc_agent_node` (`tradingagents/agents/managers/qc_agent.py`)
At the top of the node, BEFORE the existing `qc_retries >= 1` early-return and LLM audit:

```
val_retries = state.get("qc_validator_retries", 0)
if val_retries < VALIDATOR_RETRY_CAP:                 # VALIDATOR_RETRY_CAP = 2
    output_dir = Path(state["raw_dir"]).parent
    write_research_outputs(state, str(output_dir), config=<available or None>)   # preview
    results = run_phase_7_validators(str(output_dir))
    blocking = [v for v in _all_violations(results)
                if _is_blocking(v) and _file_of(v) == "decision.md"]
    if blocking:
        return {"qc_passed": False,
                "qc_feedback": format_validator_feedback(blocking),
                "qc_validator_retries": val_retries + 1}
# else fall through to the existing LLM 18-item audit (with its own qc_retries>=1 cap)
```

- `write_research_outputs`' `config` param is only used for the `state.json` `_meta` block, which validators don't need — pass whatever config the node has, or `None`. (Plan pins the exact handling; thread config into the QC-node closure if the writer requires it.)
- `_all_violations` / `_is_blocking` / `_file_of` reuse the same helpers/shape the CLI uses to aggregate `run_phase_7_validators` results (per-phase `violations` lists, each violation carrying `file` + severity). If those helpers live in `cli/research_validation.py`, import them; do not re-implement the blocking logic.
- Because the pre-check re-runs on each visit and the counter caps at 2, a PM correction is verified: visit 0 flags → PM re-drafts → visit 1 re-checks (clean → LLM audit → done; still bad → flags again, cap reached next visit).

### 3. Violation → feedback formatter (`qc_agent.py` or a small helper)
`format_validator_feedback(violations) -> str`: for each violation, emit an actionable line naming the type, the file, the claimed value, and the authoritative value from the anchor, e.g.:
> "PRICE-DATE (decision.md): cited 2026-06-29 close $359.90; raw/prices.json authoritative close is $368.57 — restate as $368.57."
> "SCENARIO-PROBABILITY (decision.md): bull/base/bear probabilities sum to 95%, must sum to exactly 100%."

The violation dataclasses already carry the needed fields (`claimed_price`/`actual_close`, `claimed_value`/`closest_canonical`, etc.). Keep the existing "re-emit the FULL document, correcting only the flagged number" framing the PM prompt already enforces.

## Data flow

```
PM → QC node:
  det pre-check (val_retries < 2?):
    write preview → run validators → filter decision.md blocking
    ├─ blocking → qc_passed=False, qc_feedback=<formatted>, qc_validator_retries+1
    │             → qc_router → Portfolio Manager (re-draft with feedback) → QC (re-check)
    └─ clean → existing LLM 18-item audit → qc_router → Executive PM (or PM on LLM fail)
```

## Error handling

- The preview write + validators are wrapped so an unexpected exception in the pre-check does NOT crash the graph — on error, log and fall through to the existing LLM audit (fail-open to current behavior, never worse than today).
- If `raw/*.json` anchors are missing (shouldn't happen at QC), `run_phase_7_validators` degrades to its own no-data handling; the pre-check treats "no blocking decision.md violations" as clean.

## Testing

- **Unit — filter & formatter:** `_file_of`/blocking filter keeps only decision.md blocking violations (drops analyst_*.md and non-blocking); `format_validator_feedback` produces the claimed-vs-authoritative line for a price-date and a scenario-probability violation fixture.
- **Unit — node behavior (mock `run_phase_7_validators`):** with a mocked blocking decision.md violation, `qc_agent_node` returns `qc_passed=False` + non-empty `qc_feedback` + `qc_validator_retries` bumped, and does NOT call the LLM audit; with clean validators, it proceeds to the LLM path (existing tests still pass); with `qc_validator_retries >= 2`, the pre-check is skipped. Mock `write_research_outputs` so no real disk write is needed.
- **Unit — plumbing:** `create_initial_state` includes `qc_validator_retries: 0`.
- **Real success measure (not a unit test):** on a run where the PM writes a wrong decision.md close, the graph self-corrects (PM re-draft) and finishes with `blocking_violations == 0` — no exit-3, no external rerun.

## Out of scope (v2 / not this phase)

- Routing `analyst_*.md`/`debate_*.md`-originated violations back to their producing node (needs new `qc_router` edges + per-analyst/debate feedback slots).
- The LangGraph checkpointer (crash recovery, a different problem).
- Any change to the LLM 18-item audit itself, or to the validators' logic/tolerances.
- Raising the LLM-QC `qc_retries` cap.
