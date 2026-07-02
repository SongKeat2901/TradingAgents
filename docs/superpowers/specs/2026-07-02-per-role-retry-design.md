# FA-101 Phase 4 — Per-Role Retry for the Fundamentals Role Nodes

**Date:** 2026-07-02
**Status:** Approved (self-approved under the standing FA-101 alignment goal), pending plan; **build gated on the Phase 3 live smoke passing**
**Scope:** Give each of the 4 fundamentals role nodes (Financial-Statement, Risk & Red-Flags, Catalysts & Ownership, Competitive-Quality) a deterministic self-check and a capped self-loop, so a role whose output is structurally incomplete is **re-run on its own** with targeted feedback — "retry the failed part, no full stack." This is the (c) component of the standing goal and builds directly on Phase 3's node split.

---

## Background

Phase 3 replaced the single fundamentals node with 4 role nodes + a deterministic aggregator (merge `d901242`). Today those nodes invoke the LLM once (`invoke_with_empty_retry`, `min_chars=1200`) and advance unconditionally — a role that silently omits a mandated section (e.g. Risk forgets to cite the Altman Z″ block, Financial-Statement skips the peer matrix) flows downstream uncorrected, and the only remedy is re-running the whole pipeline.

The pipeline already has a **capped deterministic self-correct loop**: QC's `qc_validator_retries` (cap `VALIDATOR_RETRY_CAP=2`) runs `_run_deterministic_precheck`, formats feedback, and the `qc_router` loops back to the PM on failure or advances on pass. Phase 4 generalizes exactly that pattern to each role node as a **self-loop** (LangGraph conditional edge whose "retry" target is the node itself).

## Design principles

- **Deterministic checks only** — no LLM judge in the loop. A role's output is checked for the presence of its required section headers and a non-trivial length; cheap, no file materialization, no false-positive risk from a second model.
- **Retry the failed part, not the stack** — the self-loop re-runs only that one node; the other 3 roles and all upstream work are untouched.
- **Capped + fail-open** — at most `ROLE_RETRY_CAP = 2` re-runs per role; after that the node advances with whatever it produced (never blocks the pipeline). Matches the QC-validator discipline.
- **Feedback-driven re-prompt** — on a failed check the node re-runs with a feedback line naming the missing sections, injected into its HumanMessage (mirrors `qc_feedback`).
- **No downstream change, no new QC item** — the aggregator and all 6 consumers are unaffected; the report a role finally emits is what gets aggregated.

## Mechanism (generalizes `qc_validator_retries`)

**Per-role state fields** (`agent_states.py`, init `0`/`""`/`False` in `propagation.py`), for each `<role>` in {financial, riskflags, catalysts, quality}:
- `fundamentals_<role>_retries: int`
- `fundamentals_<role>_feedback: str`
- `fundamentals_<role>_passed: bool`

**Deterministic check** — a shared helper `check_role_output(required_headers: list[str], report: str, min_chars: int = 1200) -> list[str]`: returns a list of human-readable issues — one per required header absent from `report`, plus one if `len(report.strip()) < min_chars`. Empty list ⇒ passed. Each role owns its `required_headers` (its `## `-headers from the Phase 3 prompt, e.g. Financial-Statement = `["## Business-model framing", "## Peer comparison matrix", "## Sanity check on reported numbers"]`; Risk = `["## Risk & red flags"]` + the Altman/Beneish discipline is checked only when the pm_brief block is applicable — see below).

**Optional block-citation check (soft, applicability-gated):** when `pm_brief.md` carries an *applicable* block the role must cite, require its citation token present — Risk: `"Altman"`/`"Z″"` and `"Beneish"`/`"M-score"` when those blocks are not marked "not applicable"/"unavailable"; Catalysts: a short-interest/target reference when the `## Sentiment & consensus` block is present. If the block is absent or marked unavailable, the check is skipped (no false positive on names that legitimately lack the data). v1 may ship section-presence + length first and add citation-presence in the same task if it proves low-noise on the live smoke corpus.

**Node factory change** (each role in `fundamentals_roles.py`): after `invoke_with_empty_retry`, read prior `fundamentals_<role>_feedback` from state and, if present, append it to the HumanMessage before invoking (so a re-run sees "Your previous draft was missing: …"). After invoking, run `check_role_output`; then:
- issues empty ⇒ return `{report_key: report, "...passed": True, "...feedback": ""}`.
- issues present ⇒ return `{report_key: report, "...passed": False, "...feedback": format_role_feedback(issues), "...retries": prior_retries + 1}` (keep the partial report so even a capped-out node contributes something).

**Router** — a parameterized `make_role_router(passed_key, retries_key, advance_target, cap=ROLE_RETRY_CAP)` returning a function `state -> "retry" | "advance"`: `"advance"` if `state.get(passed_key)` or `state.get(retries_key, 0) >= cap`, else `"retry"`. Wire via `add_conditional_edges("<Role> Analyst", router, {"retry": "<Role> Analyst", "advance": "<next node>"})` — the "retry" target is the node itself (self-loop). The 4 advance targets chain as today (Financial→Risk→Catalysts→Quality→Aggregator).

## Module layout

- `tradingagents/agents/analysts/fundamentals_roles.py`: add `check_role_output`, `format_role_feedback`, per-role `_REQUIRED_*` header lists, feedback-injection in each node body, and the new return keys.
- `tradingagents/graph/setup.py`: replace the 4 linear `add_edge` calls among the role nodes with `add_conditional_edges` + `make_role_router` (defined inline or imported from `conditional_logic.py`).
- `tradingagents/agents/utils/agent_states.py` + `propagation.py`: the 12 new fields + their init.

## Testing

- **`check_role_output`** (`tests/test_role_retry.py`): a report missing a required header → that header in the issues; a complete ≥min-chars report → `[]`; a too-short report → a length issue.
- **Router:** `passed=True` → "advance"; `passed=False, retries<cap` → "retry"; `retries>=cap` (regardless of passed) → "advance" (fail-open).
- **Node loop behaviour** (stub llm): first invoke returns a report missing a header → node returns `passed=False`, `retries=1`, feedback names the header; a stub whose 2nd response is complete → on re-entry the node injects the feedback into the prompt and returns `passed=True`. Cap: a stub that never satisfies → after `ROLE_RETRY_CAP` the router advances.
- **Graph:** `add_conditional_edges` present for all 4 role nodes; each router's "retry" maps to the same node name and "advance" to the correct next node; aggregator still feeds TA v2.
- **Regression:** full suite green; downstream untouched.
- **Live smoke (mini):** a real run where a role initially under-produces shows a single self-loop re-run in the logs and a complete final section — and a healthy run shows zero extra loops (no gratuitous retries).

## Out of scope (later)

- LLM-judge or Phase-7-validator (fabrication/number) checks inside the role loop — heavier (needs file materialization); a later Phase 4b if section-presence proves insufficient.
- Per-role retry for the non-fundamentals analysts (market/social/news) — same mechanism, apply later if needed.
- FA-101 coverage items still open (Phase 2b SEC-fetch 13F/13D/8-K/DEF 14A; Phase 5 red-flag screens + incremental ROIC) — separate phases.
