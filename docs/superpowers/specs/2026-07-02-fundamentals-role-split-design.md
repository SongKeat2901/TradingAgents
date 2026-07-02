# FA-101 Phase 3 — Split the Fundamentals Analyst into 4 Clear-Role Nodes

**Date:** 2026-07-02
**Status:** Approved (self-approved under the standing FA-101 alignment goal; architecture fork chosen by user — real graph nodes), pending plan
**Scope:** Replace the single overloaded `Fundamentals Analyst` node (154-line prompt, 10 output sections) with **4 focused LangGraph nodes with clear roles**, plus a deterministic aggregator that preserves the existing `fundamentals_report` contract so no downstream consumer changes. This is the "multi-agent system with clear roles" half of the standing goal, and the substrate for Phase 4 (per-role retry = re-run one node).

---

## Background

The FA-101 alignment program has closed most coverage gaps as deterministic `pm_brief.md` blocks (Phase 1 ratios/multiples/insider/surprise; WP4 Altman/Beneish; multi-year growth; WP5a sentiment/consensus). The interpretation of those blocks is done by ONE fundamentals-analyst node whose `_SYSTEM` prompt has grown to 10 output sections spanning statement analysis, red-flag screens, catalysts, ownership, and competitive quality. That single node is: (a) not "clear roles," (b) impossible to retry per-part (a failure re-runs the whole thing), (c) a large prompt that dilutes attention across unrelated tasks.

The user chose the **real-graph-nodes** architecture (over a structured single node): each role is its own node with its own LLM call and state key.

## Design principles

- **Clear roles, real nodes:** 4 sequential LangGraph nodes, each owning a disjoint slice of today's 10 sections, each with a focused `_SYSTEM` prompt and its own state key.
- **Zero-downstream-edit via aggregator:** a deterministic (no-LLM) aggregator node concatenates the 4 role reports into the existing `fundamentals_report` state key. Every current consumer (TA v2, bull, bear, 3 risk debators) reads `fundamentals_report` unchanged — so the 6×4 = 24-edit threading surface collapses to 0.
- **Preserve every mandate verbatim:** all anti-fabrication / citation-discipline / net-debt-restatement / distress-cite rules move with their section into the owning node's prompt, word-for-word. No behavioral loosening.
- **No new data, no new fetch, no new QC item.** Same raw files; each node reads only the slice it needs (smaller prompts partially offset the +3 LLM calls).
- **Free-data honesty preserved:** a role node that produces nothing still yields a labeled placeholder; the aggregator never drops a section silently.

## Role partition (today's sections → owning node)

| Node (state key) | Owns (verbatim from current `_SYSTEM`) | Reads | Cites (pm_brief blocks) |
|---|---|---|---|
| **Financial-Statement** (`fundamentals_financial_report`) | YoY pre-compute mandate; Business-model framing; Peer comparison matrix; Capital-structure compare (incl. subject net-debt-restatement discipline); Sanity check on reported numbers | pm_brief, reference, financials, peers, sec_filing | accounting-ratios, relative-multiples |
| **Risk & Red-Flags** (`fundamentals_riskflags_report`) | Distress screen (Altman Z″) discipline; Manipulation screen (Beneish M) discipline; a consolidated solvency / risk-factor red-flag narrative grounded in sec_filing risk factors | pm_brief, reference, financials, sec_filing | Altman Z″, Beneish M |
| **Catalysts & Ownership** (`fundamentals_catalysts_report`) | Deal math; Insider transactions; What management needs to prove; **Sentiment & consensus (WP5a)** citation | pm_brief, reference, news, insider | sentiment/consensus, calendar |
| **Competitive-Quality** (`fundamentals_quality_report`) | Competitive position (Porter/moat); Capital-allocation track record; Ownership & governance; the qualitative-claim-discipline mandate | pm_brief, reference, financials, sec_filing, news | accounting-ratios (ROIC) |

**No cross-node dependency:** YoY-compute and the Sanity-check table both live in Financial-Statement, so no node needs another node's output. The shared closing mandate ("Every numerical claim must trace back to … No invented numbers") is appended to all 4 prompts.

## Aggregator node

`fundamentals_aggregator_node(state)` — deterministic, no LLM. Reads the 4 role keys in a fixed order, concatenates under a top `# Fundamentals` header with each role's report beneath a `## <Role>` divider, and writes `fundamentals_report`. A role key that is empty/missing renders `_(<role> section unavailable)_` rather than being dropped. Returns `{"fundamentals_report": combined}` (no `messages` — it is not an LLM turn).

## Graph wiring (`graph/setup.py`)

The generic analyst loop wires market/social/news as today. `"fundamentals"` is handled specially: in place of the single node, register the 4 role nodes + the aggregator and wire them sequentially — `<last generic analyst> → Financial-Statement → Risk & Red-Flags → Catalysts & Ownership → Competitive-Quality → Fundamentals Aggregator → TA Agent v2`. The edge from the last analyst into TA v2 now originates at the aggregator. Node display names: `"Financial-Statement Analyst"`, `"Risk & Red-Flags Analyst"`, `"Catalysts & Ownership Analyst"`, `"Competitive-Quality Analyst"`, `"Fundamentals Aggregator"`. The old `create_fundamentals_analyst` import + node are removed.

## Module layout

- New `tradingagents/agents/analysts/fundamentals_roles.py`: the 4 `_SYSTEM_*` prompts + 4 factories (`create_financial_statement_analyst`, `create_risk_redflags_analyst`, `create_catalysts_ownership_analyst`, `create_competitive_quality_analyst`) + `create_fundamentals_aggregator`. Each role factory mirrors the current node body (build context via `format_for_prompt` with its file slice, `invoke_with_empty_retry` with a per-role `min_chars`, return its own key).
- `tradingagents/agents/utils/agent_states.py`: add the 4 new `Annotated[str, …]` fields.
- `tradingagents/agents/analysts/fundamentals_analyst.py`: retired (deleted); its two test files migrate to assert against the new role prompts.

## Testing

- **Prompt tests** (new `tests/test_fundamentals_roles.py`, absorbing the old `test_fundamentals_analyst.py` + `test_fundamentals_prompt.py` assertions): each role prompt contains its required section headers + its citation/anti-fabrication mandates (distinctive substrings — YoY-compute in Financial-Statement; Altman + Beneish in Risk; sentiment + insider in Catalysts; qualitative-claim discipline + Porter in Competitive-Quality); each role's `files=[…]` slice is correct; no `bind_tools`.
- **Aggregator test:** given a state with 4 role reports → `fundamentals_report` contains all four under their dividers, in order; a missing role → its placeholder present, others intact; never raises.
- **Graph-build smoke** (new `tests/test_graph_role_split.py`): building the graph with default analysts registers the 4 role nodes + aggregator and the aggregator (not a role node) feeds TA v2; `fundamentals_report` remains the key downstream reads.
- **Regression:** full unit suite stays green (baseline 804); the 6 downstream consumers are untouched and their existing tests still pass.
- **Live smoke (mini):** after deploy, run a single real ticker end-to-end; confirm all 4 role reports populate, the aggregated `fundamentals_report` reads coherently, and the PDF fundamentals section is intact (no regression vs the pre-split report).

## Out of scope (later phases)

- **Phase 4 — per-role retry:** each role node self-validates (deterministic precheck) and loops back to itself on failure, capped; then re-aggregate. This spec only establishes the nodes + aggregator that make that possible.
- Phase 2b SEC-fetch data (13F/13D/8-K/DEF 14A); Phase 5 red-flag screens (goodwill/commodity) + incremental ROIC; macro-in-report.
- Parallel (fan-out/merge) execution of the 4 nodes — sequential matches the existing linear chain and avoids state-merge complexity; revisit only if runtime becomes a problem.
