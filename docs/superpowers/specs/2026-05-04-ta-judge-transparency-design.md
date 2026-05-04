# TA Judge Transparency — Design

**Date:** 2026-05-04
**Status:** approved (user approved 2026-05-04)
**Predecessor:** [Quant Research Rebuild](2026-05-03-quant-research-rebuild-design.md)

## Goal

Make technical-setup disagreements between the TA Agent and downstream analysts visible and adjudicated, so the PM's final decision cannot silently route around the TA's read. Combines two interventions:

- **A.** PM Final must transcribe how it relates to the TA's classification (adopt / partially adopt / reject + reasoning).
- **C.** TA Agent runs a second pass after the four analysts have weighed in, producing a refined `technicals_v2.md` that downstream debate consumes.

## Motivation

Run #4 of the MSFT 2026-05-01 e2e exposed an opaque-disagreement pattern:

| Voice | Read |
|---|---|
| TA Agent | "Accumulation / early breakout setup" — 4.5:1 reward/risk |
| Market Analyst | "AGREE with TA Agent's accumulation classification" |
| Bull researcher | Strong buy — capex cycle thesis |
| Aggressive risk | Sell, 7.5% bull-case probability |
| Conservative | Hold/wait |
| Neutral | Trim 15-20% |
| **PM Final** | **UNDERWEIGHT, +20% bull, EV -1.0%** |

The PM picked UNDERWEIGHT but never explicitly engaged with the TA Agent's bullish classification. The decision document presents UNDERWEIGHT as if the TA's read had agreed all along. A reader cannot tell whether the PM weighed and rejected the TA's view or simply ignored it.

This design fixes the opacity. It does **not** fix the related arithmetic / numeric-consistency issues surfaced in the same audit (TA Agent's +44% percentage error, fundamentals analyst's contradictory net-cash figures). Those are scoped out — see "Out of scope".

## Architecture

```
START → PM Preflight → Researcher → TA Agent v1 →
  Market Analyst → Fundamentals Analyst → News Analyst → Social Sentiment Analyst →
  TA Agent v2 (NEW) →
  Bull Researcher ↔ Bear Researcher → Research Manager → Trader →
  Aggressive ↔ Conservative ↔ Neutral → Portfolio Manager →
    [pm_router]
       ├─ PM_RETRY_SIGNAL → RM or Risk team (existing)
       └─ default → QC Agent →
                     ├─ PASS → END
                     └─ FAIL → Portfolio Manager (with qc_feedback)
```

The new TA Agent v2 node sits between the four analysts and the bull/bear debate. It reads all four analyst reports plus the v1 technical setup and emits a refined view that surfaces (and where applicable, resolves) disagreement.

## Components

### TA Agent v2

**File:** `tradingagents/agents/analysts/ta_agent.py` — gain a `create_ta_agent_v2_node(llm)` factory alongside the existing `create_ta_agent_node`.

**Inputs (read from raw/ + state):**
- `raw/technicals.md` (v1)
- `analyst_market.md`
- `analyst_fundamentals.md`
- `analyst_news.md`
- `analyst_social.md`
- `raw/reference.json`
- `raw/prices.json`

**Output:** writes `raw/technicals_v2.md` and sets `state.technicals_report` to the v2 contents.

**Format contract:** Same five mandated sections as v1 (Major historical levels, Volume profile zones, Current technical state, Setup classification, Asymmetry), plus a prepended `## Revisions from v1` section that:

- Names each analyst whose pushback caused a revision (verbatim quote ≤30 words from that analyst's report).
- States what changed (e.g., "Setup classification revised from 'accumulation/breakout' to 'distribution range' because the Fundamentals analyst's flagged margin compression undercuts the volume-exhaustion read").
- If no revision was warranted, the section reads exactly: `No revisions — analyst reports did not surface evidence to revise v1's classification.`

**Always emits.** No conditional skip — keeps graph topology simple and downstream consumers always read v2.

**Model:** Sonnet (quick LLM tier). Cost: ~30-60s wall clock, prompt ~30-40KB.

### PM Final mandated subsection

**File:** `tradingagents/agents/managers/portfolio_manager.py` — extend `_MANDATED_SECTIONS`.

Inside the "Inputs to this decision" section, add a required subsection:

```
**Technical setup adopted:**
- TA Agent v2 classification: <verbatim quote from technicals_v2.md "Setup classification" section>
- My read: [adopt / partially adopt / reject]
- Reasoning: <≤80 words of evidence-based explanation citing specific analyst transcripts>
- Working setup for this decision: <one-line summary>
```

### QC Agent extension

**File:** `tradingagents/agents/managers/qc_agent.py` — extend `_SYSTEM` checklist with Item 14:

> 14. The "Technical setup adopted" subsection exists in the Inputs section, names the TA Agent v2 classification verbatim, picks one of {adopt, partially adopt, reject}, and provides ≥30-word reasoning that cites at least one specific analyst transcript.

A FAIL on Item 14 follows the existing QC retry path: PM gets the qc_feedback and re-emits.

### Graph wiring

**File:** `tradingagents/graph/setup.py`

- Import `create_ta_agent_v2_node`.
- Instantiate the v2 node on the quick (Sonnet) LLM tier.
- Add the node: `workflow.add_node("TA Agent v2", ta_agent_v2_node)`.
- Replace the edge from the last analyst → "Bull Researcher" with: last analyst → "TA Agent v2" → "Bull Researcher".

### Downstream consumer updates

Inspection of the current code shows the bull/bear/RM/trader/risk team prompts do **not** include `technicals.md` directly. They consume the technical view indirectly via `state.market_report` (the Market Analyst's commentary), so a refined v2 view would never reach the debate without an explicit wiring change.

The fix: TA v2 overwrites `state.technicals_report` with v2 contents, and we add that field to the prompts of every downstream agent that should reason over technicals.

**Files affected (each prompt gains a `Technicals Report: {state["technicals_report"]}` block):**
- `tradingagents/agents/researchers/bull_researcher.py`
- `tradingagents/agents/researchers/bear_researcher.py`
- `tradingagents/agents/managers/research_manager.py`
- `tradingagents/agents/trader/trader.py`
- `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- `tradingagents/agents/risk_mgmt/conservative_debator.py`
- `tradingagents/agents/risk_mgmt/neutral_debator.py`
- `tradingagents/agents/managers/portfolio_manager.py` (PM also needs to see the v2 setup verbatim so the "Technical setup adopted" subsection has source material to quote)

The Market Analyst is intentionally **excluded** from this change. Market continues to consume v1 (`raw/technicals.md`) per the T6 design — Market is the analyst empowered to agree-or-disagree with the original TA read, and that disagreement is part of what TA v2 sees and reconciles.

The QC Agent reads v2 via `raw/technicals_v2.md` (loaded analogously to how it loads `raw/reference.json` today).

## Data flow

1. TA v1 fires (existing). Writes `raw/technicals.md`. Sets `state.technicals_report` to v1 contents.
2. Four analysts fire (existing). Each writes its `analyst_*.md` and sets the corresponding state field. Market Analyst still consumes v1 — its job is to either agree or challenge the original TA read.
3. **TA v2 fires** (new). Reads v1 + four analyst reports + raw/reference.json + raw/prices.json. Writes `raw/technicals_v2.md`. Overwrites `state.technicals_report` with v2 contents.
4. Bull/Bear/RM/Trader/Risk team fire (existing topology, modified prompts). Each prompt now includes `Technicals Report: {state["technicals_report"]}` so the debate reasons over v2.
5. PM Final fires (existing topology, modified prompt). The user prompt now includes the v2 technicals report. The system prompt's `_MANDATED_SECTIONS` requires the new "Technical setup adopted" subsection.
6. QC Agent audits Items 1-14 against PM's draft. Item 14 specifically checks the new subsection's structure + that it cites v2's classification verbatim. FAIL on any item routes back to PM with `qc_feedback`. Cap at 1 retry per existing design.

## Failure modes

- **TA v2 LLM error / empty response:** existing None-content guard pattern (same as PM Pre-flight T4 fix). Falls back to v1 contents written verbatim to v2 file with a logged warning. Pipeline continues.
- **TA v2 emits malformed sections:** downstream agents still consume the file as text; the QC Agent will catch the missing "Technical setup adopted" subsection if the PM, downstream, can't extract a coherent classification.
- **PM omits the new subsection:** QC FAIL → retry → PM re-emits with subsection. Same flow as Items 1-13.
- **PM emits the subsection but reasoning is shallow (e.g., "I agree" with no cite):** QC's Item 14 word-count + analyst-cite check catches this. FAIL → retry.

## Testing

**Unit:**
- `tests/test_ta_agent_v2.py` — new file. Stubs LLM. Verifies:
  - File written to `raw/technicals_v2.md`
  - All 6 sections present (Revisions from v1 + 5 mandated)
  - State `technicals_report` set to v2 contents
- `tests/test_pm_qc_checklist.py` — extend. Verify `_MANDATED_SECTIONS` contains "Technical setup adopted" subsection requirements.
- `tests/test_qc_agent.py` — extend. Verify QC `_SYSTEM` lists Item 14, and that a draft missing the subsection fails parsing → routes back to PM.

**Integration (manual smoke test on macmini):**
- Rerun MSFT 2026-05-01 with the new wiring.
- Verify `raw/technicals_v2.md` exists with the new structure.
- Verify decision.md contains the "Technical setup adopted" subsection.
- Verify QC retry triggers if the first PM draft omits it.
- Verify final PDF includes both v1 (Technical Setup) and v2 (a new section) — or only v2 (cleaner) — see "Open question" below.

## Trade-offs accepted

- One extra Sonnet call per run (~30-60s, ~$0.05 marginal cost). The audit found this is a real reliability gap; the cost is justified.
- TA v2's view is final on technicals — bull/bear/RM disagree with v2, not v1. The PM can override v2 explicitly via the new subsection but is the only check. No v3 pass.
- TA v2 has more context than the original TA v1 (sees fundamentals + news + social), which means the v2 read may diverge from a "pure technical" read by incorporating fundamental flags. This is intended behavior — the goal is a synthesized technical view that's robust to known cross-discipline concerns.

## Out of scope (for follow-up sprints)

- **TA's arithmetic errors** (e.g., +44.14% vs actual +14.4% upside-to-200DMA in run #4). Solving this requires deterministic pre-computation of derived percentages and injection into TA prompts so the LLM doesn't compute freehand. Separate workstream — Cluster A from the brainstorm.
- **Numeric consistency across analyst reports** (e.g., fundamentals' three different net-cash figures). Requires either a numeric-extraction post-processor or a cross-section consistency QC pass. Separate workstream — Cluster B.
- **Memory consistency** (pre-flight reads from one source, PM Final from another, leading to "cold-start ticker" claim alongside SPY 2024-05-10 citation in the same run). Separate workstream — Cluster C.

## Open question (resolve during implementation)

Should the PDF render both `Technical Setup` (v1) and a new `Technical Setup — Reconciled` (v2) section, or only v2? Recommendation: only v2. v1 lives in `raw/` for audit but is not surfaced in the PDF; the v2 file's "Revisions from v1" section already names the changes, so a stakeholder reading the PDF gets both the final view and what changed. Keeps the PDF shorter.

## Rollback path

If TA v2 introduces regressions:
1. Revert the graph wiring change (single commit) — TA v2 node stops firing.
2. Revert the consumer updates (one commit) — bull/bear/RM/trader/risk read `technicals.md` again.
3. Revert the PM mandated subsection + QC Item 14 (one commit).

Each revert is independently safe because TA v2 always emits to a separate file (`technicals_v2.md`), so reverting any single layer doesn't leave the others reading a missing file.
