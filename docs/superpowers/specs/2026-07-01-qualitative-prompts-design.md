# FA-101 WP6 — Qualitative Prompt Upgrades

**Date:** 2026-07-01
**Status:** Approved (design), pending implementation plan
**Scope:** Add the qualitative fundamental-analysis coverage the FA-101 audit flagged (§1 competitive position, §6 qualitative factors) as mandated prompt sections in the existing fundamentals analyst. Prompt-only — no new graph node, no new data source, no new QC item.

---

## Background

The 2026-07-01 FA-101 audit found the pipeline strong on deterministic numbers but thin on **qualitative** coverage: Porter's Five Forces / competitive position, moat durability & disruption risk, management/capital-allocation track record, customer/supplier concentration, and corporate governance / share-class structure. These cannot be computed deterministically — they are judgment calls grounded in narrative sources (SEC filings, news). WP6 adds them as mandated sections in the fundamentals-analyst prompt.

The fundamentals analyst already receives the grounding sources — `pm_brief.md` (all deterministic blocks), `financials.json`, `peers.json`, `news.json`, `raw/sec_filing.md` (latest 10-Q/10-K excerpt), `insider.json`, `reference.json` — and already has mandated `##` sections with a strong "no invented numbers" discipline. WP6 extends that pattern to qualitative claims.

## Design principles

- **Prompt-only, no new node.** Add sections to `fundamentals_analyst._SYSTEM`; do not add a graph node or LLM call (YAGNI, avoids per-run cost).
- **Anti-fabrication is the load-bearing part.** Qualitative sections are the highest hallucination risk in the report. Every qualitative claim MUST trace to a named source (`raw/sec_filing.md`, `news.json`, `pm_brief.md`); when the free data doesn't support a claim, the analyst writes **"not determinable from available free filings"** — it must NEVER invent competitive dynamics, governance facts, share-class structures, or management history from memory.
- **Honest degradation for filing-dependent facts.** Governance / share-class / customer concentration are often only in the DEF 14A proxy or 10-K (WP5, not fetched) — so "not disclosed in the available filing" is the expected, correct output there, not a fabricated one.
- **Concise.** Each section is a few bullets — enough to add qualitative signal without ballooning the report or diluting its numeric core.
- **No new QC item** — qualitative claims can't be deterministically validated; the anti-fabrication prompt discipline is the control.

## Components

### New mandated sections in `tradingagents/agents/analysts/fundamentals_analyst.py` `_SYSTEM`
Added to the "Required sections (use the headers verbatim)" block, in the analyst's existing style:

1. **`## Competitive position`** — Porter's Five Forces (competitive rivalry, threat of new entrants, threat of substitutes, supplier power, buyer power) in brief; the company's moat type (network effects / switching costs / scale / brand / IP / regulatory) and its **durability**; and disruption risk. Ground in `raw/sec_filing.md` (business description / risk factors), `news.json`, and the peer set.
2. **`## Capital-allocation track record`** — capital-allocation discipline: buybacks vs. dividends vs. M&A, and whether reinvestment earns its cost of capital (cite the `## Accounting ratios` block's ROIC and ROIC−WACC spread); insider ownership / recent insider buying-selling (from `insider.json`, complementing the existing `## Insider transactions` section — reference it, don't duplicate).
3. **`## Ownership & governance`** — share-class / voting structure (e.g. dual-class, super-voting founder shares), board independence, and customer/supplier concentration — each grounded in `raw/sec_filing.md`/`news.json`, or explicitly "not disclosed in the available filing" when absent.

### The qualitative anti-fabrication clause
A single explicit rule near these sections (matching the file's existing "Every numerical claim must trace back to …" discipline, but for qualitative claims):

> "Every qualitative claim in these three sections must be grounded in a named source (raw/sec_filing.md, news.json, or pm_brief.md). Where the available free data does not support a claim, write 'not determinable from available free filings' — do NOT invent competitive dynamics, moats, governance/share-class facts, concentration figures, or management history from memory or general knowledge."

## Testing

- **Prompt-string tests (`tests/test_fundamentals_prompt.py`, extend):**
  - The three new section headers are present in `_SYSTEM` (`## Competitive position`, `## Capital-allocation track record`, `## Ownership & governance`).
  - The qualitative anti-fabrication clause is present — assert a distinctive substring (e.g. `"not determinable from available free filings"` and `"do not invent"`/`"do NOT invent"`), locking the discipline the way the net-debt/distress citation tests lock theirs.
- **Real success measure (not a unit test):** on a live run, the fundamentals report contains the three qualitative sections, each either grounded in the filing/news or honestly marked "not determinable" — and does NOT hallucinate governance/share-class facts absent from the fetched filing. Spot-check on the next mini run.

## Out of scope (later / not this phase)

- A dedicated qualitative/business-quality graph node (YAGNI — prompt-only is sufficient).
- WP5 data sources (DEF 14A proxy, 8-K, 13F) that would more fully ground governance/ownership — WP6 degrades honestly without them.
- A new QC-checklist item for qualitative claims (not deterministically checkable).
- Moving or restructuring the existing analyst sections.
