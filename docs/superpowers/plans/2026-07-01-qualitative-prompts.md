# Qualitative Prompt Upgrades — Implementation Plan (FA-101 WP6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three mandated qualitative sections (competitive position, capital-allocation, ownership & governance) plus a strict anti-fabrication clause to the fundamentals-analyst prompt, closing FA-101's qualitative gaps.

**Architecture:** Prompt-only — edit `fundamentals_analyst._SYSTEM` (no new graph node, data source, or QC item). Lock the additions with prompt-string tests, mirroring the existing citation-mandate tests.

**Tech Stack:** Python 3, pytest (`unit` marker), no new deps, no code logic.

## Global Constraints

- **Prompt-only** — only `tradingagents/agents/analysts/fundamentals_analyst.py` `_SYSTEM` changes (plus tests). No new node, no new data file, no new QC-checklist item.
- **Three new `##` sections**, verbatim headers: `## Competitive position`, `## Capital-allocation track record`, `## Ownership & governance`. Placed inside the existing "Required sections (use the headers verbatim):" block.
- **Anti-fabrication clause (load-bearing):** every qualitative claim must trace to a named source (`raw/sec_filing.md`, `news.json`, `pm_brief.md`); where unsupported, write **"not determinable from available free filings"**; never invent competitive dynamics, moats, governance/share-class facts, concentration figures, or management history from memory.
- **Concise** — a few bullets per section; don't restructure or remove existing sections.
- **Test marker:** the test module already starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root (baseline **789** — do not regress).

---

## File Structure

- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` — add 3 sections + anti-fabrication clause to `_SYSTEM`.
- Modify: `tests/test_fundamentals_prompt.py` — assert the 3 headers + the clause.

---

### Task 1: qualitative sections + anti-fabrication clause

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` (`_SYSTEM`)
- Test: `tests/test_fundamentals_prompt.py`

**Interfaces:**
- Consumes: nothing new (the analyst already receives `pm_brief.md`, `sec_filing.md`, `news.json`, `insider.json`).
- Produces: three mandated qualitative sections + the discipline clause in `_SYSTEM`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fundamentals_prompt.py
def test_qualitative_sections_present():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    s = fa._SYSTEM
    assert "## Competitive position" in s
    assert "## Capital-allocation track record" in s
    assert "## Ownership & governance" in s


def test_qualitative_antifabrication_clause():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    low = fa._SYSTEM.lower()
    assert "not determinable from available free filings" in low
    # forbids inventing qualitative facts from memory
    assert "do not invent" in low
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py -k qualitative -q`
Expected: FAIL — the section headers / clause aren't in `_SYSTEM` yet.

- [ ] **Step 3: Implement**

READ `fundamentals_analyst.py` `_SYSTEM` first — specifically the "Required sections (use the headers verbatim):" block (the `## `-header sections through `## What management needs to prove`). Add the three new sections inside that block (after the existing sections, matching their prose style), and the anti-fabrication clause. Use this text (adapt spacing to the file's `\`-continuation style):

```
## Competitive position

Porter's Five Forces in brief — competitive rivalry, threat of new entrants, \
threat of substitutes, supplier power, buyer power. State the moat type \
(network effects / switching costs / scale / brand / IP / regulatory) and its \
DURABILITY, plus disruption risk. Ground every claim in raw/sec_filing.md \
(business description / risk factors), news.json, or the peer set.

## Capital-allocation track record

Assess capital-allocation discipline: buybacks vs dividends vs M&A, and whether \
reinvestment earns its cost of capital — cite the "## Accounting ratios" block's \
ROIC and ROIC-WACC spread. Note insider ownership / recent buying-selling by \
reference to the "## Insider transactions" section (do not duplicate it).

## Ownership & governance

Share-class / voting structure (e.g. dual-class or super-voting founder shares), \
board independence, and customer/supplier concentration — each grounded in \
raw/sec_filing.md / news.json, or explicitly "not disclosed in the available \
filing" when absent.

Qualitative-claim discipline: every claim in the three sections above must be \
grounded in a named source (raw/sec_filing.md, news.json, or pm_brief.md). Where \
the available free data does not support a claim, write "not determinable from \
available free filings" — do NOT invent competitive dynamics, moats, governance / \
share-class facts, concentration figures, or management history from memory or \
general knowledge.
```

Ensure the committed text contains the exact substrings the tests assert: the three `## ` headers verbatim, `not determinable from available free filings`, and `do NOT invent` (the test lowercases, so `do NOT invent` satisfies `"do not invent"`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py tests/test_fundamentals_prompt.py
git commit -m "feat: add qualitative sections (competitive position/capital-allocation/governance) to fundamentals analyst"
```

---

### Task 2: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 789 + 2 new tests). Investigate any regression (esp. other `tests/test_fundamentals_prompt.py` assertions that scan `_SYSTEM` — the additions are purely additive and must not disturb existing mandated-section or citation assertions).

---

## Out of scope (later / not this plan)

- A dedicated qualitative graph node; WP5 data sources (proxy/8-K/13F); a new QC item; restructuring existing sections.

## Self-Review

- **Spec coverage:** three `##` sections (Task 1), anti-fabrication clause (Task 1 + `test_qualitative_antifabrication_clause`), prompt-only/no-new-node (only `fundamentals_analyst.py` + tests touched), honest-degradation phrasing ("not disclosed" / "not determinable") in the section text, full-suite gate (Task 2). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; the prompt text is complete; the "READ _SYSTEM first / adapt spacing" note is a concrete placement instruction, not a placeholder.
- **Type consistency:** no code interfaces; the test-asserted substrings (`## Competitive position`, `## Capital-allocation track record`, `## Ownership & governance`, `not determinable from available free filings`, `do not invent`) exactly match the section text in Step 3.
