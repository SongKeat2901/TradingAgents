# Altman Z″ Distress Screen — Implementation Plan (FA-101 WP4a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic Altman Z″ (4-variable) financial-distress score as a `pm_brief.md` block, closing FA-101 audit §10 (distress screening).

**Architecture:** Follow the established deterministic-block pattern (compute in Python → write `raw/distress_screens.json` → format a markdown block → append to `pm_brief.md` → LLM cites it). Add one field to `financials_parser`; put the score in a new `distress_screens.py` (named for the future Beneish addition); wire it into `researcher.py` after the accounting-ratios block; mandate citation in the fundamentals-analyst prompt. No new QC item.

**Tech Stack:** Python 3, pytest (`unit` marker), no new deps, no new network calls (all inputs already in `financials_parser`).

## Global Constraints

- **Formula (Altman Z″, 4-variable):** `Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4` with X1=(current_assets−current_liabilities)/total_assets, X2=retained_earnings/total_assets, X3=**ebit_ttm**/total_assets, X4=total_equity/(total_assets−total_equity). Zones: `>2.6` Safe, `1.1–2.6` Grey, `<1.1` Distress.
- **Book equity, no market cap** — X4 uses `total_equity`, not market value.
- **X3 uses `ebit_ttm`** (trailing-twelve-month), NOT the single-quarter `ebit`.
- **Skip financials:** when `fin.get("sector")` contains "financial" (case-insensitive) → not applicable, no score.
- **Free-data honesty:** any missing input (or `total_liabilities <= 0`) → `z_score = None` → block renders `n/a`, never fabricated, never crashes.
- **No new QC-checklist item** (bounded QC surface); citation mandated in the fundamentals-analyst prompt only.
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root `/Users/songkeat/Documents/Python/Trading Agent/TradingAgents` (baseline **768** — do not regress).

---

## File Structure

- Modify: `tradingagents/agents/utils/financials_parser.py` — add `retained_earnings`.
- Create: `tradingagents/agents/utils/distress_screens.py` — `compute_altman_z` + `format_distress_block`.
- Modify: `tradingagents/agents/researcher.py` — wire the block after the accounting-ratios block.
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` — cite the Z″ score/zone.
- Create test: `tests/test_distress_screens.py`.
- Modify test: `tests/test_financials_parser.py` — assert `retained_earnings`.

---

### Task 1: `retained_earnings` parser field

**Files:**
- Modify: `tradingagents/agents/utils/financials_parser.py`
- Test: `tests/test_financials_parser.py`

**Interfaces:**
- Produces: `parse_financials(...)["retained_earnings"]` (float | None). Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_financials_parser.py`. Extend the existing `_BS` balance-sheet fixture to include a `Retained Earnings` row, then assert it parses:

```python
def test_parse_financials_retained_earnings():
    from tradingagents.agents.utils.financials_parser import parse_financials
    bundle = {"ticker": "ACME", "trade_date": "2026-05-01", "financial_currency": "USD",
              "fundamentals": "# f\nMarket Cap: 1\n",
              "balance_sheet": "# BS\n\n,2026-03-31,2025-12-31\nTotal Assets,80,79\nRetained Earnings,40000000000,38000000000\n",
              "cashflow": "", "income_statement": ""}
    assert parse_financials(bundle)["retained_earnings"] == 40000000000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py::test_parse_financials_retained_earnings -q`
Expected: FAIL — `KeyError: 'retained_earnings'`.

- [ ] **Step 3: Implement**

In `financials_parser.py`, in the balance-sheet section of the returned dict (near `total_equity`), add:

```python
        "retained_earnings": _row_col0(bs, "Retained Earnings"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_financials_parser.py -q`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/financials_parser.py tests/test_financials_parser.py
git commit -m "feat: parse retained_earnings (Altman Z input)"
```

---

### Task 2: `distress_screens.py` — Altman Z″

**Files:**
- Create: `tradingagents/agents/utils/distress_screens.py`
- Test: `tests/test_distress_screens.py`

**Interfaces:**
- Consumes: `parse_financials(...)` output (needs `sector`, `total_assets`, `total_equity`, `current_assets`, `current_liabilities`, `retained_earnings`, `ebit_ttm`).
- Produces: `compute_altman_z(fin: dict) -> dict`; `format_distress_block(result: dict) -> str`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distress_screens.py
import pytest
from tradingagents.agents.utils.distress_screens import compute_altman_z, format_distress_block

pytestmark = pytest.mark.unit

# healthy: ta=100 eq=60 ca=50 cl=20 re=40 ebit=15 -> tl=40 wc=30
# x1=.3 x2=.4 x3=.15 x4=1.5 -> Z=6.56*.3+3.26*.4+6.72*.15+1.05*1.5=5.855 -> Safe
_HEALTHY = {"sector": "Technology", "total_assets": 100.0, "total_equity": 60.0,
            "current_assets": 50.0, "current_liabilities": 20.0,
            "retained_earnings": 40.0, "ebit_ttm": 15.0}
# distressed: ta=100 eq=5 ca=20 cl=40 re=-30 ebit=-10 -> Z=-2.907 -> Distress
_DISTRESS = {"sector": "Industrials", "total_assets": 100.0, "total_equity": 5.0,
             "current_assets": 20.0, "current_liabilities": 40.0,
             "retained_earnings": -30.0, "ebit_ttm": -10.0}
# grey: ta=100 eq=30 ca=40 cl=30 re=10 ebit=5 -> Z=1.768 -> Grey
_GREY = {"sector": "Consumer", "total_assets": 100.0, "total_equity": 30.0,
         "current_assets": 40.0, "current_liabilities": 30.0,
         "retained_earnings": 10.0, "ebit_ttm": 5.0}


def test_healthy_safe_zone():
    r = compute_altman_z(_HEALTHY)
    assert r["applicable"] is True
    assert r["z_score"] == 5.86  # round(5.855, 2)
    assert r["zone"] == "Safe"
    assert r["x4"] == 1.5


def test_distress_zone():
    r = compute_altman_z(_DISTRESS)
    assert r["zone"] == "Distress" and r["z_score"] < 1.1


def test_grey_zone():
    r = compute_altman_z(_GREY)
    assert r["zone"] == "Grey" and 1.1 <= r["z_score"] <= 2.6


def test_financials_skipped():
    r = compute_altman_z(dict(_HEALTHY, sector="Financial Services"))
    assert r["applicable"] is False
    assert "financial" in r["skip_reason"].lower()


def test_missing_input_na():
    r = compute_altman_z(dict(_HEALTHY, retained_earnings=None))
    assert r["applicable"] is True and r["z_score"] is None and r["zone"] is None


def test_zero_total_liabilities_na():
    # total_equity == total_assets -> total_liabilities == 0 -> x4 undefined
    r = compute_altman_z(dict(_HEALTHY, total_equity=100.0))
    assert r["z_score"] is None


def test_block_populated():
    block = format_distress_block(compute_altman_z(_HEALTHY))
    assert "## Distress screen (Altman Z″)" in block
    assert "Safe" in block and "5.86" in block and "verbatim" in block


def test_block_skipped_and_na():
    fin_block = format_distress_block(compute_altman_z(dict(_HEALTHY, sector="Financial Services")))
    assert "not applicable" in fin_block
    na_block = format_distress_block(compute_altman_z(dict(_HEALTHY, total_assets=None)))
    assert "n/a" in na_block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distress_screens.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/distress_screens.py
"""Deterministic financial-distress screens for pm_brief.md (FA-101 WP4).

Altman Z'' (4-variable, non-manufacturer) — sector-robust, uses book equity
(no market cap). Missing inputs -> None -> rendered "n/a"; financials skipped.
Beneish M-score (WP4b) will be added here once the annual-statement data layer
exists.
"""
from __future__ import annotations

from typing import Any


def _div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_altman_z(fin: dict[str, Any]) -> dict[str, Any]:
    fin = fin or {}
    sector = fin.get("sector") or ""
    if "financial" in sector.lower():
        return {"model": "Altman Z''", "applicable": False,
                "skip_reason": f"Altman Z not meaningful for financials (sector: {sector})"}

    ta = fin.get("total_assets")
    eq = fin.get("total_equity")
    ca = fin.get("current_assets")
    cl = fin.get("current_liabilities")
    re = fin.get("retained_earnings")
    ebit = fin.get("ebit_ttm")

    tl = None if (ta is None or eq is None) else ta - eq
    wc = None if (ca is None or cl is None) else ca - cl
    x1 = _div(wc, ta)
    x2 = _div(re, ta)
    x3 = _div(ebit, ta)
    x4 = _div(eq, tl) if (tl is not None and tl > 0) else None

    if None in (x1, x2, x3, x4):
        missing = [n for n, v in (("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4)) if v is None]
        return {"model": "Altman Z''", "applicable": True, "z_score": None, "zone": None,
                "unavailable_reason": f"missing/undefined inputs for {', '.join(missing)}",
                "x1": _r(x1), "x2": _r(x2), "x3": _r(x3), "x4": _r(x4)}

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    zone = "Safe" if z > 2.6 else ("Distress" if z < 1.1 else "Grey")
    return {"model": "Altman Z''", "applicable": True, "z_score": round(z, 2), "zone": zone,
            "x1": _r(x1), "x2": _r(x2), "x3": _r(x3), "x4": _r(x4)}


def _r(x):
    return None if x is None else round(x, 3)


def format_distress_block(result: dict[str, Any]) -> str:
    r = result or {}
    if not r.get("applicable", True):
        return (f"\n\n## Distress screen (Altman Z″) — not applicable "
                f"({r.get('skip_reason', 'financials')})\n\n"
                "*Altman Z is not meaningful for this sector; do not cite a Z-score for it.*\n")
    if r.get("z_score") is None:
        return (f"\n\n## Distress screen (Altman Z″) — n/a (data unavailable: "
                f"{r.get('unavailable_reason', 'missing inputs')})\n\n"
                "*Do not cite a Z-score; required inputs were missing.*\n")
    return (
        f"\n\n## Distress screen (Altman Z″) (computed from raw/financials.json)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| **Altman Z″** | **{r['z_score']}** |\n"
        f"| **Zone** | **{r['zone']}** (>2.6 Safe · 1.1-2.6 Grey · <1.1 Distress) |\n"
        f"| X1 working capital / total assets | {r['x1']} |\n"
        f"| X2 retained earnings / total assets | {r['x2']} |\n"
        f"| X3 EBIT(TTM) / total assets | {r['x3']} |\n"
        f"| X4 book equity / total liabilities | {r['x4']} |\n\n"
        "*Use the Z″ score and zone verbatim; do not recompute. Z″ (4-variable, "
        "non-manufacturer) is a relative distress indicator, not a default prediction.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_distress_screens.py -q`
Expected: PASS (8 tests). If `test_healthy_safe_zone`'s `5.86` disagrees, recompute by hand from the fixture and align the assertion to the code's rounding — do not change the formula.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/distress_screens.py tests/test_distress_screens.py
git commit -m "feat: add Altman Z'' distress screen"
```

---

### Task 3: wire into researcher + fundamentals-analyst citation

**Files:**
- Modify: `tradingagents/agents/researcher.py` (after the accounting-ratios block append)
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` (`_SYSTEM` citation mandate)
- Test: `tests/test_distress_wiring.py` + `tests/test_fundamentals_prompt.py`

**Interfaces:**
- Consumes: `fin_parsed` (already computed in `fetch_research_pack`), `raw`, `pm_brief_path`.
- Produces: `raw/distress_screens.json` + a `## Distress screen` section in `pm_brief.md`; analyst mandated to cite it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distress_wiring.py
import pytest
from tradingagents.agents.utils.distress_screens import compute_altman_z, format_distress_block

pytestmark = pytest.mark.unit


def test_block_composes_into_pm_brief(tmp_path):
    from tests.test_distress_screens import _HEALTHY
    block = format_distress_block(compute_altman_z(_HEALTHY))
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    assert "## Distress screen (Altman Z″)" in pm.read_text(encoding="utf-8")
```

```python
# add to tests/test_fundamentals_prompt.py
def test_distress_citation_mandated():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    assert "Distress screen" in fa._SYSTEM or "Altman" in fa._SYSTEM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distress_wiring.py tests/test_fundamentals_prompt.py::test_distress_citation_mandated -q`
Expected: the wiring composition test PASSES immediately (library composition — locks the contract); the fundamentals-prompt test FAILS (mandate not present yet).

- [ ] **Step 3: Implement**

READ `researcher.py` around the accounting-ratios block append first. Immediately AFTER it (where `fin_parsed`, `raw`, `pm_brief_path` are in scope), add — in its own try/except like the sibling blocks:

```python
        # --- Distress screen (Altman Z'') ---
        try:
            from tradingagents.agents.utils.distress_screens import (
                compute_altman_z, format_distress_block,
            )
            z = compute_altman_z(fin_parsed)
            (raw / "distress_screens.json").write_text(
                json.dumps(z, indent=2, default=str), encoding="utf-8")
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(format_distress_block(z))
        except Exception as exc:
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Distress screen (Altman Z″) — unavailable ({exc})\n\n"
                        "*Do not cite a Z-score.*\n")
```

Confirm `fin_parsed`/`raw`/`pm_brief_path`/`json` are the real names in scope at that point; adapt without renaming existing variables.

In `fundamentals_analyst.py` `_SYSTEM`, extend the existing accounting-ratios citation mandate (READ it first) to also cite the distress screen — e.g. add a sentence like: "When the `## Distress screen (Altman Z″)` block is applicable, cite its Z″ score and zone verbatim (Safe/Grey/Distress) as a solvency flag." Match the file's existing verbatim/anti-fabrication tone.

- [ ] **Step 4: Run tests + import smoke**

Run: `.venv/bin/python -m pytest tests/test_distress_wiring.py tests/test_fundamentals_prompt.py -q` and `.venv/bin/python -c "import tradingagents.agents.researcher"`
Then full: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all pass. The researcher node body isn't unit-tested (network deps); verification is the composition test + import smoke + full suite.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tradingagents/agents/analysts/fundamentals_analyst.py tests/test_distress_wiring.py tests/test_fundamentals_prompt.py
git commit -m "feat: wire Altman Z'' block into researcher + mandate analyst citation"
```

---

### Task 4: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 768 + new distress/parser/wiring tests). Investigate any regression.

- [ ] **Step 2: (Optional) live check**

On the mini, run one ticker and confirm `raw/pm_brief.md` shows `## Distress screen (Altman Z″)` with a sane Z″/zone (a strong balance sheet → Safe), a financial ticker shows "not applicable", and `raw/distress_screens.json` is written. Not required for merge.

---

## Out of scope (later / not this plan)

- Beneish M-score (WP4b) — needs annual current+prior-year statements the pipeline doesn't fetch.
- The original 5-variable Z / profile-selected variant.
- A new QC-checklist item for the distress score.

## Self-Review

- **Spec coverage:** Z″ formula + zones (Task 2 `compute_altman_z`), book-equity X4 no-market-cap (Task 2), skip-financials gate (Task 2 + test), degrade-to-n/a incl. tl<=0 (Task 2 + tests), `retained_earnings` field (Task 1), deterministic-block wiring + `raw/distress_screens.json` (Task 3), analyst citation without a new QC item (Task 3), tests across safe/grey/distress/financial/missing (Task 2). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step shows real code; commands have expected output.
- **Type consistency:** `compute_altman_z(fin) -> dict` keys (`applicable`, `z_score`, `zone`, `x1..x4`, `skip_reason`, `unavailable_reason`) consumed by `format_distress_block` and asserted in tests identically; `retained_earnings` produced in Task 1, read in Task 2; wiring in Task 3 uses the same function names.
