# Pin Hallucinated Numbers — Implementation Plan (Rerun-Reduction Phase C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce how often a cadence run trips the Phase-7 validators (and thus needs a full top-down rerun) by pinning the specific-date closing prices the LLM hallucinates, plus a prompt tightening for net-debt commentary.

**Architecture:** Add a deterministic `## Recent closes` block to `pm_brief.md` (last 10 sessions) built from the *same* `raw/prices.json` Close column the `phase_7_1_price_date` validator reads — so the pinned numbers agree with the validator by construction. Follow the existing `latest_session.py` deterministic-block pattern and reuse its CSV row-parser. Add a fundamentals-analyst prompt rule to restate net-debt figures verbatim from the pinned net-debt block instead of inventing a derived one.

**Tech Stack:** Python 3, pytest (`unit` marker), no new deps, no new network calls (reuses the in-memory `prices` dict).

## Global Constraints

- **Source-alignment invariant (the crux):** the recent-closes block MUST derive from `prices["ohlcv"]` Close (column index 4), reusing `tradingagents/agents/utils/latest_session.py::_parse_ohlcv`. Render with `:.2f` (matches yfinance's `.round(2)`). NEVER a fresh yfinance call. A differently-sourced/rounded number risks exceeding the validator's flat **$0.50** tolerance.
- **Deterministic-block pattern:** compute in Python → write `raw/recent_closes.json` → format a markdown block → append to `raw/pm_brief.md`; the LLM quotes it verbatim.
- **Free-data honesty:** empty/unparseable `ohlcv` → the block renders an explicit "unavailable" note, never fabricated rows; never crash the run.
- **DRY:** reuse `_parse_ohlcv` — do NOT add a third OHLCV CSV parser (`net_debt`, `latest_session`, and the validator already parse this).
- **Default N = 10** trailing sessions, most-recent-first.
- **Test marker:** every new test module starts with `pytestmark = pytest.mark.unit`.
- **Run command:** `.venv/bin/python -m pytest -q -m unit --tb=line` from repo root `/Users/songkeat/Documents/Python/Trading Agent/TradingAgents` (baseline **736** passing — do not regress).
- **pm_brief.md is NOT scanned by the validator** (so the block can't self-flag); it only helps insofar as downstream authors copy the numbers verbatim.

---

## File Structure

- Create: `tradingagents/agents/utils/recent_closes.py` — `compute_recent_closes` + `format_recent_closes_block` (reuses `latest_session._parse_ohlcv`).
- Modify: `tradingagents/agents/researcher.py` — append the block after the `latest_session` block (~line 515-516); write `raw/recent_closes.json`.
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` — add the net-debt-restatement rule to `_SYSTEM`.
- Create test: `tests/test_recent_closes.py`.
- Create test: `tests/test_recent_closes_wiring.py` (composition-level, mirrors `tests/test_researcher_blocks.py`).
- Modify test: `tests/test_fundamentals_prompt.py` — assert the new net-debt rule.

---

### Task 1: `recent_closes.py` module

**Files:**
- Create: `tradingagents/agents/utils/recent_closes.py`
- Test: `tests/test_recent_closes.py`

**Interfaces:**
- Consumes: `latest_session._parse_ohlcv(ohlcv_csv: str) -> list[dict]` (each dict has `date: str`, `close: float`, and open/high/low/volume), date-ascending.
- Produces: `compute_recent_closes(prices_data: dict, trade_date: str, n: int = 10) -> dict` and `format_recent_closes_block(rc: dict) -> str`. Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recent_closes.py
import pytest
from tradingagents.agents.utils.recent_closes import (
    compute_recent_closes,
    format_recent_closes_block,
)

pytestmark = pytest.mark.unit

# Mirrors raw/prices.json ohlcv exactly: #-comment + header + date-ascending rows,
# Close at column index 4 (Date,Open,High,Low,Close,Volume,...). Note 2026-06-26
# close is 359.90 and 2026-06-29 close is 368.57 — the real MSFT values the LLM
# confused (it hallucinated "Jun 29 close $359.90").
_OHLCV = (
    "# Stock data for MSFT\n"
    "# Total records: 12\n\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-06-15,300.0,305.0,299.0,304.10,10000000,0.0,0.0\n"
    "2026-06-16,304.0,309.0,303.0,308.20,10000000,0.0,0.0\n"
    "2026-06-17,308.0,312.0,307.0,311.30,10000000,0.0,0.0\n"
    "2026-06-18,311.0,315.0,310.0,314.40,10000000,0.0,0.0\n"
    "2026-06-19,314.0,318.0,313.0,317.50,10000000,0.0,0.0\n"
    "2026-06-22,317.0,321.0,316.0,320.60,10000000,0.0,0.0\n"
    "2026-06-23,320.0,324.0,319.0,323.70,10000000,0.0,0.0\n"
    "2026-06-24,323.0,327.0,322.0,326.80,10000000,0.0,0.0\n"
    "2026-06-25,326.0,331.0,325.0,328.77,10000000,0.0,0.0\n"
    "2026-06-26,328.0,362.0,327.0,359.90,10000000,0.0,0.0\n"
    "2026-06-29,360.0,370.0,359.0,368.57,10000000,0.0,0.0\n"
    "2026-06-30,368.0,375.0,367.0,373.02,10000000,0.0,0.0\n"
)
_PRICES = {"ohlcv": _OHLCV}


def test_returns_last_10_most_recent_first():
    rc = compute_recent_closes(_PRICES, "2026-06-30", n=10)
    assert rc["unavailable"] is False
    assert len(rc["rows"]) == 10
    assert rc["rows"][0]["date"] == "2026-06-30"      # most-recent-first
    assert rc["rows"][0]["close"] == 373.02
    assert rc["rows"][1]["date"] == "2026-06-29"
    assert rc["rows"][1]["close"] == 368.57
    dates = [r["date"] for r in rc["rows"]]
    assert "2026-06-15" not in dates and "2026-06-16" not in dates  # oldest 2 dropped by n=10


def test_on_or_before_trade_date_boundary():
    rc = compute_recent_closes(_PRICES, "2026-06-25", n=10)
    dates = [r["date"] for r in rc["rows"]]
    assert "2026-06-26" not in dates and "2026-06-29" not in dates and "2026-06-30" not in dates
    assert rc["rows"][0]["date"] == "2026-06-25"


def test_fewer_than_n_rows():
    rc = compute_recent_closes(_PRICES, "2026-06-17", n=10)
    assert len(rc["rows"]) == 3
    assert rc["rows"][0]["date"] == "2026-06-17"


def test_empty_prices_unavailable():
    rc = compute_recent_closes({"ohlcv": ""}, "2026-06-30")
    assert rc["unavailable"] is True and rc["rows"] == []


def test_block_renders_table_and_mandate():
    block = format_recent_closes_block(compute_recent_closes(_PRICES, "2026-06-30"))
    assert "## Recent closes" in block
    assert "| 2026-06-29 | $368.57 |" in block
    assert "verbatim" in block and "validator" in block


def test_block_unavailable_note():
    block = format_recent_closes_block({"unavailable": True, "reason": "x", "rows": []})
    assert "## Recent closes — unavailable" in block
    assert "Do not cite" in block


def test_source_alignment_with_validator_parse():
    """Lock the block's Close to the validator's own parse (ohlcv col 4)."""
    rc = compute_recent_closes(_PRICES, "2026-06-30", n=10)
    block_by_date = {r["date"]: f"{r['close']:.2f}" for r in rc["rows"]}
    validator_by_date = {}
    for line in _OHLCV.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) >= 5:
            validator_by_date[parts[0].strip()] = f"{float(parts[4]):.2f}"
    for d, c in block_by_date.items():
        assert validator_by_date[d] == c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_recent_closes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.agents.utils.recent_closes'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/recent_closes.py
"""Deterministic recent-closes block (rerun-reduction Phase C).

Pins the last N trailing daily closes from raw/prices.json so the LLM quotes
specific-date closes verbatim instead of hallucinating them (e.g. the MSFT
2026-06-30 run's "Jun 29 close $359.90" vs the real $368.57, which the
phase_7_1_price_date validator blocks). Built from the SAME prices.json Close
column (index 4) the validator reads — via latest_session._parse_ohlcv — so the
two agree by construction and stay inside the validator's $0.50 tolerance.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tradingagents.agents.utils.latest_session import _parse_ohlcv


def compute_recent_closes(
    prices_data: dict[str, Any], trade_date: str, n: int = 10
) -> dict[str, Any]:
    ohlcv = ""
    if isinstance(prices_data, dict):
        ohlcv = prices_data.get("ohlcv", "") or ""
    rows = _parse_ohlcv(ohlcv)  # date-ascending; close = float(parts[4])
    if not rows:
        return {"unavailable": True, "reason": "raw/prices.json has no parseable OHLCV rows", "rows": []}

    try:
        td = datetime.strptime(trade_date, "%Y-%m-%d").date()
        eligible = [r for r in rows if datetime.strptime(r["date"], "%Y-%m-%d").date() <= td]
    except (ValueError, TypeError):
        eligible = rows  # if trade_date unparseable, fall back to all rows

    if not eligible:
        return {"unavailable": True, "reason": f"no sessions on or before {trade_date}", "rows": []}

    recent = list(reversed(eligible[-n:]))  # last n, then most-recent-first
    return {
        "trade_date": trade_date,
        "as_of": recent[0]["date"],
        "rows": [{"date": r["date"], "close": r["close"]} for r in recent],
        "source": "raw/prices.json ohlcv (Close, col 4)",
        "unavailable": False,
    }


def format_recent_closes_block(rc: dict[str, Any]) -> str:
    if rc.get("unavailable") or not rc.get("rows"):
        reason = rc.get("reason", "no data")
        return (
            f"\n\n## Recent closes — unavailable ({reason})\n\n"
            "*Do not cite a closing price for any specific date; none are pinned here.*\n"
        )
    n = len(rc["rows"])
    body = "\n".join(f"| {r['date']} | ${r['close']:.2f} |" for r in rc["rows"])
    return (
        f"\n\n## Recent closes (last {n} sessions, verbatim from raw/prices.json)\n\n"
        f"| Date | Close |\n|---|---|\n{body}\n\n"
        "*Any closing price you cite for a specific date MUST be quoted verbatim "
        "from this table (source: raw/prices.json Close). Do not state a close for "
        "a date not listed here — if you need an older close, say 'not in the "
        "recent-closes table' rather than estimating. This is the same source the "
        "validator checks, so a paraphrased or rounded price will be flagged.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_recent_closes.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/recent_closes.py tests/test_recent_closes.py
git commit -m "feat: add deterministic recent-closes block (source-aligned to price validator)"
```

---

### Task 2: Wire the block into researcher.py

**Files:**
- Modify: `tradingagents/agents/researcher.py` (immediately after the `latest_session` block append, ~line 515-516)
- Test: `tests/test_recent_closes_wiring.py`

**Interfaces:**
- Consumes: in-memory `prices` (the `{"ohlcv": ...}` dict from `_fetch_prices`), `date` (trade_date str), `raw` (Path), `pm_brief_path` (Path) — all already present at that point in `fetch_research_pack`.
- Produces: `raw/recent_closes.json` + an appended `## Recent closes` section in `raw/pm_brief.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recent_closes_wiring.py
import pytest
from tradingagents.agents.utils.recent_closes import compute_recent_closes, format_recent_closes_block

pytestmark = pytest.mark.unit


def test_block_composes_into_pm_brief(tmp_path):
    from tests.test_recent_closes import _PRICES
    rc = compute_recent_closes(_PRICES, "2026-06-30")
    block = format_recent_closes_block(rc)
    pm = tmp_path / "pm_brief.md"
    pm.write_text("# brief\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    text = pm.read_text(encoding="utf-8")
    assert "## Recent closes" in text
    assert "| 2026-06-29 | $368.57 |" in text
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/test_recent_closes_wiring.py -q`
Expected: PASS immediately (pure library composition). Its purpose is to lock the composition contract before touching `researcher.py`. If it fails, fix the import/signature drift first.

- [ ] **Step 3: Wire into researcher.py**

READ `researcher.py` around the `latest_session` block append (~line 515-516) to confirm the local names (`prices`, `date`, `raw`, `pm_brief_path`) and the sibling append idiom. Immediately AFTER the latest-session block append, add — guarded by try/except like the sibling blocks so a compute failure can't crash the run:

```python
        # --- Recent closes (deterministic; pins specific-date closes) ---
        try:
            from tradingagents.agents.utils.recent_closes import (
                compute_recent_closes, format_recent_closes_block,
            )
            rc = compute_recent_closes(prices, date)
            (raw / "recent_closes.json").write_text(
                json.dumps(rc, indent=2, default=str), encoding="utf-8")
            rc_block = format_recent_closes_block(rc)
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(rc_block)
        except Exception as exc:
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Recent closes — unavailable ({exc})\n\n"
                        "*Do not cite a closing price for any specific date.*\n")
```

Confirm `prices` (not `prices_data`) and `date` are the real local names at that point; if they differ, adapt without renaming existing variables.

- [ ] **Step 4: Run tests + import smoke**

Run: `.venv/bin/python -m pytest tests/test_recent_closes_wiring.py -q` and `.venv/bin/python -c "import tradingagents.agents.researcher"`
Then full: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: all pass; researcher imports cleanly. Note in your report that the researcher node body itself is not unit-tested (network deps), so verification is composition-test + import-smoke + full-suite.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_recent_closes_wiring.py
git commit -m "feat: wire recent-closes block into researcher after latest-session"
```

---

### Task 3: Fundamentals-analyst net-debt commentary discipline (Part 2)

**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py` (`_SYSTEM`)
- Test: `tests/test_fundamentals_prompt.py` (extend)

**Interfaces:**
- Consumes: nothing new — a prompt-string edit.
- Produces: a new instruction in `_SYSTEM` forbidding invented derived net-debt figures.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_fundamentals_prompt.py
def test_net_debt_restatement_discipline_present():
    from tradingagents.agents.analysts import fundamentals_analyst as fa
    s = fa._SYSTEM
    # must instruct restating net debt from the pinned block, not inventing a derived figure
    assert "net debt" in s.lower()
    assert "raw/net_debt.json" in s or "Net debt block" in s or "## Net debt" in s
    assert "do not compute" in s.lower() or "must not" in s.lower() or "verbatim" in s.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py::test_net_debt_restatement_discipline_present -q`
Expected: FAIL (the specific net-debt-restatement instruction isn't present yet). If it happens to pass on pre-existing text, strengthen the assertion to match the exact new sentence you add in Step 3 (e.g. `assert "must not compute a novel" in s`).

- [ ] **Step 3: Add the instruction**

READ `fundamentals_analyst.py` `_SYSTEM` first. Add a sentence in the net-debt / sanity-check area (or as its own short rule), matching the existing verbatim/anti-fabrication tone, e.g.:

```
Net-debt discipline: state net debt / net cash ONLY by restating a figure
already shown in the pm_brief "## Net debt" block (or raw/net_debt.json). You
MUST NOT compute and cite a novel derived net-debt/net-cash figure (e.g. a
"~$Xbn divergence" between two definitions) — the validator only recognizes
the canonical derivations in that block, so an invented one will be flagged.
If two framings differ, name them using the block's own cells, not a new number.
```

Ensure the assertion strings in Step 1 match whatever wording you commit (adjust one or the other so they agree).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fundamentals_prompt.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/analysts/fundamentals_analyst.py tests/test_fundamentals_prompt.py
git commit -m "feat: fundamentals-analyst net-debt restatement discipline (no invented derived figures)"
```

---

### Task 4: Full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: green (baseline 736 + the new recent-closes/wiring/prompt tests). Investigate any regression before proceeding.

- [ ] **Step 2: (Optional) live parser check**

If convenient, confirm `compute_recent_closes` against a real fetched `prices` dict on the mini (via the `researcher._fetch_prices` path) and eyeball that the last 10 closes match a fresh yfinance view — but this is not required for merge (unit tests + source-alignment test cover correctness).

---

## Out of scope (later phases / not this plan)

- Phase A (reuse `raw/*.json` + enable the checkpointer / resume completed LLM stages) — separate spec.
- Phase B (move validators inline + self-correct the failing stage) — separate spec.
- Any change to the validators, their tolerances, or the TA-file exclusion.
- Pinning non-close levels (52-week high/low, intraday) — YAGNI.

## Self-Review

- **Spec coverage:** Part 1 recent-closes block (Tasks 1-2), source-alignment invariant (Task 1 `test_source_alignment_with_validator_parse` + reuse of `_parse_ohlcv`), free-data honesty degradation (Task 1 `test_empty_prices_unavailable` + Task 2 try/except), n=10 default (Task 1), wiring after latest-session (Task 2), Part 2 net-debt discipline (Task 3), full-suite gate (Task 4). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step has real code; commands have expected output.
- **Type consistency:** `compute_recent_closes(prices_data, trade_date, n=10) -> dict` with keys `rows`/`unavailable`/`as_of`/`source`; `format_recent_closes_block(rc)` consumes those exact keys; Task 2 calls them with the in-memory `prices`/`date`; `_parse_ohlcv` reused read-only (returns `close` float used verbatim). Consistent across tasks.
