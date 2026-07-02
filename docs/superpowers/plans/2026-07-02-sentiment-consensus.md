# Short Interest + Analyst Consensus — Implementation Plan (FA-101 WP5a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a deterministic short-interest + analyst-consensus/price-target block to `pm_brief.md`, cited by the social analyst — closing FA-101 §8.

**Architecture:** Add the fields to `get_fundamentals` (info already fetched, no extra call) → new `sentiment_consensus.py` parses+computes → append block in `researcher.py` → social analyst cites. Deterministic; no new QC item.

**Tech Stack:** Python 3, pytest (`unit`), stdlib `re`, no new deps/network.

## Global Constraints

- **No extra network call:** the fields come from `ticker.info` which `get_fundamentals` (`dataflows/y_finance.py:264+`) already fetches — add them to its `fields = [...]` list.
- **Field labels (must match between get_fundamentals output and the block's parser):** `Shares Short`, `Shares Short Prior Month`, `Short Ratio Days To Cover`, `Short Percent Of Float`, `Analyst Recommendation` (str), `Analyst Recommendation Mean`, `Number Of Analyst Opinions`, `Target Mean Price`, `Target Median Price`, `Target High Price`, `Target Low Price`, `Current Price`.
- **Free-data honesty:** any absent field → that metric `None` → rendered `n/a (data unavailable)`. Never fabricate/crash.
- **Citation:** social analyst (`analysts/social_media_analyst.py`, already reads `pm_brief.md`). No new QC item.
- **Test marker** `pytestmark = pytest.mark.unit`.
- **Run:** `.venv/bin/python -m pytest -q -m unit --tb=line` (baseline **795** — no regress).

---

### Task 1: surface fields in `get_fundamentals`

**Files:** Modify `tradingagents/dataflows/y_finance.py` (the `fields = [...]` list in `get_fundamentals`); Test `tests/test_sentiment_consensus.py`.

- [ ] **Step 1: Write the failing test** (source-string contract test — avoids a network mock)

```python
# tests/test_sentiment_consensus.py
import inspect
import pytest

pytestmark = pytest.mark.unit

_LABELS = ["Shares Short", "Shares Short Prior Month", "Short Ratio Days To Cover",
           "Short Percent Of Float", "Analyst Recommendation", "Analyst Recommendation Mean",
           "Number Of Analyst Opinions", "Target Mean Price", "Target Median Price",
           "Target High Price", "Target Low Price", "Current Price"]


def test_get_fundamentals_exposes_sentiment_fields():
    from tradingagents.dataflows import y_finance
    src = inspect.getsource(y_finance.get_fundamentals)
    for lbl in _LABELS:
        assert f'"{lbl}"' in src, f"missing field label: {lbl}"
```

- [ ] **Step 2: Run — FAIL** (labels absent). `.venv/bin/python -m pytest tests/test_sentiment_consensus.py -q`

- [ ] **Step 3: Implement** — add to the `fields = [...]` list in `get_fundamentals`:

```python
            ("Current Price", info.get("currentPrice")),
            ("Shares Short", info.get("sharesShort")),
            ("Shares Short Prior Month", info.get("sharesShortPriorMonth")),
            ("Short Ratio Days To Cover", info.get("shortRatio")),
            ("Short Percent Of Float", info.get("shortPercentOfFloat")),
            ("Analyst Recommendation", info.get("recommendationKey")),
            ("Analyst Recommendation Mean", info.get("recommendationMean")),
            ("Number Of Analyst Opinions", info.get("numberOfAnalystOpinions")),
            ("Target Mean Price", info.get("targetMeanPrice")),
            ("Target Median Price", info.get("targetMedianPrice")),
            ("Target High Price", info.get("targetHighPrice")),
            ("Target Low Price", info.get("targetLowPrice")),
```

(The existing `if value is not None` filter drops absent fields.)

- [ ] **Step 4: Run — PASS.**  - [ ] **Step 5: Commit** `feat: surface short-interest + analyst-consensus fields in get_fundamentals`

---

### Task 2: `sentiment_consensus.py` compute + block

**Files:** Create `tradingagents/agents/utils/sentiment_consensus.py`; Test `tests/test_sentiment_consensus.py`.

**Interfaces:** `compute_sentiment_consensus(financials: dict) -> dict`; `format_sentiment_block(result: dict) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sentiment_consensus.py
from tradingagents.agents.utils.sentiment_consensus import (
    compute_sentiment_consensus, format_sentiment_block,
)

_BLOB = (
    "# Fundamentals\nName: Acme\nCurrent Price: 100\n"
    "Shares Short: 120\nShares Short Prior Month: 100\n"
    "Short Ratio Days To Cover: 2.5\nShort Percent Of Float: 0.0128\n"
    "Analyst Recommendation: strong_buy\nAnalyst Recommendation Mean: 1.34\n"
    "Number Of Analyst Opinions: 55\nTarget Mean Price: 150\nTarget Median Price: 145\n"
    "Target High Price: 200\nTarget Low Price: 110\n"
)


def test_compute_sentiment_consensus():
    r = compute_sentiment_consensus({"fundamentals": _BLOB})
    assert r["short_pct_float"] == 1.28          # 0.0128*100
    assert r["days_to_cover"] == 2.5
    assert r["short_mom_change_pct"] == 20.0      # (120-100)/100*100
    assert r["rating"] == "strong_buy" and r["n_analysts"] == 55
    assert r["target_mean"] == 150 and r["target_upside_pct"] == 50.0  # 150/100-1


def test_missing_fields_na():
    r = compute_sentiment_consensus({"fundamentals": "# f\nName: X\n"})
    assert r["short_pct_float"] is None and r["target_upside_pct"] is None


def test_block_render():
    block = format_sentiment_block(compute_sentiment_consensus({"fundamentals": _BLOB}))
    assert "## Sentiment & consensus" in block
    assert "strong_buy" in block and "verbatim" in block
    na = format_sentiment_block(compute_sentiment_consensus({"fundamentals": ""}))
    assert "unavailable" in na.lower() or "n/a" in na.lower()
```

- [ ] **Step 2: Run — FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# tradingagents/agents/utils/sentiment_consensus.py
"""Deterministic short-interest + analyst-consensus block (FA-101 WP5a).

Parses fields already present in the get_fundamentals text blob; missing -> None
-> "n/a". Nothing fabricated."""
from __future__ import annotations

import re
from typing import Any


def _num(t: str, label: str):
    m = re.search(rf"^{re.escape(label)}:\s*(-?[0-9.]+)", t, re.MULTILINE)
    return float(m.group(1)) if m else None


def _text(t: str, label: str):
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", t, re.MULTILINE)
    return m.group(1).strip() if m else None


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def compute_sentiment_consensus(financials: dict[str, Any]) -> dict[str, Any]:
    t = financials.get("fundamentals", "") if isinstance(financials, dict) else ""
    shares_short = _num(t, "Shares Short")
    prior = _num(t, "Shares Short Prior Month")
    pct_float = _num(t, "Short Percent Of Float")
    n_analysts = _num(t, "Number Of Analyst Opinions")
    target_mean = _num(t, "Target Mean Price")
    current = _num(t, "Current Price")
    return {
        "short_pct_float": _r(pct_float * 100) if pct_float is not None else None,
        "days_to_cover": _r(_num(t, "Short Ratio Days To Cover")),
        "short_mom_change_pct": _r((shares_short - prior) / prior * 100) if (shares_short is not None and prior) else None,
        "rating": _text(t, "Analyst Recommendation"),
        "rating_mean": _r(_num(t, "Analyst Recommendation Mean")),
        "n_analysts": int(n_analysts) if n_analysts is not None else None,
        "target_mean": _r(target_mean),
        "target_median": _r(_num(t, "Target Median Price")),
        "target_upside_pct": _r((target_mean / current - 1) * 100) if (target_mean is not None and current) else None,
        "target_low": _r(_num(t, "Target Low Price")),
        "target_high": _r(_num(t, "Target High Price")),
    }


_NA = "n/a (data unavailable)"


def _c(v, suf=""):
    return _NA if v is None else f"{v}{suf}"


def format_sentiment_block(r: dict[str, Any]) -> str:
    r = r or {}
    if not any(v is not None for v in r.values()):
        return ("\n\n## Sentiment & consensus (short interest + analyst view) — unavailable\n\n"
                "*No short-interest / consensus data in the free feed; do not cite figures.*\n")
    rows = [
        ("Short % of float", _c(r.get("short_pct_float"), "%")),
        ("Days to cover", _c(r.get("days_to_cover"), "x")),
        ("Short interest MoM change", _c(r.get("short_mom_change_pct"), "%")),
        ("Analyst rating", _c(r.get("rating"))),
        ("Rating mean (1=buy..5=sell)", _c(r.get("rating_mean"))),
        ("# analysts", _c(r.get("n_analysts"))),
        ("Target mean / median", f"{_c(r.get('target_mean'))} / {_c(r.get('target_median'))}"),
        ("Target implied upside", _c(r.get("target_upside_pct"), "%")),
        ("Target low–high", f"{_c(r.get('target_low'))} – {_c(r.get('target_high'))}"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        "\n\n## Sentiment & consensus (short interest + analyst view)\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"{body}\n\n"
        "*Use these values verbatim; do not recompute. Short interest and price "
        "targets are point-in-time yfinance snapshots (bi-monthly settlement / "
        "sell-side aggregate), not real-time.*\n"
    )
```

- [ ] **Step 4: Run — PASS** (focused + full). - [ ] **Step 5: Commit** `feat: add short-interest + analyst-consensus block`

---

### Task 3: wire into researcher + social citation

**Files:** Modify `tradingagents/agents/researcher.py`, `tradingagents/agents/analysts/social_media_analyst.py`; Test `tests/test_sentiment_consensus.py`, `tests/test_social_prompt.py` (new).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sentiment_consensus.py
def test_block_composes(tmp_path):
    from tradingagents.agents.utils.sentiment_consensus import compute_sentiment_consensus, format_sentiment_block
    block = format_sentiment_block(compute_sentiment_consensus({"fundamentals": _BLOB}))
    pm = tmp_path / "pm_brief.md"; pm.write_text("# b\n", encoding="utf-8")
    with open(pm, "a", encoding="utf-8") as f:
        f.write(block)
    assert "## Sentiment & consensus" in pm.read_text(encoding="utf-8")
```

```python
# tests/test_social_prompt.py
import pytest
pytestmark = pytest.mark.unit

def test_social_cites_sentiment_block():
    from tradingagents.agents.analysts import social_media_analyst as sa
    low = sa._SYSTEM.lower()
    assert "sentiment & consensus" in low or ("short interest" in low and "consensus" in low)
```

- [ ] **Step 2: Run** — composition PASSES; social-prompt test FAILS.

- [ ] **Step 3: Implement**

READ `researcher.py` around a sibling block append (e.g. the accounting-ratios block, where `financials`/`raw`/`pm_brief_path`/`json` are in scope). Add, in its own try/except (fail-open, matching siblings):

```python
        try:
            from tradingagents.agents.utils.sentiment_consensus import (
                compute_sentiment_consensus, format_sentiment_block,
            )
            sc = compute_sentiment_consensus(financials)
            (raw / "sentiment_consensus.json").write_text(
                json.dumps(sc, indent=2, default=str), encoding="utf-8")
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(format_sentiment_block(sc))
        except Exception as exc:
            with open(pm_brief_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n## Sentiment & consensus — unavailable ({exc})\n\n*Do not cite figures.*\n")
```

READ `social_media_analyst.py` `_SYSTEM`; add a mandate to cite the `## Sentiment & consensus` block's short-interest %/days-to-cover and analyst rating + target upside verbatim when present (else "not reported"), matching the file's style. Ensure the social-prompt test's asserted substring matches.

- [ ] **Step 4: Run** focused + import smoke (`.venv/bin/python -c "import tradingagents.agents.researcher, tradingagents.agents.analysts.social_media_analyst"`) + full suite. All pass.

- [ ] **Step 5: Commit** `feat: wire sentiment/consensus block into researcher + social-analyst citation`

---

### Task 4: Full-suite gate
- [ ] `.venv/bin/python -m pytest -q -m unit --tb=line` — green (baseline 795 + new).

---

## Out of scope
13F/13D-G/8-K/proxy; revision-trend; role restructure; per-role retry; macro; red-flag screens.

## Self-Review
- Spec coverage: fields (Task 1), compute+block (Task 2), wiring+citation (Task 3), gate (Task 4). Mapped.
- Placeholders: none; the "READ sibling block / social _SYSTEM" notes are concrete placement instructions.
- Type consistency: `compute_sentiment_consensus(financials)->dict` keys consumed by `format_sentiment_block` and tests identically; label strings in Task 1 == parser labels in Task 2.
