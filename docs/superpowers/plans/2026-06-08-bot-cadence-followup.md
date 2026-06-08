# Bot cadence-followup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Python orchestrator (`tradingcadencefollowup`) that lets the TrueKnot bot, from a private DM, QC the latest research cadence and publish the grade-A passes — auto-dismissing the known validator false-positive patterns, escalating only novel flags to LLM judgement.

**Architecture:** A new `tradingagents/cadence/` package holds the pure, testable core (batch detection, run loading, FP-classifier, grading) plus thin gog/filesystem side-effect wrappers (token guard, idempotent PDF publish, promote, summary refresh). `cli/cadence_followup.py` wires them and emits a JSON result contract. A bot `SKILL.md` (source in `ops/openclaw-skills/`) invokes the CLI, adjudicates residual flags, and composes the DM.

**Tech Stack:** Python 3.13, stdlib (`dataclasses`, `enum`, `re`, `json`, `subprocess`, `pathlib`), pytest (`-m unit`), `gog` CLI (shelled), yfinance (only via existing reuse). No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-06-08-bot-cadence-followup-design.md`

**Conventions:** Commits go directly on `main`, Conventional-Commits prefixes, footer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (no "Generated with" line). Tests run with `.venv/bin/python -m pytest -q -m unit`.

---

### Task 1: Scaffold the `cadence` package + data models

**Files:**
- Create: `tradingagents/cadence/__init__.py`
- Create: `tradingagents/cadence/models.py`
- Create: `tests/cadence/__init__.py`
- Test: `tests/cadence/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_models.py
import pytest
from tradingagents.cadence.models import (
    FlagDisposition, FlagVerdict, RunData, RunVerdict,
)

pytestmark = pytest.mark.unit


def test_flag_disposition_values():
    assert FlagDisposition.DISMISS.value == "dismiss"
    assert FlagDisposition.CORRECT_BY_DESIGN.value == "correct"
    assert FlagDisposition.NEEDS_ADJUDICATION.value == "adjudicate"


def test_flag_verdict_defaults():
    fv = FlagVerdict(phase="phase_7_5_net_debt",
                     disposition=FlagDisposition.DISMISS, reason="x")
    assert fv.detail == {}


def test_run_verdict_holds_lists():
    rv = RunVerdict(ticker="AAPL", grade="A", flag_verdicts=[],
                    auto_dismissed=[], needs_adjudication=[])
    assert rv.grade == "A"
    assert rv.needs_adjudication == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.cadence'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/cadence/__init__.py
"""Cadence follow-up: deterministic QC + publish orchestration for a research batch."""
```

```python
# tradingagents/cadence/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FlagDisposition(Enum):
    DISMISS = "dismiss"                 # known-safe validator false positive
    CORRECT_BY_DESIGN = "correct"       # e.g. non-USD reporter skip
    NEEDS_ADJUDICATION = "adjudicate"   # novel — escalate to the bot LLM


@dataclass
class FlagVerdict:
    phase: str
    disposition: FlagDisposition
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class RunData:
    ticker: str
    trade_date: str
    run_dir: str
    validation: dict
    intrinsic_value: dict
    peer_ratios: dict
    financials: dict
    reference_price: float | None


@dataclass
class RunVerdict:
    ticker: str
    grade: str                       # "A" or "HOLD"
    flag_verdicts: list[FlagVerdict]
    auto_dismissed: list[FlagVerdict]
    needs_adjudication: list[FlagVerdict]
```

Create empty `tests/cadence/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_models.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/__init__.py tradingagents/cadence/models.py tests/cadence/__init__.py tests/cadence/test_models.py
git commit -m "feat(cadence): scaffold package + QC data models

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Batch detector — newest preaudit trade_date, completed only

**Files:**
- Create: `tradingagents/cadence/batch.py`
- Test: `tests/cadence/test_batch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_batch.py
import pytest
from tradingagents.cadence.batch import find_latest_batch

pytestmark = pytest.mark.unit


def _mk(base, name, with_decision=True):
    d = base / name
    d.mkdir(parents=True)
    if with_decision:
        (d / "decision.md").write_text("- **Reference price:** $1.00\n")
    return d


def test_picks_newest_date_completed_only_excludes_dotted(tmp_path):
    pre = tmp_path / "preaudit"
    # older date, completed
    _mk(pre, "2026-06-04-AAA")
    # newest date: one completed, one in-flight (no decision.md), one archived (dotted)
    _mk(pre, "2026-06-05-BBB")
    _mk(pre, "2026-06-05-CCC", with_decision=False)
    _mk(pre, "2026-06-05-DDD.pre-cadence")
    date, runs = find_latest_batch(pre)
    assert date == "2026-06-05"
    assert [r.name for r in runs] == ["2026-06-05-BBB"]


def test_empty_base_returns_none(tmp_path):
    pre = tmp_path / "preaudit"
    pre.mkdir()
    assert find_latest_batch(pre) == (None, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_batch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.cadence.batch'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/cadence/batch.py
from __future__ import annotations

import re
from pathlib import Path

_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-([A-Z0-9.]+)$")


def find_latest_batch(preaudit_base: Path) -> tuple[str | None, list[Path]]:
    """Return (trade_date, [completed run dirs]) for the newest date present.

    A run is 'completed' iff it contains decision.md. Archived dirs whose name
    contains a dot (e.g. '<date>-<T>.pre-cadence') are excluded — the historical
    glob bug that matched them caused false 'batch complete' signals.
    """
    preaudit_base = Path(preaudit_base)
    if not preaudit_base.is_dir():
        return None, []
    by_date: dict[str, list[Path]] = {}
    for d in preaudit_base.iterdir():
        if not d.is_dir() or "." in d.name:
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        if not (d / "decision.md").is_file():
            continue
        by_date.setdefault(m.group(1), []).append(d)
    if not by_date:
        return None, []
    newest = max(by_date)
    return newest, sorted(by_date[newest], key=lambda p: p.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_batch.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/batch.py tests/cadence/test_batch.py
git commit -m "feat(cadence): batch detector (newest date, completed, excl. dotted)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Run loader — parse validation_report + raw/ + reference price

**Files:**
- Modify: `tradingagents/cadence/batch.py` (add `load_run`)
- Test: `tests/cadence/test_load_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_load_run.py
import json
import pytest
from tradingagents.cadence.batch import load_run

pytestmark = pytest.mark.unit


def _build(tmp_path):
    rd = tmp_path / "2026-06-05-AAPL"
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text(
        "intro\n- **Reference price:** $177.00 (yfinance close of 2026-06-05)\n")
    (rd / "validation_report.json").write_text(json.dumps(
        {"total_violations": 1, "blocking_violations": 1,
         "phase_7_5_net_debt": {"violations": [{"severity": "MATERIAL"}]}}))
    (rd / "raw" / "intrinsic_value.json").write_text(
        json.dumps({"ticker": "AAPL", "inputs": {"net_debt": 39139000000.0}}))
    (rd / "raw" / "peer_ratios.json").write_text(json.dumps({"COHR": {"ttm_pe": 178.67}}))
    (rd / "raw" / "financials.json").write_text(json.dumps({"ticker": "AAPL"}))
    return rd


def test_load_run_populates_fields(tmp_path):
    rd = _build(tmp_path)
    run = load_run(rd)
    assert run.ticker == "AAPL"
    assert run.trade_date == "2026-06-05"
    assert run.reference_price == 177.00
    assert run.validation["blocking_violations"] == 1
    assert run.intrinsic_value["inputs"]["net_debt"] == 39139000000.0
    assert "COHR" in run.peer_ratios


def test_load_run_missing_raw_is_empty_dict(tmp_path):
    rd = tmp_path / "2026-06-05-XYZ"
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text("no ref price here\n")
    (rd / "validation_report.json").write_text("{}")
    run = load_run(rd)
    assert run.peer_ratios == {}
    assert run.reference_price is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_load_run.py -q`
Expected: FAIL — `ImportError: cannot import name 'load_run'`

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/cadence/batch.py`:

```python
import json
from tradingagents.cadence.models import RunData

_REF_RE = re.compile(r"Reference price:\**\s*\$([0-9][0-9,]*\.?[0-9]*)")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _reference_price(decision_md: Path) -> float | None:
    try:
        text = decision_md.read_text()
    except OSError:
        return None
    m = _REF_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    m = _DIR_RE.match(run_dir.name)
    trade_date = m.group(1) if m else ""
    ticker = m.group(2) if m else run_dir.name
    raw = run_dir / "raw"
    return RunData(
        ticker=ticker,
        trade_date=trade_date,
        run_dir=str(run_dir),
        validation=_read_json(run_dir / "validation_report.json"),
        intrinsic_value=_read_json(raw / "intrinsic_value.json"),
        peer_ratios=_read_json(raw / "peer_ratios.json"),
        financials=_read_json(raw / "financials.json"),
        reference_price=_reference_price(run_dir / "decision.md"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_load_run.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/batch.py tests/cadence/test_load_run.py
git commit -m "feat(cadence): load_run parses validation_report + raw/ + ref price

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: FP-classifier — iterate violations + net-debt $-grab + non-USD skip

**Files:**
- Create: `tradingagents/cadence/fp_classifier.py`
- Test: `tests/cadence/test_fp_classifier_netdebt.py`

The classifier walks every `phase_*` block in `validation.report`, yielding one
`FlagVerdict` per violation. This task handles two dispositions:
`definitional_drift` net-debt $-grab (DISMISS) and `skipped_non_usd_reporter`
(CORRECT_BY_DESIGN). All other types default to NEEDS_ADJUDICATION for now.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_fp_classifier_netdebt.py
import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None):
    return RunData(ticker="AAPL", trade_date="2026-06-05", run_dir="/x",
                   validation=validation,
                   intrinsic_value=intrinsic or {"inputs": {"net_debt": 39139000000.0}},
                   peer_ratios={}, financials={}, reference_price=177.0)


def test_buyback_dollar_grab_is_dismissed():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 163000000000.0,
        "match_text": "$39.14B Net Debt against a >$163B authorized buyback and "
                      "$82.6B H1 operating cash flow indicates no balance-sheet stress."}]}}
    verdicts = classify_run_flags(_run(v))
    assert len(verdicts) == 1
    assert verdicts[0].disposition is FlagDisposition.DISMISS
    assert "buyback" in verdicts[0].reason.lower()


def test_operating_cashflow_fcf_grab_is_dismissed():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net cash", "claimed_dollars": 2540000000.0,
        "match_text": "FCF Q1 2026 = -$2,540M: Net cash from operating activities "
                      "$1,096M - capex $3,636M = -$2,540M (cashflow col 0)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_non_usd_skip_is_correct_by_design():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MINOR", "type": "skipped_non_usd_reporter"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.CORRECT_BY_DESIGN


def test_unknown_definitional_drift_escalates():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_label": "net debt", "claimed_dollars": 50000000000.0,
        "match_text": "Net debt is $50B per the balance sheet."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_netdebt.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.cadence.fp_classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/cadence/fp_classifier.py
from __future__ import annotations

import re
from tradingagents.cadence.models import FlagDisposition, FlagVerdict, RunData

# Words that, when adjacent to the flagged dollar figure, prove it is NOT a
# net-debt/net-cash *balance* (so phase_7_5 mis-grabbed it).
_NON_NETDEBT_NEARBY = (
    "buyback", "repurchase", "authoriz",            # share-buyback authorization
    "operating activities", "operating cash flow",  # cash-flow statement line
    "fcf", "free cash flow",                        # free cash flow
    "capex", "capital expenditure",
)


def _phase_violations(validation: dict):
    for phase, block in validation.items():
        if isinstance(block, dict):
            for v in block.get("violations", []) or []:
                yield phase, v


def _dollars_to_tokens(d: float | None) -> list[str]:
    """Plausible textual renderings of a dollar figure for window matching."""
    if not d:
        return []
    toks = []
    billions = d / 1e9
    millions = d / 1e6
    toks.append(f"{billions:.2f}".rstrip("0").rstrip("."))   # 163 / 39.14
    toks.append(f"{int(round(millions)):,}")                  # 2,540
    return [t for t in toks if t]


def _is_netdebt_dollar_grab(v: dict) -> bool:
    mt = (v.get("match_text") or "").lower()
    for tok in _dollars_to_tokens(v.get("claimed_dollars")):
        idx = mt.find(tok.lower())
        if idx == -1:
            continue
        window = mt[max(0, idx - 40): idx + 40]
        if any(w in window for w in _NON_NETDEBT_NEARBY):
            return True
    return False


def classify_violation(phase: str, v: dict, run: RunData) -> FlagVerdict:
    vtype = v.get("type")
    detail = {k: v.get(k) for k in ("file", "line_no", "claimed_value",
                                    "claimed_dollars", "match_text", "ticker",
                                    "metric", "actual_close", "claimed_price")
              if v.get(k) is not None}

    if vtype == "skipped_non_usd_reporter":
        return FlagVerdict(phase, FlagDisposition.CORRECT_BY_DESIGN,
                           "non-USD reporter: net-debt check correctly skipped", detail)

    if vtype == "definitional_drift" and _is_netdebt_dollar_grab(v):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "flagged $ is a non-net-debt quantity (buyback / operating "
                           "cash flow / FCF) sitting near 'net debt/cash' wording", detail)

    return FlagVerdict(phase, FlagDisposition.NEEDS_ADJUDICATION,
                       f"{vtype or 'unknown'}: no known false-positive pattern matched", detail)


def classify_run_flags(run: RunData) -> list[FlagVerdict]:
    return [classify_violation(phase, v, run)
            for phase, v in _phase_violations(run.validation)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_netdebt.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/fp_classifier.py tests/cadence/test_fp_classifier_netdebt.py
git commit -m "feat(cadence): FP-classifier — net-debt \$-grab + non-USD skip

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: FP-classifier — price-date from→to cross-wire

**Files:**
- Modify: `tradingagents/cadence/fp_classifier.py` (extend `classify_violation`)
- Test: `tests/cadence/test_fp_classifier_pricedate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_fp_classifier_pricedate.py
import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation):
    return RunData(ticker="ASX", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value={}, peer_ratios={},
                   financials={}, reference_price=34.03)


def test_from_to_miswire_is_dismissed():
    # validator paired the May-28 date with $34.03 (the 'to' endpoint),
    # but the text says 'from the $40.60 May 28 close to $34.03'.
    v = {"phase_7_1_price_date": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_close",
        "claimed_date": "2026-05-28", "claimed_price": 34.03, "actual_close": 40.6,
        "match_text": "ine from the $40.60 May 28 close to $34.03 compressed RSI from"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS
    assert "from" in verdicts[0].reason.lower()


def test_genuine_wrong_close_escalates():
    # actual_close not present as a 'from' value -> not the known miswire
    v = {"phase_7_1_price_date": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_close",
        "claimed_date": "2026-05-28", "claimed_price": 34.03, "actual_close": 40.6,
        "match_text": "the May 28 close was $34.03 per my notes"}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_pricedate.py -q`
Expected: FAIL — both default to NEEDS_ADJUDICATION, so `test_from_to_miswire_is_dismissed` fails on the DISMISS assertion.

- [ ] **Step 3: Write minimal implementation**

Add this helper to `fp_classifier.py` and a branch in `classify_violation` BEFORE the final return:

```python
def _fmt_price(p) -> str | None:
    if p is None:
        return None
    s = f"{float(p):.2f}".rstrip("0").rstrip(".")
    return s


def _is_from_to_miswire(v: dict) -> bool:
    """True when match_text reads 'from $<actual_close> ... to $<claimed_price>',
    i.e. the validator paired the date with the 'to' endpoint, not the 'from'."""
    mt = (v.get("match_text") or "").lower()
    frm = _fmt_price(v.get("actual_close"))
    to = _fmt_price(v.get("claimed_price"))
    if not frm or not to:
        return False
    pat = r"from\b[^.]*?\$?" + re.escape(frm) + r"[^.]*?\bto\b[^.]*?\$?" + re.escape(to)
    return re.search(pat, mt) is not None
```

In `classify_violation`, add before the final `return`:

```python
    if vtype == "wrong_close" and _is_from_to_miswire(v):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "price-date 'from $A (date) to $B' cross-wire: validator "
                           "paired the date with the 'to' endpoint, not the 'from' value",
                           detail)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_pricedate.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/fp_classifier.py tests/cadence/test_fp_classifier_pricedate.py
git commit -m "feat(cadence): FP-classifier — price-date from->to cross-wire

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: FP-classifier — peer-metric "respectively" mis-map

**Files:**
- Modify: `tradingagents/cadence/fp_classifier.py`
- Test: `tests/cadence/test_fp_classifier_peer.py`

The validator blames `v["ticker"]` for `v["metric"] = v["claimed_value"]`. It is a
known FP when the claimed value actually matches a DIFFERENT peer's cell and the
prose used "respectively". Map metric label → `peer_ratios` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_fp_classifier_peer.py
import pytest
from tradingagents.cadence.models import RunData, FlagDisposition
from tradingagents.cadence.fp_classifier import classify_run_flags

pytestmark = pytest.mark.unit


def _run(validation):
    peers = {"ASX": {"nd_ebitda": 1.27}, "TSM": {"nd_ebitda": 0.33}}
    return RunData(ticker="AMKR", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value={}, peer_ratios=peers,
                   financials={}, reference_price=64.95)


def test_respectively_mismap_is_dismissed():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "1.27x",
        "actual_value": "0.33",
        "match_text": "Net Debt $293M vs. ASX $159.79B, TSM $936.16B; "
                      "ND/EBITDA 1.27x and 0.33x respectively)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.DISMISS


def test_peer_metric_no_respectively_escalates():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "1.27x",
        "actual_value": "0.33",
        "match_text": "TSM ND/EBITDA is 1.27x."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION


def test_peer_metric_value_matches_no_other_peer_escalates():
    v = {"phase_7_3_peer_metric": {"violations": [{
        "severity": "MATERIAL", "type": "wrong_peer_metric",
        "ticker": "TSM", "metric": "ND/EBITDA", "claimed_value": "9.99x",
        "actual_value": "0.33",
        "match_text": "ND/EBITDA 9.99x and 0.33x respectively)."}]}}
    verdicts = classify_run_flags(_run(v))
    assert verdicts[0].disposition is FlagDisposition.NEEDS_ADJUDICATION
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_peer.py -q`
Expected: FAIL — `test_respectively_mismap_is_dismissed` expects DISMISS, gets NEEDS_ADJUDICATION.

- [ ] **Step 3: Write minimal implementation**

Add to `fp_classifier.py`:

```python
_METRIC_TO_KEY = {
    "nd/ebitda": "nd_ebitda", "net debt/ebitda": "nd_ebitda", "leverage": "nd_ebitda",
    "ttm pe": "ttm_pe", "ttm p/e": "ttm_pe", "forward pe": "forward_pe",
    "forward p/e": "forward_pe", "ttm ebitda": "ttm_ebitda", "net debt": "net_debt",
    "op margin": "op_margin", "operating margin": "op_margin",
    "capex/revenue": "capex_revenue",
}


def _parse_metric_value(s) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    return float(m.group(0).replace(",", "")) if m else None


def _is_respectively_mismap(v: dict, run: RunData) -> bool:
    if "respectively" not in (v.get("match_text") or "").lower():
        return False
    key = _METRIC_TO_KEY.get((v.get("metric") or "").strip().lower())
    claimed = _parse_metric_value(v.get("claimed_value"))
    blamed = (v.get("ticker") or "").upper()
    if not key or claimed is None:
        return False
    for peer, cells in (run.peer_ratios or {}).items():
        if peer.upper() == blamed or not isinstance(cells, dict):
            continue
        cell = cells.get(key)
        if isinstance(cell, (int, float)) and abs(cell - claimed) <= 0.01:
            return True   # claimed value belongs to a DIFFERENT peer -> mis-map
    return False
```

In `classify_violation`, add before the final `return`:

```python
    if vtype == "wrong_peer_metric" and _is_respectively_mismap(v, run):
        return FlagVerdict(phase, FlagDisposition.DISMISS,
                           "peer 'X and Y respectively': claimed value matches the OTHER "
                           "peer's cell; validator mapped the metric to the wrong ticker",
                           detail)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_peer.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/fp_classifier.py tests/cadence/test_fp_classifier_peer.py
git commit -m "feat(cadence): FP-classifier — peer 'respectively' mis-map

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Grader — blocking-only grade with null-FV tolerance

**Files:**
- Create: `tradingagents/cadence/grader.py`
- Test: `tests/cadence/test_grader.py`

Grade A iff every BLOCKING violation classifies as DISMISS or CORRECT_BY_DESIGN
(no NEEDS_ADJUDICATION). MINOR-only violations never block. Null/suppressed
intrinsic value is NOT a defect. The grader builds the final `RunVerdict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_grader.py
import pytest
from tradingagents.cadence.models import RunData
from tradingagents.cadence.grader import grade_run

pytestmark = pytest.mark.unit


def _run(validation, intrinsic=None):
    return RunData(ticker="T", trade_date="2026-06-05", run_dir="/x",
                   validation=validation, intrinsic_value=intrinsic or {},
                   peer_ratios={"ASX": {"nd_ebitda": 1.27}, "TSM": {"nd_ebitda": 0.33}},
                   financials={}, reference_price=10.0)


def test_all_dismissed_grades_A():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_dollars": 163000000000.0,
        "match_text": "$163B authorized buyback"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "A"
    assert len(rv.auto_dismissed) == 1
    assert rv.needs_adjudication == []


def test_null_fair_value_still_A():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MINOR", "type": "skipped_non_usd_reporter"}]}}
    rv = grade_run(_run(v, intrinsic={"fair_value": {"base": None}}))
    assert rv.grade == "A"


def test_unknown_blocking_flag_holds():
    v = {"phase_7_5_net_debt": {"violations": [{
        "severity": "MATERIAL", "type": "definitional_drift",
        "claimed_dollars": 50000000000.0, "match_text": "Net debt is $50B"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "HOLD"
    assert len(rv.needs_adjudication) == 1


def test_minor_unknown_does_not_block():
    v = {"phase_x": {"violations": [{"severity": "MINOR", "type": "whatever"}]}}
    rv = grade_run(_run(v))
    assert rv.grade == "A"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_grader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.cadence.grader'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/cadence/grader.py
from __future__ import annotations

from tradingagents.cadence.models import FlagDisposition, RunData, RunVerdict
from tradingagents.cadence.fp_classifier import classify_run_flags


def grade_run(run: RunData) -> RunVerdict:
    verdicts = classify_run_flags(run)
    auto_dismissed = [v for v in verdicts
                      if v.disposition in (FlagDisposition.DISMISS,
                                           FlagDisposition.CORRECT_BY_DESIGN)]
    # Only MATERIAL/blocking NEEDS_ADJUDICATION flags hold the grade.
    blocking_open = [v for v in verdicts
                     if v.disposition is FlagDisposition.NEEDS_ADJUDICATION
                     and (v.detail.get("severity") != "MINOR")]
    grade = "A" if not blocking_open else "HOLD"
    return RunVerdict(ticker=run.ticker, grade=grade, flag_verdicts=verdicts,
                      auto_dismissed=auto_dismissed, needs_adjudication=blocking_open)
```

Note: `severity` is read from `detail`. Update `classify_violation`'s `detail`
dict (Task 4) to also carry severity — add `"severity"` to the tuple of keys it
copies. Make that one-line edit now and re-run Task 4's tests to confirm still green:

Run: `.venv/bin/python -m pytest tests/cadence/test_fp_classifier_netdebt.py -q`
Expected: PASS (4 passed)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_grader.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/grader.py tradingagents/cadence/fp_classifier.py tests/cadence/test_grader.py
git commit -m "feat(cadence): grader — blocking-only grade, null-FV tolerant

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Side-effects — gog token guard + idempotent publish + promote

**Files:**
- Create: `tradingagents/cadence/publish.py`
- Test: `tests/cadence/test_publish.py`

All gog/Drive ops shell out and are tested with a fake runner injected via a
`run` callable (default `subprocess.run`). `promote` is pure filesystem.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_publish.py
import json
import pytest
from pathlib import Path
from tradingagents.cadence import publish

pytestmark = pytest.mark.unit


class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_token_valid_true_when_grep_hits():
    runner = lambda args, **kw: FakeProc(out="trueknotsg@gmail.com  valid")
    assert publish.gog_token_valid(run=runner) is True


def test_token_valid_false_on_invalid_grant():
    runner = lambda args, **kw: FakeProc(rc=1, err='oauth2: "invalid_grant"')
    assert publish.gog_token_valid(run=runner) is False


def test_publish_pdf_appends_when_absent(tmp_path):
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("AAA\tID_A\n")
    calls = []
    def runner(args, **kw):
        calls.append(args)
        return FakeProc(out=json.dumps({"file": {"id": "NEW_ID"}}))
    fid = publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                              parent="PARENT", account="acct", run=runner)
    assert fid == "NEW_ID"
    rows = dict(l.split("\t") for l in manifest.read_text().splitlines())
    assert rows["BBB"] == "NEW_ID"
    assert rows["AAA"] == "ID_A"   # untouched


def test_publish_pdf_replaces_when_present(tmp_path):
    manifest = tmp_path / "pdf_ids.tsv"
    manifest.write_text("BBB\tOLD_ID\n")
    seen = []
    def runner(args, **kw):
        seen.append(args[:3])
        return FakeProc(out=json.dumps({"file": {"id": "REPL_ID"}}))
    fid = publish.publish_pdf("BBB", tmp_path / "b.pdf", manifest,
                              parent="PARENT", account="acct", run=runner)
    assert fid == "REPL_ID"
    assert any("trash" in " ".join(a) for a in seen)   # old id trashed
    rows = dict(l.split("\t") for l in manifest.read_text().splitlines())
    assert rows["BBB"] == "REPL_ID"


def test_promote_moves_only_on_call(tmp_path):
    run_dir = tmp_path / "preaudit" / "2026-06-05-BBB"
    (run_dir).mkdir(parents=True)
    (run_dir / "decision.md").write_text("x")
    final_base = tmp_path / "final"
    dest = publish.promote(run_dir, final_base, "wk 24 2026")
    assert dest == final_base / "wk 24 2026" / "2026-06-05-BBB"
    assert (dest / "decision.md").is_file()
    assert not run_dir.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_publish.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.cadence.publish'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/cadence/publish.py
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

GOG = "/opt/homebrew/bin/gog"


def _run_default(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def gog_token_valid(account: str = "trueknotsg@gmail.com", run=_run_default) -> bool:
    proc = run([GOG, "auth", "list"])
    out = f"{getattr(proc, 'stdout', '')}{getattr(proc, 'stderr', '')}"
    if "invalid_grant" in out:
        return False
    return account.split("@")[0] in out and getattr(proc, "returncode", 1) == 0


def _read_manifest(manifest: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            if "\t" in line:
                t, i = line.split("\t", 1)
                rows[t.strip()] = i.strip()
    return rows


def _write_manifest(manifest: Path, rows: dict[str, str]) -> None:
    manifest.write_text("".join(f"{t}\t{i}\n" for t, i in rows.items()))


def publish_pdf(ticker: str, pdf: Path, manifest: Path, *, parent: str,
                account: str, run=_run_default) -> str:
    """Idempotent: replace by known file ID (never name-search)."""
    rows = _read_manifest(manifest)
    old = rows.get(ticker)
    if old:
        run([GOG, "drive", "trash", old, "-a", account])
    proc = run([GOG, "drive", "upload", str(pdf), "--parent", parent,
                "-a", account, "-j"])
    file_id = json.loads(proc.stdout)["file"]["id"]
    rows[ticker] = file_id
    _write_manifest(manifest, rows)
    return file_id


def promote(run_dir: Path, final_base: Path, week: str) -> Path:
    """Move a graded-A run into final/<week>/. Only destructive step."""
    dest_dir = Path(final_base) / week
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(run_dir).name
    shutil.move(str(run_dir), str(dest))
    return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_publish.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/publish.py tests/cadence/test_publish.py
git commit -m "feat(cadence): token guard + idempotent publish + promote

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Summary-sheet refresh wrapper (reuse, don't reimplement)

**Files:**
- Modify: `tradingagents/cadence/publish.py` (add `refresh_summary_sheet`)
- Test: `tests/cadence/test_refresh_summary.py`

The native Research Summary sheet is rebuilt by the existing on-mini updater
(`~/gsheet-tool/update_register.py`, run with the repo venv). The orchestrator
shells it; we do NOT duplicate gsheet logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_refresh_summary.py
import pytest
from tradingagents.cadence import publish

pytestmark = pytest.mark.unit


class FakeProc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_refresh_invokes_updater_with_python(tmp_path):
    calls = []
    def runner(args, **kw):
        calls.append(args)
        return FakeProc(out="updated 9 rows")
    ok = publish.refresh_summary_sheet(python="/venv/bin/python",
                                       script="/u/update_register.py",
                                       account="acct", run=runner)
    assert ok is True
    assert calls[0][0] == "/venv/bin/python"
    assert "/u/update_register.py" in calls[0]


def test_refresh_returns_false_on_nonzero(tmp_path):
    runner = lambda args, **kw: FakeProc(rc=2, err="boom")
    assert publish.refresh_summary_sheet(python="p", script="s",
                                         account="a", run=runner) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_refresh_summary.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'refresh_summary_sheet'`

- [ ] **Step 3: Write minimal implementation**

Append to `tradingagents/cadence/publish.py`:

```python
def refresh_summary_sheet(*, python: str, script: str, account: str,
                          run=_run_default) -> bool:
    proc = run([python, script, "-a", account])
    return getattr(proc, "returncode", 1) == 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_refresh_summary.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/cadence/publish.py tests/cadence/test_refresh_summary.py
git commit -m "feat(cadence): summary-sheet refresh wrapper (reuse on-mini updater)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: CLI orchestrator + JSON result contract + console script

**Files:**
- Create: `cli/cadence_followup.py`
- Modify: `pyproject.toml:42` (add console script under `[project.scripts]`)
- Test: `tests/cadence/test_cli.py`

The CLI ties it together: detect batch → grade each → if token valid, publish +
promote grade-A tickers → refresh sheet once → emit the JSON contract. A
`--no-write` flag (test/safety only; NOT a user-facing mode) computes without
side effects.

- [ ] **Step 1: Write the failing test**

```python
# tests/cadence/test_cli.py
import json
import pytest
from pathlib import Path
from cli import cadence_followup as cf

pytestmark = pytest.mark.unit


def _mk_run(pre, name, validation):
    rd = pre / name
    (rd / "raw").mkdir(parents=True)
    (rd / "decision.md").write_text("- **Reference price:** $10.00 (yfinance close)\n")
    (rd / "validation_report.json").write_text(json.dumps(validation))
    for f in ("intrinsic_value", "peer_ratios", "financials"):
        (rd / "raw" / f"{f}.json").write_text("{}")
    return rd


def test_no_write_emits_contract(tmp_path, capsys):
    pre = tmp_path / "preaudit"
    _mk_run(pre, "2026-06-05-AAA", {"blocking_violations": 1,
        "phase_7_5_net_debt": {"violations": [{"severity": "MATERIAL",
            "type": "definitional_drift", "claimed_dollars": 163000000000.0,
            "match_text": "$163B authorized buyback"}]}})
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trade_date"] == "2026-06-05"
    assert out["tickers"][0]["ticker"] == "AAA"
    assert out["tickers"][0]["grade"] == "A"
    assert out["tickers"][0]["published"] is False   # --no-write


def test_empty_base_reports_no_batch(tmp_path, capsys):
    pre = tmp_path / "preaudit"; pre.mkdir()
    rc = cf.main(["--preaudit-base", str(pre), "--no-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trade_date"] is None
    assert out["tickers"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cadence/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli.cadence_followup'`

- [ ] **Step 3: Write minimal implementation**

```python
# cli/cadence_followup.py
"""Autonomous cadence follow-up: QC the newest preaudit batch and publish passes.

Emits a JSON result contract on stdout for the OpenClaw `cadence-followup` skill
to consume (adjudicate residual flags + compose the DM). Deterministic core lives
in tradingagents/cadence/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tradingagents.cadence.batch import find_latest_batch, load_run
from tradingagents.cadence.grader import grade_run
from tradingagents.cadence import publish as pub

DEFAULT_PREAUDIT = Path.home() / "tkresearch" / "preaudit"
PDF_PARENT = "1-sX6LyPafUFKMdy9sNZh-7wh6YT-5wXs"
MANIFEST = Path.home() / "gsheet-tool" / "pdf_ids.tsv"
ACCOUNT = "trueknotsg@gmail.com"
FINAL_BASE = (Path.home() / "Library/CloudStorage"
              / "GoogleDrive-trueknotsg@gmail.com/My Drive/TK Research/final")
REGISTER_PY = str(Path.home() / "gsheet-tool" / "update_register.py")
VENV_PY = str(Path.home() / "tradingagents" / ".venv" / "bin" / "python")


def _iso_week_folder(date_str: str) -> str:
    import datetime as _dt
    y, m, d = (int(x) for x in date_str.split("-"))
    wk = _dt.date(y, m, d).isocalendar().week
    return f"wk {wk} {y}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preaudit-base", default=str(DEFAULT_PREAUDIT))
    ap.add_argument("--no-write", action="store_true",
                    help="QC + emit contract without Drive/sheet writes (test/safety)")
    args = ap.parse_args(argv)

    date, run_dirs = find_latest_batch(Path(args.preaudit_base))
    result = {"trade_date": date, "batch_size": len(run_dirs),
              "completed": len(run_dirs), "token_valid": None,
              "writes_held": False, "reauth_url": None, "tickers": []}
    if not date:
        print(json.dumps(result, indent=2))
        return 0

    token_ok = (not args.no_write) and pub.gog_token_valid(ACCOUNT)
    result["token_valid"] = token_ok
    if not args.no_write and not token_ok:
        result["writes_held"] = True
        result["reauth_url"] = ("re-auth required: run `gog auth add "
                                f"{ACCOUNT} --services sheets,drive` in the mini browser")

    week = _iso_week_folder(date)
    for rd in run_dirs:
        run = load_run(rd)
        rv = grade_run(run)
        row = {
            "ticker": rv.ticker, "grade": rv.grade,
            "auto_dismissed": [{"phase": v.phase, "reason": v.reason}
                               for v in rv.auto_dismissed],
            "needs_adjudication": [{"phase": v.phase, "reason": v.reason, **v.detail}
                                   for v in rv.needs_adjudication],
            "published": False, "promoted_to": None,
        }
        can_write = rv.grade == "A" and token_ok and not args.no_write
        if can_write:
            pdf = Path(run.run_dir) / f"research-{run.trade_date}-{run.ticker}.pdf"
            if pdf.is_file():
                pub.publish_pdf(run.ticker, pdf, MANIFEST,
                                parent=PDF_PARENT, account=ACCOUNT)
                dest = pub.promote(Path(run.run_dir), FINAL_BASE, week)
                row["published"] = True
                row["promoted_to"] = str(dest)
        result["tickers"].append(row)

    if any(t["published"] for t in result["tickers"]):
        pub.refresh_summary_sheet(python=VENV_PY, script=REGISTER_PY, account=ACCOUNT)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `pyproject.toml` under `[project.scripts]` (after line 43):

```toml
tradingcadencefollowup = "cli.cadence_followup:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cadence/test_cli.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full cadence suite + commit**

Run: `.venv/bin/python -m pytest tests/cadence -q`
Expected: PASS (all cadence tests green)

```bash
git add cli/cadence_followup.py pyproject.toml tests/cadence/test_cli.py
git commit -m "feat(cadence): CLI orchestrator + JSON contract + console script

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Bot SKILL.md source + deploy note

**Files:**
- Create: `ops/openclaw-skills/cadence-followup/SKILL.md`
- Modify: `.claude/skills/deploy-mini/SKILL.md` (add copy + reinstall step)

No automated test (it is a runbook for the bot LLM). Verification is manual,
described in the final section.

- [ ] **Step 1: Write the SKILL.md**

```markdown
---
name: cadence-followup
description: >-
  CANONICAL handler for "follow up the cadence", "QC and publish the latest
  cadence", "publish the passes", "finalize the latest research batch". Runs the
  deterministic QC + publish orchestrator over the newest preaudit batch and
  posts a per-ticker verdict + batch summary. DM (SK) only. Distinct from
  `trading-research` (runs a NEW research) and `research-summary` (reads reports).
---

# Cadence follow-up (autonomous QC → publish)

Invoke when SK DMs anything matching: "follow up the cadence", "QC + publish the
latest cadence", "finalize the batch", "publish the passes". DM only.

## How to run

1. Run the orchestrator (it auto-detects the newest preaudit batch):

       GOG_KEYRING_PASSWORD=$GOG_KEYRING_PASSWORD ~/tradingagents/.venv/bin/tradingcadencefollowup

   It prints a JSON object. Parse it — do NOT re-derive the grades yourself.

2. For each ticker with `needs_adjudication` non-empty: open the cited `file` at
   `line_no` in the run dir and the relevant `raw/*.json`, decide real-vs-FP using
   the same patterns the classifier knows (net-debt $-grab, price-date from→to,
   peer "respectively"). If it is genuinely a real defect, leave the ticker
   unpublished and call it out for manual fix. If it is in fact a false positive
   the classifier missed, note it (so the pattern can be added later) and you MAY
   re-run with that ticker forced — but only when you are confident.

3. If `writes_held` is true, the gog token is expired: post the `reauth_url` to SK,
   list the grade-A tickers awaiting publish, and stop (do not retry writes until
   SK confirms re-auth).

4. Compose the DM:
   - Per ticker: `<T>: <grade>` + one-line reason (published / held: <defect>).
   - Batch summary: N graded A & published, M held, any re-auth needed.

## Guardrails

- Idempotent: re-running cannot create Drive duplicates (publish is by file ID).
- Promote (`mv` to final/) happens only for grade A — never touch a held ticker.
- Never run this against a batch the laptop session is actively QC-ing (ask SK if
  unsure which batch is live).
```

- [ ] **Step 2: Add the deploy step**

In `.claude/skills/deploy-mini/SKILL.md`, add to the deploy sequence:

```bash
# Deploy the cadence-followup bot skill + reinstall so the console script exists
ssh macmini-trueknot 'mkdir -p ~/.openclaw/workspace/skills/cadence-followup && \
  cp ~/tradingagents/ops/openclaw-skills/cadence-followup/SKILL.md \
     ~/.openclaw/workspace/skills/cadence-followup/SKILL.md && \
  cd ~/tradingagents && .venv/bin/pip install -e . --quiet'
# then kickstart the daemon so the new skill loads:
ssh macmini-superqsp "sudo launchctl kickstart -k system/com.trueknot.openclaw.gateway"
```

- [ ] **Step 3: Commit**

```bash
git add ops/openclaw-skills/cadence-followup/SKILL.md .claude/skills/deploy-mini/SKILL.md
git commit -m "feat(cadence): bot SKILL.md + deploy-mini wiring

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Full-suite green + manual mini verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole unit suite**

Run: `.venv/bin/python -m pytest -q -m unit --tb=line`
Expected: PASS — existing suite + the new `tests/cadence/*` all green.

- [ ] **Step 2: Deploy to the mini** via the `deploy-mini` skill (now includes the
  SKILL.md copy + reinstall + daemon kickstart from Task 11).

- [ ] **Step 3: Smoke-test the CLI on the mini against a REAL past batch in
  `--no-write` mode** (no writes — this is a one-off verification, not a standing
  mode):

Run: `ssh macmini-trueknot '~/tradingagents/.venv/bin/tradingcadencefollowup --no-write'`
Expected: JSON contract listing the newest batch's tickers with grades; spot-check
that the wk24 tickers already adjudicated by the laptop session (AAPL/AMKR/ASX/INTC
auto-dismiss, FUTU/IFNNY correct-by-design) come back grade A.

- [ ] **Step 4: Live DM test** (next cadence, per the spec's wk24 boundary): DM
  `@TrueKnotBot` "follow up the latest cadence" and confirm it returns a per-ticker
  verdict, publishes grade-A passes idempotently, and refreshes the summary sheet.

- [ ] **Step 5: Update memory** — add a `project_bot_cadence_followup_deployed.md`
  note (skill name, CLI entry point, the FP-pattern catalogue location) and a
  `MEMORY.md` pointer.

---

## Self-Review

**Spec coverage:** batch detection (T2), run loading (T3), FP-classifier 3 patterns
+ non-USD + plausibility tolerance (T4–T7, grader tolerates null FV), idempotent
publish (T8), promote gated on A (T8/T10), gsheet refresh reuse (T9), token guard +
graceful hold (T8/T10), JSON contract (T10), DM/SKILL.md + adjudication + re-auth
relay (T11), wk24 boundary (T11 guardrail + T12 step 4), secrets via
`GOG_KEYRING_PASSWORD` env (T11). All spec sections map to a task.

**Placeholder scan:** no TBD/TODO; every code step is complete and runnable.

**Type consistency:** `RunData`/`RunVerdict`/`FlagVerdict`/`FlagDisposition` defined
in T1 and used consistently; `classify_run_flags`/`classify_violation`/`grade_run`/
`find_latest_batch`/`load_run`/`gog_token_valid`/`publish_pdf`/`promote`/
`refresh_summary_sheet` signatures match across tasks. T7 adds `"severity"` to the
T4 `detail` keys — called out explicitly to keep `grade_run`'s severity read valid.

**Out of scope (unchanged):** fixing the validators themselves; running research;
dry-run mode; WhatsApp agent.
```
