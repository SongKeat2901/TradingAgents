# Liquidity-Anchored 12-Month Probability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM-eyeballed volume zones and guessed scenario probabilities with deterministic Python: a volume profile that defines the liquidity levels, and a block-bootstrap Monte Carlo that computes how likely those levels are to be reached over 12 months — both hard-anchored into the agents.

**Architecture:** Two new pure-Python modules (`volume_profile.py`, `forward_distribution.py`) computed in the Researcher node, written to `raw/*.json`, rendered as Markdown blocks appended to `pm_brief.md`, and injected verbatim into the TA Agents (levels) and Portfolio Manager (scenario probabilities) — the same pattern as `classifier.py` / `peer_ratios.py` / the Phase 7.15 PM injection. The existing `classifier.py` BREAKOUT/BREAKDOWN rules are augmented with volume-node confirmation. A new validator gates delivery on scenario-probability fidelity.

**Tech Stack:** Python 3.12, stdlib only (`statistics`, `random`, `math`, `json`) — no numpy/scipy dependency added; pytest; existing yfinance-sourced `prices.json` (`{"ohlcv": "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\\n…"}`).

**Spec:** `docs/superpowers/specs/2026-05-28-liquidity-anchored-probability-design.md`

**Refinement vs spec:** scenario probabilities use a **first-barrier-touch** partition (each simulated path classified by which target it reaches first: bull-first / bear-first / neither=base). This is touch-based (per the approved decision) AND naturally mutually exclusive summing to 100%, avoiding the arbitrary normalization the spec sketched. Independent touch probabilities and terminal quantiles are also stored as cross-checks.

---

## File Structure

- **Create** `tradingagents/agents/utils/volume_profile.py` — OHLCV parsing, window selection, volume-by-price histogram, POC/Value-Area/HVN/LVN extraction, JSON builder, Markdown block renderer.
- **Create** `tradingagents/agents/utils/forward_distribution.py` — log-return extraction, block-bootstrap path simulation, first-barrier-touch scenario probabilities (targets from volume profile), JSON builder, Markdown block renderer.
- **Create** `tradingagents/validators/scenario_probability_validator.py` — validate PM scenario targets/probabilities against `forward_probabilities.json`.
- **Modify** `tradingagents/agents/utils/classifier.py` — add volume-node confirmation to BREAKOUT/BREAKDOWN.
- **Modify** `tradingagents/agents/researcher.py` — compute + write the two new JSONs and append the two blocks to `pm_brief.md`.
- **Modify** `tradingagents/agents/analysts/ta_agent.py` — inject the volume-profile block; retire the eyeballed "Volume profile zones" instruction.
- **Modify** `tradingagents/agents/managers/portfolio_manager.py` — inject the forward-probability block (mirror the Phase 7.15 peer-ratios injection).
- **Modify** `cli/research_validation.py` — wire the new validator into `_collect_violations` + report schema.
- **Create tests** `tests/test_volume_profile.py`, `tests/test_forward_distribution.py`, `tests/test_scenario_probability_validator.py`; **extend** `tests/test_classifier.py`.

---

## Phase 1 — Volume Profile

### Task 1: OHLCV parsing + window selection

**Files:**
- Create: `tradingagents/agents/utils/volume_profile.py`
- Test: `tests/test_volume_profile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_volume_profile.py
import pytest
pytestmark = pytest.mark.unit

_OHLCV = (
    "# Stock data for X\n"
    "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    "2026-05-20,10,12,9,11,1000,0.0,0.0\n"
    "2026-05-21,11,13,10,12,2000,0.0,0.0\n"
)

def test_parse_ohlcv_rows():
    from tradingagents.agents.utils.volume_profile import parse_ohlcv
    rows = parse_ohlcv(_OHLCV)
    assert len(rows) == 2
    assert rows[0] == ("2026-05-20", 10.0, 12.0, 9.0, 11.0, 1000.0)
    assert rows[1][2] == 13.0  # high

def test_select_window_takes_trailing_rows():
    from tradingagents.agents.utils.volume_profile import parse_ohlcv, select_window
    rows = parse_ohlcv(_OHLCV)
    # 1 "month" ≈ 21 trading days; request more than available → returns all
    assert select_window(rows, months=36) == rows
    # request 0 → empty
    assert select_window(rows, months=0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.agents.utils.volume_profile'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/volume_profile.py
"""Deterministic volume-by-price profile (liquidity levels).

Replaces the LLM-eyeballed "Volume profile zones" section with a computed
volume-by-price histogram and the levels traders care about: Point of
Control, Value Area (70%), High/Low Volume Nodes. Output is consumed
verbatim by the TA Agents and is the target source for the forward
distribution model.

stdlib only — no numpy. See docs/superpowers/specs/
2026-05-28-liquidity-anchored-probability-design.md.
"""
from __future__ import annotations

Bar = tuple[str, float, float, float, float, float]  # date, o, h, l, c, volume

_TRADING_DAYS_PER_MONTH = 21


def parse_ohlcv(ohlcv_csv: str) -> list[Bar]:
    """Parse the prices.json ohlcv CSV string into typed bars (skips
    comment/header lines and malformed rows)."""
    rows: list[Bar] = []
    for line in ohlcv_csv.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append((parts[0], float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4]), float(parts[5])))
        except ValueError:
            continue
    return rows


def select_window(rows: list[Bar], months: int) -> list[Bar]:
    """Return the trailing `months` of bars (≈21 trading days/month)."""
    if months <= 0:
        return []
    n = months * _TRADING_DAYS_PER_MONTH
    return rows[-n:] if len(rows) > n else list(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/volume_profile.py tests/test_volume_profile.py
git commit -m "feat(volume-profile): OHLCV parsing + trailing-window selection"
```

### Task 2: Volume-by-price histogram + POC + Value Area

**Files:**
- Modify: `tradingagents/agents/utils/volume_profile.py`
- Test: `tests/test_volume_profile.py`

- [ ] **Step 1: Write the failing test**

```python
def test_histogram_poc_and_value_area():
    from tradingagents.agents.utils.volume_profile import (
        parse_ohlcv, build_histogram, point_of_control, value_area,
    )
    # Two bars: most volume transacts in the 10-12 band.
    ohlcv = (
        "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
        "2026-05-20,10,12,10,11,9000,0.0,0.0\n"   # heavy volume, tight 10-12
        "2026-05-21,18,20,18,19,1000,0.0,0.0\n"   # light volume, far 18-20
    )
    rows = parse_ohlcv(ohlcv)
    bins = build_histogram(rows, n_bins=20)
    poc = point_of_control(bins)
    assert 10.0 <= poc <= 12.0          # control sits in the heavy band
    val, vah = value_area(bins, pct=0.70)
    assert val <= poc <= vah
    assert vah < 18.0                    # the light far band is outside value area
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py::test_histogram_poc_and_value_area -v`
Expected: FAIL — `ImportError: cannot import name 'build_histogram'`

- [ ] **Step 3: Write minimal implementation** (append to `volume_profile.py`)

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    mid: float
    volume: float


def build_histogram(rows: list[Bar], n_bins: int = 50) -> list[Bin]:
    """Volume-by-price histogram. Each bar's volume is spread uniformly
    across the price bins its [low, high] range overlaps."""
    if not rows:
        return []
    lo = min(r[3] for r in rows)
    hi = max(r[2] for r in rows)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / n_bins
    edges = [lo + i * width for i in range(n_bins + 1)]
    vols = [0.0] * n_bins
    for _d, _o, h, low, _c, vol in rows:
        if h <= low:
            h = low + 1e-9
        span = h - low
        # distribute this bar's volume across the bins it overlaps,
        # proportional to the overlap length
        for i in range(n_bins):
            overlap = min(h, edges[i + 1]) - max(low, edges[i])
            if overlap > 0:
                vols[i] += vol * (overlap / span)
    return [Bin(edges[i], edges[i + 1], (edges[i] + edges[i + 1]) / 2, vols[i])
            for i in range(n_bins)]


def point_of_control(bins: list[Bin]) -> float | None:
    """Mid-price of the highest-volume bin."""
    if not bins:
        return None
    return max(bins, key=lambda b: b.volume).mid


def value_area(bins: list[Bin], pct: float = 0.70) -> tuple[float | None, float | None]:
    """Expand outward from the POC bin until `pct` of total volume is
    captured; return (value_area_low, value_area_high)."""
    if not bins:
        return (None, None)
    total = sum(b.volume for b in bins)
    if total <= 0:
        return (None, None)
    poc_idx = max(range(len(bins)), key=lambda i: bins[i].volume)
    captured = bins[poc_idx].volume
    lo_idx = hi_idx = poc_idx
    target = pct * total
    while captured < target and (lo_idx > 0 or hi_idx < len(bins) - 1):
        below = bins[lo_idx - 1].volume if lo_idx > 0 else -1.0
        above = bins[hi_idx + 1].volume if hi_idx < len(bins) - 1 else -1.0
        if above >= below:
            hi_idx += 1
            captured += bins[hi_idx].volume
        else:
            lo_idx -= 1
            captured += bins[lo_idx].volume
    return (bins[lo_idx].low, bins[hi_idx].high)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/volume_profile.py tests/test_volume_profile.py
git commit -m "feat(volume-profile): histogram + point-of-control + value area"
```

### Task 3: High/Low Volume Node extraction

**Files:**
- Modify: `tradingagents/agents/utils/volume_profile.py`
- Test: `tests/test_volume_profile.py`

- [ ] **Step 1: Write the failing test**

```python
def test_hvn_lvn_extraction():
    from tradingagents.agents.utils.volume_profile import Bin, high_volume_nodes, low_volume_nodes
    # volume shape: peak, trough, peak  → 1 HVN each side, 1 LVN in the gap
    bins = [
        Bin(9, 10, 9.5, 100), Bin(10, 11, 10.5, 900), Bin(11, 12, 11.5, 100),
        Bin(12, 13, 12.5, 50),  Bin(13, 14, 13.5, 80),  Bin(14, 15, 14.5, 800),
        Bin(15, 16, 15.5, 90),
    ]
    hvn = high_volume_nodes(bins, max_nodes=3)
    lvn = low_volume_nodes(bins, max_nodes=3)
    assert 10.5 in [round(p, 1) for p in hvn]   # left peak
    assert 14.5 in [round(p, 1) for p in hvn]   # right peak
    assert any(12.0 <= p <= 13.0 for p in lvn)  # the gap/trough
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py::test_hvn_lvn_extraction -v`
Expected: FAIL — `ImportError: cannot import name 'high_volume_nodes'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def _local_extrema(bins: list[Bin], want_peaks: bool) -> list[tuple[float, float]]:
    """Return [(mid_price, volume)] for local maxima (peaks) or minima
    (troughs) of the volume histogram, comparing each interior bin to its
    immediate neighbours."""
    out: list[tuple[float, float]] = []
    for i in range(1, len(bins) - 1):
        v, lft, rgt = bins[i].volume, bins[i - 1].volume, bins[i + 1].volume
        is_peak = v >= lft and v >= rgt
        is_trough = v <= lft and v <= rgt
        if (want_peaks and is_peak) or (not want_peaks and is_trough):
            out.append((bins[i].mid, v))
    return out


def high_volume_nodes(bins: list[Bin], max_nodes: int = 5) -> list[float]:
    """Mid-prices of the strongest local volume peaks (descending volume)."""
    peaks = _local_extrema(bins, want_peaks=True)
    peaks.sort(key=lambda pv: -pv[1])
    return [round(p, 2) for p, _v in peaks[:max_nodes]]


def low_volume_nodes(bins: list[Bin], max_nodes: int = 5) -> list[float]:
    """Mid-prices of the deepest local volume troughs (ascending volume)."""
    troughs = _local_extrema(bins, want_peaks=False)
    troughs.sort(key=lambda pv: pv[1])
    return [round(p, 2) for p, _v in troughs[:max_nodes]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/volume_profile.py tests/test_volume_profile.py
git commit -m "feat(volume-profile): HVN/LVN local-extrema extraction"
```

### Task 4: compute_volume_profile (dual window) + JSON shape

**Files:**
- Modify: `tradingagents/agents/utils/volume_profile.py`
- Test: `tests/test_volume_profile.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_volume_profile_dual_window():
    from tradingagents.agents.utils.volume_profile import compute_volume_profile
    ohlcv = "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n" + "\n".join(
        f"2026-{(i//28)+1:02d}-{(i%28)+1:02d},10,12,10,11,{1000+i}" for i in range(60)
    )
    vp = compute_volume_profile(ohlcv, n_bins=20)
    for win in ("structural_36mo", "tactical_6mo"):
        assert win in vp
        assert vp[win]["poc"] is not None
        assert vp[win]["vah"] >= vp[win]["val"]
        assert isinstance(vp[win]["hvn"], list)
        assert isinstance(vp[win]["lvn"], list)
    assert vp["n_bins"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py::test_compute_volume_profile_dual_window -v`
Expected: FAIL — `ImportError: cannot import name 'compute_volume_profile'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def _profile_one_window(rows: list[Bar], n_bins: int) -> dict:
    bins = build_histogram(rows, n_bins=n_bins)
    val, vah = value_area(bins)
    return {
        "poc": point_of_control(bins),
        "vah": vah,
        "val": val,
        "hvn": high_volume_nodes(bins),
        "lvn": low_volume_nodes(bins),
        "n_bars": len(rows),
    }


def compute_volume_profile(ohlcv_csv: str, n_bins: int = 50) -> dict:
    """Compute the 36-month structural and 6-month tactical volume profiles.
    Returns a JSON-serialisable dict consumed by researcher.py."""
    rows = parse_ohlcv(ohlcv_csv)
    return {
        "n_bins": n_bins,
        "structural_36mo": _profile_one_window(select_window(rows, 36), n_bins),
        "tactical_6mo": _profile_one_window(select_window(rows, 6), n_bins),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/volume_profile.py tests/test_volume_profile.py
git commit -m "feat(volume-profile): dual-window compute_volume_profile + JSON shape"
```

### Task 5: format_volume_profile_block (Markdown)

**Files:**
- Modify: `tradingagents/agents/utils/volume_profile.py`
- Test: `tests/test_volume_profile.py`

- [ ] **Step 1: Write the failing test**

```python
def test_format_volume_profile_block():
    from tradingagents.agents.utils.volume_profile import format_volume_profile_block
    vp = {
        "n_bins": 50,
        "structural_36mo": {"poc": 100.0, "vah": 110.0, "val": 90.0,
                             "hvn": [108.0, 92.0], "lvn": [101.0], "n_bars": 700},
        "tactical_6mo": {"poc": 105.0, "vah": 112.0, "val": 98.0,
                          "hvn": [111.0], "lvn": [106.0], "n_bars": 120},
    }
    block = format_volume_profile_block(vp)
    assert "## Liquidity / Volume profile" in block
    assert "100.0" in block and "110.0" in block  # POC + VAH appear
    assert "Use these levels verbatim" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py::test_format_volume_profile_block -v`
Expected: FAIL — `ImportError: cannot import name 'format_volume_profile_block'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def _fmt(v: float | None) -> str:
    return f"${v:.2f}" if isinstance(v, (int, float)) else "(n/a)"


def format_volume_profile_block(vp: dict) -> str:
    """Render the computed profile as a Markdown block for pm_brief.md and
    TA-agent injection. Mirrors the peer_ratios block's verbatim-use footer."""
    def _row(label: str, w: dict) -> str:
        hvn = ", ".join(_fmt(x) for x in w.get("hvn", [])) or "(none)"
        lvn = ", ".join(_fmt(x) for x in w.get("lvn", [])) or "(none)"
        return (f"| {label} | {_fmt(w.get('poc'))} | {_fmt(w.get('val'))}"
                f" | {_fmt(w.get('vah'))} | {hvn} | {lvn} |")

    return (
        "\n\n## Liquidity / Volume profile (computed from raw/prices.json)\n\n"
        "| Window | POC | Value-Area Low | Value-Area High | High-Volume Nodes | Low-Volume Nodes |\n"
        "|---|---|---|---|---|---|\n"
        + _row("36-mo structural", vp.get("structural_36mo", {})) + "\n"
        + _row("6-mo tactical", vp.get("tactical_6mo", {})) + "\n\n"
        "*POC = most-transacted price (acceptance). Value Area = 70% of volume. "
        "High-Volume Nodes = liquidity magnets (support/resistance). Low-Volume "
        "Nodes = thin 'slip-through' zones. **Use these levels verbatim** — they "
        "are computed from actual volume-by-price, not eyeballed. Do not invent "
        "alternative 'accumulation zones' from memory.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_volume_profile.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/volume_profile.py tests/test_volume_profile.py
git commit -m "feat(volume-profile): Markdown block renderer"
```

### Task 6: Wire volume profile into the Researcher node

**Files:**
- Modify: `tradingagents/agents/researcher.py` (in `fetch_research_pack`, immediately after the `classification.json` write — around line 224)
- Test: manual integration check (no unit test for I/O glue; covered by the module tests)

- [ ] **Step 1: Add the compute + write + append, after the classification block**

Insert after the `(raw / "classification.json").write_text(...)` call:

```python
    # Phase 8.x: deterministic volume profile (liquidity levels). Computed
    # here, written to raw/, and appended to pm_brief.md so TA agents and
    # the forward-distribution model consume real volume-by-price levels.
    from tradingagents.agents.utils.volume_profile import (
        compute_volume_profile, format_volume_profile_block,
    )
    volume_profile = compute_volume_profile(prices.get("ohlcv", ""))
    (raw / "volume_profile.json").write_text(
        json.dumps(volume_profile, indent=2, default=str), encoding="utf-8"
    )
```

- [ ] **Step 2: Append the block to pm_brief.md alongside the peer-ratios append**

Find where `peer_block` is appended to `pm_brief.md` (the `with pm_brief_path.open("a")` or `pm_brief_path.write_text(existing + ...)` call near the end of `fetch_research_pack`) and add, in the same append, after the peer block:

```python
    vp_block = format_volume_profile_block(volume_profile)
    with (raw / "pm_brief.md").open("a", encoding="utf-8") as fh:
        fh.write(vp_block)
```

(Match the existing append idiom in the file — if it rebuilds the whole string, append `vp_block` to that string instead.)

- [ ] **Step 3: Verify on a real run dir**

Run:
```bash
.venv/bin/python -c "
import json, sys; sys.path.insert(0,'.')
from tradingagents.agents.utils.volume_profile import compute_volume_profile, format_volume_profile_block
d=json.load(open('/Users/trueknot/.openclaw/data/research/2026-05-26-GOOGL/raw/prices.json'))
vp=compute_volume_profile(d['ohlcv']); print(format_volume_profile_block(vp))
"
```
Expected: a populated Liquidity/Volume-profile table with real GOOGL POC/VAH/VAL/HVN values.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/researcher.py
git commit -m "feat(volume-profile): wire compute + pm_brief append into Researcher"
```

### Task 7: Inject volume profile into the TA Agent; retire eyeballed zones

**Files:**
- Modify: `tradingagents/agents/analysts/ta_agent.py` (the `_SYSTEM` / `_SYSTEM_V2` "## Volume profile zones" instruction, and the `format_for_prompt(... files=[...])` lists to include `volume_profile.json`)

- [ ] **Step 1: Add `volume_profile.json` to the TA-agent context files**

In `create_ta_agent_node` and `create_ta_agent_v2_node`, add `"volume_profile.json"` to the `format_for_prompt(raw_dir, files=[...])` list.

- [ ] **Step 2: Replace the eyeballed instruction**

In `_SYSTEM` and `_SYSTEM_V2`, change the "## Volume profile zones" section instruction to:

```text
## Volume profile zones

Use the computed POC, Value Area (VAH/VAL), High-Volume Nodes and
Low-Volume Nodes from raw/volume_profile.json VERBATIM. State the 36-mo
structural levels and the 6-mo tactical levels. Do NOT estimate
"accumulation bands" by eye — cite the computed cells.
```

- [ ] **Step 3: Verify the prompt assembles**

Run: `.venv/bin/python -c "import tradingagents.agents.analysts.ta_agent as m; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/analysts/ta_agent.py
git commit -m "feat(volume-profile): TA agents consume computed levels, retire eyeballed zones"
```

---

## Phase 2 — Classifier breakout/breakdown augmentation

### Task 8: Add volume-node confirmation to BREAKOUT/BREAKDOWN

**Files:**
- Modify: `tradingagents/agents/utils/classifier.py` (`compute_classification` signature + the BREAKOUT / BREAKDOWN branches + the return dict)
- Modify: `tradingagents/agents/researcher.py` (pass `volume_profile` into `compute_classification`)
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier.py (append)
def test_breakout_records_cleared_hvn_when_volume_profile_supplied():
    from tradingagents.agents.utils.classifier import compute_classification
    # Construct a fresh-golden-cross BREAKOUT setup
    reference = {
        "reference_price": 105.0, "spot_50dma": 104.0, "spot_200dma": 103.0,
        "ytd_high": 110.0, "ytd_low": 80.0, "atr_14": 2.0,
    }
    ohlcv = "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits\n" + "\n".join(
        f"2026-01-{(i%28)+1:02d},100,106,99,105,{5000 if i>250 else 1000}" for i in range(260)
    )
    vp = {"structural_36mo": {"vah": 102.0, "hvn": [101.5, 98.0], "val": 97.0}}
    out = compute_classification(reference, ohlcv, volume_profile=vp)
    # spot 105 cleared the 102 VAH / 101.5 HVN below it
    assert out.get("volume_confirmed") in (True, False)  # field present
    if out["setup_class"] == "BREAKOUT":
        assert out["broken_level"] is not None
        assert out["broken_level_type"] in ("HVN", "VAH")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_classifier.py::test_breakout_records_cleared_hvn_when_volume_profile_supplied -v`
Expected: FAIL — `TypeError: compute_classification() got an unexpected keyword argument 'volume_profile'`

- [ ] **Step 3: Implement**

In `compute_classification`, change the signature to:
```python
def compute_classification(reference: dict, ohlcv_csv: str, history_window: int = 90,
                           volume_profile: dict | None = None) -> dict[str, Any]:
```

Add this helper near the top of `classifier.py`:
```python
def _confirm_break(spot: float, direction: str, vp: dict | None
                   ) -> tuple[float | None, str | None]:
    """Return (broken_level, level_type) for the nearest structural level the
    spot has cleared (direction='up' for breakout, 'down' for breakdown)."""
    if not vp:
        return (None, None)
    w = vp.get("structural_36mo", {})
    candidates: list[tuple[float, str]] = []
    if w.get("vah") is not None:
        candidates.append((w["vah"], "VAH"))
    if w.get("val") is not None:
        candidates.append((w["val"], "VAL"))
    for hvn in w.get("hvn", []) or []:
        candidates.append((hvn, "HVN"))
    if direction == "up":
        cleared = [(lvl, t) for lvl, t in candidates if lvl <= spot]
        return max(cleared, key=lambda x: x[0]) if cleared else (None, None)
    cleared = [(lvl, t) for lvl, t in candidates if lvl >= spot]
    return min(cleared, key=lambda x: x[0]) if cleared else (None, None)
```

In the BREAKOUT branch (where `setup = "BREAKOUT"` is assigned) add:
```python
        broken_level, broken_level_type = _confirm_break(spot, "up", volume_profile)
```
In the BREAKDOWN branch add:
```python
        broken_level, broken_level_type = _confirm_break(spot, "down", volume_profile)
```
At the top of `compute_classification` (after the INDETERMINATE guard) initialise defaults so every return path has the fields:
```python
    broken_level, broken_level_type = None, None
```
Add to the final return dict:
```python
        "broken_level": broken_level,
        "broken_level_type": broken_level_type,
        "volume_confirmed": broken_level is not None,
```
And add the same three keys (all `None`/`False`) to the `_INDETERMINATE` dict.

In `researcher.py`, change the classifier call to pass the profile (compute `volume_profile` BEFORE the classifier call, reorder if needed):
```python
    classification = compute_classification(reference, prices.get("ohlcv", ""),
                                            volume_profile=volume_profile)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_classifier.py -v`
Expected: PASS (existing classifier tests + the new one)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/classifier.py tradingagents/agents/researcher.py tests/test_classifier.py
git commit -m "feat(classifier): volume-node confirmation for breakout/breakdown"
```

---

## Phase 3 — 12-Month Forward Distribution

### Task 9: Log returns + block-bootstrap path simulation

**Files:**
- Create: `tradingagents/agents/utils/forward_distribution.py`
- Test: `tests/test_forward_distribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forward_distribution.py
import pytest
pytestmark = pytest.mark.unit


def test_log_returns_and_simulation_is_deterministic():
    from tradingagents.agents.utils.forward_distribution import (
        daily_log_returns, simulate_paths,
    )
    closes = [100.0 * (1.001 ** i) for i in range(400)]  # gentle uptrend
    rets = daily_log_returns(closes)
    assert len(rets) == 399
    paths_a = simulate_paths(spot=closes[-1], returns=rets, horizon=252,
                             n_paths=500, block=10, seed=42)
    paths_b = simulate_paths(spot=closes[-1], returns=rets, horizon=252,
                             n_paths=500, block=10, seed=42)
    assert paths_a == paths_b                       # deterministic on seed
    assert len(paths_a) == 500
    assert all(len(p) == 252 for p in paths_a)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradingagents.agents.utils.forward_distribution'`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/agents/utils/forward_distribution.py
"""Block-bootstrap Monte Carlo 12-month forward distribution.

Resamples contiguous blocks of the trailing 36 months of daily log-returns
to build forward price paths, then classifies each path by the first
liquidity level it reaches (first-barrier-touch). The resulting Bull/Base/
Bear probabilities hard-anchor the Portfolio Manager's scenarios.

Deterministic on (ticker, trade_date) seed — same input → same probabilities,
the same consistency requirement that motivated the Python classifier. stdlib
only. See docs/superpowers/specs/2026-05-28-liquidity-anchored-probability-design.md.
"""
from __future__ import annotations

import math
import random


def daily_log_returns(closes: list[float]) -> list[float]:
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0 and cur > 0:
            out.append(math.log(cur / prev))
    return out


def simulate_paths(spot: float, returns: list[float], horizon: int = 252,
                   n_paths: int = 10000, block: int = 10,
                   seed: int = 0) -> list[list[float]]:
    """Block-bootstrap: assemble each path from random contiguous blocks of
    the historical return series, then exponentiate the cumulative sum off
    `spot`. Returns n_paths lists of `horizon` prices."""
    rng = random.Random(seed)
    if not returns or spot <= 0:
        return [[spot] * horizon for _ in range(n_paths)]
    max_start = max(len(returns) - block, 0)
    paths: list[list[float]] = []
    for _ in range(n_paths):
        seq: list[float] = []
        while len(seq) < horizon:
            start = rng.randint(0, max_start)
            seq.extend(returns[start:start + block])
        seq = seq[:horizon]
        price = spot
        path: list[float] = []
        for r in seq:
            price *= math.exp(r)
            path.append(price)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/forward_distribution.py tests/test_forward_distribution.py
git commit -m "feat(forward-dist): log returns + deterministic block-bootstrap paths"
```

### Task 10: First-barrier-touch scenario probabilities

**Files:**
- Modify: `tradingagents/agents/utils/forward_distribution.py`
- Test: `tests/test_forward_distribution.py`

- [ ] **Step 1: Write the failing test**

```python
def test_first_barrier_touch_partitions_to_100pct():
    from tradingagents.agents.utils.forward_distribution import (
        simulate_paths, first_barrier_probabilities,
    )
    # synthetic paths: half go up to 120, half down to 80
    up = [[100 + i * (20/252) for i in range(1, 253)] for _ in range(5)]
    down = [[100 - i * (20/252) for i in range(1, 253)] for _ in range(5)]
    probs = first_barrier_probabilities(up + down, bull=115.0, bear=85.0)
    assert abs(probs["bull"] + probs["base"] + probs["bear"] - 1.0) < 1e-9
    assert probs["bull"] == pytest.approx(0.5, abs=0.01)
    assert probs["bear"] == pytest.approx(0.5, abs=0.01)
    assert probs["base"] == pytest.approx(0.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py::test_first_barrier_touch_partitions_to_100pct -v`
Expected: FAIL — `ImportError: cannot import name 'first_barrier_probabilities'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def first_barrier_probabilities(paths: list[list[float]], bull: float,
                                bear: float) -> dict[str, float]:
    """Classify each path by the FIRST barrier it touches: bull-first,
    bear-first, or neither (base). Mutually exclusive → sums to 1.0."""
    if not paths:
        return {"bull": 0.0, "base": 1.0, "bear": 0.0}
    n_bull = n_bear = n_base = 0
    for path in paths:
        outcome = "base"
        for px in path:
            if bull is not None and px >= bull:
                outcome = "bull"
                break
            if bear is not None and px <= bear:
                outcome = "bear"
                break
        if outcome == "bull":
            n_bull += 1
        elif outcome == "bear":
            n_bear += 1
        else:
            n_base += 1
    total = len(paths)
    return {"bull": n_bull / total, "base": n_base / total, "bear": n_bear / total}


def touch_probabilities(paths: list[list[float]], bull: float,
                        bear: float) -> dict[str, float]:
    """Independent (non-exclusive) probability each level is touched at all —
    stored as a cross-check alongside the first-barrier partition."""
    if not paths:
        return {"bull": 0.0, "bear": 0.0}
    tb = sum(1 for p in paths if any(px >= bull for px in p)) / len(paths)
    tr = sum(1 for p in paths if any(px <= bear for px in p)) / len(paths)
    return {"bull": tb, "bear": tr}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/forward_distribution.py tests/test_forward_distribution.py
git commit -m "feat(forward-dist): first-barrier-touch scenario partition + touch cross-check"
```

### Task 11: compute_forward_probabilities (targets from volume profile) + JSON + block

**Files:**
- Modify: `tradingagents/agents/utils/forward_distribution.py`
- Test: `tests/test_forward_distribution.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_forward_probabilities_picks_levels_and_sums_100():
    from tradingagents.agents.utils.forward_distribution import (
        compute_forward_probabilities, format_forward_block,
    )
    closes = [100.0 + (i % 7) for i in range(800)]  # noisy flat ~100-106
    vp = {"structural_36mo": {"poc": 103.0, "hvn": [112.0, 95.0], "vah": 106.0, "val": 100.0},
          "tactical_6mo": {"poc": 103.0, "hvn": [109.0, 99.0], "vah": 105.0, "val": 101.0}}
    out = compute_forward_probabilities("XYZ", "2026-05-28", spot=103.0,
                                        closes=closes, volume_profile=vp,
                                        n_paths=500)
    s = out["scenarios"]
    assert abs(s["bull"]["probability"] + s["base"]["probability"]
               + s["bear"]["probability"] - 1.0) < 1e-9
    assert s["bull"]["target"] > 103.0 > s["bear"]["target"]
    block = format_forward_block(out)
    assert "## 12-month scenario probabilities" in block
    assert "Use these targets and probabilities verbatim" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py::test_compute_forward_probabilities_picks_levels_and_sums_100 -v`
Expected: FAIL — `ImportError: cannot import name 'compute_forward_probabilities'`

- [ ] **Step 3: Write minimal implementation** (append)

```python
def _pick_targets(spot: float, vp: dict) -> tuple[float, float, float]:
    """Bull = nearest HVN above spot (fallback VAH, then spot*1.15);
    Base = POC (fallback spot);
    Bear = nearest HVN below spot (fallback VAL, then spot*0.85)."""
    w = vp.get("structural_36mo", {}) if vp else {}
    hvn = w.get("hvn", []) or []
    above = sorted(x for x in hvn if x > spot)
    below = sorted((x for x in hvn if x < spot), reverse=True)
    bull = above[0] if above else (w.get("vah") if (w.get("vah") or 0) > spot else spot * 1.15)
    bear = below[0] if below else (w.get("val") if (w.get("val") or 1e18) < spot else spot * 0.85)
    base = w.get("poc") or spot
    return (round(bull, 2), round(base, 2), round(bear, 2))


def _seed_for(ticker: str, trade_date: str) -> int:
    return abs(hash(f"{ticker}:{trade_date}")) % (2 ** 31)


def compute_forward_probabilities(ticker: str, trade_date: str, spot: float,
                                  closes: list[float], volume_profile: dict,
                                  horizon: int = 252, n_paths: int = 10000,
                                  block: int = 10) -> dict:
    """Full pipeline: targets from volume profile → block-bootstrap paths →
    first-barrier-touch scenario probabilities. Deterministic on ticker+date."""
    bull, base, bear = _pick_targets(spot, volume_profile)
    rets = daily_log_returns(closes)
    paths = simulate_paths(spot, rets, horizon=horizon, n_paths=n_paths,
                           block=block, seed=_seed_for(ticker, trade_date))
    fb = first_barrier_probabilities(paths, bull=bull, bear=bear)
    touch = touch_probabilities(paths, bull=bull, bear=bear)
    terminals = sorted(p[-1] for p in paths)
    def q(frac: float) -> float:
        return round(terminals[min(int(frac * len(terminals)), len(terminals) - 1)], 2)
    return {
        "ticker": ticker, "trade_date": trade_date, "spot": spot,
        "method": "block-bootstrap MC, first-barrier-touch",
        "n_paths": n_paths, "block": block, "horizon": horizon,
        "seed": _seed_for(ticker, trade_date),
        "scenarios": {
            "bull": {"target": bull, "probability": round(fb["bull"], 4),
                     "touch_prob": round(touch["bull"], 4)},
            "base": {"target": base, "probability": round(fb["base"], 4)},
            "bear": {"target": bear, "probability": round(fb["bear"], 4),
                     "touch_prob": round(touch["bear"], 4)},
        },
        "terminal_quantiles": {"p05": q(0.05), "p25": q(0.25), "p50": q(0.50),
                               "p75": q(0.75), "p95": q(0.95)},
    }


def format_forward_block(out: dict) -> str:
    s = out["scenarios"]
    def pct(x): return f"{x * 100:.0f}%"
    return (
        "\n\n## 12-month scenario probabilities (block-bootstrap MC on 36-mo history)\n\n"
        "| Scenario | Target | Probability (first-barrier touch) |\n|---|---|---|\n"
        f"| Bull | ${s['bull']['target']:.2f} | {pct(s['bull']['probability'])} |\n"
        f"| Base | ${s['base']['target']:.2f} | {pct(s['base']['probability'])} |\n"
        f"| Bear | ${s['bear']['target']:.2f} | {pct(s['bear']['probability'])} |\n\n"
        f"Terminal price quantiles: p05 ${out['terminal_quantiles']['p05']:.2f} · "
        f"p50 ${out['terminal_quantiles']['p50']:.2f} · "
        f"p95 ${out['terminal_quantiles']['p95']:.2f}.\n\n"
        "*Targets are volume-profile liquidity levels; probabilities are the "
        "fraction of simulated 12-month paths whose first barrier touch is that "
        "level (Base = neither touched). **Use these targets and probabilities "
        "verbatim** in the Bull/Base/Bear scenario table — do not substitute "
        "judgement-based numbers. They sum to 100% by construction.*\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_forward_distribution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/forward_distribution.py tests/test_forward_distribution.py
git commit -m "feat(forward-dist): targets-from-profile + scenario JSON + Markdown block"
```

### Task 12: Wire forward distribution into Researcher + PM injection

**Files:**
- Modify: `tradingagents/agents/researcher.py` (after the volume-profile write)
- Modify: `tradingagents/agents/managers/portfolio_manager.py` (mirror the Phase 7.15 `peer_ratios_block` injection)

- [ ] **Step 1: Compute + write + append in Researcher**

After the volume-profile write/append, add:
```python
    from tradingagents.agents.utils.forward_distribution import (
        compute_forward_probabilities, format_forward_block,
    )
    from tradingagents.agents.utils.volume_profile import parse_ohlcv
    _closes = [r[4] for r in parse_ohlcv(prices.get("ohlcv", ""))]
    fwd = compute_forward_probabilities(
        ticker, date, spot=close_on_date, closes=_closes,
        volume_profile=volume_profile,
    )
    (raw / "forward_probabilities.json").write_text(
        json.dumps(fwd, indent=2, default=str), encoding="utf-8"
    )
    with (raw / "pm_brief.md").open("a", encoding="utf-8") as fh:
        fh.write(format_forward_block(fwd))
```

- [ ] **Step 2: Inject into the PM prompt**

In `portfolio_manager.py`, locate the Phase 7.15 `peer_ratios_block` construction (reads `raw_dir/pm_brief.md`, slices `## Peer ratios` + `## Net debt`). Extend the `for header in (...)` tuple to also slice `"## 12-month scenario probabilities"` and `"## Liquidity / Volume profile"`, so those computed tables are injected verbatim into the PM prompt the same way. Add to the directive text: "The Bull/Base/Bear scenario targets and probabilities in the '## 12-month scenario probabilities' block are computed and authoritative — use them verbatim in your scenario table."

- [ ] **Step 3: Verify end-to-end assembly**

Run:
```bash
.venv/bin/python -c "import tradingagents.agents.researcher, tradingagents.agents.managers.portfolio_manager; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/researcher.py tradingagents/agents/managers/portfolio_manager.py
git commit -m "feat(forward-dist): wire into Researcher + inject scenarios into PM prompt"
```

---

## Phase 4 — Scenario-probability validator

### Task 13: scenario_probability_validator

**Files:**
- Create: `tradingagents/validators/scenario_probability_validator.py`
- Test: `tests/test_scenario_probability_validator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario_probability_validator.py
import json
import pytest
pytestmark = pytest.mark.unit


def _write_fwd(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(parents=True, exist_ok=True)
    (raw / "forward_probabilities.json").write_text(json.dumps({
        "scenarios": {
            "bull": {"target": 120.0, "probability": 0.30},
            "base": {"target": 103.0, "probability": 0.50},
            "bear": {"target": 90.0, "probability": 0.20},
        }
    }), encoding="utf-8")
    return tmp_path


def test_passes_when_decision_matches_anchor(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_fwd(tmp_path)
    decision = ("| Bull | 30% | $120.00 |\n| Base | 50% | $103.00 |\n"
                "| Bear | 20% | $90.00 |")
    vios = validate_scenario_probabilities(decision, tmp_path)
    assert vios == []


def test_flags_probability_drift_from_anchor(tmp_path):
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
    _write_fwd(tmp_path)
    decision = ("| Bull | 55% | $120.00 |\n| Base | 30% | $103.00 |\n"
                "| Bear | 15% | $90.00 |")   # bull 55 vs anchor 30
    vios = validate_scenario_probabilities(decision, tmp_path)
    assert any(v.severity == "MATERIAL" for v in vios)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenario_probability_validator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# tradingagents/validators/scenario_probability_validator.py
"""Validate the PM's Bull/Base/Bear scenario probabilities/targets against the
deterministic raw/forward_probabilities.json anchor (Phase 8.x)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PROB_TOL = 0.05   # 5 percentage points
_ROW = re.compile(r"\|\s*(Bull|Base|Bear)\s*\|\s*(\d+(?:\.\d+)?)%\s*\|", re.IGNORECASE)


@dataclass(frozen=True)
class ScenarioViolation:
    severity: Literal["MATERIAL", "MINOR"]
    type: str
    scenario: str
    claimed: float
    anchor: float
    match_text: str


def validate_scenario_probabilities(decision_text: str, run_dir) -> list[ScenarioViolation]:
    fwd_path = Path(run_dir) / "raw" / "forward_probabilities.json"
    if not fwd_path.exists():
        return []
    try:
        anchor = json.loads(fwd_path.read_text(encoding="utf-8"))["scenarios"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []
    claimed = {m.group(1).lower(): float(m.group(2)) / 100.0
               for m in _ROW.finditer(decision_text)}
    vios: list[ScenarioViolation] = []
    for scen in ("bull", "base", "bear"):
        if scen not in claimed or scen not in anchor:
            continue
        a = anchor[scen]["probability"]
        c = claimed[scen]
        if abs(c - a) > _PROB_TOL:
            vios.append(ScenarioViolation(
                severity="MATERIAL", type="scenario_probability_drift",
                scenario=scen, claimed=c, anchor=a,
                match_text=f"{scen} claimed {c:.0%} vs anchor {a:.0%}",
            ))
    return vios
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scenario_probability_validator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/validators/scenario_probability_validator.py tests/test_scenario_probability_validator.py
git commit -m "feat(validator): scenario-probability drift vs forward_probabilities.json"
```

### Task 14: Wire validator into research_validation + delivery gate

**Files:**
- Modify: `cli/research_validation.py` (`_collect_violations` + `run_phase_7_validators` report dict + blocking count)

- [ ] **Step 1: Add import + call in `_collect_violations`**

In the imports block add:
```python
    from tradingagents.validators.scenario_probability_validator import (
        validate_scenario_probabilities,
    )
```
After the loop that builds `peer_violations`, add (decision.md is the scenario source):
```python
    decision_path = rd / "decision.md"
    scenario_violations = (
        validate_scenario_probabilities(decision_path.read_text(encoding="utf-8"), rd)
        if decision_path.exists() else []
    )
```
Add `"scenario_violations": scenario_violations,` to the returned dict.

- [ ] **Step 2: Surface in `run_phase_7_validators`**

Add a `phase_8_scenario_probability` section to the serialised report (mirror the existing `phase_7_x` entries with `count` + `violations`) and include `sum(1 for v in raw["scenario_violations"] if v.severity == "MATERIAL")` in the `blocking_violations` and `total_violations` totals.

- [ ] **Step 3: Run the full validator suite**

Run: `.venv/bin/python -m pytest tests/test_scenario_probability_validator.py tests/test_volume_profile.py tests/test_forward_distribution.py tests/test_classifier.py -v`
Expected: PASS (all)

- [ ] **Step 4: Re-validate a real run dir (smoke)**

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from cli.research_validation import run_phase_7_validators
r=run_phase_7_validators('/Users/trueknot/.openclaw/data/research/2026-05-26-GOOGL')
print('blocking', r['blocking_violations'])
"
```
Expected: runs without error (the old GOOGL dir has no forward_probabilities.json → scenario validator no-ops, returns prior count).

- [ ] **Step 5: Commit**

```bash
git add cli/research_validation.py
git commit -m "feat(validator): gate delivery on scenario-probability fidelity"
```

---

## Final integration smoke test

- [ ] Run one full ticker end-to-end and confirm `raw/volume_profile.json`, `raw/forward_probabilities.json` exist, `pm_brief.md` carries both new blocks, the PM scenario table matches the anchor, and `validation_report.json` shows `phase_8_scenario_probability`.

```bash
$HOME/local/bin/tradingresearch --ticker NVDA --date 2026-05-27 \
  --output-dir $HOME/.openclaw/data/research/2026-05-27-NVDA-vptest
```

- [ ] Run the complete suite: `.venv/bin/python -m pytest -q` — expect all prior tests + the new modules' tests green.
