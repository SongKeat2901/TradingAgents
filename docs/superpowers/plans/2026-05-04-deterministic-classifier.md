# Deterministic Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM-judged "Setup classification" + "Asymmetry" sections in TA Agents v1 and v2 with deterministic Python rules. The classifier runs in the Researcher, writes `raw/classification.json`, and is injected into both TA agents' SystemMessages as ground truth. Removes inter-run divergence on the load-bearing technical-classification call.

**Architecture:** New pure-Python module `tradingagents/agents/utils/classifier.py` exposes `compute_classification(reference, ohlcv_csv) -> dict`. Researcher calls it after building `reference.json`. TA Agents v1 and v2 read the result and substitute six placeholders (`$SETUP_CLASS`, `$UP_TARGET`, `$UP_PCT`, `$DOWN_TARGET`, `$DOWN_PCT`, `$RR`) into a new "DETERMINISTIC CLASSIFICATION" block in their SystemMessages.

**Tech Stack:** Python 3.13, pure stdlib (no langchain), pytest with `unit` marker.

**Spec:** `docs/superpowers/specs/2026-05-04-deterministic-classifier-design.md`

---

## File Structure

**Files to create:**

| File | Responsibility |
|---|---|
| `tradingagents/agents/utils/classifier.py` | Pure-Python rule engine (~120 lines): 6-class taxonomy + asymmetry math + INDETERMINATE fallback |
| `tests/test_classifier.py` | Unit tests: one per class with synthetic OHLCV; first-match-wins; asymmetry math; INDETERMINATE |

**Files to modify:**

| File | Why |
|---|---|
| `tradingagents/agents/researcher.py` | Call `compute_classification` + write `raw/classification.json` |
| `tradingagents/agents/analysts/ta_agent.py` | Add DETERMINISTIC CLASSIFICATION block to both `_SYSTEM` constants; substitute placeholders in both node bodies |
| `tests/test_researcher.py` | Assert `classification.json` written + has documented keys |
| `tests/test_ta_agent.py` | Stub `classification.json` in tmp_path; verify substitution in SystemMessage |
| `tests/test_ta_agent_v2.py` | Same |

---

## Task 1: Classifier module + unit tests

**Files:**
- Create: `tradingagents/agents/utils/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
"""Tests for the deterministic technical classifier (Phase-6 stochasticity mitigation)."""
import pytest

pytestmark = pytest.mark.unit


def _ohlcv(rows):
    """Build a get_stock_data-style CSV string from (date, open, high, low, close, volume) rows."""
    header = "# Stock data\nDate,Open,High,Low,Close,Volume,Dividends,Stock Splits\n"
    body = "\n".join(
        f"{d},{o},{h},{l},{c},{v},0.0,0.0" for (d, o, h, l, c, v) in rows
    )
    return header + body + "\n"


def _ref(reference_price, spot_50dma, spot_200dma, ytd_high, ytd_low, atr_14, trade_date="2026-05-01"):
    return {
        "ticker": "MSFT",
        "trade_date": trade_date,
        "reference_price": reference_price,
        "spot_50dma": spot_50dma,
        "spot_200dma": spot_200dma,
        "ytd_high": ytd_high,
        "ytd_low": ytd_low,
        "atr_14": atr_14,
    }


def _flat_history(spot, days=100, vol=20_000_000):
    """100 days of low-vol flat trading at `spot`. Used as a base for tests that
    want one specific day to stand out."""
    from datetime import date, timedelta
    end = date(2026, 5, 1)
    rows = []
    for i in range(days):
        d = (end - timedelta(days=days - 1 - i)).isoformat()
        rows.append((d, spot, spot + 0.5, spot - 0.5, spot, vol))
    return rows


def test_capitulation_top_decile_volume_with_large_move(tmp_path):
    """CAPITULATION: top-decile volume, >1.5σ move, bear MA alignment."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=25_000_000)
    # Last day: 6% drop on 100M volume (top decile vs 25M avg)
    rows[-1] = ("2026-05-01", 410.0, 411.0, 385.0, 385.5, 100_000_000)
    ref = _ref(
        reference_price=385.5,
        spot_50dma=405.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CAPITULATION"
    assert out["recent_volume_signal"] == "capitulation"
    assert out["upside_target"] == 460.0  # 200-DMA
    assert out["downside_target"] == 380.0  # YTD low
    assert "rationale" in out
    assert "100" in out["rationale"] or "top decile" in out["rationale"].lower()


def test_breakdown_below_50dma_with_volume_spike():
    """BREAKDOWN: spot < 50-DMA, 50-DMA < 200-DMA, gap > 8%, vol > 1.5× 50d avg."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    rows[-1] = ("2026-05-01", 410.0, 411.0, 405.0, 408.0, 35_000_000)
    ref = _ref(
        reference_price=408.0,
        spot_50dma=420.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "BREAKDOWN"
    assert out["upside_target"] == 420.0  # 50-DMA recapture


def test_downtrend_catchall_when_bear_alignment_no_other_trigger():
    """DOWNTREND: spot < 200-DMA + 50<200, no capitulation/breakdown trigger."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=410.0,
        spot_50dma=400.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "DOWNTREND"
    assert out["upside_target"] == 460.0
    assert out["downside_target"] == 380.0


def test_consolidation_when_near_both_mas_and_tight_range():
    """CONSOLIDATION: |gap_to_50dma| < 3%, |gap_to_200dma| < 8%, range < 1.5× ATR."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=460.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=460.0,
        spot_50dma=458.0,
        spot_200dma=465.0,
        ytd_high=475.0,
        ytd_low=445.0,
        atr_14=4.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CONSOLIDATION"


def test_uptrend_when_bull_alignment_above_200dma():
    """UPTREND: spot > 200-DMA, 50-DMA > 200-DMA."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=480.0, days=100, vol=20_000_000)
    ref = _ref(
        reference_price=480.0,
        spot_50dma=470.0,
        spot_200dma=440.0,
        ytd_high=485.0,
        ytd_low=400.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "UPTREND"


def test_breakout_recent_50over200_cross_with_volume():
    """BREAKOUT: recent 50/200 golden cross + 5d vol > 90d median."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=470.0, days=100, vol=20_000_000)
    # Last 5 days: high volume
    for i in range(95, 100):
        d, o, h, l, c, _ = rows[i]
        rows[i] = (d, o, h, l, c, 30_000_000)
    ref = _ref(
        reference_price=470.0,
        spot_50dma=465.0,
        spot_200dma=464.0,  # golden cross within last 10d (50 just above 200)
        ytd_high=475.0,
        ytd_low=420.0,
        atr_14=6.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "BREAKOUT"


def test_indeterminate_when_reference_has_nulls():
    """If reference_price or any DMA is None, return INDETERMINATE."""
    from tradingagents.agents.utils.classifier import compute_classification

    ref = _ref(
        reference_price=None,
        spot_50dma=400.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(_flat_history(410.0)))
    assert out["setup_class"] == "INDETERMINATE"


def test_first_match_wins_capitulation_over_breakdown():
    """A setup that satisfies BOTH capitulation and breakdown triggers
    should classify as CAPITULATION (earlier rule wins)."""
    from tradingagents.agents.utils.classifier import compute_classification

    rows = _flat_history(spot=410.0, days=100, vol=20_000_000)
    # Last day: 8% drop on 80M volume (top decile, large move)
    # AND spot is below 50-DMA, 50<200, gap > 8%
    rows[-1] = ("2026-05-01", 410.0, 412.0, 375.0, 377.0, 80_000_000)
    ref = _ref(
        reference_price=377.0,
        spot_50dma=420.0,    # spot < 50-DMA
        spot_200dma=460.0,   # gap > 8% (377 vs 460 = 18% below)
        ytd_high=480.0,
        ytd_low=370.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(rows))
    assert out["setup_class"] == "CAPITULATION"


def test_asymmetry_math_basic():
    """Spot-check the upside/downside arithmetic for a DOWNTREND case."""
    from tradingagents.agents.utils.classifier import compute_classification

    ref = _ref(
        reference_price=400.0,
        spot_50dma=395.0,
        spot_200dma=460.0,
        ytd_high=480.0,
        ytd_low=380.0,
        atr_14=8.0,
    )
    out = compute_classification(ref, _ohlcv(_flat_history(400.0)))
    assert out["setup_class"] == "DOWNTREND"
    # Upside to 200-DMA: (460 - 400) / 400 = 15.0%
    assert abs(out["upside_pct"] - 15.0) < 0.01
    # Downside to YTD low: (380 - 400) / 400 = -5.0%
    assert abs(out["downside_pct"] - (-5.0)) < 0.01
    # R/R = 15.0 / 5.0 = 3.0
    assert abs(out["reward_risk_ratio"] - 3.0) < 0.01
```

- [ ] **Step 2: Run tests to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_classifier.py -v
```

Expected: ImportError on `tradingagents.agents.utils.classifier`.

- [ ] **Step 3: Implement the classifier**

Create `tradingagents/agents/utils/classifier.py`:

```python
"""Deterministic technical-setup classifier (Phase-6 stochasticity mitigation).

The Anthropic API does not produce deterministic outputs at temperature=0
(verified empirically on 2026-05-04). To remove inter-run variance from the
load-bearing setup-classification call, we replace the LLM's free-form
classification with a pure-Python rule engine.

The classifier reads `reference.json` (spot price + DMAs + YTD high/low +
ATR) and the OHLCV CSV string (from `get_stock_data`). It applies a
6-rule taxonomy in priority order (first match wins) and computes
asymmetry math (upside/downside targets and reward/risk ratio).

Output is consumed verbatim by both TA Agent v1 and TA Agent v2 — they
inject the result into their SystemMessage as ground truth and write
prose around it.
"""

from __future__ import annotations

import statistics
from typing import Any


_INDETERMINATE = {
    "setup_class": "INDETERMINATE",
    "gap_to_50dma_pct": None,
    "gap_to_200dma_pct": None,
    "ma_alignment": "unknown",
    "recent_volume_signal": "unknown",
    "upside_target": None,
    "upside_pct": None,
    "downside_target": None,
    "downside_pct": None,
    "reward_risk_ratio": None,
    "rationale": "Reference data missing or null; cannot classify deterministically.",
}


def _parse_ohlcv(ohlcv_csv: str) -> list[tuple[str, float, float, float, float, float]]:
    """Parse get_stock_data CSV into (date, open, high, low, close, volume) rows.

    Skips comment lines (#...) and the header. Malformed rows are dropped.
    """
    rows = []
    for line in ohlcv_csv.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append((
                parts[0],
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            ))
        except ValueError:
            continue
    return rows


def _gap_pct(spot: float, target: float) -> float:
    return (spot - target) / spot * 100.0


def _is_top_decile_volume(latest_volume: float, recent_volumes: list[float]) -> bool:
    if not recent_volumes:
        return False
    sorted_vols = sorted(recent_volumes)
    p90_idx = int(len(sorted_vols) * 0.9)
    p90 = sorted_vols[min(p90_idx, len(sorted_vols) - 1)]
    return latest_volume >= p90


def _daily_move_in_sigmas(latest_close: float, prev_close: float, recent_closes: list[float]) -> float:
    if len(recent_closes) < 2:
        return 0.0
    daily_moves = [
        (recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1] * 100.0
        for i in range(1, len(recent_closes))
    ]
    if not daily_moves:
        return 0.0
    sigma = statistics.stdev(daily_moves) if len(daily_moves) > 1 else 0.0
    if sigma == 0:
        return 0.0
    move = (latest_close - prev_close) / prev_close * 100.0
    return abs(move / sigma)


def _compute_asymmetry(
    setup_class: str,
    spot: float,
    spot_50dma: float,
    spot_200dma: float,
    ytd_high: float,
    ytd_low: float,
    rows: list[tuple[str, float, float, float, float, float]],
) -> dict[str, float | None]:
    """Per-class upside/downside target selection."""
    recent_30 = rows[-30:] if len(rows) >= 30 else rows
    recent_30_high = max((r[2] for r in recent_30), default=spot)
    recent_30_low = min((r[3] for r in recent_30), default=spot)

    if setup_class == "CAPITULATION":
        upside, downside = spot_200dma, ytd_low
    elif setup_class == "BREAKDOWN":
        upside = spot_50dma
        downside = max(ytd_low, spot * 0.90)
    elif setup_class == "DOWNTREND":
        upside, downside = spot_200dma, ytd_low
    elif setup_class == "CONSOLIDATION":
        upside, downside = recent_30_high, recent_30_low
    elif setup_class == "UPTREND":
        upside = max(recent_30_high, spot_200dma * 1.05)
        downside = spot_50dma
    elif setup_class == "BREAKOUT":
        upside = recent_30_high * 1.08
        downside = spot_50dma
    else:
        return {"upside_target": None, "upside_pct": None,
                "downside_target": None, "downside_pct": None,
                "reward_risk_ratio": None}

    upside_pct = (upside - spot) / spot * 100.0
    downside_pct = (downside - spot) / spot * 100.0
    rr = abs(upside_pct) / abs(downside_pct) if downside_pct != 0 else None

    return {
        "upside_target": round(upside, 2),
        "upside_pct": round(upside_pct, 2),
        "downside_target": round(downside, 2),
        "downside_pct": round(downside_pct, 2),
        "reward_risk_ratio": round(rr, 1) if rr is not None else None,
    }


def compute_classification(
    reference: dict,
    ohlcv_csv: str,
    history_window: int = 90,
) -> dict[str, Any]:
    """Apply the 6-class rule engine in priority order; return classification dict.

    Required reference keys: reference_price, spot_50dma, spot_200dma,
    ytd_high, ytd_low, atr_14. If any is None, returns INDETERMINATE.

    Returns a dict with the documented schema (see module docstring) suitable
    for serialization to raw/classification.json.
    """
    spot = reference.get("reference_price")
    ma50 = reference.get("spot_50dma")
    ma200 = reference.get("spot_200dma")
    ytd_high = reference.get("ytd_high")
    ytd_low = reference.get("ytd_low")
    atr = reference.get("atr_14")

    if any(v is None for v in (spot, ma50, ma200, ytd_high, ytd_low, atr)):
        return dict(_INDETERMINATE)

    rows = _parse_ohlcv(ohlcv_csv)
    if not rows:
        return dict(_INDETERMINATE)

    gap_50 = _gap_pct(spot, ma50)
    gap_200 = _gap_pct(spot, ma200)
    bear_aligned = ma50 < ma200
    bull_aligned = ma50 > ma200

    last = rows[-1]
    last_close = last[4]
    last_volume = last[5]
    recent = rows[-history_window:] if len(rows) >= history_window else rows
    recent_volumes = [r[5] for r in recent[:-1]]
    recent_closes = [r[4] for r in recent]

    avg_50d_volume = (
        sum(r[5] for r in rows[-50:-1]) / max(len(rows[-50:-1]), 1)
        if len(rows) >= 2 else 0
    )
    median_90d_volume = (
        statistics.median(r[5] for r in rows[-90:-1])
        if len(rows) >= 2 else 0
    )

    top_decile_vol = _is_top_decile_volume(last_volume, recent_volumes)
    sigmas = _daily_move_in_sigmas(last_close, rows[-2][4] if len(rows) >= 2 else last_close, recent_closes)
    big_move = sigmas > 1.5

    # Rule 1: CAPITULATION
    if top_decile_vol and big_move and bear_aligned:
        setup = "CAPITULATION"
        rationale = (
            f"Top-decile volume ({last_volume:.0f}) on a {sigmas:.1f}σ move; "
            f"50-DMA below 200-DMA (bear alignment)."
        )
        vol_signal = "capitulation"
    # Rule 2: BREAKDOWN
    elif (
        spot < ma50 and bear_aligned and gap_200 < -8.0
        and last_volume > 1.5 * avg_50d_volume
    ):
        setup = "BREAKDOWN"
        rationale = (
            f"Spot below 50-DMA, bear MA alignment, {abs(gap_200):.1f}% below "
            f"200-DMA, latest volume {last_volume / avg_50d_volume:.1f}× 50-day avg."
        )
        vol_signal = "above_average"
    # Rule 3: DOWNTREND (catch-all bear)
    elif spot < ma200 and bear_aligned:
        setup = "DOWNTREND"
        rationale = (
            f"Spot {abs(gap_200):.1f}% below 200-DMA with bear MA alignment "
            f"(50-DMA below 200-DMA); no breakdown/capitulation trigger."
        )
        vol_signal = "normal"
    # Rule 4: CONSOLIDATION
    elif (
        abs(gap_50) < 3.0 and abs(gap_200) < 8.0
        and (max(r[2] for r in rows[-10:]) - min(r[3] for r in rows[-10:])) < 1.5 * atr
    ):
        setup = "CONSOLIDATION"
        rationale = (
            f"Spot near both MAs (gap-50: {gap_50:+.1f}%, gap-200: {gap_200:+.1f}%); "
            f"10-day range tight (< 1.5× ATR-14)."
        )
        vol_signal = "below_average"
    # Rule 5: UPTREND
    elif spot > ma200 and bull_aligned:
        # Check BREAKOUT (Rule 6) takes priority — recent cross + volume confirmation
        # 50/200 golden cross within last 10 trading days
        recent_10 = rows[-10:]
        recent_10_5d_avg = sum(r[5] for r in rows[-5:]) / max(len(rows[-5:]), 1)
        # Approximation: if 50-DMA is just above 200-DMA AND recent 5-day avg
        # volume > 90-day median, treat as fresh breakout.
        if (
            ma50 > ma200 and (ma50 - ma200) / ma200 < 0.02  # close cross
            and recent_10_5d_avg > median_90d_volume
        ):
            setup = "BREAKOUT"
            rationale = (
                f"Recent 50/200 golden cross ({ma50:.2f} just above {ma200:.2f}); "
                f"5-day volume average above 90-day median."
            )
            vol_signal = "breakout_volume"
        else:
            setup = "UPTREND"
            rationale = (
                f"Spot {abs(gap_200):.1f}% above 200-DMA with bull MA alignment "
                f"(50-DMA above 200-DMA)."
            )
            vol_signal = "normal"
    else:
        return dict(_INDETERMINATE) | {
            "rationale": "No rule matched; spot/MA configuration is ambiguous."
        }

    asym = _compute_asymmetry(setup, spot, ma50, ma200, ytd_high, ytd_low, rows)
    alignment = "bullish_aligned" if bull_aligned else ("bearish_aligned" if bear_aligned else "mixed")

    return {
        "setup_class": setup,
        "gap_to_50dma_pct": round(gap_50, 2),
        "gap_to_200dma_pct": round(gap_200, 2),
        "ma_alignment": alignment,
        "recent_volume_signal": vol_signal,
        "upside_target": asym["upside_target"],
        "upside_pct": asym["upside_pct"],
        "downside_target": asym["downside_target"],
        "downside_pct": asym["downside_pct"],
        "reward_risk_ratio": asym["reward_risk_ratio"],
        "rationale": rationale,
    }
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
.venv/bin/python -m pytest tests/test_classifier.py -v
```

Expected: 9 passed.

If a specific test fails, inspect the rule's threshold logic. Common issues:
- BREAKOUT vs UPTREND ordering (BREAKOUT is checked inside UPTREND branch; the test fixture must satisfy the breakout-specific conditions, or it falls through to UPTREND).
- Top-decile volume math: ensure the synthetic CSV has enough flat-volume history for the 90-day window.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/utils/classifier.py tests/test_classifier.py
git commit -m "feat(classifier): pure-Python 6-class technical-setup classifier"
```

---

## Task 2: Researcher writes classification.json

**Files:**
- Modify: `tradingagents/agents/researcher.py`
- Modify: `tests/test_researcher.py`

- [ ] **Step 1: Write failing test**

Open `tests/test_researcher.py` and add this test at the bottom:

```python
def test_researcher_writes_classification_json(tmp_path, monkeypatch):
    """The Researcher must write raw/classification.json with the expected schema."""
    from tradingagents.agents import researcher
    monkeypatch.setattr(researcher, "_fetch_financials", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_news", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_insider", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_social", lambda t, d: {})
    monkeypatch.setattr(researcher, "_fetch_prices", lambda t, d: {"ohlcv": _OHLCV_STUB})
    monkeypatch.setattr(researcher, "_fetch_indicators", lambda t, d: {
        "close_50_sma": _INDICATOR_STUB(405.0),
        "close_200_sma": _INDICATOR_STUB(460.0),
        "rsi": _INDICATOR_STUB(58.0),
        "macd": _INDICATOR_STUB(1.2),
        "boll_ub": _INDICATOR_STUB(430.0),
        "boll_lb": _INDICATOR_STUB(390.0),
        "atr": _INDICATOR_STUB(8.0),
    })

    state = _stub_state(tmp_path)
    researcher.fetch_research_pack(state)

    cls_path = Path(state["raw_dir"]) / "classification.json"
    assert cls_path.exists()
    cls = json.loads(cls_path.read_text())
    # Schema check — every documented key present
    for key in ("setup_class", "gap_to_50dma_pct", "gap_to_200dma_pct",
                "ma_alignment", "recent_volume_signal", "upside_target",
                "upside_pct", "downside_target", "downside_pct",
                "reward_risk_ratio", "rationale"):
        assert key in cls, f"classification.json missing key: {key}"
    # The stubbed fixture (spot=410, 50-DMA=405, 200-DMA=460) is bear-aligned
    # downtrend; should be DOWNTREND or one of the bear classes.
    assert cls["setup_class"] in {"CAPITULATION", "BREAKDOWN", "DOWNTREND"}
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_researcher.py::test_researcher_writes_classification_json -v
```

Expected: FAIL — `classification.json` not yet written.

- [ ] **Step 3: Modify researcher.py to call the classifier**

Open `tradingagents/agents/researcher.py`. Find the existing block that builds `reference` and writes `reference.json` (around the bottom of `fetch_research_pack`):

```python
    (raw / "reference.json").write_text(json.dumps(reference, indent=2, default=str), encoding="utf-8")
```

Add immediately after:

```python
    # Phase-6 stochasticity mitigation: pure-Python deterministic classifier.
    # See tradingagents/agents/utils/classifier.py + the design spec at
    # docs/superpowers/specs/2026-05-04-deterministic-classifier-design.md
    from tradingagents.agents.utils.classifier import compute_classification
    classification = compute_classification(reference, prices.get("ohlcv", ""))
    (raw / "classification.json").write_text(
        json.dumps(classification, indent=2, default=str), encoding="utf-8"
    )
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_researcher.py -v
```

Expected: all researcher tests pass (including the new `test_researcher_writes_classification_json`).

Then full suite:

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 142 + 9 (classifier) + 1 (researcher addition) = 152 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/agents/researcher.py tests/test_researcher.py
git commit -m "feat(researcher): write classification.json alongside reference.json"
```

---

## Task 3: TA Agent v1 + v2 inject DETERMINISTIC CLASSIFICATION block

**Files:**
- Modify: `tradingagents/agents/analysts/ta_agent.py`
- Modify: `tests/test_ta_agent.py`
- Modify: `tests/test_ta_agent_v2.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_ta_agent.py`, add at the bottom:

```python
def test_ta_agent_v1_substitutes_deterministic_classification(tmp_path):
    """If raw/classification.json exists, the TA agent must substitute its
    values into the SystemMessage so the LLM sees the deterministic class."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "prices.json").write_text(_json.dumps({"history_5y": []}), encoding="utf-8")
    (raw / "pm_brief.md").write_text("# Brief", encoding="utf-8")
    (raw / "reference.json").write_text(_json.dumps({
        "ticker": "MSFT", "reference_price": 410.0,
        "spot_50dma": 405.0, "spot_200dma": 460.0,
    }), encoding="utf-8")
    (raw / "classification.json").write_text(_json.dumps({
        "setup_class": "DOWNTREND",
        "upside_target": 460.0, "upside_pct": 12.20,
        "downside_target": 380.0, "downside_pct": -7.32,
        "reward_risk_ratio": 1.7,
        "rationale": "Stub rationale for test",
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="## Major historical levels\n\n# v1 stub")

    node = create_ta_agent_node(fake_llm)
    node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
    })

    # Inspect the system message that was sent to the LLM
    call_args = fake_llm.invoke.call_args
    system_msg = call_args.args[0][0].content
    assert "DETERMINISTIC CLASSIFICATION" in system_msg
    assert "DOWNTREND" in system_msg
    assert "460.0" in system_msg or "460.00" in system_msg
    assert "12.2" in system_msg or "+12.2" in system_msg
    assert "1.7" in system_msg
    assert "Stub rationale" in system_msg
```

In `tests/test_ta_agent_v2.py`, add at the bottom:

```python
def test_ta_agent_v2_substitutes_deterministic_classification(tmp_path):
    """TA Agent v2 must inject the same DETERMINISTIC CLASSIFICATION block
    so the v2 review can never flip the classification."""
    import json as _json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from tradingagents.agents.analysts.ta_agent import create_ta_agent_v2_node

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "technicals.md").write_text("# v1", encoding="utf-8")
    (raw / "reference.json").write_text(_json.dumps({"reference_price": 410.0}), encoding="utf-8")
    (raw / "prices.json").write_text(_json.dumps({"ohlcv": "..."}), encoding="utf-8")
    (raw / "classification.json").write_text(_json.dumps({
        "setup_class": "BREAKDOWN",
        "upside_target": 420.0, "upside_pct": 2.44,
        "downside_target": 380.0, "downside_pct": -7.32,
        "reward_risk_ratio": 0.3,
        "rationale": "v2 stub rationale",
    }), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = AIMessage(content="## Revisions from v1\n\n# v2 stub")

    node = create_ta_agent_v2_node(fake_llm)
    node({
        "company_of_interest": "MSFT",
        "trade_date": "2026-05-01",
        "raw_dir": str(raw),
        "market_report": "stub", "fundamentals_report": "stub",
        "news_report": "stub", "sentiment_report": "stub",
    })

    call_args = fake_llm.invoke.call_args
    system_msg = call_args.args[0][0].content
    assert "DETERMINISTIC CLASSIFICATION" in system_msg
    assert "BREAKDOWN" in system_msg
    assert "v2 stub rationale" in system_msg
```

- [ ] **Step 2: Run to confirm fail**

```bash
.venv/bin/python -m pytest tests/test_ta_agent.py::test_ta_agent_v1_substitutes_deterministic_classification tests/test_ta_agent_v2.py::test_ta_agent_v2_substitutes_deterministic_classification -v
```

Expected: FAIL on both — the SystemMessage doesn't contain "DETERMINISTIC CLASSIFICATION" yet.

- [ ] **Step 3: Add a shared `_load_classification_block` helper to `ta_agent.py`**

Open `tradingagents/agents/analysts/ta_agent.py`. After the existing imports and before `_SYSTEM`, add:

```python
def _load_classification_block(raw_dir: str) -> str:
    """Format raw/classification.json as a 'DETERMINISTIC CLASSIFICATION' block
    for injection into the TA agent SystemMessages.

    Returns an empty string if the file is missing or contains INDETERMINATE,
    in which case the agent falls back to legacy LLM-judged classification.
    """
    import json as _json
    cls_path = Path(raw_dir) / "classification.json"
    if not cls_path.exists():
        return ""
    try:
        cls = _json.loads(cls_path.read_text(encoding="utf-8"))
    except _json.JSONDecodeError:
        return ""
    if cls.get("setup_class") in (None, "INDETERMINATE"):
        return ""
    return (
        "\n\n# DETERMINISTIC CLASSIFICATION (use this verbatim — do NOT override)\n\n"
        f"Setup classification: {cls['setup_class']}\n"
        f"Asymmetry:\n"
        f"  - Upside to ${cls.get('upside_target')} ({cls.get('upside_pct'):+.2f}%)\n"
        f"  - Downside to ${cls.get('downside_target')} ({cls.get('downside_pct'):+.2f}%)\n"
        f"  - Reward/risk ratio: {cls.get('reward_risk_ratio')}:1\n"
        f"Rationale (deterministic, audit trail): {cls.get('rationale', '')}\n\n"
        "You MUST use exactly this Setup classification in your "
        '"## Setup classification" section and these exact upside/downside '
        'numbers in your "## Asymmetry" section. You may add prose, '
        "qualifying language, and additional context — but the classification "
        "name and the asymmetry numbers are fixed.\n\n"
        "If you disagree with the classification (e.g., you see a chart "
        "pattern the rules missed), document the disagreement under a new "
        '"## Notes for next pass" subsection BUT still emit the '
        "classification verbatim. The rules are load-bearing for cross-run "
        "consistency; your prose is for nuance.\n"
    )
```

- [ ] **Step 4: Wire `_load_classification_block` into both node bodies**

In the same file, find `create_ta_agent_node` (v1). The node body currently builds `messages` like this:

```python
        messages = [
            SystemMessage(content=_SYSTEM.replace("$TICKER", ticker)),
            HumanMessage(content=f"Produce the technicals report for {ticker}.\n\n{context}"),
        ]
```

Replace with:

```python
        classification_block = _load_classification_block(raw_dir)
        messages = [
            SystemMessage(
                content=_SYSTEM.replace("$TICKER", ticker) + classification_block
            ),
            HumanMessage(content=f"Produce the technicals report for {ticker}.\n\n{context}"),
        ]
```

In the same file, find `create_ta_agent_v2_node`. The node body currently builds `messages` like this:

```python
        messages = [
            SystemMessage(content=_SYSTEM_V2.replace("$TICKER", ticker).replace("$DATE", date)),
            HumanMessage(content=(
                f"Produce the v2 technicals report for {ticker} on {date}. "
                f"Below are the v1 setup, the four analyst reports, and the "
                f"reference snapshot. Refine and emit v2.\n\n"
                f"{v1_context}\n{analyst_block}"
            )),
        ]
```

Replace with:

```python
        classification_block = _load_classification_block(raw_dir)
        messages = [
            SystemMessage(
                content=_SYSTEM_V2.replace("$TICKER", ticker).replace("$DATE", date)
                + classification_block
            ),
            HumanMessage(content=(
                f"Produce the v2 technicals report for {ticker} on {date}. "
                f"Below are the v1 setup, the four analyst reports, and the "
                f"reference snapshot. Refine and emit v2.\n\n"
                f"{v1_context}\n{analyst_block}"
            )),
        ]
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ta_agent.py tests/test_ta_agent_v2.py -v
```

Expected: all pass.

Then full suite:

```bash
.venv/bin/python -m pytest -q -m unit --tb=line
```

Expected: 152 + 2 (TA agent v1/v2 classification tests) = 154 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/analysts/ta_agent.py tests/test_ta_agent.py tests/test_ta_agent_v2.py
git commit -m "feat(ta-agent): inject DETERMINISTIC CLASSIFICATION block from raw/classification.json"
```

---

## Task 4: E2E validation on macmini

**Files:** none (operator step)

- [ ] **Step 1: Push + redeploy**

```bash
git push origin main
ssh macmini-trueknot 'cd ~/tradingagents && git pull origin main && .venv/bin/pip install -e . --quiet'
ssh macmini-trueknot 'cd ~/tradingagents && git rev-parse --short HEAD'
```

Expected: HEAD matches the commit from Task 3.

- [ ] **Step 2: Refresh OAuth + run MSFT 2026-05-01**

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min. Verify done:

```bash
ssh macmini-trueknot 'pgrep -fl tradingresearch | head -3'  # empty
```

- [ ] **Step 3: Inspect classification.json**

```bash
ssh macmini-trueknot 'cat ~/.openclaw/data/research/2026-05-01-MSFT/raw/classification.json'
```

Expected output: a valid JSON dict with `setup_class` set to one of {CAPITULATION, BREAKDOWN, DOWNTREND, CONSOLIDATION, UPTREND, BREAKOUT}, populated upside/downside targets, and a non-empty rationale. For MSFT 2026-05-01 (spot $407.78, 50-DMA $396.11, 200-DMA $466.64), expect DOWNTREND or CAPITULATION.

- [ ] **Step 4: Verify TA Agent v1 + v2 use the deterministic classification**

```bash
ssh macmini-trueknot '
echo "=== TA v1 setup classification section ==="
grep -A 2 "## Setup classification" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals.md | head -5
echo
echo "=== TA v2 setup classification section ==="
grep -A 2 "## Setup classification" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals_v2.md | head -5
echo
echo "=== TA v1 asymmetry section ==="
grep -A 5 "## Asymmetry" ~/.openclaw/data/research/2026-05-01-MSFT/raw/technicals.md | head -8
'
```

Expected: both TA v1 and TA v2 list the same `setup_class` value as `classification.json`. The asymmetry numbers in TA v1 should match `upside_target`/`downside_target`/`reward_risk_ratio` from `classification.json`.

- [ ] **Step 5: Run a SECOND time and confirm classification.json is byte-identical**

```bash
ssh macmini-trueknot '~/.nvm/versions/node/v24.14.1/bin/claude -p hi'
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-classifierA'
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT
~/local/bin/tradingresearch \
  --ticker MSFT --date 2026-05-01 \
  --output-dir ~/.openclaw/data/research/2026-05-01-MSFT \
  --telegram-notify=-1003753140043
'
```

Wait ~14-18 min. Then:

```bash
ssh macmini-trueknot 'cp -R ~/.openclaw/data/research/2026-05-01-MSFT ~/.openclaw/data/research/2026-05-01-MSFT-classifierB'
ssh macmini-trueknot '
diff -q \
  ~/.openclaw/data/research/2026-05-01-MSFT-classifierA/raw/classification.json \
  ~/.openclaw/data/research/2026-05-01-MSFT-classifierB/raw/classification.json
'
```

Expected output: no diff (byte-identical). The classifier is pure Python from deterministic inputs, so this MUST pass.

- [ ] **Step 6: Confirm the LLM honored the classification**

For each archived run, verify the TA v1's "## Setup classification" section names the same value as the classifier:

```bash
ssh macmini-trueknot '
for run in classifierA classifierB; do
  echo "=== Run $run ==="
  echo "Classifier said: $(python3 -c "import json; print(json.load(open(\"~/.openclaw/data/research/2026-05-01-MSFT-${run}/raw/classification.json\".replace(\"~\", \"/Users/trueknot\")))[\"setup_class\"])")"
  echo "TA v1 said: $(grep -A 2 "## Setup classification" ~/.openclaw/data/research/2026-05-01-MSFT-${run}/raw/technicals.md | tail -1)"
done
'
```

Expected: classifier and TA v1 strings match for both runs.

- [ ] **Step 7: Cleanup archived dirs**

```bash
ssh macmini-trueknot '
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-classifierA
rm -rf ~/.openclaw/data/research/2026-05-01-MSFT-classifierB
'
```

- [ ] **Step 8: Report findings**

Convergence check:
- ✅ `classification.json` byte-identical between runs
- ✅ Both runs' TA v1 + TA v2 cite the same `setup_class`
- ✅ Final ratings still vary (LLM still writes synthesis prose), but the load-bearing technical classification is now stable

If classification.json differs between runs → debug (should never happen; pure Python from deterministic inputs).

If TA v1 or v2 emits a DIFFERENT `setup_class` than classification.json → the LLM ignored the "use verbatim" instruction. Add to follow-up: extend the QC Agent's Item 14 to also verify `technicals.md` and `technicals_v2.md` cite the deterministic classification, OR move the substitution to a post-LLM verifier.

If everything is consistent: ship as the new default. Stochasticity in the technical-classification axis is now eliminated; remaining variance lives in synthesis prose (acceptable).

---

## Self-review notes

**Spec coverage:**

- ✅ Pure-Python module (`tradingagents/agents/utils/classifier.py`) — Task 1
- ✅ 6-class taxonomy with documented thresholds — Task 1, Step 3
- ✅ Asymmetry math per class — Task 1, Step 3 (`_compute_asymmetry`)
- ✅ INDETERMINATE fallback when reference fields are null — Task 1, `_INDETERMINATE` constant + null check
- ✅ Researcher writes `raw/classification.json` — Task 2
- ✅ TA Agent v1 + v2 substitute the DETERMINISTIC CLASSIFICATION block — Task 3
- ✅ HARD ground-truth instruction in the prompt block — Task 3, `_load_classification_block` template
- ✅ Fall back to legacy LLM judgment when `classification.json` is missing or INDETERMINATE — Task 3, `_load_classification_block` returns `""` in those cases
- ✅ Unit tests per class — Task 1
- ✅ E2E validation: classification.json byte-identical across runs — Task 4

**Type / signature consistency:**

- `compute_classification(reference: dict, ohlcv_csv: str) -> dict` — defined in Task 1, called in Task 2 with `prices.get("ohlcv", "")`, output structure matches Task 3's `_load_classification_block` reads (`cls["setup_class"]`, `cls.get("upside_target")`, etc.)
- All 11 documented classification dict keys are present in `_INDETERMINATE`, in `compute_classification`'s return, and in the test's `must_contain` assertion (Task 2 test).
- `_load_classification_block(raw_dir: str) -> str` — defined in Task 3 Step 3, called in both `create_ta_agent_node` (v1) and `create_ta_agent_v2_node` (v2) Step 4.

**Placeholder scan:**

- No "TBD" / "implement later" / "similar to Task N" patterns.
- Each step shows complete code, exact file paths, and exact verification commands.
- Asymmetry math is fully specified per class (no "tune later" notes).

**Out-of-scope confirmation:** Multi-run consensus, empirical historical anchoring, soft-anchor mode, pattern recognition (head-and-shoulders, etc.), and multi-ticker calibration are all explicitly noted in the spec as separate workstreams. Nothing in this plan extends into those.

**Rollback path:** Each task commits independently. Reverting Task 3 (TA agent integration) leaves `classification.json` written but unused. Reverting Task 2 (Researcher integration) leaves the classifier module orphan but unused. Reverting Task 1 deletes the module entirely.
