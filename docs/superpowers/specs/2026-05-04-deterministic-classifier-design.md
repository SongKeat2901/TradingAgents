# Deterministic Classification Helper — Design

**Date:** 2026-05-04
**Status:** approved (user approved 2026-05-04)
**Predecessor:** [Temperature Pin Experiment (FAILED)](2026-05-04-temperature-pin-experiment-design.md)

## Goal

Remove the LLM's load-bearing technical-classification call from the variance window. The temperature-pin experiment proved Anthropic's API doesn't produce deterministic outputs at temperature=0 (Run A: UNDERWEIGHT, Run B: OVERWEIGHT on identical input). Replace the LLM-judged "Setup classification" + "Asymmetry" sections with a pure-Python rule engine that runs in the Researcher and is consumed by both TA Agents as ground truth. The LLM still writes prose (Volume profile, Current technical state narrative, Major historical levels with crowd psychology), but the directional classification and reward/risk arithmetic become deterministic.

## Motivation

The temperature-pin experiment (commit `69f7ff4`, reverted in `fdfebd6`) verified:

- `chat.temperature == 0.0` reaches the actual ChatAnthropic instance ✓
- Outputs still differ across runs (TA v1 113 diff lines, TA v2 163 diff lines, decision.md 317 diff lines)
- Final ratings flipped UNDERWEIGHT ↔ OVERWEIGHT on identical input

The conclusion: **the Anthropic API doesn't produce deterministic outputs at temperature=0** (hardware-level non-determinism in mixed-precision GPU arithmetic). The mitigation has to come from outside the LLM.

## Architecture

```
Researcher
  ├─ writes raw/reference.json (existing — numeric MA + ATR + YTD)
  ├─ writes raw/classification.json (NEW — deterministic class + asymmetry)
  ├─ writes raw/prices.json (existing — OHLCV CSV)
  └─ writes the other raw/ files (existing)

TA Agent v1
  ├─ reads raw/classification.json
  ├─ injects DETERMINISTIC CLASSIFICATION block into SystemMessage
  └─ LLM writes prose; classification + asymmetry are fixed

TA Agent v2 (post-analyst reconciliation)
  ├─ reads raw/classification.json (same value as v1; no recompute)
  ├─ injects same block + Revisions-from-v1 reasoning
  └─ LLM still writes prose; v2 cannot flip the classification
```

The classifier is **pure Python** — no LLM call, no I/O beyond reading `reference.json` + the OHLCV CSV string already in state. This makes it bit-exactly deterministic across runs.

## Components

### Rule taxonomy (6 classes, first match wins)

Order matters — earlier rules take precedence over later ones. Each rule names the data points it inspects, all of which exist in `raw/reference.json` or in the OHLCV CSV.

**1. CAPITULATION**
- Latest day's volume in top decile of last 90 trading days
- Latest day's price move > 1.5 × rolling 14-day standard deviation
- 50-DMA < 200-DMA (bear alignment confirms it's not just a one-day spike)

**2. BREAKDOWN**
- Spot < 50-DMA
- 50-DMA < 200-DMA
- gap_to_200dma > 8% below
- Latest-day volume > 1.5 × 50-day average volume

**3. DOWNTREND** (catch-all for the bear regime that isn't capitulation/breakdown)
- Spot < 200-DMA
- 50-DMA < 200-DMA

**4. CONSOLIDATION**
- |gap_to_50dma| < 3%
- |gap_to_200dma| < 8%
- Latest 10-day range < 1.5 × ATR-14 (price coiling)

**5. UPTREND**
- Spot > 200-DMA
- 50-DMA > 200-DMA

**6. BREAKOUT**
- Spot > 50-DMA
- 50-DMA crossed above 200-DMA in last 10 trading days
- Latest 5-day volume average > 90-day median volume

### Asymmetry math (per class)

| Class | Upside target | Downside target |
|---|---|---|
| CAPITULATION | 200-DMA | YTD low |
| BREAKDOWN | 50-DMA (recapture) | max(YTD low, spot × 0.90) |
| DOWNTREND | 200-DMA | YTD low |
| CONSOLIDATION | recent 30-day high | recent 30-day low |
| UPTREND | max(recent 30-day high, 200-DMA × 1.05) | 50-DMA |
| BREAKOUT | recent 30-day high × 1.08 | 50-DMA |

Computed quantities:
- `upside_pct = (upside_target - spot) / spot × 100`
- `downside_pct = (downside_target - spot) / spot × 100`
- `reward_risk_ratio = abs(upside_pct) / abs(downside_pct)` rounded to 1 decimal

### `tradingagents/agents/utils/classifier.py` (new module)

Pure function:

```python
def compute_classification(
    reference: dict,
    ohlcv_csv: str,
    history_window: int = 90,
) -> dict:
    """Deterministic technical-setup classifier.

    Inputs:
      reference: raw/reference.json contents (reference_price, spot_50dma,
                 spot_200dma, ytd_high, ytd_low, atr_14, trade_date)
      ohlcv_csv: get_stock_data return string (CSV with comment header)
      history_window: lookback for "top decile" and "50-day average" calcs

    Output dict (keys are stable; downstream agents read these verbatim):
      setup_class:           one of {CAPITULATION, BREAKDOWN, DOWNTREND,
                             CONSOLIDATION, UPTREND, BREAKOUT, INDETERMINATE}
      gap_to_50dma_pct:      float, signed (negative = below)
      gap_to_200dma_pct:     float, signed
      ma_alignment:          one of {bullish_aligned, bearish_aligned, mixed}
      recent_volume_signal:  one of {capitulation, breakout_volume,
                             above_average, below_average, normal}
      upside_target:         float (dollars)
      upside_pct:            float (signed)
      downside_target:       float
      downside_pct:          float (signed)
      reward_risk_ratio:     float
      rationale:             1-2 sentence string naming which rule fired
                             and the data points that triggered it
    """
```

INDETERMINATE: returned when reference data is missing/null OR none of the 6 rules match. Downstream agents fall back to LLM judgment in this case (rare — the rules cover the common regimes).

### Researcher integration

In `tradingagents/agents/researcher.py`, after building `reference`, add:

```python
from tradingagents.agents.utils.classifier import compute_classification

classification = compute_classification(reference, prices.get("ohlcv", ""))
(raw / "classification.json").write_text(
    json.dumps(classification, indent=2, default=str), encoding="utf-8"
)
```

Existing tests (`tests/test_researcher.py`) gain assertions on the new file's contents.

### TA Agent integration

Both `_SYSTEM` (v1) and `_SYSTEM_V2` (v2) gain a new section near the top:

```
# DETERMINISTIC CLASSIFICATION (use this verbatim — do NOT override)

Setup classification: $SETUP_CLASS
Asymmetry:
  - Upside to $UP_TARGET (+$UP_PCT%)
  - Downside to $DOWN_TARGET ($DOWN_PCT%)
  - Reward/risk ratio: $RR:1
Rationale (deterministic, audit trail): $RATIONALE

You MUST use exactly this Setup classification in your "## Setup classification"
section and these exact upside/downside numbers in your "## Asymmetry" section.
You may add prose, qualifying language, and additional context — but the
classification name and the asymmetry numbers are fixed.

If you disagree with the classification (e.g., you see a chart pattern the
rules missed), document the disagreement under a new "## Notes for next pass"
subsection BUT still emit the classification verbatim. The rules are
load-bearing for cross-run consistency; your prose is for nuance.
```

Both TA agents' node functions, before composing the prompt, read `raw/classification.json` and substitute `$SETUP_CLASS`, `$UP_TARGET`, etc. with the actual values.

### Files affected

| File | Change |
|---|---|
| `tradingagents/agents/utils/classifier.py` | NEW (~120 lines) — the rule engine |
| `tradingagents/agents/researcher.py` | MODIFY — call `compute_classification` + write `classification.json` |
| `tradingagents/agents/analysts/ta_agent.py` | MODIFY — both `_SYSTEM` constants gain the DETERMINISTIC CLASSIFICATION block; node bodies substitute values from `raw/classification.json` |
| `tests/test_classifier.py` | NEW (~120 lines) — one test per class with synthetic OHLCV fixtures |
| `tests/test_researcher.py` | MODIFY — assert `classification.json` written + has the documented keys |
| `tests/test_ta_agent.py`, `tests/test_ta_agent_v2.py` | MODIFY — verify the substitution lands in the prompt (with stub `classification.json` in tmp_path) |

## Data flow

1. Researcher fetches `prices.json` (OHLCV CSV string) + computes `reference.json` (numeric MA + ATR)
2. Researcher calls `compute_classification(reference, ohlcv_csv)` → produces a dict
3. Researcher writes `raw/classification.json`
4. TA Agent v1 fires:
   - reads `raw/classification.json`
   - substitutes the 6 placeholders into the SystemMessage
   - calls LLM
   - LLM writes the report — classification name and asymmetry numbers are constrained, prose is free
5. 4 analysts run (existing)
6. TA Agent v2 fires:
   - reads `raw/classification.json` (same file, same value)
   - same substitution + the existing "Revisions from v1" mandate
   - v2 prose may differ but the classification class itself cannot flip

## Tests

### Unit (`tests/test_classifier.py`)

One test per class, each with a synthetic OHLCV CSV that triggers exactly that class:

- `test_classifies_capitulation`: top-decile volume + >1.5σ daily move + 50<200 DMA → CAPITULATION
- `test_classifies_breakdown`: spot < 50-DMA < 200-DMA + 8%+ gap + volume spike → BREAKDOWN
- `test_classifies_downtrend`: bear MA alignment, no breakdown trigger → DOWNTREND
- `test_classifies_consolidation`: spot near both MAs + tight range → CONSOLIDATION
- `test_classifies_uptrend`: spot > 200-DMA + bull MA alignment → UPTREND
- `test_classifies_breakout`: recent 50-over-200 cross + volume confirmation → BREAKOUT
- `test_indeterminate_when_reference_has_nulls`: returns INDETERMINATE
- `test_first_match_wins`: a CAPITULATION-ish setup that also satisfies BREAKDOWN should classify as CAPITULATION
- `test_asymmetry_math_per_class`: spot-checks the upside/downside arithmetic for at least 3 classes

### Existing test updates

- `tests/test_researcher.py`: assert `classification.json` exists and has the documented keys after `fetch_research_pack`
- `tests/test_ta_agent.py` + `tests/test_ta_agent_v2.py`: stub `classification.json` in `tmp_path / "raw"` and assert the SystemMessage contains the substituted classification + numbers

### E2E validation

After ship, run MSFT 2026-05-01 twice on macmini:

- `raw/classification.json` should be **byte-identical** between runs (it's pure Python from identical inputs)
- The TA v1 + TA v2 reports should both contain the same `## Setup classification` line (e.g., "CAPITULATION") and the same asymmetry numbers
- The PM Final's "Technical setup adopted" subsection should cite the same v2 classification verbatim
- Final rating may still vary somewhat (the LLM still writes the synthesis prose) but the load-bearing technical read is now stable

## Failure modes

- **Rule misclassifies**: The classifier names a class that's plausibly wrong (e.g., calls a clear breakout "CONSOLIDATION" because the volume threshold isn't met). Trade-off accepted: deterministic > LLM-judgment-but-flips. Mitigation: tune thresholds on follow-up tickers.
- **OHLCV parse fails**: classifier returns INDETERMINATE; TA Agents fall back to LLM judgment for those sections (we can detect this case in the substitution code and use a "(no deterministic classification available)" fallback prompt).
- **Reference fields are null**: same fallback to INDETERMINATE.
- **LLM ignores the "must use verbatim" instruction**: real risk. Mitigation: QC Agent's Item 14 already verifies the PM cites v2's classification; we can add a similar check that TA's Setup classification matches `raw/classification.json`. Out of scope for v1; add if observed.

## Out of scope

- **Pattern recognition** (head-and-shoulders, double-tops, etc.) — pure rules can't classify these well; LLM stays in charge of those nuances via the prose
- **Multi-ticker calibration** — thresholds may need tuning for biotechs/SPACs/crypto-adjacent names; v1 is calibrated to mega-cap equities like MSFT
- **Soft-anchor mode** (LLM allowed to override with reasoning) — explicitly rejected; we're going hard ground truth
- **Multi-run consensus** (Option C from earlier) — separate workstream if even rule-based gives unstable downstream PM ratings
- **Empirical historical anchoring** (Option D) — separate workstream

## Trade-offs accepted

- 6 classes don't capture every chart pattern; edge cases fall to LLM prose
- Hand-calibrated thresholds may need tuning per asset class
- Hard ground truth means a wrong rule produces a wrong classification; we improve the rule rather than let the LLM override
- The Volume profile + Current technical state + Major historical levels sections remain LLM-written and stochastic; the deterministic part is bounded to classification + asymmetry math

## Rollback path

The change is additive: classifier.json is written but TA agent's substitution falls back to legacy LLM-judged sections if the file is missing or has INDETERMINATE class.

To roll back fully: revert one commit (the classifier integration commit). The new `raw/classification.json` files become orphan artifacts; harmless.

## Estimated effort

- Classifier module: ~3 hours (taxonomy + asymmetry math + edge cases)
- Researcher integration: ~30 min
- TA Agent prompt + node integration: ~1 hour (both v1 + v2 + substitution)
- Tests: ~2 hours (8-10 unit tests for classifier + updates to researcher/TA tests)
- E2E validation on macmini: ~30 min (one run; confirm classification.json byte-identical, TA reports match)
- Total: ~7 hours
